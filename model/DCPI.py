import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
import torch.nn.functional as F

from model.darkcnn import DarkNet
import warnings

warnings.filterwarnings("ignore")


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.ln = nn.Linear(out_features, 1)
        '''self.mlp = Mlp(
            in_features=out_features,
            hidden_features=out_features * 2,
            out_features=1,
            act_layer=nn.GELU
         )'''

    def forward(self, x):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        cosine = cosine
        return self.ln(cosine)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, image_token):
        B, N, C = image_token.shape
        kv = (
            self.kv(image_token)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        B, N, C = query.shape
        q = (
            self.q(query)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 定义自定义的融合模块（插入融合的部分）
class InjectModule(nn.Module):
    def __init__(self, dim, hid_dim):
        super(InjectModule, self).__init__()
        self.down_proj = Mlp(
            in_features=dim,
            hidden_features=int(hid_dim),
            out_features=int(hid_dim),
            act_layer=nn.GELU
        )
        self.cross_attn = CrossAttention(dim=int(hid_dim), num_heads=int(hid_dim / 4))
        self.up_proj = Mlp(
            in_features=int(hid_dim),
            hidden_features=dim,
            out_features=dim,
            act_layer=nn.GELU
        )
        self.scale_factor = nn.Parameter(
            torch.randn(dim) * 0.02
        )

    def forward(self, x, y):
        x_down = self.down_proj(x)  # x.shape(bs, patch_num^2, 64)
        x_down = x_down + self.cross_attn(x_down, y)
        x_up = self.up_proj(x_down)
        x = x + x_up * self.scale_factor

        return x


def darkchannel(image, window_size=1):
    min_channel, _ = torch.min(image, dim=1, keepdim=True)
    return min_channel


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


class ModifiedSwinTransformer(nn.Module):
    def __init__(self, base_model, inject_modules):
        super().__init__()
        self.darknet = DarkNet()
        self.base_model = base_model
        self.inject_modules = nn.ModuleList(inject_modules)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        in_channels = [96, 192, 384, 768]
        kernel_sizes = [3, 4, 2, 1]
        strides = [8, 4, 2, 1]
        paddings = [1, 0, 0, 0]

        self.len = len(in_channels)
        self.final_channel = in_channels[0] + in_channels[3] + in_channels[2] + in_channels[1]
        # self.ln = NormedLinear(self.final_channel, self.final_channel)
        self.ln = nn.Linear(self.final_channel, 1)
        self.dp = nn.Dropout(0)
        self.BRCBlock = nn.ModuleList(
            [
                nn.Sequential(
                    # nn.BatchNorm2d(in_channels[i]),

                    nn.Conv2d(in_channels[i], in_channels[i], kernel_size=kernel_sizes[i]
                              , stride=strides[i], padding=paddings[i], bias=False),
                    nn.ReLU(),
                )
                for i in range(len(in_channels))
            ]
        )
        self.se = SE_Block(self.final_channel)

    def forward(self, x, y):
        dark_x = darkchannel(x)
        dark_additional = darkchannel(y)

        # 计算差分
        differential = dark_x - dark_additional

        # 使用差分和原始x继续模型的正常操作
        y = self.darknet(differential)
        x = self.base_model.patch_embed(x)
        for i, layer in enumerate(self.base_model.layers):
            # 使用注入模块在原有模型的每个阶段前融合额外输入
            B, H, W, C = x.shape
            # x = torch.reshape(x, (B, -1, C))
            x = x.permute(0, 3, 1, 2)
            x = self.inject_modules[i](x, y[i])
            x = x.permute(0, 2, 3, 1)
            x = torch.reshape(x, (B, H, W, C))
            n = x.permute(0, 3, 1, 2)
            if i == 1:
                catdata = self.BRCBlock[i - 1](n)
            elif i > 1:
                c = self.BRCBlock[i - 1](n)
                catdata = torch.cat((catdata, c), dim=1)
            x = layer(x)
        n = x.permute(0, 3, 1, 2)
        x = torch.cat((catdata, self.BRCBlock[self.len - 1](n)), dim=1)
        x = self.se(x)
        x = self.dp(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.ln(x)


        return x


def create_model():
    # 加载预训练模型
    base_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
    base_model.load_state_dict(torch.load('./model/model_weights.pth'))


    dim = [base_model.embed_dim, base_model.embed_dim, base_model.embed_dim * 2, base_model.embed_dim * 4]
    hid_dim = [64, 128, 256, 512]
    inject_modules = [
        InjectModule(dim=dim[i], hid_dim=hid_dim[i]) for i in range(4)
    ]

    modified = ModifiedSwinTransformer(base_model, inject_modules)
    return modified
