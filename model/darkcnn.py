from torch import nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        feature_channels = [256, 512, 1024, 2048]
        out_channels = [64, 128, 256, 512]
        pool_sizes = [7, 7, 7, 7]
        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            feature_channels[i],
                            out_channels[i],
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                        nn.GELU(),
                        nn.Conv2d(
                            out_channels[i],
                            out_channels[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        nn.AdaptiveAvgPool2d(pool_sizes[i]),
                    ]
                )
                for i in range(4)
            ]
        )

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def resize(self, y):
        B, C, _, _ = y.shape
        y = y.permute(0, 2, 3, 1).reshape(
            B, -1, C
        )
        return y

    def forward(self, x):
        y = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = self.conv[0](x)
        x1 = self.resize(x1)
        y.append(x1)

        x = self.layer2(x)
        x2 = self.conv[1](x)
        x2 = self.resize(x2)
        y.append(x2)

        x = self.layer3(x)
        x3 = self.conv[2](x)
        x3 = self.resize(x3)
        y.append(x3)

        x = self.layer4(x)
        x4 = self.conv[3](x)
        x4 = self.resize(x4)
        y.append(x4)

        return y


def DarkNet():
    return ResNet(Bottleneck, [3, 4, 6, 3])

