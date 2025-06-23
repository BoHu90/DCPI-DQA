import os
import random
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader


class DehazeDataset(Dataset):
    def __init__(self, root_dir, locations, index, patch_num, transformc=None, transformd=None):
        self.root_dir = root_dir
        self.transformc = transformc
        self.transformd = transformd
        self.dehaze_images = []
        self.fog_images = []
        self.scores = []
        self.index = index
        self.patch_num = patch_num

        # 提前加载所有数据到内存
        self.load_data(locations)

    def load_data(self, locations):
        for s, (location, num_subfolders) in enumerate(locations.items()):
            if s in self.index:
                location_dir = os.path.join(self.root_dir, location)
                fog_dir = os.path.join(location_dir, 'fog')
                if os.path.isdir(fog_dir):
                    fog_images = [f for f in os.listdir(fog_dir) if f.endswith('.png')]
                    fog_images = sorted(fog_images, key=lambda x: int(x.split('_')[1].split('.')[0]))
                    for i, fog_image in enumerate(fog_images):
                        fog_image_path = os.path.join(fog_dir, fog_image)
                        subfolder_name = f"{location}_{i + 1}"
                        subfolder_path = os.path.join(location_dir, subfolder_name)
                        mat_path = os.path.join(subfolder_path, f"{subfolder_name}_scores.mat")
                        mat_content = scipy.io.loadmat(mat_path)
                        image_names = mat_content['imageScores'][0][0][0].tolist()
                        scores = mat_content['imageScores'][0][0][1]

                        for j in range(len(image_names)):
                            dehaze_image_name = image_names[j][0][0].replace("'", '')
                            sc = scores[j][0]
                            dehaze_image_path = os.path.join(subfolder_path, dehaze_image_name)
                            if os.path.isfile(dehaze_image_path):
                                # 加载图像并存储在内存中
                                dehaze_image = Image.open(dehaze_image_path).convert('RGB')
                                fog_image = Image.open(fog_image_path).convert('RGB')

                                # 存储图像和分数
                                self.dehaze_images.append(dehaze_image)
                                self.fog_images.append(fog_image)
                                self.scores.append((1.0001-sc))

        # 创建数据索引
        self.data = []
        for i in range(len(self.dehaze_images)):
            for j in range(self.patch_num):
                self.data.append((i, j))  # 只存储索引，避免重复数据

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 使用索引访问存储在内存中的图像和分数
        img_index, _ = self.data[idx]
        dehaze_image = self.dehaze_images[img_index]
        fog_image = self.fog_images[img_index]
        score = self.scores[img_index]

        # 应用变换
        if self.transformc:
            dehaze_image = self.transformc(dehaze_image)
        if self.transformd:
            fog_image = self.transformd(fog_image)

        return dehaze_image, fog_image, torch.tensor(score, dtype=torch.float)




def getexBeDDE(root_dir, patch_num, patch_size, index, batch_size=16, istrain=True):
    locations = {
        'beijing': 15,
        'changsha': 8,
        'chengdu': 26,
        'hangzhou': 11,
        'hefei': 9,
        'hongkong': 13,
        'lanzhou': 15,
        'nanchang': 23,
        'shanghai': 8,
        'shenyang': 20,
        'tianjing': 8,
        'wuhan': 11
    }
    if istrain:
        transformc = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transformd = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transformc = transforms.Compose([
            transforms.RandomCrop(size=patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transformd = transforms.Compose([
            transforms.RandomCrop(size=patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = DehazeDataset(root_dir=root_dir, locations=locations, index=index, transformc=transformc,
                            transformd=transformd, patch_num=patch_num)

    # resolutions = dataset.get_image_resolutions()
    # save_resolutions_to_csv(resolutions, 'exBeDDE_resolutions_count.csv')
    if istrain:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=istrain, num_workers=18, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=istrain, num_workers=18, pin_memory=True)

    return dataloader

if __name__ == "__main__":
    root_dir = '/home/dataset/exBeDDE'
    index = list(range(0, 12))
    batch_size = 1
    patch_num = 1  # Assume some number for patch_num
    patch_size = (224, 224)  # Assume some size for patch_size
    random.shuffle(index)
    train_index = index[0:int(round(0.8 * len(index)))]
    test_index = index[int(round(0.8 * len(index))):len(index)]
    dataloader = getexBeDDE(root_dir, patch_num, patch_size, train_index, batch_size)
    testloader = getexBeDDE(root_dir, 1, patch_size, test_index, batch_size, istrain=False)

    for dehaze_image, dcd_image, score in dataloader:
        print(dehaze_image.shape, dcd_image.shape, score)
        for i in range(batch_size):
            image = dcd_image[i]
            de = dehaze_image[i]
            image = image.permute(1, 2, 0).numpy()
            de = de.permute(1, 2, 0).numpy()
            plt.imshow(image)
            plt.show()
            plt.imshow(de)
            plt.show()
        break