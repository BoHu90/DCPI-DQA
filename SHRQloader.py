# import csv
# import os
import random
# from itertools import chain
from concurrent.futures import ThreadPoolExecutor

import cv2
import scipy.io
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
# import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, dehaze_dirs, index, num_ways, transformc=None, transformd=None, fl=1, patch_num=1):
        self.dehaze_dirs = dehaze_dirs
        self.transformc = transformc
        self.transformd = transformd
        self.index = index
        self.num_ways = num_ways
        self.patch_num = patch_num
        self.dehaze_images = []
        self.hazy_images = []
        self.mos = []

        for dehaze_dir in dehaze_dirs:
            mat_file = dehaze_dir + "/" + "MOS.mat"
            mat_data = scipy.io.loadmat(mat_file)
            dehaze_names = [str(name[0]) for name in mat_data['Dehaze_name'].squeeze()]
            dehaze_names = [name.replace('\\', '/') for name in dehaze_names]
            if fl == 1:
                hazy_names = [str(name[0]) for name in mat_data['Hazy_name'].squeeze()]
            else:
                hazy_names = [str(name[0]) for name in mat_data['Haze_name'].squeeze()]
            hazy_names = [name.replace('\\', '/') for name in hazy_names]
            mos_scores = mat_data['MOS'].squeeze()

            for dehaze_name, hazy_name, mos in zip(dehaze_names, hazy_names, mos_scores):
                dehaze_path = dehaze_dir + '/' + dehaze_name
                hazy_path = dehaze_dir + '/' + hazy_name

                # 读取图像并存储在内存中
                dehaze_image = Image.open(dehaze_path).convert('RGB')
                hazy_image = Image.open(hazy_path).convert('RGB')

                self.dehaze_images.append(dehaze_image)
                self.hazy_images.append(hazy_image)
                self.mos.append(mos)

        self.data = []
        for i in self.index:
            for x in range(self.num_ways):
                for j in range(self.patch_num):
                    self.data.append((i * self.num_ways + x))

    def __len__(self):
        return len(self.index) * self.patch_num * self.num_ways

    def __getitem__(self, idx):
        data_idx = self.data[idx]
        dehaze_image = self.dehaze_images[data_idx]
        hazy_image = self.hazy_images[data_idx]
        mos_score = self.mos[data_idx]

        # 设置随机种子
        seed = torch.random.seed()

        torch.random.manual_seed(seed)
        dehaze_image = self.transformc(dehaze_image)

        torch.random.manual_seed(seed)
        hazy_image = self.transformd(hazy_image)

        return dehaze_image, hazy_image, torch.tensor(mos_score, dtype=torch.float)


def getSHRQ(dehaze_dirs, patch_num, patch_size, index, num_ways, batch_size=32, istrain=True):
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

    dataset = ImageDataset(dehaze_dirs, index, num_ways, transformc=transformc,
                           transformd=transformd, fl=1, patch_num=patch_num)
    if istrain:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)
    else:
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=20)

    return data_loader


# 主函数示例
if __name__ == "__main__":
    dehaze_dirs = ['/home/dataset/SHRQ/Regular']
    batch_size = 1
    index = list(range(0, 45))  # 600 0 1 2 3 4 5 6 7, 8 9 10 11 12 13 14 15
    patch_num = 2  # Assume some number for patch_num
    patch_size = (224, 224)  # Assume some size for patch_size
    num_ways = 8
    random.shuffle(index)
    train_index = index[0:int(round(0.8 * len(index)))]
    test_index = index[int(round(0.8 * len(index))):len(index)]
    data_loader = getSHRQ(dehaze_dirs, patch_num, patch_size, train_index, num_ways, batch_size)
    test_loader = getSHRQ(dehaze_dirs, patch_num, patch_size, test_index, num_ways, batch_size, istrain=False)

    # 测试数据加载器
    print(len(data_loader))
    train_list = []
    for dehaze_image, dcd_image, mos_score in data_loader:
        '''image = dcd_image[i]
        image = image.permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.show()
        rgb = dehaze_image[i]
        rgb = rgb.permute(1, 2, 0).numpy()
        plt.imshow(rgb)
        plt.show()'''


    print(len(test_loader))
    test_list = []
    for dehaze_image, dcd_image, mos_score in test_loader:
        '''image = dcd_image[i]
        image = image.permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.show()
        rgb = dehaze_image[i]
        rgb = rgb.permute(1, 2, 0).numpy()
        plt.imshow(rgb)
        plt.show()'''



