import random
from itertools import chain

from matplotlib import pyplot as plt

from SHRQloader import ImageDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def getDHQ(dehaze_dirs, patch_num, patch_size, index, num_ways, batch_size=32, istrain=True):
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

    datasets = []
    dataset = ImageDataset(dehaze_dirs, index, num_ways, transformc=transformc,
                               transformd=transformd, fl=0, patch_num=patch_num)
    datasets.append(dataset)

    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    if istrain:
        data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=istrain, num_workers=16)
    else:
        data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return data_loader


if __name__ == "__main__":
    dehaze_dirs = ['/home/dataset/DHQ/DHD']
    batch_size = 1
    index = list(range(0, 75))
    num_ways = 7
    random.shuffle(index)
    train_index = index[0:int(round(0.8 * len(index)))]
    test_index = index[int(round(0.8 * len(index))):len(index)]
    data_loader = getDHQ(dehaze_dirs, 1, (224, 224), train_index, num_ways, batch_size)
    test_loader = getDHQ(dehaze_dirs, 1, (224, 224), test_index, num_ways, batch_size, istrain=False)
    train_list = []
    for dehaze_image, dcd_image, mos_score, imgindex in data_loader:
        train_list.append(imgindex)
        '''i = 1
        print(dehaze_image.size(), dcd_image.size(), mos_score.size(), mos_score)
        image = dcd_image[0]
        image = image.permute(1, 2, 0).numpy()
        plt.imshow(image, cmap='gray')
        plt.show()
        break'''
    test_list = []
    for dehaze_image, dcd_image, mos_score, imgindex in test_loader:
        test_list.append(imgindex)

    list1 = list(chain.from_iterable(train_list))
    list2 = list(chain.from_iterable(test_list))
    set1 = set(list1)
    set2 = set(list2)

    common_elements = set1.intersection(set2)
    set1 = sorted(list(set1))
    set2 = sorted(list(set2))
    print(common_elements)

