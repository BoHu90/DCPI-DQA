import csv
import random
import warnings
import time
import numpy as np
import pandas as pd
from scipy import stats
from SHRQloader import getSHRQ
from DHQloader import getDHQ
from exBeDDEloader import getexBeDDE

import torch
import torch.nn as nn
from model.DCPI import create_model as DCPI
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
locations = [15, 8, 26, 11, 9, 13, 15, 23, 8, 20, 8, 11]

def RMSE(list1, list2):
    # 确保两个列表长度相同
    if len(list1) != len(list2): raise ValueError("两个列表长度不相同")
    # 计算均方根误差
    mse = np.mean((np.array(list1) - np.array(list2)) ** 2)
    rmse = np.sqrt(mse)

    return rmse



class NetSolver(object):
    def __init__(self, path, config, image_list):
        self.image_list = image_list
        self.path = path
        self.epochs = config.epochs
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.patch_size = config.patch_size
        self.patch_num = config.patch_num
        self.test_patch_num = config.test_patch_num
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.train_test_num = config.train_test_num
        self.model_type = config.model_type
        self.flag = config.flag

        self.model_net = DCPI()
        print("创建DCPI")
        self.model_net = nn.DataParallel(self.model_net).to(device)
        # self.model_net = self.model_net.to(device)
        self.model_net.train(True)

        self.use_dark = config.use_dark

        self.l1_loss = nn.L1Loss().to(device)
        self.solver = torch.optim.Adam(self.model_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.solver, T_max=20, eta_min=0)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.solver, gamma=0.91)

        self.train_data, self.test_data = self.__load__()

    def __load__(self):
        random.shuffle(self.image_list)
        if self.dataset == 'exBeDDE':
            self.train_index = self.image_list[0:int(round(0.8 * len(self.image_list)))-1]
            self.test_index = self.image_list[int(round(0.8 * len(self.image_list)))-1:len(self.image_list)]
        else:
            self.train_index = self.image_list[0:int(round(0.8 * len(self.image_list)))]
            self.test_index = self.image_list[int(round(0.8 * len(self.image_list))):len(self.image_list)]
        if self.dataset == 'DHQ':
            data = getDHQ(self.path, self.patch_num, self.patch_size, self.train_index, 7, self.batch_size, True)
            data_test = getDHQ(self.path, self.test_patch_num, self.patch_size, self.test_index, 7, int(self.batch_size), False)
        elif self.dataset == 'SHRQA':
            data = getSHRQ(self.path, self.patch_num, self.patch_size, self.train_index, 8, self.batch_size, True)
            data_test = getSHRQ(self.path, self.test_patch_num, self.patch_size, self.test_index, 8, self.batch_size, False)
        elif self.dataset == 'SHRQR':
            data = getSHRQ(self.path, self.patch_num, self.patch_size, self.train_index, 8, self.batch_size, True)
            data_test = getSHRQ(self.path, self.test_patch_num, self.patch_size, self.test_index, 8, 1, False)
        elif self.dataset == 'SHRQ':
            data = getSHRQ(self.path, self.patch_num, self.patch_size, self.train_index, 8, self.batch_size, True)
            data_test = getSHRQ(self.path, self.test_patch_num, self.patch_size, self.test_index, 8, 1, False)
        else:
            data = getexBeDDE(self.path, self.patch_num, self.patch_size, self.train_index, self.batch_size, True)
            data_test = getexBeDDE(self.path, self.test_patch_num, self.patch_size, self.test_index, 1, False)
        return data, data_test

    def train(self, round):
        best_srcc = 0.0
        best_plcc = 0.0
        result = []
        headname = []
        data_len = len(self.train_data)
        print('Start training...')
        for t in range(self.epochs):

            epoch_loss = []
            pre_scores = []
            scores = []
            # if t >= 1:
            #     self.train_data, self.test_data = self.__load__()
            time1 = time.time()
            for i, (dehaze, darkc, labels) in enumerate(self.train_data):
                dehaze = dehaze.float().to(device)
                darkc = darkc.float().to(device)
                labels = labels.float().to(device)

                self.solver.zero_grad()

                pre_score = self.model_net(dehaze, darkc)
                pre_scores = pre_scores + pre_score.cpu().squeeze(1).tolist()
                # pre_score = self.model_net(dehaze)

                scores = scores + labels.cpu().tolist()
                loss = self.l1_loss(pre_score.squeeze(), labels.detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()
                n = 0
                if i == len(self.train_data) / 3 - 1 or i == (len(self.train_data) / 3) * 2 - 1 and n<=15:
                # if i == len(self.train_data) / 2 - 1 and r <= :
                    self.scheduler.step()
                    n+=1
                if ((i + 1) % 100) == 0 and i != 0:
                    steps = data_len
                    print(f'epoch: {t + 1}, steps: {i + 1}/{steps}, train_loss: {sum(epoch_loss) / len(epoch_loss)}')

            train_srcc = stats.spearmanr(pre_scores, scores)[0]
            test_srcc, test_plcc, test_rmse = self.test()
            result.append({"epoch": t + 1, "train_loss": sum(epoch_loss) / len(epoch_loss), "train_srcc": train_srcc,
                           "test_srcc": test_srcc, "test_plcc": test_plcc, "test_rmse": test_rmse})
            print(f"current loss: {loss.item()}")
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
            print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_RMSE')
            print('%d\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc,
                   test_srcc, test_plcc, test_rmse))
            if t == 0:
                print(
                    f"one epoch cost {time.time() - time1}, each batch cost {(time.time() - time1) / data_len} seconds")
            self.scheduler.step()

        dict = result[0]
        for headers in sorted(dict.keys()):  # 把字典的键取出来
            headname.append(headers)

        with open('%s%d_%s_%s_round%d_results.csv' % (self.flag, self.version, self.model_type, self.dataset, round + 1)
                , 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headname)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
            writer.writeheader()  # 写入列名
            writer.writerows(result)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self):
        self.model_net.train(False)
        pre_scores = []
        scores = []
        for dehaze, darkc, labels in self.test_data:
            dehaze = dehaze.float().to(device)
            darkc = darkc.float().to(device)
            labels = labels.float().to(device)

            pre_score = self.model_net(dehaze, darkc)
            pre_scores = pre_scores + pre_score.cpu().tolist()
            # pre_score = self.model_net(dehaze)

            scores = scores + labels.cpu().tolist()
        pre_scores = np.reshape(np.array(pre_scores), (-1, self.test_patch_num))
        pre_scores = np.mean(pre_scores, axis=1)
        df = pd.DataFrame(data=pre_scores, columns=['pre_score'])
        df.to_csv('scores.csv', index=False)
        scores = np.mean(np.reshape(np.array(scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pre_scores, scores)
        test_plcc, _ = stats.pearsonr(pre_scores, scores)
        test_rmse= RMSE(pre_scores, scores)
        self.model_net.train(True)
        return test_srcc, test_plcc, test_rmse
