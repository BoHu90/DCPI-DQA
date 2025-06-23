import csv
import argparse
import random
import numpy as np
from netsolver import NetSolver

import warnings
warnings.filterwarnings("ignore")

def main(config):
    datasets_path = {
        "DHQ": ['./dataset/DHQ/DHD'],
        "SHRQR": ['./dataset/SHRQ/Regular'],
        'SHRQA': ['./dataset/SHRQ/Aerial'],
        "exBeDDE": './dataset/exBeDDE',
        'SHRQ': ['./dataset/SHRQ/Regular', './dataset/SHRQ/Aerial'],
    }

    img_num = {
        "DHQ": list(range(0, 250)),
        "SHRQA": list(range(0, 3)),
        "SHRQR": list(range(0, 45)),
        "exBeDDE": list(range(0, 12)),
        'SHRQ': list(range(0, 75)),
    }
    image_list = img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=float)
    plcc_all = np.zeros(config.train_test_num, dtype=float)

    print(f'Training and testing on {config.dataset} for {config.train_test_num} rounds....')
    results = []
    head = []
    for i in range(config.train_test_num):
        print(f'Round {i + 1}')

        result = NetSolver(datasets_path[config.dataset], config, image_list)
        srcc_all[i], plcc_all[i] = result.train(i)
        results.append({"round": i + 1, "srcc": srcc_all[i], "plcc": plcc_all[i]})

    srcc_median = np.median(srcc_all)
    plcc_median = np.median(plcc_all)
    name = results[0]
    for i in sorted(name.keys()):
        head.append(i)

    with open('%s_%s_%s_results.csv' % (config.flag, config.dataset, config.model_type), 'w', newline='',
              encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=head)
        writer.writeheader()
        writer.writerows(results)

    print('Final test median SRCC: %4.4f, PLCC: %4.4f ' % (srcc_median, plcc_median))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='DHQ', help='Support datasets:DHQ|SHRQA'
                                                                                     '|exBeDDE|SHRQR|SHRQ')
    parser.add_argument('--patch_num', dest='patch_num', type=int, default=25,  help='Patch number')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Patch size')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Test Patch number')
    parser.add_argument('--use_dark', dest='use_dark', action='store_true', default=True, help='Use dark channel')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate  2e-5')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-6, help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=4, help='Epochs for training')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=6, help='Train-test times')
    parser.add_argument('--flag', dest='flag', type=str, default='PIM(iAFF)', help='Flag')

    config = parser.parse_args()
    main(config)

