# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 9:40
# @Author : 张宝宇
# @Version：V 0.0
# @File : re_cls_test.py
# @desc :
import argparse
import random

import numpy as np
import torch

from config import *
from frame.recframe import RecFrame


def seed_torch(m_seed=2021):
    random.seed(m_seed)
    np.random.seed(m_seed)
    torch.manual_seed(m_seed)


if __name__ == '__main__':
    # 设置参数，train 和 dataset
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--dataset', default='food_data', type=str,
                        help='specify the dataset from ["food_data","公开数据集"]')
    args = parser.parse_args()
    dataset = args.dataset
    is_train = args.train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(seed)
    framework = RecFrame(batch_size, lr,
                         epoch, train_path, dev_path, rel2id=rel2id_path, num_class=num_class, hidden_size=hidden_size)
    # framework = RecFrame(batch_size, lr, epoch, train_path, dev_path, test=test_path, num_class=num_class,
    #                      hidden_size=hidden_size)
    framework.train_start()
