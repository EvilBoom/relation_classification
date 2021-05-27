# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 9:47
# @Author : 张宝宇
# @Version：V 0.0
# @File : config.py
# @desc :
import os

num_class = 24
hidden_size = 768
batch_size = 6
max_length = 300
epoch = 100
lr = 0.00003
seed = 2021
root_path = 'datasets'
train_path = os.path.join(root_path, 'nyt/train_triples_small.json')
dev_path = os.path.join(root_path, 'nyt/dev_triples_small.json')
# test_path =
rel2id_path = os.path.join(root_path, 'nyt/rel2id.json')
