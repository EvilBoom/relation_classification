# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 9:47
# @Author : 张宝宇
# @Version：V 0.0
# @File : config.py
# @desc :
import os
# nyt 24 pp 48
num_class = 48
hidden_size = 768
batch_size = 24
max_length = 300
epoch = 100
lr = 0.00003
seed = 2021
root_path = 'datasets'
# train_path = os.path.join(root_path, 'nyt/train_triples.json')
# dev_path = os.path.join(root_path, 'nyt/dev_triples.json')
# # test_path =
# rel2id_path = os.path.join(root_path, 'nyt/rel2id.json')
train_path = os.path.join(root_path, 'pp/train_triples_small.json')
dev_path = os.path.join(root_path, 'pp/dev_triples_small.json')
# test_path =
rel2id_path = os.path.join(root_path, 'pp/rel2id.json')

