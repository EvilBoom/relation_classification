# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 9:47
# @Author : 张宝宇
# @Version：V 0.0
# @File : config.py
# @desc :
import os
# nyt 24 pp 48
num_class = 24
hidden_size = 768
batch_size = 16
max_length = 300
epoch = 100
lr = 0.00003
seed = 2021
# root_path = 'datasets'
# ************* nyt
train_path = os.path.join('D:/Projects/datasets/nyt/train_triples.json')
dev_path = os.path.join('D:/Projects/datasets/nyt/dev_triples.json')
# test_path =
rel2id_path = os.path.join('D:/Projects/datasets/nyt/rel2id.json')
# ************* baidu
# train_path = os.path.join(root_path, 'D:/Projects/datasets/pp/train_triples.json')
# dev_path = os.path.join(root_path, 'D:/Projects/datasets/pp/dev_triples.json')
# # test_path =
# rel2id_path = os.path.join(root_path, 'pp/rel2id.json')
# ************* 情感分析
# train_path = os.path.join('D:/Projects/datasets/sentiment/train_triples.json')
# dev_path = os.path.join('D:/Projects/datasets/sentiment/dev_triples.json')
# test_path = os.path.join('D:/Projects/datasets/sentiment/test_triples.json')
