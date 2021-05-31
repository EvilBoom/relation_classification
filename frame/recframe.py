# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 10:07
# @Author : 张宝宇
# @Version：V 0.0
# @File : recframe.py
# @desc :
import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from model.recmodel import RecModel
from dataloader.dataloaders import rec_dataloader


class RecFrame(nn.Module):
    def __init__(self, batch_size, lr,
                 max_epoch, train, val, num_class, hidden_size, rel2id=None, test=None):
        super().__init__()
        self.model = RecModel(num_class, hidden_size).cuda()
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.train_path = train
        self.eva_path = val
        self.rel_dict = rel2id
        self.train_loader = rec_dataloader(path=self.train_path, rel_dict_path=self.rel_dict, shuffle=True,
                                           batch_size=self.batch_size)
        self.dev_loader = rec_dataloader(path=self.eva_path, rel_dict_path=self.rel_dict, shuffle=True,
                                         batch_size=self.batch_size)
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_start(self):
        for epoch in range(self.max_epoch):
            # Train
            self.model.train()
            train_loss = 0
            print(f"=== Epoch {epoch} train ===")
            t = tqdm(self.train_loader)
            # 把数据放进GPU
            for t_iter, data in enumerate(t):
                tokens, tokens_id, att_mask, labels = data
                # tokens.cuda()
                tokens_id = tokens_id.cuda()
                att_mask = att_mask.cuda()
                labels = torch.stack(list(labels), dim=0)
                labels = labels.cuda()
                out = self.model(tokens_id, attention_mask=att_mask, labels=labels)
                loss = self.loss_func(out, labels.float())
                train_loss += loss.item()
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            avg_loss = train_loss / len(t)
            logging.info(f"Epoch: {epoch}, train loss: {avg_loss}")
            self.model.eval()
            t = tqdm(self.dev_loader)
            pred_num, gold_num, correct_num = 1e-10, 1e-10, 1e-10
            # dev_losses = 0
            with torch.no_grad():
                for iter_s, batch_samples in enumerate(t):
                    tokens, tokens_id, att_mask, labels = batch_samples
                    tokens_id = tokens_id.cuda()
                    att_mask = att_mask.cuda()
                    labels = torch.stack(list(labels), dim=0)
                    labels = labels.cuda()
                    rel_out = self.model(tokens_id, attention_mask=att_mask)
                    # dev_losses += loss.item()
                    # 计算评价指标
                    labels = labels.to('cpu').numpy()
                    rel_out = rel_out.to('cpu').numpy()
                    for pre, gold in zip(rel_out, labels):
                        # 获取里面为 > 0.6 或者等于 1 的坐标
                        pre_set = set([int(i) for i in np.argwhere(pre > 0.6)])
                        gold_set = set([int(i) for i in np.argwhere(gold == 1)])
                        pred_num += len(pre_set)
                        gold_num += len(gold_set)
                        correct_num += len(pre_set & gold_set)
            print('正确个数', correct_num)
            print('预测个数', pred_num)
            precision = correct_num / pred_num
            recall = correct_num / gold_num
            f1_score = 2 * precision * recall / (precision + recall)
            print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1_score, precision, recall))
