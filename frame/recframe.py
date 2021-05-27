# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 10:07
# @Author : 张宝宇
# @Version：V 0.0
# @File : recframe.py
# @desc :
import logging

import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader.dataloaders import rec_dataloader


class RecFrame(nn.Module):
    def __init__(self, re_model, batch_size, lr,
                 max_epoch, train, val, rel2id):
        super().__init__()
        self.model = re_model
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
        self.optimizer = torch.optim.Adam(
            [{'params': self.ner_model.parameters()}, {'params': self.er_model.parameters()}], lr=self.lr)

    def train_start(self):
        for epoch in range(self.max_epoch):
            # Train
            self.train()
            train_loss = 0
            print(f"=== Epoch {epoch} train ===")
            t = tqdm(self.train_loader)
            # 把数据放进GPU
            for t_iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                self.model.zero_grad()
                tokens, tokens_id, att_mask, labels = data
                loss = self.model(tokens_id, attention_mask=att_mask, labels=labels)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            avg_loss = train_loss / len(t)
            logging.info(f"Epoch: {epoch}, train loss: {avg_loss}")
