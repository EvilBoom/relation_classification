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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_start(self):
        for epoch in range(self.max_epoch):
            # Train
            self.train()
            train_loss = 0
            print(f"=== Epoch {epoch} train ===")
            t = tqdm(self.train_loader)
            # 把数据放进GPU
            for t_iter, data in enumerate(t):
                self.model.zero_grad()
                tokens, tokens_id, att_mask, labels = data
                # tokens.cuda()
                tokens_id = tokens_id.cuda()
                att_mask = att_mask.cuda()
                labels = torch.stack(list(labels), dim=0)
                labels = labels.cuda()
                loss, _ = self.model(tokens_id, attention_mask=att_mask, labels=labels)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            avg_loss = train_loss / len(t)
            logging.info(f"Epoch: {epoch}, train loss: {avg_loss}")
            self.eval()
            t = tqdm(self.dev_loader)
            predict_num, correct_num = 1e-10, 1e-10
            dev_losses = 0
            with torch.no_grad():
                for iter_s, batch_samples in enumerate(t):
                    tokens, tokens_id, att_mask, labels = data = batch_samples
                    batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
                    label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
                    loss, _ = self.model(tokens_id, attention_mask=batch_masks, labels=labels)
                    dev_losses += loss.item()

                    # (batch_size, max_len - padding_label_len)
                    batch_output = self.model.crf.decode(batch_output, mask=label_masks)
                    # (batch_size, max_len)
                    batch_tags = batch_tags.to('cpu').numpy()
                    pred_tags = []
                    true_tags = []
                    pred_tags.extend([[self.id2rel.get(idx) for idx in indices] for indices in batch_output])
                    # (batch_size, max_len - padding_label_len)
                    true_tags.extend([[self.id2rel.get(idx) for idx in indices if idx > -1] for indices in batch_tags])
                    assert len(pred_tags) == len(true_tags)
                    pred_set = find_all(pred_tags)
                    true_set = find_all(true_tags)
                    predict_num += len(pred_tags)
                    correct_num += len(pred_set & true_set)
            print('正确个数', correct_num)
            print('预测个数', predict_num)
            precision = correct_num / predict_num
            recall = correct_num / predict_num
            f1_score = 2 * precision * recall / (precision + recall)
            print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1_score, precision, recall))
            return precision, recall, f1_score
