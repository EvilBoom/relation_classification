# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 9:59
# @Author : 张宝宇
# @Version：V 0.0
# @File : recmodel.py
# @desc :
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class RecModel(nn.Module):
    def __init__(self, num_class, hidden_size):
        super(RecModel, self).__init__()
        self.num_labels = num_class
        model_config = BertConfig.from_pretrained("bert-base-uncased")
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.bert = BertModel(model_config)
        self.multi_conv = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.lin1 = nn.Linear(hidden_size, self.num_labels)
        self.f_cnn = nn.Linear(self.num_filters*len(self.filter_sizes),self.num_labels)
        # self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, inputs, attention_mask, labels=None):
        outputs, pooled = self.bert(inputs, attention_mask=attention_mask)
        # # 加CNN
        outputs = outputs.unsqueeze(1)
        outputs = torch.cat([self.conv_and_pool(outputs, conv) for conv in self.multi_conv], 1)
        # 无CNN的时候
        # sequence_output = pooled
        # CNN
        sequence_output = outputs
        sequence_output = self.dropout(sequence_output)
        # 无CNN
        # rel_out = self.lin1(sequence_output)
        # 有CNN
        rel_out = self.f_cnn(sequence_output)
        rel_out = rel_out.sigmoid()
        outputs = (rel_out,)

        if labels is not None:
            # 获取label mask
            loss_mask = labels.gt(0)
            loss = F.binary_cross_entropy(rel_out, labels.float(), reduction='none')
            loss = (loss * loss_mask.float()).sum() / loss_mask.float().sum()
            # print(loss)
            outputs = (loss,) + outputs
        # contain: (loss), scores
        return outputs
