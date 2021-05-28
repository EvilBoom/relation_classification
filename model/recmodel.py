# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 9:59
# @Author : 张宝宇
# @Version：V 0.0
# @File : recmodel.py
# @desc :
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class RecModel(nn.Module):
    def __init__(self, num_class, hidden_size):
        super(RecModel, self).__init__()
        self.num_labels = num_class
        model_config = BertConfig.from_pretrained("bert-base-chinese")
        self.bert = BertModel(model_config)
        self.dropout = nn.Dropout(0.1)
        self.lin1 = nn.Linear(hidden_size, self.num_labels)
        # self.init_weights()

    def forward(self, inputs, attention_mask, labels):
        outputs, pooled = self.bert(inputs, attention_mask=attention_mask)
        # sequence_output = outputs[0]
        sequence_output = pooled
        sequence_output = self.dropout(sequence_output)
        rel_out = self.lin1(sequence_output)
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
