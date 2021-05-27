# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 9:59
# @Author : 张宝宇
# @Version：V 0.0
# @File : recmodel.py
# @desc :
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig


class RecModel(nn.Module):
    def __init__(self, num_class, hidden_size):
        super(RecModel, self).__init__()
        self.num_labels = num_class
        model_config = BertConfig.from_pretrained("bert-base-chinese")
        self.bert = BertModel(model_config)
        self.dropout = nn.Dropout(0.1)
        self.lin1 = nn.Linear(hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, inputs, attention_mask, labels):
        outputs = self.bert(inputs, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        rel_out = self.lin1(sequence_output)
        rel_out = rel_out.sigmoid()
        outputs = (rel_out,)

        if labels is not None:
            # 获取label mask
            loss_mask = labels.gt(0)
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if loss_mask is not None:
                # 只留下label存在的位置计算loss
                active_loss = loss_mask.view(-1) == 1
                active_logits = rel_out.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(rel_out.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        # contain: (loss), scores
        return outputs
