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
        model_config = BertConfig.from_pretrained("bert-base-uncased")  # uncased chinese
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.bert = BertModel(model_config)

        self.dropout = nn.Dropout(0.1)
        self.lin1 = nn.Linear(hidden_size, self.num_labels)
        self.loss_f = nn.BCEWithLogitsLoss()

        # CNN
        self.multi_conv = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.f_cnn = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_labels)
        # bi-GRU
        self.bi_gru = nn.GRU(768, hidden_size=300, num_layers=2, bidirectional=True, dropout=0.01)
        self.gru_lin1 = nn.Linear(600, self.num_labels)

        # self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, inputs, attention_mask, labels=None):
        outputs, pooled = self.bert(inputs, attention_mask=attention_mask)
        # ******************** gru
        # out, h_n = self.bi_gru(outputs)
        # sequence_output = out[:, -1]
        # ********************* 加CNN
        # outputs = outputs.unsqueeze(1)
        # outputs = torch.cat([self.conv_and_pool(outputs, conv) for conv in self.multi_conv], 1)
        # sequence_output = outputs
        # *********************
        # 无CNN的时候
        sequence_output = pooled
        # *********************
        sequence_output = self.dropout(sequence_output)
        # gru
        # rel_out = self.gru_lin1(sequence_output)
        # ********************* 无CNN
        rel_out = self.lin1(sequence_output)
        # ********************* 有CNN
        # rel_out = self.f_cnn(sequence_output)
        # *********************
        rel_out = rel_out.sigmoid()
        outputs = (rel_out,)

        if labels is not None:
            # 获取label mask
            loss_mask = labels.gt(0)
            # loss = F.binary_cross_entropy(rel_out, labels.float(), reduction='none')
            # bce 加 sigmoid 综合
            # loss = self.loss_f(rel_out, labels.float())
            loss = self.mul_cat_cross(labels, rel_out)
            loss = loss.sum()/len(loss)
            # loss = (loss * loss_mask.float()).sum() / loss_mask.float().sum()
            outputs = (loss,) + outputs
        return outputs

    def mul_cat_cross(self, y_true, y_pred):
        """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
             1表示对应的类为目标类，0表示对应的类为非目标类。
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss
