# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 10:37
# @Author : 张宝宇
# @Version：V 0.0
# @File : dev_metric.py
# @desc :

def cat_word_piece(self, x):
    new_x = []
    for i in range(len(x) - 1):
        sub_x = x[i]
        rear = x[i + 1]
        new_x.append(sub_x)
        if "##" not in rear:
            new_x.append("[blank]")
    if len(x) > 0:
        new_x.append(x[-1])
    new_x = ''.join([i.lstrip("##") for i in new_x])
    new_x = ' '.join(new_x.split('[blank]'))
    return new_x


def metric_com():
    pass
