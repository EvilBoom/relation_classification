# _*_ coding: utf-8 _*_
# @Time : 2021/5/30 19:59
# @Author : 张宝宇
# @Version：V 0.0
# @File : sentiment.py
# @desc :
import codecs
import json

import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('D:/Projects/datasets/sentiment/sentiment.valid.data', sep="\t", names=["text", "label"])
    data = data.values.tolist()
    temp_data = []
    for item in data:
        temp = {"text": item[0], "triple_list": [item[1], ]}
        temp_data.append(temp)
    # print(temp_data)
    with codecs.open('D:/Projects/datasets/sentiment/dev_triples.json', 'w', encoding='utf-8') as f:
        json.dump(temp_data, f, indent=4, ensure_ascii=False)
