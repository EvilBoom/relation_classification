# _*_ coding: utf-8 _*_
# @Time : 2021/5/28 20:58
# @Author : 张宝宇
# @Version：V 0.0
# @File : clean.py
# @desc :
import json
import codecs
import re
if __name__ == '__main__':
    data = json.load(open('../datasets/nyt/train_triples.json', encoding='utf-8'))
    ret_data = []
    pattern = '/(.*?)/(.*?)/(.*?)$'
    for item in data:
        spo_list = item['triple_list']
        flag = True
        for spo in spo_list:
            relation = re.search(pattern, spo[1]).group(3)
            spo[1] = relation
        item['triple_list'] = spo_list
    with codecs.open('../datasets/nyt/train_triples.json', 'w', encoding='utf-8') as f:
        json.dump(ret_data, f, indent=4, ensure_ascii=False)
