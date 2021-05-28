# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 10:26
# @Author : 张宝宇
# @Version：V 0.0
# @File : datas.py
# @desc :
import json

import numpy as np
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

BERT_MAX_LEN = 500


def seq_padding(batch, padding=0):
    length_batch = [len(seq) for seq in batch]
    max_length = max(length_batch)
    return np.array([
        np.concatenate([seq, [padding] * (max_length - len(seq))]) if len(seq) < max_length else seq for seq in batch
    ])


class RecDataset(data.Dataset):
    def __init__(self, path=None, rel_dict_path=None):
        super().__init__()
        self.max_len = 300
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.path = path
        # 生成 id 和 relation 映射
        print("加载数据")
        id_and_rel = json.load(open(rel_dict_path, encoding='utf-8'))
        id2rel = id_and_rel["id2rel"]
        rel2id = id_and_rel["rel2id"]
        id2rel = {int(i): j for i, j in id2rel.items()}
        self.id2rel = id2rel
        self.rel2id = rel2id
        print("加载完成")
        self.dataset = json.load(open(path, encoding='utf-8'))
        self.word_pad_idx = 0
        self.label_pad_idx = -1
        self.device = torch.device("cuda")
        # 初始化 sentences，labels，rel2id, id2rel
        self.num_rel = len(id2rel)
        print('加载数据\n')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        ret = self._tokenizer(item)
        return ret

    def _tokenize(self, tokens):
        re_tokens = ['[CLS]']
        for token in tokens.strip().split():
            re_tokens += self.bert_tokenizer.tokenize(token)
        re_tokens.append('[SEP]')
        return re_tokens

    def _tokenizer(self, line):
        # token 是分字后
        text = ' '.join(line['text'].split()[:self.max_len])
        tokens = self._tokenize(text)
        if len(tokens) > BERT_MAX_LEN:
            tokens = tokens[:BERT_MAX_LEN]
        text_len = len(tokens)
        # token_id
        token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) > text_len:
            token_ids = token_ids[:text_len]
        # attention_mask
        att_mask = torch.ones(len(token_ids)).long()
        # labels
        temps_labels = torch.zeros(24).long()
        labels = line['triple_list']
        for rel in labels:
            temps_labels[self.rel2id[rel[1]]] = 1
        return [tokens, token_ids, att_mask, temps_labels]

    @staticmethod
    def collate_fn(data):
        tokens, token_ids, att_mask, labels = list(zip(*data))
        tokens_batch = torch.from_numpy(seq_padding(token_ids)).long()
        att_mask_batch = pad_sequence(att_mask, batch_first=True, padding_value=0)
        return tokens, tokens_batch, att_mask_batch, labels
