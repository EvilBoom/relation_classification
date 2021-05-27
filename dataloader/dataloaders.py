# _*_ coding: utf-8 _*_
# @Time : 2021/5/27 10:26
# @Author : 张宝宇
# @Version：V 0.0
# @File : dataloaders.py
# @desc :
import torch.utils.data as data

from dataloader.datas import RecDataset


def rec_dataloader(batch_size=None, shuffle=None, num_workers=0, path=None,
                   rel_dict_path=None, collate_fn=RecDataset.collate_fn):
    dataset = RecDataset(path=path, rel_dict_path=rel_dict_path)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader
