#!/public/home/elpt_2024_000543/miniconda3/envs/grm/bin/python
# -*- encoding: utf-8 -*-



'''
:工具文件utils.py
:函数create_data_lists           创造训练集和测试集验证集列表文件
:类preTransforms                 对输入的降水数据进行变换
:类AverageMeter                  用于统计一组数据的平均值、累加和、数据个数
:函数clip_gradient               丢弃梯度防止计算过程中梯度爆炸
:函数adjust_learning_rate        调整学习率
'''

import os
import json
import numpy as np
import torch
import xarray as xr 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_data_lists(train_folders, eval_folders,output_folder,obj_folders):
    """
    :创建训练集和测试集列表文件.
    :train_folders: 训练文件夹集合; 各文件夹中的降水数据将被合并到一个图片列表文件里面
    :eval_folders: 评估文件夹集合; 每个文件夹将形成一个图片列表文件
    :output_folder: 最终生成的文件列表,json格式
    :obj_folders: 需要降尺度的数据文件放置的文件夹
    """
    print("\n正在创建文件列表... 请耐心等待.\n")

    train_pre = list()
    for d in train_folders:
        for i in os.listdir(d):
            pre_path = os.path.join(d, i)
            train_pre.append(pre_path)
    print("训练集中共有 %d 个训练降水数据\n" % len(train_pre))
    with open(os.path.join(output_folder, 'PreTrain.json'), 'w') as j:
        json.dump(train_pre, j)

    eval_pre = list()
    for d in eval_folders:
        for i in os.listdir(d):
            pre_path = os.path.join(d, i)
            eval_pre.append(pre_path)
    print("评估集中共有 %d 个评估降水数据\n" %(len(eval_pre)))
    with open(os.path.join(output_folder,'PreEval.json'),'w') as j:
        json.dump(eval_pre, j)

    obj_pre = list()
    for d in obj_folders:
        for i in os.listdir(d):
            pre_path = os.path.join(d, i)
            obj_pre.append(pre_path)
    print("目标集中共有 %d 个目标降水数据\n" %(len(obj_pre)))
    with open(os.path.join(output_folder,'PreObj.json'),'w') as j:
        json.dump(obj_pre, j)

    print("生成完毕。\n训练集、测试集、评估集、目标文件列表已保存在 %s 下\n" % output_folder)


class PreTransforms(object):
    """
    :降水数据处理,后续给dataset.py中的SRDataset类__getitem__方法调用,使与PyTorch的DataLoader兼容
    :利用train、test、eval训练
    :对obj降尺度
    >>>data=xr.open_dataset("./Data/DataTrain/1950_08_01.nc")
    >>>pre=data['tp']
    >>>lr_pre,hr_pre=PreTransforms(split='train')(pre)
    """

    def __init__(self, split):
        """
        :类的输入参数是split
        :split: 'train' ,'test', 'eval','obj'
        :train分辨率1°*1°
        :eval分辨率1°*1°
        :obj分辨率0.4°*0.4°
        """
        self.split = split.lower()
        assert self.split in {'train','eval','obj'}

    def __call__(self, pre):
        """
        :函数的输入参数pre
        :对降水数据进行裁剪和下采样形成低分辨率降水数据
        :pre: 由xarray库读取的降水数据
        :lonw: pre经度起始
        :lone: pre经度结束
        :lats: pre纬度起始
        :latn: pre纬度结束
        :返回DataArray形式的低分辨率和高分辨率降水数据
        
        """
        
        select_lonw=106
        select_lone=110
        select_lats=26
        select_latn=30
        select_lon = np.linspace(select_lonw, select_lone, (select_lone-select_lonw+1)*100)
        select_lat = np.linspace(select_lats, select_latn, (select_latn-select_lats+1)*100)
        
        # 如果pre的数据类型不是xarray.core.dataarray.DataArray将报错，因为方便sel处理，
        if not isinstance(pre, xr.DataArray):
            raise TypeError("pre must be of type xarray.core.dataarray.DataArray")
        
        if (self.split == 'train'or self.split == 'eval' or self.split == 'obj'):
            lr_pre = pre.sel(lon=slice(select_lonw, select_lone), lat=slice(select_latn, select_lats))
            hr_pre = lr_pre.interp(lon=select_lon, lat=select_lat,method='linear')
            
        else:
            raise ValueError("split must be one of {'train','eval','obj'}")
        
        return lr_pre, hr_pre


class AverageMeter(object):
    """
    跟踪记录类，用于统计一组数据的平均值、累加和、数据个数.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    丢弃梯度防止计算过程中梯度爆炸.

    :参数 optimizer: 优化器，其梯度将被截断
    :参数 grad_clip: 截断值
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    调整学习率.

    :optimizer: 需要调整的优化器
    :shrink_factor: 调整因子，范围在 (0, 1) 之间，用于乘上原学习率.
    """

    print("\n调整学习率.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("新的学习率为 %f\n" % (optimizer.param_groups[0]['lr'], ))
