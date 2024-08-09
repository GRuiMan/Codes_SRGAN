#!/public/home/elpt_2024_000543/miniconda3/envs/grm/bin/python
# -*- encoding: utf-8 -*-


'''
:数据集文件datasets.py
:SRDataset类    
:数据集加载器,
:根据json路径中存着的降水数据路径读取数据,
:返回能被torch.utils.data.DataLoader加载的数据
'''

import xarray as xr
from torch.utils.data import Dataset
import json
import os
from utils import PreTransforms
import numpy as np
 
 
class SRDataset(Dataset):
    """
    :超分数据集加载器
    >>> data_folder = './Data/JsonPath/'
    >>> dataset = SRDataset(data_folder,split='obj')
    >>> lr_pre, hr_pre = dataset[0]
    """
 
    def __init__(self, data_folder, split,dataarray=False):
        """
        :data_folder:           Json数据文件所在文件夹路径
        :split:                 'train','eval','obj'
        """
 
        self.data_folder = data_folder
        self.dataarray = dataarray

        # 读取数据文件
        self.split = split.lower()
        if self.split == 'train':
            with open(os.path.join(data_folder,'PreTrain.json'), 'r') as j:
                self.Pre = json.load(j)
        elif self.split == 'eval':
            with open(os.path.join(data_folder,'PreEval.json'), 'r') as j:
                self.Pre = json.load(j)
        elif self.split == 'obj':
            with open(os.path.join(data_folder,'PreObj.json'), 'r') as j:
                self.Pre = json.load(j)
        else:
            raise ValueError("split must be one of {'train','eval','obj'}")

        # 数据处理方式
        self.transform = PreTransforms(split=self.split)
 
    def __getitem__(self, i):
        """
        :i: 降水数据检索号
        :降水数据读取,读取后用utils.py中的PreTransforms进行处理
        :使与PyTorch的DataLoader兼容
        :返回第i个低分辨率和高分辨率的降水数据对
        :后续将返回的hr_Pre, lr_Pre输入到网络中用于训练,np.array类型
        """
        Pre = xr.open_dataset(self.Pre[i])['tp']
        lr_pre, hr_pre = self.transform(Pre)
        
        if self.dataarray == True:
            return lr_pre, hr_pre
        
        else:
            lr_pre = np.array(lr_pre, dtype=np.float32)
            hr_pre = np.array(hr_pre, dtype=np.float32)
            return lr_pre, hr_pre
            
        
        
 
    def __len__(self):
        """
        :使与PyTorch的DataLoader兼容
        :加载的降水数据总数
        """
        return len(self.Pre)