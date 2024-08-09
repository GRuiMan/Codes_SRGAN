#!/public/home/elpt_2024_000543/miniconda3/envs/grm/bin/python
# -*- encoding: utf-8 -*-

'''
:生成路径文件create_data_lists.py
:运行后./Data/Path文件夹下生成4个json文件
:分别为PreTrain.json,PreTest.json,PreEval.json,PreObj.json
'''

from utils import create_data_lists
import multiprocessing

if __name__ == '__main__':
    create_data_lists(train_folders=['./Data/DataTrain'],
                      eval_folders=['./Data/DataEval'],
                      obj_folders=['./Data/DataObj'],
                      output_folder='./Data/JsonPath')
    
# 查看CPU核心数,确定工作线程数
num_cores = multiprocessing.cpu_count()
print(f'Number of CPU cores: {num_cores}')



