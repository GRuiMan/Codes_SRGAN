'''
:处理spilt的数据集
:lr_pre是从元数据中截取的低分辨率数据
:hr_pre是lr_pre插值后的高分辨率数据
'''

from datasets import SRDataset

split = 'eval'  # 'train','eval','obj'

data_folder = './Data/JsonPath/'
downscale_dataset = SRDataset(data_folder,split=split,dataarray=True)

if split == 'obj':
    
    for i in range(len(downscale_dataset)):
        lr_pre, hr_pre = downscale_dataset[i]
        lr_pre.to_netcdf(f"./Data/DataResults/ResultObj/{i:02d}_pre_lr.nc")
        hr_pre.to_netcdf(f"./Data/DataResults/ResultObj/{i:02d}_pre_hr.nc")
        
        
elif split == 'eval':
    
    for i in range(len(downscale_dataset)):
        lr_pre, hr_pre = downscale_dataset[i]
        lr_pre.to_netcdf(f"./Data/DataResults/ResultEval/{i:02d}_pre_lr.nc")
        hr_pre.to_netcdf(f"./Data/DataResults/ResultEval/{i:02d}_pre_hr.nc")




