#!/public/home/elpt_2024_000543/miniconda3/envs/grm/bin/python
# -*- encoding: utf-8 -*-


'''
:目标数据集
:用./Data/DataObj/2023_08_11_18_08_12_china.nc进行测试
:0.4*0.4-->0.01*0.01
'''


from utils import *
import time
import xarray as xr
import torch 
from datasets import SRDataset
from models import Generator
from scipy.ndimage import gaussian_filter


# 模型参数，一定要与训练时的生成器参数一致，否则会导致mismatched keys
large_kernel_size_g = 9    # 第一层卷积和最后一层卷积的核大小
small_kernel_size_g = 3    # 中间层卷积的核大小
large_stride_g = 1         # 第一层卷积的步长
small_stride_g = 1         # 中间层卷积的步长
large_padding_g = 4        # 第一层卷积的填充
small_padding_g = 1        # 中间层卷积的填充
n_channels_g = 1           # 中间层通道数
n_blocks_g = 8             # 残差模块数量  
upscale_factor = 100       # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载训练好的模型
pre_srgan = "./Data/SavedModels/pre_srgan.pth"
checkpoint = torch.load(pre_srgan, weights_only=True)
model=Generator(in_channels=n_channels_g,
                out_channels=n_channels_g,
                    upscale_factor=upscale_factor,
                    large_kernel_size=large_kernel_size_g,
                    small_kernel_size=small_kernel_size_g,
                    large_stride=large_stride_g,
                    small_stride=small_stride_g,
                    large_padding=large_padding_g,
                    small_padding=small_padding_g,
                    n_res=n_blocks_g)
model_state_dict = checkpoint['generator']
model = model.to(device)
model.load_state_dict(model_state_dict)
model.eval()

# 数据集参数
data_folder = './Data/JsonPath/'
batch_size = 1
downscale_dataset = SRDataset(data_folder,split='obj')
downscale_loader = torch.utils.data.DataLoader(downscale_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,         # 数据加载单线程
        pin_memory=True) 

# 加载数据
select_lonw = 106
select_lone = 110
select_lats = 26
select_latn = 30
select_lon = np.linspace(select_lonw, select_lone, (select_lone-select_lonw+1)*100)
select_lat = np.linspace(select_lats, select_latn, (select_latn-select_lats+1)*100)


# 记录时间
start = time.time()

# 高斯滤波参数
sigma = 15

# 用训练好的模型推理，注意，是需要裁剪后的数据
# unsqueeze(0)是加0，squeeze是加0
with torch.no_grad():

    for i, (lr_pre, hr_pre) in enumerate(downscale_loader):
        hr_pre = hr_pre.to(device)   
        hr_pre = hr_pre.unsqueeze(1) 
        hr_pre_cropped = hr_pre[0,0,80:401:80,80:401:80].unsqueeze(0)
        hr_pre_cropped = hr_pre_cropped.unsqueeze(0)
        hr_pre=hr_pre.squeeze(0).squeeze(0).cpu().numpy()

        
        sr_pre = model(hr_pre_cropped).squeeze(0).cpu().detach()
        sr_pre = torch.relu(sr_pre)
        sr_pre = sr_pre.squeeze(0)
        sr_pre = sr_pre.numpy()
        sr_pre = np.array(sr_pre, dtype=np.float32)
        sr_pre = gaussian_filter(sr_pre, sigma=sigma)
        sr_pre = sr_pre*0.3+hr_pre*0.7
        sr_pre=xr.Dataset({"tp": (["lat", "lon"], sr_pre)},
                    coords={"lon": select_lon, "lat": select_lat})
        
        sr_pre.to_netcdf(f"./Data/DataResults/ResultObj/{i:02d}_pre_sr.nc")

print('用时  {:.5f} 秒'.format(time.time()-start))
