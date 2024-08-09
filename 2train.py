#!/public/home/elpt_2024_000543/miniconda3/envs/grm/bin/python
# -*- encoding: utf-8 -*-

'''
:超分生成对抗网络训练文件2train.py
:batch_size可以依据显存大小更改
'''

import numpy as np
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from models import TruncatedVGG19
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from models import Generator, Discriminator
from datasets import SRDataset
from utils import AverageMeter, adjust_learning_rate


# 数据集参数
data_folder = './Data/JsonPath/'    # 数据存放路径,Json文件存放路径
upscale_factor = 100                # 放大比例  

# 生成器模型参数(与SRResModel相同)
large_kernel_size_g = 9    # 第一层卷积和最后一层卷积的核大小
small_kernel_size_g = 3    # 中间层卷积的核大小
large_stride_g = 1         # 第一层卷积的步长
small_stride_g = 1         # 中间层卷积的步长
large_padding_g = 4        # 第一层卷积的填充
small_padding_g = 1        # 中间层卷积的填充
n_channels_g = 1           # 中间层通道数
n_blocks_g = 8             # 残差模块数量
vgg19_i = 5                # VGG19网络第i个池化层
vgg19_j = 4                # VGG19网络第j个卷积层


# 判别器模型参数
kernel_size_d = 3        # 所有卷积模块的核大小
n_channels_d = 1         # 第1层卷积模块的通道数, 后续每隔1个模块通道数翻倍
n_blocks_d = 8           # 卷积模块数量,由8改为6
n_fc_d = 1024            # 全连接层连接数,由1024改为768

# 学习参数
batch_size = 16     # 批大小
start_epoch = 1     # 迭代起始位置
epochs = 100         # 迭代轮数
checkpoint = None   # SRGAN预训练模型, 如果没有则填None
workers = 4         # 加载数据线程数量
beta = 1e-3         # 判别损失乘子
lr = 1e-4           # 学习率

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = 1                          # 用来运行的gpu数量，我的机器上只有一个gpu
cudnn.benchmark = True            # 对卷积进行加速
writer = SummaryWriter('./runs/') # 实时监控,默认在runs下,使用命令 tensorboard --logdir runs  进行查看

# 定制化的dataloaders
train_dataset = SRDataset(data_folder,split='train')
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=workers,
                                            pin_memory=True) 



def main():
    """
    训练.
    """
    global checkpoint,start_epoch,writer

    # 生成器和判别器,vgg19模型
    generator = Generator(in_channels=n_channels_g,
                          out_channels=n_channels_g,
                          upscale_factor=upscale_factor,
                          large_kernel_size=large_kernel_size_g,
                          small_kernel_size=small_kernel_size_g,
                          large_stride=large_stride_g,
                          small_stride=small_stride_g,
                          large_padding=large_padding_g,
                          small_padding=small_padding_g,
                          n_res=n_blocks_g)
    generator = generator.to(device)
                            

    discriminator = Discriminator(kernel_size=kernel_size_d,
                                    n_channels=n_channels_d,
                                    n_res=n_blocks_d,
                                    n_fc=n_fc_d)
    discriminator = discriminator.to(device)
    
    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19.eval()
    truncated_vgg19 = truncated_vgg19.to(device)

    # 单机多GPU训练
    if torch.cuda.is_available() and ngpu > 1:
        generator = nn.DataParallel(generator, device_ids=list(range(ngpu)))
        discriminator = nn.DataParallel(discriminator, device_ids=list(range(ngpu)))

    # 损失函数
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # 优化器
    optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad,generator.parameters()),lr=lr)
    optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad,discriminator.parameters()),lr=lr)

    # 开始逐轮训练
    for epoch in range(start_epoch, epochs+1):
        
        if epoch == int(epochs / 2):  # 执行到一半时降低学习率
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        generator.train()   # 开启训练模式：允许使用批样本归一化
        discriminator.train()

        losses_c = AverageMeter()  # 内容损失
        losses_g = AverageMeter()  # 对抗损失(生成损失)
        losses_d = AverageMeter()  # 对抗损失(判别损失)

        n_iter = len(train_loader) # 训练批次数

        # 按批处理
        for i, (lr_pre, hr_pre) in enumerate(train_loader):

            # 数据移至默认设备进行训练
            lr_pre = lr_pre.to(device)          # (batch_size (4),  width(11), height(11))
            hr_pre = hr_pre.to(device)          # (batch_size (4),  width(500), height(500))
            lr_pre = lr_pre.unsqueeze(1)        # (batch_size (4), channels(1), width(11), height(11))
            hr_pre = hr_pre.unsqueeze(1)        # (batch_size (4), channels(1), width(1100), height(1100))
           
            #-----------------------1. 生成器更新----------------------------
            # 生成
            sr_pre = generator(lr_pre)          # (batch_size (4), channels(1), width(500), height(500))
            
            sr_pre_in_vgg = truncated_vgg19(sr_pre)  # batchsize X 512 X 6 X 6
            hr_pre_in_vgg = truncated_vgg19(hr_pre)  # batchsize X 512 X 6 X 6
            
            
            # 计算内容损失
            content_loss = content_loss_criterion(sr_pre_in_vgg,hr_pre_in_vgg)

            # 计算对抗损失
            sr_discriminated = discriminator(sr_pre)                   # (batch,1)   
            adversarial_loss = adversarial_loss_criterion(
                sr_discriminated, torch.ones_like(sr_discriminated)) # 生成器希望生成的图像能够完全迷惑判别器，因此它的预期所有图片真值为1

            # 计算总的感知损失
            perceptual_loss = content_loss + beta * adversarial_loss

            # 后向传播.
            optimizer_g.zero_grad()
            perceptual_loss.backward()

            # 更新生成器参数
            optimizer_g.step()

            # 记录损失值
            losses_c.update(content_loss.item(), lr_pre.size(0))
            losses_g.update(adversarial_loss.item(), lr_pre.size(0))

            print(f'Epoch [{epoch}/{epochs}], Step [{i+1}/{n_iter}] started')
            print(f'Content Loss                    : {losses_c.avg:.4f}')
            print(f'Generator Adversarial Loss      : {losses_g.avg:.4f}') 
                           


            #-----------------------2. 判别器更新----------------------------
            # 判别器
            hr_discriminated = discriminator(hr_pre)
            sr_discriminated = discriminator(sr_pre.detach())

            # 对抗损失，二值交叉熵损失
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                            adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))  # 判别器希望能够准确的判断真假，因此凡是生成器生成的都设置为0，原始图像均设置为1

            # 后向传播
            optimizer_d.zero_grad()
            adversarial_loss.backward()

            # 更新判别器
            optimizer_d.step()

            # 记录损失
            losses_d.update(adversarial_loss.item(), hr_pre.size(0))

            # 判别器损失输出,
            print(f'Discriminator Adversarial Loss  : {losses_d.avg:.4f}') 

            # 监控图像变化，每一轮的最后一次保存
            if i==(n_iter-2):
                writer.add_image('SRGAN/epoch_'+str(epoch)+'_1_original', make_grid(lr_pre[:batch_size,:1,:,:].cpu(), nrow=4, normalize=False),epoch)
                writer.add_image('SRGAN/epoch_'+str(epoch)+'_2_original', make_grid(sr_pre[:batch_size,:1,:,:].cpu(), nrow=4, normalize=False),epoch)
                writer.add_image('SRGAN/epoch_'+str(epoch)+'_3_original', make_grid(hr_pre[:batch_size,:1,:,:].cpu(), nrow=4, normalize=False),epoch)

            # 打印结果
            print(f'Epoch [{epoch}/{epochs}], Step [{i+1}/{len(train_loader)}] completed')
 
        # 手动释放内存              
        del lr_pre, hr_pre, sr_pre, hr_discriminated, sr_discriminated  # 手工清除掉缓存

        # 监控损失值变化
        writer.add_scalar('SRGAN/Loss_c', losses_c.val, epoch) 
        writer.add_scalar('SRGAN/Loss_g', losses_g.val, epoch)  
        writer.add_scalar('SRGAN/Loss_d', losses_d.val, epoch)  


        # 保存预训练模型,多核运算用generator.model.state_dict()等
        torch.save({
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
        }, './Data/SavedModels/pre_srgan.pth')
    
    # 训练结束关闭监控
    writer.close()


if __name__ == '__main__':
    main()