#!/public/home/elpt_2024_000543/miniconda3/envs/grm/bin/python
# -*- encoding: utf-8 -*-


'''
:模型模块文件 models.py
:需要按照降尺度的模型需求予以更改网络参数与网络结构!!!!
:包含了类:
:ConModel                                 卷积模块
:SRConModel                               超分卷积模块
:ResModel                                 残差模块
:SRResModel                               超分残差模型
:LineariZation                            线性化模块
:Generator                                生成器模块
:Discriminator                            判别器模块
:TruncatedVGG19                           截断VGG19模块

'''

from torch import nn
import torchvision
from torchvision.models import vgg19, VGG19_Weights

class ConModel(nn.Module):
    """
    :卷积模块,由卷积层, BN归一化层, 激活层构成.
    :输入和输出尺寸相同, 通道数不同.
    :(height,width) = (height,width - kernel_size + 2 * padding) // stride + 1
    :Model=ConModel(in_channels=1,out_channels=64,kernel_size=3,stride=1,batch_norm=True,activation='PReLu')
    :output = Model(input)
    """

    def __init__(self,in_channels,out_channels,
                 kernel_size, stride,padding,
                 batch_norm=False, activation=None):
        """
        :in_channels:                      输入通道数
        :out_channels:                     输出通道数
        :kernel_size:                      卷积核大小
        :stride:                           步长
        :padding:                          填充
        :batch_norm:                       是否包含BN层
        :activation:                       激活层类型; 如果没有则为None
        """
        super(ConModel, self).__init__()

        if activation ==False:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # 层列表
        layers = list()

        # 1个卷积层: 
        layers.append(  nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size, 
                      stride=stride,
                      padding=padding))

        # 1个BN归一化层,stabilize and speed up training
        if batch_norm == True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # 1个激活层
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # 按上述出现顺序合并层
        self.conv_Model = nn.Sequential(*layers)

    def forward(self, input):
        """
        :卷积模块如何处理输入.
        :input:输入图像集,张量表示,大小为(Num, in_channels, Width, Height)
        :output:输出图像集,张量表示,大小为(Num, in_channels, Width, Height)
        :It processes the input by passing it through each layer in sequence
        :the result is stored in output
        """
        output = self.conv_Model(input)
        return output


class SRConModel(nn.Module):
    """
    :超分卷积模型,包含3个卷积层,像素清洗层和激活层.
    :Model=SRConModel(kernel_size=3,padding=1,upscale_factor=100)
    :output = Model(input)

    """

    def __init__(self, in_channels,out_channels,upscale_factor,kernel_size,stride,padding):
        """
        :in_channels:                      输入通道数
        :out_channels:                     输出通道数
        :upscale_factor:                   放大系数
        :kernel_size:                      卷积核大小
        :stride:                           步长
        :padding:                          填充
        """
        super(SRConModel, self).__init__()

        # 首先通过卷积将通道数扩展为 scaling factor^2 倍
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * (upscale_factor ** 2),
                              kernel_size=kernel_size,stride=stride, padding=padding)

        # 进行像素清洗(b,r^2*C,W,H)->(b,C,W*r,H*r)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # 最后添加激活层
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        :前向传播.
        :input: 输入图像数据集,张量表示,大小为(N, n_channels, w, h)
        :ouputs: 输出图像数据集,张量表示,大小为 (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)               # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)     # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)             # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResModel(nn.Module):
    """
    :残差模型, 包含两个卷积模块和一个跳连.
    :Model=ResModel()
    :output = Model(input)
    """

    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        """
        :in_channels:                      输入通道数
        :out_channels:                     输出通道数
        :kernel_size:                      卷积核大小
        :stride:                           步长
        :padding:                          填充
        """

        super(ResModel, self).__init__()

        # 第一个卷积块
        self.conv_Model1 = ConModel(in_channels=in_channels,out_channels=out_channels,
                                    kernel_size=kernel_size,stride=stride,padding=padding,
                                    batch_norm=True, activation='PReLu')
        self.conv_Model2 = ConModel(in_channels=in_channels,out_channels=out_channels,
                                    kernel_size=kernel_size,stride=stride,padding=padding,
                                    batch_norm=True, activation=None)

    def forward(self, input):
        """
        :残差模块如何处理输入.
        :input          (N, n_channels, w, h)
        :output         (N, n_channels, w, h)
        """
        residual = input                    # (N, n_channels, w, h),跳过卷积层
        output = self.conv_Model1(input)    # (N, n_channels, w, h)
        output = self.conv_Model2(output)   # (N, n_channels, w, h)
        output = output + residual          # (N, n_channels, w, h)
        return output


class SRResModel(nn.Module):
    """
    :超分残差模型
    :Model=SRResModel(in_channels=in_channels,,,,,n_res=16,upscale_factor=8)
    :sr_Pre = Model(lr_Pre)
    """
    def __init__(self, in_channels,out_channels,upscale_factor,
                 large_kernel_size,small_kernel_size,
                 large_stride,small_stride,
                 large_padding,small_padding,
                 n_res):
        """
        :in_channels                 输入通道数
        :out_channels                输出通道数
        :upscale_factor              放大系数
        :large_kernel_size           第一个以及最后一层卷积块卷积核大小
        :small_kernel_size           中间残差模块卷积核大小
        :large_stride                第一个以及最后一层卷积块步长
        :small_stride                中间残差模块步长
        :large_padding               第一个以及最后一层卷积块填充
        :small_padding               中间残差模块填充
        :n_res                       残差模块数
        """

        super(SRResModel, self).__init__()

        # 第一个卷积块
        self.conv_Model1 = ConModel(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=large_kernel_size,
                                    stride=large_stride,
                                    padding=large_padding,
                                    batch_norm=False, 
                                    activation='PReLu')

        # 残差模块, 每个残差模块包含两个卷积层一个跳连层,这些层通过Sequential组合在一起
        self.residual_Models = nn.Sequential(*[ResModel(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=small_kernel_size,
                                                        stride=small_stride,
                                                        padding=small_padding) 
                                               for _ in range(n_res)])

        # 第二个卷积块
        self.conv_Model2 = ConModel(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=small_kernel_size,
                                    stride=small_stride,
                                    padding=small_padding,
                                    batch_norm=True, 
                                    activation=None)

        # 超分卷积模块
        self.sr_con_Models = nn.Sequential(*[SRConModel(in_channels=in_channels,
                                                         out_channels=out_channels,
                                                         upscale_factor=upscale_factor,
                                                         kernel_size=small_kernel_size,
                                                         stride=small_stride,
                                                         padding=small_padding)])

        # 最后一个卷积模块
        self.conv_Model3 = ConModel(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=large_kernel_size,
                                    stride=large_stride,
                                    padding=large_padding,
                                    batch_norm=False, 
                                    activation='ReLu')

    def forward(self, lr_Pre):
        """
        :超分残差模型处理输入.
        :lr_Pre: 低分辨率张量大小为 (1,1,71,41)
        :sr_Pre: 高分辨率张量大小为 (1,1,71*100,41*100)
        """
        output = self.conv_Model1(lr_Pre)                   #(1,1,71,41)
        residual = output                                   #(1,1,71,41)
        output = self.residual_Models(output)               #(1,1,71,41)
        output = self.conv_Model2(output)                   #(1,1,71,41)
        output = output + residual                          #(1,1,71,41)
        output = self.sr_con_Models(output)                #(1,1,71*100,41*100)  
        sr_Pre = self.conv_Model3(output)                   #(1,1,71*100,41*100)

        return sr_Pre

    


class Generator(nn.Module):
    """
    :生成器模型,其结构与SRResModel完全一致.
    """

    def __init__(self, in_channels,out_channels,
                    upscale_factor,
                    large_kernel_size,small_kernel_size,
                    large_stride,small_stride,
                    large_padding,small_padding,
                    n_res):

        """
        :in_channels                    输入通道数
        :out_channels                   输出通道数
        :upscale_factor                 放大比例
        :large_kernel_size              第一层和最后一层卷积核大小
        :small_kernel_size              中间层卷积核大小
        :stride                         步长
        :padding                       填充
        :n_res                          残差模块数量
        """
        super(Generator, self).__init__()
        self.net = SRResModel(in_channels=in_channels,
                            out_channels=out_channels,
                            upscale_factor=upscale_factor,
                            large_kernel_size=large_kernel_size, 
                            small_kernel_size=small_kernel_size, 
                            large_stride=large_stride,
                            small_stride=small_stride, 
                            large_padding=large_padding,
                            small_padding=small_padding,
                            n_res=n_res)

    def forward(self, lr_Pre):
        """
        :lr_Pre: 低精度图像                         (1, 1, 71, 41)
        :sr_Pre: 超分重建图像                       (1, 1, 71 * 100, 41 * 100)
        """
        sr_Pre = self.net(lr_Pre)                 #(N, n_channels, w * scaling factor, h * scaling factor)
        return sr_Pre


class Discriminator(nn.Module):
    """
    :SRGAN判别器
    """

    def __init__(self, kernel_size, n_channels, n_res, n_fc):
        """
        :kernel_size        所有卷积层的核大小
        :n_channels         初始卷积层输出通道数, 后面每隔一个卷积层通道数翻倍
        :n_res              卷积块数量
        :n_fc               全连接层连接数
        """
        super(Discriminator, self).__init__()

        # 卷积系列,参照论文SRGAN进行设计
        conv_blocks = list()
        in_channels = 1
        for i in range(n_res):
            out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels
            conv_blocks.append(
                ConModel(in_channels=in_channels, 
                         out_channels=out_channels, 
                         kernel_size=kernel_size,
                         stride=1 if i % 2 == 0 else 2, 
                         padding=1,
                         batch_norm=i != 0, 
                         activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # 固定输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, n_fc)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(n_fc, 1)

        # 最后不需要添加sigmoid层,因为PyTorch的nn.BCEWithLogitsLoss()已经包含了这个步骤

    def forward(self, input):
        """
        前向传播.

        :input         用于作判别的原始高清图或超分重建图,张量表示,大小为(N, 1, w * scaling factor, h * scaling factor)
        :logit       一个评分值, 用于判断一副图像是否是高清图, 张量表示,大小为 (N)
        """

        batch_size = input.size(0)
        output = self.conv_blocks(input)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit


class TruncatedVGG19(nn.Module):
    """
    :truncated VGG19网络
    :用于在训练中循环计算VGG特征空间的MSE损失
    """

    def __init__(self, i, j):
        """
        :i              第 i 个池化层
        :j              第 j 个卷积层
        """
        super(TruncatedVGG19, self).__init__()

        # 加载预训练的VGG模型C:\Users\hp/.cache\torch\hub\checkpoints\vgg19-dcbb9e9d.pth
        vgg19 = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        vgg19.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 改变输入为单通道

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # 迭代搜索
        for layer in vgg19.features.children():
            truncate_at += 1

            # 统计
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # 截断位置在第(i-1)个池化层之后（第 i 个池化层之前）的第 j 个卷积层
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # 检查是否满足条件
        assert maxpool_counter == i - 1 and conv_counter == j, "当前 i=%d 、 j=%d 不满足 VGG19 模型结构" % (
            i, j)
        

        # 截取网络
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])
        

    def forward(self, input):
        """
        :input      超分重建图,大小为 (N, 1, w * scaling factor, h * scaling factor)
        :output     VGG19特征图,大小为 (N, feature_map_channels, feature_map_w, feature_map_h)
        """

        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)
        return output
    





