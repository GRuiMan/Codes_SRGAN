# Code Structure && Documentation

鸣谢：@钱彬，代码根据 @钱彬更改得来，来源于一下几篇博客与文章，基本框架与思路一致。有问题可联系3588430252@qq.com or 202183300708@nuist.edu.cn

## Idea Sources：

### Blogs:

[图像超分辨率重建：SRResNet算法原理详解-CSDN博客](https://blog.csdn.net/SehN1/article/details/135050472)

[图像超分辨率重建——SRGAN/SRResNet论文精读笔记-CSDN博客](https://blog.csdn.net/zency/article/details/127754042)

[图像超分辨率重建算法，让模糊图像变清晰(附数据和代码)_提高分辨率的算法-CSDN博客、](https://blog.csdn.net/qq_35054151/article/details/112415601)

[一文掌握图像超分辨率重建（算法原理、Pytorch实现）——含完整代码和数据_基于拼接图像超分辨率重构方法的设计代码-CSDN博客](https://blog.csdn.net/qianbin3200896/article/details/104181552)

### Papers

[Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf (thecvf.com)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)

[High-resolution downscaling with interpretable deep learning: Rainfall extremes over New Zealand - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2212094722001049?via%3Dihub)

## Data Files：

### DataTrain

训练数据，1950-2022年8月降水数据，31*73=2263个

### DataEval

评估数据，2023年8月降水数据,31*1=31个

### DataObj

**2023_08_11_18_08_12_china.nc**2023年8月11号18时预报的8月12号的中国全天的预报降水

**2023_08_12_06_08_13_china.nc**

**2023_08_13_12_08_14_china.nc**

**2023_08_14_18_08_15_china.nc**



### DataResults

**Historical_Records**之前由于损失函数以及batch_size选取有误导致的降尺度失败的文件

**ResultEval**根据评估数据集DataEval文件夹下进行降尺度后的数据文件，对于每个文件有pre_lr,pre_hr,pre_sr，00_pre_lr是2023年8月1日的降水低分辨率数据，以此类推。

**ResultObj**根据目标数据集DataObj文件夹下进行降尺度后的数据文件，命名规则与上类似。

2023_08_11_18_08_12_china.nc降尺度后的数据文件

### SavedModels

保存的模型参数为.pth文件

### JsonPath

保存降水数据文件路径

**PreTrain.json,**

**PreEval.json,**

**PreObj.json**

## Running Files

### 1create_data_lists.py

create the json path for the training,eval,test data.

### 2train.py

**Define the Models (Generator and Discriminator)**

**Define the Loss Functions**，

**Content Loss:instead of Mean Squared Error (MSE) Loss,Perceptual Loss**: Uses a pretrained network (e.g., VGG) to compute the difference between the feature representations of the generated and true images. **This can be useful for capturing perceptual differences that MSE might miss.**

**Define the Optimizers**

**Training Loop**

- Calculate the Discriminator Loss

- Update the Discriminator

- Calculate the Generator Loss

- Update the Generator

**Model–>Loss Function–>Optimizers–>Update Models(Implentation by loop)**



### 3downcale_eval.py

same to downscale_obj.py,but from 1$\cdot$1–>0.01$\cdot$0.01

**Load the Trained Model**

**Prepare the Target Data(eval data)**

**Apply the Model to Downscale the Data** 

**Save or Display the Downscaled Data**

### 4downscale_obj.py

same to downscale_obj.py,but from 0.4 $\cdot$0.4  to  0.01 $\cdot$0.01

**Load the Trained Model**

**Prepare the Target Data(obj data)**

**Apply the Model to Downscale the Data** 

**Save or Display the Downscaled Data**

### 5interpolation.py

produce low revolution and high revolution data

**Low resolution data**select the certain area(lonw=106E,lone=110E,lats=26N,latn=30N) of origin data(lonw=70E,lone=140E,lats=15N,latn=55N).

**High resolution data**interpolation form low resolution data.

if split='eval/train/',from 1$\cdot$1  to 0.01$\cdot$0.01

if split=‘obj’,from 0.4 $\cdot$0.4  to 0.01 $\cdot$0.01

### Eval_Correctness.ipynb

Correctness assessment of downscaled data(split=‘eval’ or ‘obj’).

Creat the Q-Q plots of eval and obj.

The sites selected for the downscaled data for the eval data were similar to the actual data, **eval files is good:smile:.**

but the sites selected for the downscaled data for the target data were far from the actual data.**Obj files seems to be bad:tired_face:.**

## Model Files

### datasets.py

**SRDataset：** 数据加载器，加载自己需要的文件，dataarray默认为False，即当训练、对目标数据集、评估数据集进行降尺度时，dataarray=False，返回np.array格式。当进行插值时，dataarray=True,返回xr.DataArray格式。用法：

```
data_folder = './Data/JsonPath/'
dataset = SRDataset(data_folder,split='obj',dataarray=true)
lr_pre, hr_pre = dataset[0]
lr_pre = datasets[0][0]
hr_pre = datasets[0][1]
lr_pre.to_netcdf("pre_lr.nc")
hr_pre.to_netcdf("pre_hr.nc")
```

### models.py

**ConModel**：卷积模型。卷积神经网络（CNN）的基本构件。它由一个卷积层组成，可选择在卷积层之后加入批量归一化和激活层。构造函数参数允许自定义输入和输出通道、内核大小、跨度、是否包含批量归一化以及激活函数的类型。该模块用途广泛，可用作更复杂模型的一个组件。

**SRConModel**：超分辨率卷积模型，可用于提高图像的分辨率。它首先通过卷积层增加输入的通道数，系数等于缩放因子的平方。然后，它使用像素洗牌操作，将输出重新排列成分辨率更高的图像，并减少通道数。最后，应用 ReLU 激活函数，对应降水数据来说确保生成都是正值。

**ResModel**：残差模型，是 ResNet 架构中的一个关键组件。它包含两个卷积块，带有一个跳转连接，可将残差块的输入添加到其输出中。这种设计有助于缓解深度网络中的梯度消失问题，并允许训练更深的模型。

**SRResModel**：超分辨率残差模型。它是一个专为提高图像分辨率而设计的模型。该网络从一个卷积块开始，然后是一系列残差块、另一个卷积块、一系列用于上采样的亚像素卷积块，最后是一个卷积块，以产生高分辨率输出。该模型的第一个和最后一个卷积块使用较大的内核尺寸，中间的卷积块使用较小的内核尺寸。

**Generator**：生成器模型,其结构与SRResModel完全一致。造数据用的

**Discriminator**：SRGAN判别器，判断数据是造的还是真的。

**TruncatedVGG19**：truncated VGG19网络。用于在训练中循环计算VGG特征空间的MSE损失。如果需要提高图像成超分辨率，则需要利用VGG计算特征图以实现降低感知损失(Preceived Loss)，如果为了提高像素准确性(pixel accuracy)，则不需利用VGG计算特征图。vgg19-dcbb9e9d.pth 需要首先放在主目录下.

WINDOWS下

```
 C:\Users\hp\.cache\torch\hub\checkpoints\vgg19-dcbb9e9d.pth 
```

LINUX下

```
/public/home/elpt_2024_000543/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
```



### utils.py

工具函数文件，所有项目中涉及到的一些自定义函数均放置在该文件中；

**create_data_lists** 函数用于扫描指定目录中符合最小尺寸要求的图像，并生成列出这些图像的 JSON 文件。该函数尤其适用于将数据集组织为训练集和测试集，这是机器学习工作流程中的常见要求。

**PreTransforms** 类提供了对数据应用特定变换（如裁剪和调整大小）的方法，以生成低分辨率和高分辨率图像对。这对于超分辨率等任务特别有用，因为在这些任务中，需要训练模型将图像从低分辨率提升到高分辨率。用法：

```
data=xr.open_dataset("./Data/DataTrain/1950_08_01.nc")
pre=data['tp']
lr_pre,hr_pre=PreTransforms(split='train')(pre)
```

**AverageMeter** 类是跟踪和计算数字流的平均值、总和和计数的实用程序。它可用于监控训练过程中的损失和准确度等指标，从而提供对模型性能的深入了解。

**clip_gradient **  其中 clip_gradient 可以将梯度剪切到指定范围，防止梯度爆炸

**adjust_learning_rate** 可以修改优化器的学习率，这对于执行学习率计划非常有用。



## Events Files

**train**     **train.e14516687**训练错误提示文件，直接文本文件打开。**train.o14516687**训练过程提示文件，直接文本文件打开。

**runs **      runs文件夹下存有事件文件：events.out.tfevents.1722695704.b3301r8n2.109176.0。保存训练过程中的内容损失loss_c，对抗损失中的生成器损失loss_g，对抗损失中的判别器损失loss_d。









