B-CNN-MFM-SK for Peanut Leaf Disease Identification

官方 TensorFlow/Keras 实现 | 论文标题：《Spectral Classification of Symptomatic Peanut Leaves Based on Deep Learning Algorithms》提出 B-CNN-MFM-SK 融合模型，基于 TensorFlow 框架实现花生叶片四类状态（健康、炭疽病、焦斑病、疮痂病）高精度识别，兼顾轻量化部署与田间检测适应性，助力智慧农业作物病害防控。

Table of Contents

研究背景与模型定位

B-CNN-MFM-SK 核心创新点

实验数据集：花生叶片双模态光谱数据集

实验环境配置

代码使用说明

项目文件结构

已知问题与注意事项

引用与联系方式

1\. 研究背景与模型定位

花生作为重要经济作物，叶片病害（炭疽病、焦斑病、疮痂病等）直接导致产量下降 10%-30%。传统病害检测依赖人工目视鉴别与实验室生化分析，存在效率低（单样本耗时 3-5 分钟）、误判率高（依赖经验，误判率超 20%）、难以规模化田间监测的问题。

光谱检测技术凭借非破坏性、客观性、多波段特征等优势，已成为作物病害检测的核心技术。本文提出 B-CNN-MFM-SK（Bilinear Convolutional Neural Network-Maximum Feature Mapping-Adaptive Selection Kernel） 融合模型，整合三大核心模块协同优化：

双线性 CNN（B-CNN）适配双模态光谱特征融合；

最大特征映射（MFM）层避免光谱特征信息丢失；

自适应选择核（SK）模块强化病害光谱差异区分。

模型基于 TensorFlow 1.14.0 + Keras 2.2.4 框架实现，针对花生叶片四类状态（健康、炭疽病、焦斑病、疮痂病）的反射 / 荧光双模态光谱数据进行高精度分类，为田间移动检测设备部署、作物病害早期防控提供技术支撑。

2\. B-CNN-MFM-SK 核心创新点

2.1 双线性 CNN（B-CNN）：双模态光谱特征深度融合

针对花生叶片反射光谱（400-800nm）与荧光光谱（650-800nm）的信息互补特性，设计双并行 CNN 骨干网络：

单模态特征提取：CNN-A 处理反射光谱 GAF 图像，捕捉叶绿素吸收相关特征；CNN-B 处理荧光光谱 GAF 图像，聚焦光合效率关联特征；

双线性聚合运算：通过外积运算融合双模态特征，建立波段间关联，解决单一光谱特征区分度不足的问题，特征利用率提升 40%。

2.2 MFM 层：光谱特征信息保全

花生病害光谱的细微峰位差异（如 760nm 红边效应、685nm 荧光峰）是分类关键，MFM 层替代传统激活函数：

避免梯度消失：通过最大特征映射机制，保留浅层卷积的光谱基线特征与深层的病害特异性特征；

强化弱信号提取：针对感染初期叶片的光谱微弱变化，提升特征响应强度，为早期病害检测奠定基础。

2.3 自适应 SK 模块：动态匹配光谱特征

针对不同病害光谱的波段特异性（如焦斑病荧光强度最低、健康叶片反射光谱红边效应显著），设计 SK 卷积模块：

多核自适应选择：内置 2×2、3×3 小尺寸卷积核，自动匹配光谱局部特征（如病害特征峰）与全局趋势；

通道注意力机制：通过全局平均池化（GAP）与全局最大池化（GMP）并行压缩，聚焦高贡献通道，降低环境噪声（如光照波动）干扰。

2.4 GAF 光谱转换：1D 到 2D 的特征适配

引入 Gramian Angular Field（GAF）算法，将一维反射 / 荧光光谱序列转换为 32×32 像素的 2D 图像，保留光谱波段的时序关联性与强度差异，解决深度学习模型难以直接处理一维光谱数据的痛点，分类准确率提升 8%-10%。

3\. 实验数据集：花生叶片双模态光谱数据集

3.1 数据集概况

本研究数据集通过集成式双模态光谱检测系统采集，涵盖花生叶片四类核心状态，数据具有高一致性与非破坏性：

数据集名称	包含类别	光谱类型	样本总数	数据格式	图像分辨率（GAF 转换后）	数据分布（训练：验证：测试）

花生叶片光谱数据集	健康（Healthy）、炭疽病（Anthracnose）、焦斑病（Scorch spot）、疮痂病（Scab）	反射光谱（400-800nm）、荧光光谱（650-800nm）	9700+（反射光谱 5060+、荧光光谱 4640+）	原始光谱数据（.csv）+ GAF 2D 图像（.png）	32×32	训练：验证：测试 = 3:1:0.45（按总样本 10% 为测试集划分）

3.2 数据集获取与结构

3.2.1 下载方式

数据集可联系通讯作者获取（邮箱：caizhaopeng@huuc.edu.cn；zhaojunminhuuc@yeah.net），或通过科研数据平台：https://github.com/XLX-SHY/Symptom-Peanut-Leaf-Spectral-Classification.git

3.2.2 文件夹组织

解压后放置于项目根目录，结构如下：

plaintext

Peanut\_Leaf\_Dataset/

├── Reflectance\_spectra/  # 反射光谱数据

│   ├── Healthy/

│   ├── Anthracnose/

│   ├── Scorch\_spot/

│   └── Scab/

├── Fluorescence\_spectra/ # 荧光光谱数据

│   ├── Healthy/

│   ├── Anthracnose/

│   ├── Scorch\_spot/

│   └── Scab/

└── GAF\_images/           # 预处理后的GAF 2D图像（自动生成）

    ├── Reflectance/

    └── Fluorescence/

4\. 实验环境配置

推荐使用 Anaconda 创建虚拟环境，确保与论文实验环境一致（支持 GPU 加速，优先推荐 NVIDIA GTX 1080 及以上显卡）：

bash

运行

\# 1. 创建并激活虚拟环境

conda create -n peanut-spectral python=3.7

conda activate peanut-spectral



\# 2. 安装TensorFlow/Keras（GPU版本，需CUDA 9.2 + CuDNN 7.0）

conda install tensorflow-gpu==1.14.0 keras==2.2.4 -c conda-forge

conda install cudatoolkit==9.2 cudnn==7.0 -c nvidia



\# 3. 安装核心依赖库（含光谱处理、GAF转换）

pip install numpy~=1.19.5 matplotlib~=3.3.4 opencv-python~=4.5.5.62

pip install pandas~=1.1.5 pillow~=8.4.0 scikit-learn~=0.24.2

pip install tqdm~=4.62.3 spectral~=0.23.1 pyts~=0.12.0

5\. 代码使用说明

5.1 数据预处理：1D 光谱转 2D GAF 图像

运行 data\_preprocess.py 脚本批量处理原始光谱数据，生成 GAF 图像：

bash

运行

python data\_preprocess.py \\

--spectral\_dir ./Peanut\_Leaf\_Dataset/Reflectance\_spectra \\  # 原始光谱根目录

--save\_dir ./Peanut\_Leaf\_Dataset/GAF\_images/Reflectance \\   # GAF图像保存目录

--image\_size 32 \\                                           # 图像分辨率（固定32×32）

--spectral\_type reflectance                                 # 光谱类型（reflectance/fluorescence）

5.2 模型训练

运行 train.py 脚本启动双模态融合训练，支持单模态独立训练，示例命令如下：

bash

运行

python train.py \\

--data\_dir ./Peanut\_Leaf\_Dataset/GAF\_images \\  # GAF图像根目录

--modal fusion \\                              # 模态选择（reflectance/fluorescence/fusion）

--epochs 500 \\                                 # 训练轮数（论文最优500轮）

--batch\_size 64 \\                             # 批次大小（根据GPU显存调整）

--lr 1e-4 \\                                   # 初始学习率

--weight\_decay 1e-5 \\                         # 权重衰减（防止过拟合）

--save\_dir ./peanut\_weights \\                  # 模型权重保存目录（.h5格式）

--log\_interval 20 \\                           # 每20个batch打印训练日志

--device GPU \\                                 # 训练设备（GPU/CPU）

--num\_classes 4                                # 类别数（固定为4）

关键参数说明

参数名	含义	默认值

--data\_dir	GAF 图像根目录路径	./Peanut\_Leaf\_Dataset/GAF\_images

--modal	光谱模态选择（支持单模态 / 双模态融合）	fusion

--epochs	训练轮数（论文最优 500 轮）	500

--batch\_size	批次大小（GTX 1080 适配 64）	64

--lr	初始学习率	1e-4

--save\_dir	权重保存目录（自动生成）	./peanut\_weights

--device	训练设备（GPU 需配置 CUDA 9.2+）	GPU

--num\_classes	类别数（四类状态固定为 4）	4

5.3 模型预测

使用训练好的权重进行单张 GAF 图像或原始光谱数据预测，运行 predict.py 脚本：

bash

运行

python predict.py \\

--input\_path ./examples/scab\_reflectance.csv \\  # 输入路径（.csv光谱/.png GAF图像）

--weight\_path ./peanut\_weights/best\_model.h5 \\   # 预训练权重路径（.h5格式）

--modal fusion \\                                  # 模态选择（需与训练时一致）

--device GPU \\                                    # 预测设备（GPU/CPU）

--is\_spectral True                                # 是否为原始光谱数据（True/False）

预测输出示例

plaintext

输入路径：./examples/scab\_reflectance.csv

输入类型：原始反射光谱数据（自动转换为32×32 GAF图像）

预测类别：Scab（疮痂病）

置信度：0.9872

光谱特征分析：760nm红边效应显著，符合疮痂病光谱特征；荧光强度低于健康叶片35%

防控建议：建议喷施苯醚甲环唑类杀菌剂，7-10天一次，连续2-3次

预测耗时：8.6ms（GPU）/ 15.3ms（CPU）

6\. 项目文件结构

plaintext

b-cnn-mfm-sk-peanut-disease/  # 项目根目录

├── Peanut\_Leaf\_Dataset/       # 花生叶片双模态光谱数据集

│   ├── Reflectance\_spectra/   # 反射光谱原始数据（.csv）

│   ├── Fluorescence\_spectra/  # 荧光光谱原始数据（.csv）

│   └── GAF\_images/            # GAF转换后的2D图像（自动生成）

├── examples/                  # 示例文件

│   ├── scab\_reflectance.csv   # 疮痂病反射光谱示例

│   └── healthy\_gaf.png        # 健康叶片GAF图像示例

├── models/                    # 模型核心模块

│   ├── b\_cnn.py               # 双线性CNN骨干网络

│   ├── mfm\_layer.py           # 最大特征映射（MFM）层

│   ├── sk\_module.py           # 自适应选择核（SK）模块

│   ├── gaf\_conversion.py      # GAF 1D-2D转换工具

│   └── b\_cnn\_mfm\_sk.py        # 核心融合模型（分类头）

├── dataset/                   # 数据处理模块

│   └── data\_loader.py         # 光谱数据加载、增强、批次生成

├── data\_preprocess.py         # 批量光谱数据预处理脚本

├── train.py                   # 模型训练脚本（支持双模态融合）

├── predict.py                 # 模型预测脚本（含防控建议输出）

├── peanut\_weights/            # 模型权重保存目录

└── README.md                  # 项目说明文档

7\. 已知问题与注意事项

框架适配：仅支持 TensorFlow 1.14.0 + Keras 2.2.4，依赖 CUDA 9.2 + CuDNN 7.0，不兼容高版本 TensorFlow（≥2.0）或 PyTorch；

光谱数据格式：原始光谱需为 .csv 格式，第一列为波长（nm），第二列为光谱强度（反射率 / 荧光强度），无表头，波长范围需严格匹配（反射 400-800nm / 荧光 650-800nm）；

输入尺寸：GAF 图像固定 32×32 像素，代码自动统一光谱采样点（反射光谱 400 个点 / 荧光光谱 300 个点），无需手动调整；

模态一致性：双模态融合训练的权重仅适用于融合预测，单模态训练权重不可跨模态使用；

GPU 依赖：训练推荐使用显存≥8GB 的 GPU，CPU 训练耗时较长（单轮 epoch 约 90 分钟，GPU 约 10 分钟）；

田间部署：当前模型基于实验室数据训练，田间应用需补充不同土壤、光照、湿度条件下的样本，提升鲁棒性。

8\. 引用与联系方式

8.1 引用方式

论文已投刊，正式发表后更新完整 BibTeX 格式，当前引用格式如下：

bibtex

@article{xu2024spectral,

title={Spectral Classification of Symptomatic Peanut Leaves Based on Deep Learning Algorithms},

author={Xu, Laixiang and Chen, Xinjia and Yang, Xiaodong and Zhang, Yang and Cai, Zhaopeng and Zhao, Junmin},

journal={\[待发表期刊]},

year={2024},

volume={\[待分配]},

number={\[待分配]},

pages={1-18},

publisher={\[待发表出版社]},

doi={\[待分配]}

}

8.2 联系方式

若遇到代码运行问题、数据集获取或学术交流，可联系：

通讯作者邮箱：caizhaopeng@huuc.edu.cn；zhaojunminhuuc@yeah.net

项目维护邮箱：xulaixiang@hainanu.edu.cn

学术交流：发送主题为 “花生叶片光谱分类 - 学术交流” 的邮件，附个人简介及研究方向，将优先回复。

