import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


class MFM_Layer(nn.Module):


    def __init__(self, mask_radius1=10, mask_radius2=20):
        super().__init__()
        self.mask_radius1 = mask_radius1  # 频率掩码内环半径
        self.mask_radius2 = mask_radius2  # 频率掩码外环半径

    def frequency_transform(self, x, mask):

        # 2D傅里叶变换
        x_freq = torch.fft.fft2(x)
        # 低频分量移到中心
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
        # 应用频率掩码
        x_freq_masked = x_freq * mask
        # 恢复频率顺序
        x_freq_masked = torch.fft.ifftshift(x_freq_masked, dim=(-2, -1))
        # 逆傅里叶变换（仅保留实部）
        x_corrupted = torch.fft.ifft2(x_freq_masked).real
        # 数值裁剪，避免溢出
        return torch.clamp(x_corrupted, min=0., max=1.)

    def generate_mask(self, x):

        B, C, H, W = x.shape
        # 创建全1掩码（复数类型适配傅里叶变换）
        mask = torch.ones((B, C, H, W), dtype=torch.complex64, device=x.device)
        # 计算中心坐标
        center_h, center_w = H // 2, W // 2

        # 生成坐标网格，计算每个像素到中心的距离
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'
        )
        dist_to_center = torch.sqrt((y_grid - center_h) ** 2 + (x_grid - center_w) ** 2)

        # 环形区域（radius1~radius2）设为0（掩码该区域频率）
        mask_region = (dist_to_center >= self.mask_radius1) & (dist_to_center <= self.mask_radius2)
        mask[:, :, mask_region] = 0.0
        return mask

    def forward(self, x):
        # 生成适配当前输入的掩码
        mask = self.generate_mask(x)
        # 频率域处理并返回
        return self.frequency_transform(x, mask)



class SKConv(nn.Module):
    """
    SK卷积模块：多分支特征融合+自适应权重分配
    核心：通过不同扩张率的卷积提取多尺度特征，再自适应选择重要特征
    """

    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super().__init__()
        # 计算降维维度（保证不低于L）
        d = max(in_channels // r, L)
        self.M = M  # 分支数
        self.out_channels = out_channels

        # 构建M个分支（不同dilation的卷积）
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 3, stride,
                    padding=1 + i, dilation=1 + i, groups=32, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for i in range(M)
        ])

        # 全局平均池化（GAP）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 降维全连接层
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_channels, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        # 升维全连接层（生成各分支权重）
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, bias=False)
        # Softmax归一化权重
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        # 1. 多分支特征提取
        branch_outs = [conv(x) for conv in self.conv_branches]
        # 2. 特征融合（逐元素相加）
        U = reduce(lambda a, b: a + b, branch_outs)
        # 3. 全局信息提取
        s = self.global_pool(U)
        # 4. 降维+升维生成权重
        z = self.fc1(s)
        a_b = self.fc2(z).reshape(batch_size, self.M, self.out_channels, -1)
        # 5. 权重归一化
        a_b = self.softmax(a_b)
        # 6. 拆分权重并调整形状
        weight_branches = a_b.chunk(self.M, dim=1)
        weight_branches = [w.reshape(batch_size, self.out_channels, 1, 1) for w in weight_branches]
        # 7. 加权融合各分支特征
        V = reduce(lambda a, b: a + b, [o * w for o, w in zip(branch_outs, weight_branches)])
        return V


# ======================== 3. 双分支CNN（CNN-A/CNN-B） ========================
class CNNBranch(nn.Module):


    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            # Conv1 + MFM + Pool1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            MFM_Layer(mask_radius1=8, mask_radius2=16),
            nn.MaxPool2d(2, 2),

            # Conv2 + SKConv
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SKConv(in_channels=128, out_channels=128),

            # Conv3 + MFM + Pool3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            MFM_Layer(mask_radius1=4, mask_radius2=8),
            nn.MaxPool2d(2, 2),

            # Conv4（最终输出512通道）
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)


# ======================== 4. 融合MFM/SK的B-CNN主模型 ========================
class FusionBCNN(nn.Module):

    def __init__(self, num_classes=14):
        super().__init__()
        # 双分支CNN（参数独立，特征互补）
        self.branch_a = CNNBranch()
        self.branch_b = CNNBranch()

        # 分类头（双线性池化后维度为512*512）
        self.classifier = nn.Sequential(
            nn.Linear(512 * 512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 1. 双分支特征提取
        feat_a = self.branch_a(x)  # CNN-A输出: [B, 512, H, W]
        feat_b = self.branch_b(x)  # CNN-B输出: [B, 512, H, W]

        # 2. 特征展平（适配双线性池化）
        batch_size = feat_a.size(0)
        feat_size = feat_a.size(2) * feat_a.size(3)  # H*W
        feat_a_flat = feat_a.view(batch_size, 512, feat_size)
        feat_b_flat = feat_b.view(batch_size, 512, feat_size)

        # 3. 双线性池化（外积 + 缩放 + 归一化）
        # 外积计算: (B,512,feat_size) × (B,feat_size,512) = (B,512,512)
        bilinear = torch.bmm(feat_a_flat, feat_b_flat.transpose(1, 2)) / feat_size
        # 展平为一维特征
        bilinear_flat = bilinear.view(batch_size, -1)
        # L2归一化（原B-CNN逻辑，避免数值爆炸）
        bilinear_norm = F.normalize(
            torch.sign(bilinear_flat) * torch.sqrt(torch.abs(bilinear_flat) + 1e-10),
            dim=1
        )

        # 4. 分类预测
        output = self.classifier(bilinear_norm)
        return output


if __name__ == '__main__':
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型
    model = FusionBCNN(num_classes=14).to(device)

    # 生成测试输入（batch=2, 3通道, 224×224）
    test_input = torch.randn(2, 3, 224, 224).to(device)

    # 前向传播
    with torch.no_grad():
        output = model(test_input)

    # 输出结果验证
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")  # 应为 [2, 14]
    print(f"输出概率和: {output.sum(dim=1)}")  # Softmax后每行和≈1
    print("\n模型结构验证通过！")