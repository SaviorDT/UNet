import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """
        經典 UNet 模型 - 四次下採樣和四次上採樣，包含 Batch Normalization
        
        Args:
            in_channels: 輸入圖像的通道數 (RGB=3)
            out_channels: 輸出的通道數 (二分類分割=1)
        """
        super(UNet, self).__init__()
        
        # Encoder (下採樣部分)
        # 第一層: 3->64
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)        
        # 第二層: 64->128
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        # 第三層: 128->256
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        
        # 第四層: 256->512
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        
        # 底部 (Bridge): 512->1024
        self.conv5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(1024)
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(1024)
        
        # Max pooling: 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
          # Decoder (上採樣部分)
        # 第一次上採樣: 1024->512
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)  # 1024 = 512 + 512 (skip connection)
        self.bn_up1_1 = nn.BatchNorm2d(512)
        self.conv_up1_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_up1_2 = nn.BatchNorm2d(512)
        
        # 第二次上採樣: 512->256
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 512 = 256 + 256 (skip connection)
        self.bn_up2_1 = nn.BatchNorm2d(256)
        self.conv_up2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_up2_2 = nn.BatchNorm2d(256)
        
        # 第三次上採樣: 256->128
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 256 = 128 + 128 (skip connection)
        self.bn_up3_1 = nn.BatchNorm2d(128)
        self.conv_up3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_up3_2 = nn.BatchNorm2d(128)
        
        # 第四次上採樣: 128->64
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 128 = 64 + 64 (skip connection)
        self.bn_up4_1 = nn.BatchNorm2d(64)
        self.conv_up4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_up4_2 = nn.BatchNorm2d(64)
          # 最終輸出層: 1x1 conv (不使用 BN，因為這是最終輸出)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # 激活函數
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.intermediate_features = []  # 用於存儲中間層特徵
        
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入圖像 tensor, 形狀為 (batch_size, channels, height, width)
            
        Returns:
            輸出分割結果, 形狀為 (batch_size, out_channels, height, width)
        """
        self.intermediate_features = []  # 每次前向傳播清空中間層特徵

        # 第一層: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Pool
        conv1_1 = self.relu(self.bn1_1(self.conv1_1(x)))
        self.intermediate_features.append(conv1_1)
        conv1_2 = self.relu(self.bn1_2(self.conv1_2(conv1_1)))
        self.intermediate_features.append(conv1_2)
        pool1 = self.pool(conv1_2)

        # 第二層
        conv2_1 = self.relu(self.bn2_1(self.conv2_1(pool1)))
        self.intermediate_features.append(conv2_1)
        conv2_2 = self.relu(self.bn2_2(self.conv2_2(conv2_1)))
        self.intermediate_features.append(conv2_2)
        pool2 = self.pool(conv2_2)

        # 第三層
        conv3_1 = self.relu(self.bn3_1(self.conv3_1(pool2)))
        self.intermediate_features.append(conv3_1)
        conv3_2 = self.relu(self.bn3_2(self.conv3_2(conv3_1)))
        self.intermediate_features.append(conv3_2)
        pool3 = self.pool(conv3_2)

        # 第四層
        conv4_1 = self.relu(self.bn4_1(self.conv4_1(pool3)))
        self.intermediate_features.append(conv4_1)
        conv4_2 = self.relu(self.bn4_2(self.conv4_2(conv4_1)))
        self.intermediate_features.append(conv4_2)
        pool4 = self.pool(conv4_2)

        # 底部 (Bridge)
        conv5_1 = self.relu(self.bn5_1(self.conv5_1(pool4)))
        conv5_2 = self.relu(self.bn5_2(self.conv5_2(conv5_1)))

        # Decoder (上採樣路徑)
        # 第一次上採樣: UpConv -> Concatenation -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU
        up1 = self.upconv1(conv5_2)
        # 確保尺寸匹配
        # if up1.size() != conv4_2.size():
        #     up1 = F.interpolate(up1, size=conv4_2.shape[2:], mode='bilinear', align_corners=False)
        # Skip connection + concatenation
        merge1 = torch.cat([conv4_2, up1], dim=1)
        conv_up1_1 = self.relu(self.bn_up1_1(self.conv_up1_1(merge1)))
        self.intermediate_features.append(conv_up1_1)
        conv_up1_2 = self.relu(self.bn_up1_2(self.conv_up1_2(conv_up1_1)))
        self.intermediate_features.append(conv_up1_2)

        # 第二次上採樣
        up2 = self.upconv2(conv_up1_2)
        # if up2.size() != conv3_2.size():
        #     up2 = F.interpolate(up2, size=conv3_2.shape[2:], mode='bilinear', align_corners=False)
        merge2 = torch.cat([conv3_2, up2], dim=1)
        conv_up2_1 = self.relu(self.bn_up2_1(self.conv_up2_1(merge2)))
        self.intermediate_features.append(conv_up2_1)
        conv_up2_2 = self.relu(self.bn_up2_2(self.conv_up2_2(conv_up2_1)))
        self.intermediate_features.append(conv_up2_2)

        # 第三次上採樣
        up3 = self.upconv3(conv_up2_2)
        # if up3.size() != conv2_2.size():
        #     up3 = F.interpolate(up3, size=conv2_2.shape[2:], mode='bilinear', align_corners=False)
        merge3 = torch.cat([conv2_2, up3], dim=1)
        conv_up3_1 = self.relu(self.bn_up3_1(self.conv_up3_1(merge3)))
        self.intermediate_features.append(conv_up3_1)
        conv_up3_2 = self.relu(self.bn_up3_2(self.conv_up3_2(conv_up3_1)))
        self.intermediate_features.append(conv_up3_2)

        # 第四次上採樣
        up4 = self.upconv4(conv_up3_2)
        # if up4.size() != conv1_2.size():
        #     up4 = F.interpolate(up4, size=conv1_2.shape[2:], mode='bilinear', align_corners=False)
        merge4 = torch.cat([conv1_2, up4], dim=1)
        conv_up4_1 = self.relu(self.bn_up4_1(self.conv_up4_1(merge4)))
        conv_up4_2 = self.relu(self.bn_up4_2(self.conv_up4_2(conv_up4_1)))
        self.intermediate_features.append(conv_up4_2)

        # 最終輸出 (不使用 BN 和 ReLU，因為這是最終輸出層)
        output = self.final_conv(conv_up4_2)
        # output = self.sigmoid(output)
        
        return output
