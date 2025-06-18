import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DepthwiseConvBlock(nn.Module):
    """
    深度可分離卷積塊，包含殘差連接
    """
    def __init__(self, channels, kernel_size=3, padding=1):
        super(DepthwiseConvBlock, self).__init__()
        
        # 深度可分離卷積 = 深度卷積 + 點卷積
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                                 padding=padding, groups=channels)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 保存輸入用於殘差連接
        residual = x
        
        # 深度可分離卷積
        out = self.depthwise(x)
        out = self.bn(out)
        
        out = out + residual  # pixel-wise 相加
        out = self.relu(out)
        
        return out

class AttentionGate(nn.Module):
    """
    自定義注意力門機制，按照以下6個步驟實現:
    Step 1: g → sigmoid → b
    Step 2: g → tanh → c  
    Step 3: b ⊙ c → e (pixel-wise multiply)
    Step 4: (g + x) → sigmoid → a (pixel-wise add then sigmoid)
    Step 5: a ⊙ x → d (pixel-wise multiply)
    Step 6: d + e → output (pixel-wise add)
    """
    def __init__(self, F_g, F_l):
        """
        Args:
            F_g: gating signal的通道數 (來自decoder)
            F_l: feature map的通道數 (來自encoder)
        """
        super(AttentionGate, self).__init__()
        
        # 確保 x 和 g 有相同的通道數進行運算
        # 如果通道數不同，需要調整
        self.inter_channels = min(F_g, F_l)
        
        # # 用於調整 g 的通道數到 inter_channels
        # self.W_g = nn.Sequential(
        #     nn.Conv2d(F_g, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(self.inter_channels)
        # )
        
        # # 用於調整 x 的通道數到 inter_channels  
        # self.W_x = nn.Sequential(
        #     nn.Conv2d(F_l, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(self.inter_channels)
        # )
        
        # Step 1 & 2: g → sigmoid/tanh 的轉換層
        self.sigmoid_conv = nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, bias=True)
        self.tanh_conv = nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, bias=True)
        
        # Step 4: (g + x) → sigmoid 的轉換層
        self.combine_conv = nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, bias=True)
        
        # 最終輸出層，保持原始 x 的通道數
        self.output_conv = nn.Conv2d(self.inter_channels, F_l, kernel_size=1, bias=True)
    
    def forward(self, g, x):
        """
        實現自定義的6步注意力機制
        
        Args:
            g: gating signal (來自decoder的上採樣結果) [B, F_g, H, W]
            x: feature map (來自encoder的skip connection) [B, F_l, H, W]
        
        Returns:
            注意力加權後的特徵圖 [B, F_l, H, W]
        """
        # 獲取輸入尺寸
        batch_size, _, height, width = x.size()
        
        # 如果 g 和 x 的空間尺寸不同，需要調整 g 的尺寸
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=(height, width), mode='bilinear', align_corners=True)
        
        # 調整通道數到相同維度
        # g_adjusted = self.W_g(g)  # [B, inter_channels, H, W]
        # x_adjusted = self.W_x(x)  # [B, inter_channels, H, W]
        g_adjusted = g
        x_adjusted = x

        # Step 1: g → sigmoid → b
        b = torch.sigmoid(self.sigmoid_conv(g_adjusted))
        
        # Step 2: g → tanh → c
        c = torch.tanh(self.tanh_conv(g_adjusted))
        
        # Step 3: b ⊙ c → e (pixel-wise multiply)
        e = b * c
        
        # Step 4: (g + x) → sigmoid → a (pixel-wise add then sigmoid)
        combined = g_adjusted + x_adjusted
        a = torch.sigmoid(self.combine_conv(combined))
        
        # Step 5: a ⊙ x → d (pixel-wise multiply)
        d = a * x_adjusted
        
        # Step 6: d + e → output (pixel-wise add)
        attention_output = d + e
        
        # 將結果轉換回原始 x 的通道數
        # final_output = self.output_conv(attention_output)
        
        return attention_output

class LUNeXt(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """
        LUNeXt 模型 - 改進的 UNet，使用深度可分離卷積、殘差連接和注意力門
        
        Args:
            in_channels: 輸入圖像的通道數 (RGB=3)
            out_channels: 輸出的通道數 (二分類分割=1)
        """
        super(LUNeXt, self).__init__()
        
        # Encoder (下採樣部分) - 使用深度可分離卷積塊
        # 第一層: 3->64
        self.conv1_1 = DepthwiseConvBlock(in_channels)
        self.conv1_2 = DepthwiseConvBlock(in_channels)
        self.downconv1 = nn.Conv2d(in_channels, 64, kernel_size=2, stride=2)
        
        # 第二層: 64->128
        self.conv2_1 = DepthwiseConvBlock(64)
        self.conv2_2 = DepthwiseConvBlock(64)
        self.downconv2 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        
        # 第三層: 128->256
        self.conv3_1 = DepthwiseConvBlock(128)
        self.conv3_2 = DepthwiseConvBlock(128)
        self.downconv3 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        
        # 第四層: 256->512
        self.conv4_1 = DepthwiseConvBlock(256)
        self.conv4_2 = DepthwiseConvBlock(256)
        self.downconv4 = nn.Conv2d(256, 512, kernel_size=2, stride=2)
        
        # 底部 (Bridge): 
        self.conv5_1 = DepthwiseConvBlock(512)
        self.conv5_2 = DepthwiseConvBlock(512)
          # Decoder (上採樣部分)
        # 第一次上採樣: 512->256
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=256, F_l=256)
        self.conv_up1_1 = DepthwiseConvBlock(256)  # 注意力門輸出 256 通道
        self.conv_up1_2 = DepthwiseConvBlock(256)
        
        # 第二次上採樣: 256->128
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128)
        self.conv_up2_1 = DepthwiseConvBlock(128)  # 注意力門輸出 128 通道
        self.conv_up2_2 = DepthwiseConvBlock(128)
        
        # 第三次上採樣: 128->64
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=64, F_l=64)
        self.conv_up3_1 = DepthwiseConvBlock(64)  # 注意力門輸出 64 通道
        self.conv_up3_2 = DepthwiseConvBlock(64)
        
        # 第四次上採樣: 64->in_channels
        self.upconv4 = nn.ConvTranspose2d(64, in_channels, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=in_channels, F_l=in_channels)
        self.conv_up4_1 = DepthwiseConvBlock(in_channels)  # 注意力門輸出 in_channels 通道
        self.conv_up4_2 = DepthwiseConvBlock(in_channels)
        
        # 最終輸出層: 1x1 conv
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入圖像 tensor, 形狀為 (batch_size, channels, height, width)
            
        Returns:
            輸出分割結果, 形狀為 (batch_size, out_channels, height, width)
        """
        # Encoder (下採樣路徑)
        # 第一層
        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)
        pool1 = self.downconv1(conv1)
        
        # 第二層
        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.downconv2(conv2)
        
        # 第三層
        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = self.downconv3(conv3)
        
        # 第四層
        conv4 = self.conv4_1(pool3)
        conv4 = self.conv4_2(conv4)
        pool4 = self.downconv4(conv4)
        
        # 底部 (Bridge)
        conv5 = self.conv5_1(pool4)
        conv5 = self.conv5_2(conv5)
          # Decoder (上採樣路徑) - 使用注意力門
        # 第一次上採樣
        up1 = self.upconv1(conv5)
        # 確保尺寸匹配
        if up1.size() != conv4.size():
            up1 = F.interpolate(up1, size=conv4.shape[2:], mode='bilinear', align_corners=False)
        # 注意力門機制 - 現在直接輸出加權的特徵圖
        merge1 = self.att1(up1, conv4)
        conv_up1 = self.conv_up1_1(merge1)
        conv_up1 = self.conv_up1_2(conv_up1)
        
        # 第二次上採樣
        up2 = self.upconv2(conv_up1)
        if up2.size() != conv3.size():
            up2 = F.interpolate(up2, size=conv3.shape[2:], mode='bilinear', align_corners=False)
        merge2 = self.att2(up2, conv3)
        conv_up2 = self.conv_up2_1(merge2)
        conv_up2 = self.conv_up2_2(conv_up2)
        
        # 第三次上採樣
        up3 = self.upconv3(conv_up2)
        if up3.size() != conv2.size():
            up3 = F.interpolate(up3, size=conv2.shape[2:], mode='bilinear', align_corners=False)
        merge3 = self.att3(up3, conv2)
        conv_up3 = self.conv_up3_1(merge3)
        conv_up3 = self.conv_up3_2(conv_up3)
        
        # 第四次上採樣
        up4 = self.upconv4(conv_up3)
        if up4.size() != conv1.size():
            up4 = F.interpolate(up4, size=conv1.shape[2:], mode='bilinear', align_corners=False)
        merge4 = self.att4(up4, conv1)
        conv_up4 = self.conv_up4_1(merge4)
        conv_up4 = self.conv_up4_2(conv_up4)
        
        # 最終輸出
        output = self.final_conv(conv_up4)
        
        return output