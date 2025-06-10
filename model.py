import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

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
        
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入圖像 tensor, 形狀為 (batch_size, channels, height, width)
            
        Returns:
            輸出分割結果, 形狀為 (batch_size, out_channels, height, width)
        """        # Encoder (下採樣路徑)
        # 第一層: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Pool
        conv1 = self.relu(self.bn1_1(self.conv1_1(x)))
        conv1 = self.relu(self.bn1_2(self.conv1_2(conv1)))
        pool1 = self.pool(conv1)
        
        # 第二層
        conv2 = self.relu(self.bn2_1(self.conv2_1(pool1)))
        conv2 = self.relu(self.bn2_2(self.conv2_2(conv2)))
        pool2 = self.pool(conv2)
        
        # 第三層
        conv3 = self.relu(self.bn3_1(self.conv3_1(pool2)))
        conv3 = self.relu(self.bn3_2(self.conv3_2(conv3)))
        pool3 = self.pool(conv3)
        
        # 第四層
        conv4 = self.relu(self.bn4_1(self.conv4_1(pool3)))
        conv4 = self.relu(self.bn4_2(self.conv4_2(conv4)))
        pool4 = self.pool(conv4)
        
        # 底部 (Bridge)
        conv5 = self.relu(self.bn5_1(self.conv5_1(pool4)))
        conv5 = self.relu(self.bn5_2(self.conv5_2(conv5)))
          # Decoder (上採樣路徑)
        # 第一次上採樣: UpConv -> Concatenation -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU
        up1 = self.upconv1(conv5)
        # 確保尺寸匹配
        if up1.size() != conv4.size():
            up1 = F.interpolate(up1, size=conv4.shape[2:], mode='bilinear', align_corners=False)
        # Skip connection + concatenation
        merge1 = torch.cat([conv4, up1], dim=1)
        conv_up1 = self.relu(self.bn_up1_1(self.conv_up1_1(merge1)))
        conv_up1 = self.relu(self.bn_up1_2(self.conv_up1_2(conv_up1)))
        
        # 第二次上採樣
        up2 = self.upconv2(conv_up1)
        if up2.size() != conv3.size():
            up2 = F.interpolate(up2, size=conv3.shape[2:], mode='bilinear', align_corners=False)
        merge2 = torch.cat([conv3, up2], dim=1)
        conv_up2 = self.relu(self.bn_up2_1(self.conv_up2_1(merge2)))
        conv_up2 = self.relu(self.bn_up2_2(self.conv_up2_2(conv_up2)))
        
        # 第三次上採樣
        up3 = self.upconv3(conv_up2)
        if up3.size() != conv2.size():
            up3 = F.interpolate(up3, size=conv2.shape[2:], mode='bilinear', align_corners=False)
        merge3 = torch.cat([conv2, up3], dim=1)
        conv_up3 = self.relu(self.bn_up3_1(self.conv_up3_1(merge3)))
        conv_up3 = self.relu(self.bn_up3_2(self.conv_up3_2(conv_up3)))
        
        # 第四次上採樣
        up4 = self.upconv4(conv_up3)
        if up4.size() != conv1.size():
            up4 = F.interpolate(up4, size=conv1.shape[2:], mode='bilinear', align_corners=False)
        merge4 = torch.cat([conv1, up4], dim=1)
        conv_up4 = self.relu(self.bn_up4_1(self.conv_up4_1(merge4)))
        conv_up4 = self.relu(self.bn_up4_2(self.conv_up4_2(conv_up4)))
          # 最終輸出 (不使用 BN 和 ReLU，因為這是最終輸出層)
        output = self.final_conv(conv_up4)
        
        return output

def create_unet(in_channels=3, out_channels=1):
    """
    創建 UNet 模型的輔助函數
    
    Args:
        in_channels: 輸入通道數
        out_channels: 輸出通道數
        
    Returns:
        UNet 模型實例
    """
    model = UNet(in_channels, out_channels)
    return model

def train_model(train_data, validation_data, epochs=50, batch_size=16, learning_rate=0.001, 
                patience=10, device=None):
    """
    訓練 UNet 模型
    
    Args:
        train_data: tuple (train_images, train_masks)
        validation_data: tuple (val_images, val_masks)
        epochs: 訓練輪數
        batch_size: 批次大小
        learning_rate: 學習率
        patience: 早停耐心值
        device: 訓練設備 (cuda/cpu)
        
    Returns:
        trained_model: 訓練好的模型
    """
    # 設置設備
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 準備數據
    train_images, train_masks = train_data
    val_images, val_masks = validation_data
    
    # 轉換為 torch tensors
    train_images = torch.FloatTensor(train_images).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    train_masks = torch.FloatTensor(train_masks).permute(0, 3, 1, 2)    # (N, H, W, C) -> (N, C, H, W)
    val_images = torch.FloatTensor(val_images).permute(0, 3, 1, 2)
    val_masks = torch.FloatTensor(val_masks).permute(0, 3, 1, 2)
    
    # 創建數據載入器
    train_dataset = TensorDataset(train_images, train_masks)
    val_dataset = TensorDataset(val_images, val_masks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 創建模型
    model = create_unet(in_channels=3, out_channels=1)
    model = model.to(device)
    
    # 定義損失函數和優化器
    criterion = nn.BCEWithLogitsLoss()  # 適用於二分類分割
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 訓練歷史記錄
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"開始訓練 UNet 模型...")
    print(f"訓練樣本數: {len(train_images)}, 驗證樣本數: {len(val_images)}")
    print(f"批次大小: {batch_size}, 學習率: {learning_rate}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        epoch_train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        
        for batch_idx, (images, masks) in enumerate(train_progress):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向傳播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # 更新進度條
            train_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{epoch_train_loss/(batch_idx+1):.4f}'
            })
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 驗證階段
        model.eval()
        epoch_val_loss = 0.0
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_progress):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()
                
                # 更新進度條
                val_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{epoch_val_loss/(batch_idx+1):.4f}'
                })
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 學習率調整
        scheduler.step(avg_val_loss)
        
        # 打印訓練結果
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 早停檢查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f'  ✓ 新的最佳模型已保存 (Val Loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  ⚠ 驗證損失未改善 ({patience_counter}/{patience})')
        
        print("-" * 50)
        
        # 早停
        if patience_counter >= patience:
            print(f'早停觸發！在第 {epoch+1} 輪停止訓練')
            break
    
    # 載入最佳模型
    model.load_state_dict(torch.load('best_unet_model.pth'))
    model.eval()
    
    print("訓練完成！")
    print(f"最佳驗證損失: {best_val_loss:.4f}")
    
    return model

def dice_coefficient(pred, target, smooth=1e-6):
    """
    計算 Dice 係數 (用於評估分割性能)
    
    Args:
        pred: 預測結果 tensor
        target: 真實標籤 tensor
        smooth: 平滑因子
        
    Returns:
        dice_score: Dice 係數
    """
    pred = torch.sigmoid(pred)  # 將 logits 轉為概率
    pred = (pred > 0.5).float()  # 二值化
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()

def calculate_metrics(pred, target, smooth=1e-6):
    """
    計算各種評估指標
    
    Args:
        pred: 預測結果 tensor (經過 sigmoid 和二值化)
        target: 真實標籤 tensor
        smooth: 平滑因子
        
    Returns:
        dict: 包含各種指標的字典
    """
    # 確保都是 float tensor
    pred = pred.float()
    target = target.float()
    
    # 計算 True Positive, False Positive, True Negative, False Negative
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    # Dice Coefficient (F1-Score)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    
    # IoU (Intersection over Union)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    
    # Accuracy
    accuracy = (tp + tn + smooth) / (tp + fp + tn + fn + smooth)
    
    # Precision
    precision = (tp + smooth) / (tp + fp + smooth)
    
    # Recall (Sensitivity)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    # Specificity
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'specificity': specificity.item(),
        'tp': tp.item(),
        'fp': fp.item(),
        'tn': tn.item(),
        'fn': fn.item()
    }

def save_model(model, filepath):
    """
    保存訓練好的模型
    
    Args:
        model: 訓練好的模型
        filepath: 保存路徑
    """
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存至: {filepath}")

def save_results(model, test_data, train_time, filepath):
    """
    保存測試結果，包含詳細的評估指標
    
    Args:
        model: 訓練好的模型
        test_data: 測試數據
        train_time: 訓練時間
        filepath: 結果保存路徑
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    test_images, test_masks = test_data
    
    # 檢查是否有測試數據
    if len(test_images) == 0:
        print("警告: 沒有測試數據")
        return
    
    test_images = torch.FloatTensor(test_images).permute(0, 3, 1, 2).to(device)
    test_masks = torch.FloatTensor(test_masks).permute(0, 3, 1, 2).to(device)
    
    # 初始化累積指標
    total_metrics = {
        'dice': 0.0, 'iou': 0.0, 'accuracy': 0.0, 
        'precision': 0.0, 'recall': 0.0, 'specificity': 0.0,
        'tp': 0.0, 'fp': 0.0, 'tn': 0.0, 'fn': 0.0
    }
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    print("開始評估模型...")
    
    with torch.no_grad():
        batch_size = 16
        num_batches = (len(test_images) + batch_size - 1) // batch_size
        
        for i in range(0, len(test_images), batch_size):
            batch_images = test_images[i:i+batch_size]
            batch_masks = test_masks[i:i+batch_size]
            
            # 前向傳播
            outputs = model(batch_images)
            loss = criterion(outputs, batch_masks)
            total_loss += loss.item()
            
            # 將 logits 轉換為概率並二值化
            pred_probs = torch.sigmoid(outputs)
            pred_binary = (pred_probs > 0.5).float()
            
            # 計算每個樣本的指標
            for j in range(len(batch_images)):
                metrics = calculate_metrics(pred_binary[j], batch_masks[j])
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
            
            print(f"處理進度: {min(i+batch_size, len(test_images))}/{len(test_images)}")
    
    # 計算平均指標
    num_samples = len(test_images)
    avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
    avg_loss = total_loss / num_batches
    
    # 保存結果到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("UNet 模型測試結果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("基本信息:\n")
        f.write(f"訓練時間: {train_time:.2f} 秒\n")
        f.write(f"測試樣本數: {num_samples}\n")
        f.write(f"使用設備: {device}\n")
        f.write(f"平均測試損失: {avg_loss:.6f}\n\n")
        
        f.write("評估指標:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Dice Coefficient (DC): {avg_metrics['dice']:.6f}\n")
        f.write(f"Intersection over Union (IoU): {avg_metrics['iou']:.6f}\n")
        f.write(f"Accuracy: {avg_metrics['accuracy']:.6f}\n")
        f.write(f"Precision: {avg_metrics['precision']:.6f}\n")
        f.write(f"Recall (Sensitivity): {avg_metrics['recall']:.6f}\n")
        f.write(f"Specificity: {avg_metrics['specificity']:.6f}\n\n")
        
        f.write("混淆矩陣統計:\n")
        f.write("-" * 30 + "\n")
        f.write(f"True Positive (TP): {avg_metrics['tp']:.2f}\n")
        f.write(f"False Positive (FP): {avg_metrics['fp']:.2f}\n")
        f.write(f"True Negative (TN): {avg_metrics['tn']:.2f}\n")
        f.write(f"False Negative (FN): {avg_metrics['fn']:.2f}\n\n")
        
        # 計算 F1-Score (等同於 Dice)
        f1_score = avg_metrics['dice']
        f.write(f"F1-Score: {f1_score:.6f}\n")
        
        # 計算 Jaccard Index (等同於 IoU)
        jaccard = avg_metrics['iou']
        f.write(f"Jaccard Index: {jaccard:.6f}\n")
    
    # 在控制台也顯示結果
    print("\n" + "=" * 50)
    print("模型評估完成！")
    print("=" * 50)
    print(f"測試樣本數: {num_samples}")
    print(f"平均測試損失: {avg_loss:.6f}")
    print("-" * 30)
    print("主要評估指標:")
    print(f"  Dice Coefficient: {avg_metrics['dice']:.6f}")
    print(f"  IoU: {avg_metrics['iou']:.6f}")
    print(f"  Accuracy: {avg_metrics['accuracy']:.6f}")
    print(f"  Precision: {avg_metrics['precision']:.6f}")
    print(f"  Recall: {avg_metrics['recall']:.6f}")
    print("=" * 50)
    
    print(f"詳細結果已保存至: {filepath}")

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
        self.downconv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=2)
        
        # 第二層: 64->128
        self.conv2_1 = DepthwiseConvBlock(64)
        self.conv2_2 = DepthwiseConvBlock(64)
        self.downconv2 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        
        # 第三層: 128->256
        self.conv3_1 = DepthwiseConvBlock(128)
        self.conv3_2 = DepthwiseConvBlock(128)
        self.downconv3 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        
        # 第四層: 256->512
        self.conv4_1 = DepthwiseConvBlock(256)
        self.conv4_2 = DepthwiseConvBlock(256)
        self.downconv4 = nn.Conv2d(256, 512, kernel_size=1, stride=2)
        
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

def create_lunext(in_channels=3, out_channels=1):
    """
    創建 LUNeXt 模型的輔助函數
    
    Args:
        in_channels: 輸入通道數
        out_channels: 輸出通道數
        
    Returns:
        LUNeXt 模型實例
    """
    model = LUNeXt(in_channels, out_channels)
    return model

def train_lunext_model(train_data, validation_data, epochs=50, batch_size=16, learning_rate=0.001, 
                       patience=10, device=None):
    """
    訓練 LUNeXt 模型
    
    Args:
        train_data: tuple (train_images, train_masks)
        validation_data: tuple (val_images, val_masks)
        epochs: 訓練輪數
        batch_size: 批次大小
        learning_rate: 學習率
        patience: 早停耐心值
        device: 訓練設備 (cuda/cpu)
        
    Returns:
        trained_model: 訓練好的模型
    """
    # 設置設備
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 準備數據
    train_images, train_masks = train_data
    val_images, val_masks = validation_data
    
    # 轉換為 torch tensors
    train_images = torch.FloatTensor(train_images).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    train_masks = torch.FloatTensor(train_masks).permute(0, 3, 1, 2)    # (N, H, W, C) -> (N, C, H, W)
    val_images = torch.FloatTensor(val_images).permute(0, 3, 1, 2)
    val_masks = torch.FloatTensor(val_masks).permute(0, 3, 1, 2)
    
    # 創建數據載入器
    train_dataset = TensorDataset(train_images, train_masks)
    val_dataset = TensorDataset(val_images, val_masks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 創建模型
    model = create_lunext(in_channels=3, out_channels=1)
    model = model.to(device)
    
    # 定義損失函數和優化器
    criterion = nn.BCEWithLogitsLoss()  # 適用於二分類分割
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 訓練歷史記錄
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"開始訓練 LUNeXt 模型...")
    print(f"訓練樣本數: {len(train_images)}, 驗證樣本數: {len(val_images)}")
    print(f"批次大小: {batch_size}, 學習率: {learning_rate}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        epoch_train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        
        for batch_idx, (images, masks) in enumerate(train_progress):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向傳播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # 更新進度條
            train_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{epoch_train_loss/(batch_idx+1):.4f}'
            })
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 驗證階段
        model.eval()
        epoch_val_loss = 0.0
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_progress):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()
                
                # 更新進度條
                val_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{epoch_val_loss/(batch_idx+1):.4f}'
                })
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 學習率調整
        scheduler.step(avg_val_loss)
        
        # 打印訓練結果
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 早停檢查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_lunext_model.pth')
            print(f'  ✓ 新的最佳 LUNeXt 模型已保存 (Val Loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  ⚠ 驗證損失未改善 ({patience_counter}/{patience})')
        
        print("-" * 50)
        
        # 早停
        if patience_counter >= patience:
            print(f'早停觸發！在第 {epoch+1} 輪停止訓練')
            break
    
    # 載入最佳模型
    model.load_state_dict(torch.load('best_lunext_model.pth'))
    model.eval()
    
    print("LUNeXt 訓練完成！")
    print(f"最佳驗證損失: {best_val_loss:.4f}")
    
    return model


def save_prediction_images(model, test_data, output_folder="predictions", model_name="Model"):
    """
    保存模型預測結果圖像到指定資料夾
    
    Args:
        model: 訓練好的模型
        test_data: 測試數據 (images, masks)
        output_folder: 輸出資料夾路徑
        model_name: 模型名稱，用於子資料夾命名
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    test_images, test_masks = test_data
    
    if len(test_images) == 0:
        print("警告: 沒有測試數據可供保存")
        return
    
    # 創建輸出資料夾結構
    base_folder = os.path.join(output_folder, model_name)
    folders = {
        'ground_truth': os.path.join(base_folder, 'ground_truth'), 
        'predictions': os.path.join(base_folder, 'predictions')
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    print(f"正在保存 {model_name} 預測結果到: {base_folder}")
    
    test_images_tensor = torch.FloatTensor(test_images).permute(0, 3, 1, 2).to(device)
    test_masks_tensor = torch.FloatTensor(test_masks).permute(0, 3, 1, 2).to(device)
    
    with torch.no_grad():
        batch_size = 16
        saved_count = 0
        
        # 使用 tqdm 顯示保存進度
        total_images = len(test_images)
        pbar = tqdm(total=total_images, desc=f"保存 {model_name} 預測結果", unit="images")
        
        for i in range(0, len(test_images_tensor), batch_size):
            batch_images = test_images_tensor[i:i+batch_size]
            batch_masks = test_masks_tensor[i:i+batch_size]
            
            # 獲取模型預測
            outputs = model(batch_images)
            pred_probs = torch.sigmoid(outputs)
            pred_binary = (pred_probs > 0.5).float()
            
            # 將 tensor 轉換為 numpy 用於保存
            batch_images_np = batch_images.cpu().permute(0, 2, 3, 1).numpy()
            batch_masks_np = batch_masks.cpu().permute(0, 2, 3, 1).numpy()
            pred_binary_np = pred_binary.cpu().permute(0, 2, 3, 1).numpy()
            pred_probs_np = pred_probs.cpu().permute(0, 2, 3, 1).numpy()
            
            # 保存每張圖像
            for j in range(len(batch_images_np)):
                img_idx = i + j
                
                # 準備數據 (將值範圍調整到 0-255)
                ground_truth = (batch_masks_np[j][:,:,0] * 255).astype(np.uint8)  # 移除通道維度
                prediction = (pred_binary_np[j][:,:,0] * 255).astype(np.uint8)
                prediction_prob = (pred_probs_np[j][:,:,0] * 255).astype(np.uint8)
                
                # 保存ground truth
                gt_img = Image.fromarray(ground_truth)
                gt_img.save(os.path.join(folders['ground_truth'], f"{img_idx:04d}_ground_truth.png"))
                
                # 保存二值化預測
                pred_img = Image.fromarray(prediction)
                pred_img.save(os.path.join(folders['predictions'], f"{img_idx:04d}_prediction.png"))
                
                saved_count += 1
                pbar.update(1)
        
        pbar.close()
    
    print(f"\n{model_name} 預測結果保存完成!")
    print(f"共保存了 {saved_count} 張圖像")
    print(f"保存位置:")
    print(f"  原始圖像: {folders['original']}")
    print(f"  真實遮罩: {folders['ground_truth']}")
    print(f"  預測結果: {folders['predictions']}")
    print(f"  對比圖像: {folders['comparison']}")

def save_results_with_images(model, test_data, train_time, filepath, model_name="Model"):
    """
    增強版的結果保存函數，同時保存評估指標和預測圖像
    
    Args:
        model: 訓練好的模型
        test_data: 測試數據
        train_time: 訓練時間
        filepath: 結果文本文件保存路徑
        model_name: 模型名稱
    """
    # 首先保存評估指標
    save_results(model, test_data, train_time, filepath)
    
    # 然後保存預測圖像
    save_prediction_images(model, test_data, output_folder="predictions", model_name=model_name)
