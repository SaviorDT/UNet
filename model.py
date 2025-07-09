import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from models.unet import UNet
from models.lunext import LUNeXt

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

def train_model(train_data, validation_data, model_type="UNet", epochs=50, batch_size=16, learning_rate=0.001, 
                patience=10, device=None, custom_loss=None):
    """
    訓練模型（支援 UNet 和 LUNeXt）

    Args:
        train_data: tuple (train_images, train_masks)
        validation_data: tuple (val_images, val_masks)
        model_type: 模型類型 ("UNet" 或 "LUNeXt")
        epochs: 訓練輪數
        batch_size: 批次大小
        learning_rate: 學習率
        patience: 早停耐心值
        device: 訓練設備 (cuda/cpu)
        custom_loss: 自訂損失函數 (默認為 None)

    Returns:
        trained_model: 訓練好的模型
    """
    # 設置設備
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 獲取 CUDA 設備詳細訊息
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"CUDA 設備: {torch.cuda.get_device_name(current_device)}")
        print(f"CUDA 運算能力: {torch.cuda.get_device_capability(current_device)}")
        print(f"CUDA 記憶體總量: {torch.cuda.get_device_properties(current_device).total_memory / (1024**3):.2f} GB")
        # 預熱 GPU
        torch.cuda.empty_cache()
        warm_tensor = torch.randn(100, 100, device=device)
        del warm_tensor
        torch.cuda.empty_cache()
    
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
    
    # 使用多進程加速數據載入，避免成為訓練瓶頸
    num_workers = 4  # 可根據 CPU 核心數調整
    pin_memory = True if torch.cuda.is_available() else False
    prefetch_factor = 2  # 預取因子: 預取 num_workers * prefetch_factor 個樣本
    persistent_workers = True  # 保持 worker 進程活著，避免每個 epoch 重新創建
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    
    # 創建模型
    if model_type == "UNet":
        model = create_unet(in_channels=3, out_channels=1)
        model_save_path = 'best_unet_model.pth'
    else:  # LUNeXt
        model = create_lunext(in_channels=3, out_channels=1)
        model_save_path = 'best_lunext_model.pth'
    
    model = model.to(device)
    
    # 定義損失函數和優化器
    if custom_loss == "self_reg":
        criterion = SelfRegLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()  # 適用於二分類分割

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
      # 訓練歷史記錄
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 啟用混合精度訓練 (Mixed Precision)以加速計算
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    print(f"開始訓練 {model_type} 模型...")
    print(f"訓練樣本數: {len(train_images)}, 驗證樣本數: {len(val_images)}")
    print(f"批次大小: {batch_size}, 學習率: {learning_rate}")
    print(f"混合精度訓練: {'啟用' if scaler else '停用'}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        epoch_train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        
        for batch_idx, (images, masks) in enumerate(train_progress):
            images = images.to(device, non_blocking=True)  # 使用non_blocking加速數據傳輸
            masks = masks.to(device, non_blocking=True)
            
            # 前向傳播 (使用混合精度)
            optimizer.zero_grad()
            
            if scaler:  # 使用混合精度 (float16)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    # 如果使用自訂損失函數，傳入中間層特徵
                    if custom_loss == "self_reg":
                        loss = criterion(outputs, masks, model.intermediate_features)
                    else:
                        loss = criterion(outputs, masks)
                
                # 混合精度反向傳播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # 常規訓練 (float32)
                outputs = model(images)
                # 如果使用自訂損失函數，傳入中間層特徵
                if custom_loss == "self_reg":
                    loss = criterion(outputs, masks, model.intermediate_features)
                else:
                    loss = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # 釋放一些記憶體
            if batch_idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 更新進度條
            train_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{epoch_train_loss/(batch_idx+1):.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
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
                
                # 如果使用自訂損失函數，傳入中間層特徵
                if custom_loss == "self_reg":
                    loss = criterion(outputs, masks, model.intermediate_features)
                else:
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
            torch.save(model.state_dict(), model_save_path)
            print(f'  ✓ 新的最佳 {model_type} 模型已保存 (Val Loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  ⚠ 驗證損失未改善 ({patience_counter}/{patience})')
        
        print("-" * 50)
        
        # 早停
        if patience_counter >= patience:
            print(f'早停觸發！在第 {epoch+1} 輪停止訓練')
            break
    
    # 載入最佳模型
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    print(f"{model_type} 訓練完成！")
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
        batch_size = 8
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
        'original': os.path.join(base_folder, 'original'),
        'ground_truth': os.path.join(base_folder, 'ground_truth'), 
        'predictions': os.path.join(base_folder, 'predictions'),
        'comparison': os.path.join(base_folder, 'comparison')
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    print(f"正在保存 {model_name} 預測結果到: {base_folder}")
    
    test_images_tensor = torch.FloatTensor(test_images).permute(0, 3, 1, 2).to(device)
    test_masks_tensor = torch.FloatTensor(test_masks).permute(0, 3, 1, 2).to(device)
    
    with torch.no_grad():
        batch_size = 8
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
                original = (batch_images_np[j] * 255).astype(np.uint8)
                ground_truth = (batch_masks_np[j][:,:,0] * 255).astype(np.uint8)  # 移除通道維度
                prediction = (pred_binary_np[j][:,:,0] * 255).astype(np.uint8)
                
                # 保存原始圖像
                original_img = Image.fromarray(original)
                original_img.save(os.path.join(folders['original'], f"{img_idx:04d}_original.png"))
                
                # 保存ground truth
                gt_img = Image.fromarray(ground_truth)
                gt_img.save(os.path.join(folders['ground_truth'], f"{img_idx:04d}_ground_truth.png"))
                
                # 保存二值化預測
                pred_img = Image.fromarray(prediction)
                pred_img.save(os.path.join(folders['predictions'], f"{img_idx:04d}_prediction.png"))
                
                # 創建比對圖像 (原圖 | Ground Truth | Prediction) - 128x128 大小
                # 水平拼接三張圖像，每張 128x128
                comparison_width = 128 * 3  # 384
                comparison_height = 128
                
                # 調整圖像大小到 128x128
                original_resized = Image.fromarray(original).resize((128, 128))
                gt_resized = Image.fromarray(ground_truth).resize((128, 128))
                pred_resized = Image.fromarray(prediction).resize((128, 128))
                
                # 創建比對圖像畫布
                comparison_img = Image.new('RGB', (comparison_width, comparison_height))
                
                # 將原圖轉換為RGB（如果需要）
                if original_resized.mode != 'RGB':
                    original_resized = original_resized.convert('RGB')
                
                # 將灰度圖轉換為RGB
                gt_rgb = Image.fromarray(np.stack([ground_truth, ground_truth, ground_truth], axis=-1)).resize((128, 128))
                pred_rgb = Image.fromarray(np.stack([prediction, prediction, prediction], axis=-1)).resize((128, 128))
                
                # 拼接圖像
                comparison_img.paste(original_resized, (0, 0))          # 原圖在左邊
                comparison_img.paste(gt_rgb, (128, 0))                  # Ground Truth在中間
                comparison_img.paste(pred_rgb, (256, 0))                # Prediction在右邊
                
                # 保存比對圖像
                comparison_img.save(os.path.join(folders['comparison'], f"{img_idx:04d}_comparison.png"))
                
                saved_count += 1
                pbar.update(1)
        
        pbar.close()
    print(f"\n{model_name} 預測結果保存完成!")
    print(f"共保存了 {saved_count} 張圖像")
    print(f"保存位置:")
    print(f"  原始圖像: {folders['original']}")
    print(f"  真實遮罩: {folders['ground_truth']}")
    print(f"  預測結果: {folders['predictions']}")
    print(f"  比對圖像: {folders['comparison']}")

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

class SelfRegLoss(nn.Module):
    def __init__(self):
        """
        自定義損失函數類，支持 Dice Loss 和中間層特徵組合
        優化批次處理和 GPU 效率，加強記憶體管理
        """
        super(SelfRegLoss, self).__init__()
        self.ce_loss_fn = CrossEntropyLoss()
        # 清除 CUDA 快取以確保初始化時記憶體乾淨
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def dice_loss(self, predictions, targets, smooth=1e-6):
        """
        計算 Dice Loss - 向量化處理以提高效率

        Args:
            predictions: 模型的預測結果
            targets: 真實標籤
            smooth: 平滑因子

        Returns:
            dice_loss: Dice Loss 值
        """
        # 確保 sigmoid 運算在正確的設備上進行
        predictions = torch.sigmoid(predictions)
        
        # 直接使用批次級矩陣操作
        intersection = torch.sum(predictions * targets, dim=[1, 2, 3])
        union = torch.sum(predictions * predictions, dim=[1, 2, 3]) + torch.sum(targets * targets, dim=[1, 2, 3])
        
        # 批次級 dice 係數
        dice_coef = (2. * intersection + smooth) / (union + smooth)
        
        # 取批次平均值
        return 1 - torch.mean(dice_coef)

    def forward(self, predictions, targets, intermediate_features):
        """
        計算損失並支持 backward

        Args:
            predictions: 模型的預測結果
            targets: 真實標籤
            intermediate_features: 中間層特徵列表

        Returns:
            loss: 計算出的損失值
        """

        # 使用 CUDA 自動混合精度計算主損失
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            ce_loss = self.ce_loss_fn(predictions, targets)
            dice_loss = self.dice_loss(predictions, targets)
        
        # 為了節省記憶體，使用 no_grad 計算輔助損失
        # 這樣可以避免保存中間計算的梯度，顯著減少記憶體使用
        with torch.no_grad():
            lsrc = self.LSRC(intermediate_features, intermediate_features[-1])
            lifd = self.LIFD(intermediate_features)
            
            # 轉換為純量值，避免保存計算圖
            lsrc_value = lsrc.item()
            lifd_value = lifd.item()
            
        # 使用純量值合併損失，避免梯度傳播到輔助損失
        loss = 0.4 * ce_loss + 0.6 * dice_loss + lsrc_value * 0.015 + lifd_value * 0.015

        del predictions, targets, intermediate_features
        
        # 釋放不需要的中間變數
        torch.cuda.empty_cache()
        
        return loss
        
    def LSRC(self, intermediate_features, final_layer):
        """
        LSRC (Layer-wise Self-Regularization Comparison) 函數
        批次化處理以提高效率

        Args:
            intermediate_features: 中間層特徵列表
            final_layer: 最後一層特徵

        Returns:
            lsrc_loss: LSRC 損失值
        """
        if len(intermediate_features) <= 1:
            # 確保 final_detached 被初始化，即使中間層特徵數量不足
            final_detached = final_layer.detach()
            return torch.tensor(0.0, device=final_layer.device)
        
        lsrc_loss = 0
        # 批次處理中間層
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            # 限制處理的層數，減少記憶體使用
            indices_to_process = list(range(0, len(intermediate_features) - 1))
            # 將 final_layer 分離一次，避免重複操作和記憶體浪費
            final_detached = final_layer.detach()
            
            for idx in indices_to_process:
                feature = intermediate_features[idx]
                
                # 使用 detach 分離特徵，避免保存梯度
                feature_detached = feature.detach()
                
                # 驗證特徵是否在相同設備
                if feature_detached.device != final_detached.device:
                    feature_detached = feature_detached.to(final_detached.device)
                
                # 使用自適應池化調整特徵大小
                pooled_final_layer = F.adaptive_avg_pool2d(final_detached, feature_detached.shape[2:])
                
                # 處理通道數不同的情況
                if feature_detached.size(1) != pooled_final_layer.size(1):
                    # 找出較小的通道數
                    min_channels = min(feature_detached.size(1), pooled_final_layer.size(1))
                    # 隨機選擇通道，確保兩者通道數相同
                    # 使用不同的隨機種子，避免每次選擇相同的通道子集
                    # 但在同一批次內保持一致性
                    seed = hash(str(idx) + "feature") % 10000
                    
                    if feature_detached.size(1) > min_channels:
                        torch.manual_seed(seed)
                        indices = torch.randperm(feature_detached.size(1))[:min_channels].to(feature_detached.device)
                        feature_channels = torch.index_select(feature_detached, 1, indices)
                    else:
                        feature_channels = feature_detached
                    
                    if pooled_final_layer.size(1) > min_channels:
                        torch.manual_seed(seed + 1)  # 使用不同的種子
                        indices = torch.randperm(pooled_final_layer.size(1))[:min_channels].to(pooled_final_layer.device)
                        pooled_channels = torch.index_select(pooled_final_layer, 1, indices)
                    else:
                        pooled_channels = pooled_final_layer
                else:
                    # 通道數相同，直接使用
                    feature_channels = feature_detached
                    pooled_channels = pooled_final_layer
                
                # 使用向量化操作計算損失
                l2_norm = F.mse_loss(feature_channels, pooled_channels)
                lsrc_loss += l2_norm
                
                # 釋放記憶體
                del feature_detached, pooled_final_layer
                del feature_channels, pooled_channels
                torch.cuda.empty_cache()
        
        return lsrc_loss / max(1, len(intermediate_features) - 1)
    
    def LIFD(self, intermediate_features):
        """
        LIFD (Layer-wise Intermediate Feature Decomposition) 函數
        優化批次處理效率

        Args:
            intermediate_features: 中間層特徵列表

        Returns:
            lifd_loss: LIFD 損失值
        """
        if not intermediate_features or len(intermediate_features) <= 1:
            # 處理空列表或只有一個元素的情況
            return torch.tensor(0.0, device=intermediate_features[0].device if intermediate_features else 'cpu')
        
        lifd_loss = 0
        
        indices_to_process = list(range(0, len(intermediate_features) - 1))
        
        # 更高效的批次處理
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for idx in indices_to_process:
                feature = intermediate_features[idx]
                # 分離特徵，避免保存梯度
                feature_detached = feature.detach()
                
                # 使用張量操作進行正規化和排序
                # 歸一化特徵（跨空間維度）
                norm = torch.norm(feature_detached, p=2, dim=(2, 3), keepdim=True)
                norm_features = feature_detached / (norm + 1e-8)
                
                # 計算通道重要性
                importance = norm_features.mean([0, 2, 3])
                
                # 排序通道
                sorted_indices = torch.argsort(importance, descending=True)
                
                # 重新排列特徵
                f_s = torch.index_select(feature_detached, 1, sorted_indices)
                
                # 分割特徵
                mid_point = f_s.shape[1] // 2
                
                # 確保有足夠的通道進行分割
                if mid_point > 0:
                    # 計算分解損失
                    first_half = f_s[:, :mid_point]
                    second_half = f_s[:, mid_point:2*mid_point] if 2*mid_point <= f_s.shape[1] else f_s[:, mid_point:]
                    
                    intra_fd_loss = F.mse_loss(first_half, second_half)
                    lifd_loss += intra_fd_loss
                    
                    # 釋放記憶體
                    del first_half, second_half
                
                # 釋放記憶體
                del feature_detached, norm, norm_features, importance, sorted_indices, f_s
                torch.cuda.empty_cache()
        
        # 防止除以零
        return lifd_loss / max(1, len(intermediate_features) - 1)