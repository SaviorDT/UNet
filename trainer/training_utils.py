import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.model_utils import create_unet, create_lunext
from models.losses import SelfRegLoss
from losses.cl_dice import soft_dice_cldice

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
    num_workers = 3  # 可根據 CPU 核心數調整
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
    elif custom_loss == "cl_dice":
        criterion = soft_dice_cldice()
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
