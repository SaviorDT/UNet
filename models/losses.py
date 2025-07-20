import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

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
