import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

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
        batch_size = 1
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
        batch_size = 1
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
                
                # 獲取原始圖像尺寸
                height, width = original.shape[:2]
                
                # 保存原始圖像
                original_img = Image.fromarray(original)
                original_img.save(os.path.join(folders['original'], f"{img_idx:04d}_original.png"))
                
                # 保存ground truth
                gt_img = Image.fromarray(ground_truth)
                gt_img.save(os.path.join(folders['ground_truth'], f"{img_idx:04d}_ground_truth.png"))
                
                # 保存二值化預測
                pred_img = Image.fromarray(prediction)
                pred_img.save(os.path.join(folders['predictions'], f"{img_idx:04d}_prediction.png"))
                
                # 創建比對圖像 - 保持原始畫質
                # 水平拼接三張圖像，使用原始尺寸
                comparison_width = width * 3
                comparison_height = height
                
                # 創建比對圖像畫布
                comparison_img = Image.new('RGB', (comparison_width, comparison_height))
                
                # 將原圖轉換為RGB（如果需要）
                if original_img.mode != 'RGB':
                    original_rgb = original_img.convert('RGB')
                else:
                    original_rgb = original_img
                
                # 將灰度圖轉換為RGB，保持原始尺寸
                gt_rgb = Image.fromarray(np.stack([ground_truth, ground_truth, ground_truth], axis=-1))
                pred_rgb = Image.fromarray(np.stack([prediction, prediction, prediction], axis=-1))
                
                # 拼接圖像，保持原始畫質
                comparison_img.paste(original_rgb, (0, 0))           # 原圖在左邊
                comparison_img.paste(gt_rgb, (width, 0))             # Ground Truth在中間
                comparison_img.paste(pred_rgb, (width * 2, 0))       # Prediction在右邊
                
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