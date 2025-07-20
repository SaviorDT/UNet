import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import skeletonize
from skimage import measure
import cv2
from test_helper.utils import calculate_metrics
from tqdm import tqdm

def skeletonize_mask(mask):
    """
    对二值化掩码进行骨架化处理
    
    Args:
        mask: 二值化掩码 (numpy array, 值为 0 或 1)
        
    Returns:
        skeleton: 骨架化后的掩码 (numpy array, 值为 0 或 1)
    """
    # 确保输入是二值化的
    if mask.max() > 1:
        mask = (mask > 0.5).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # 使用 scikit-image 的 skeletonize 函数
    skeleton = skeletonize(mask)
    
    return skeleton.astype(np.float32)

def calculate_skeleton_metrics(pred_skeleton, gt_skeleton, smooth=1e-6):
    """
    计算骨架相似度的各种评估指标
    
    Args:
        pred_skeleton: 预测结果的骨架 (numpy array)
        gt_skeleton: ground truth的骨架 (numpy array)
        smooth: 平滑因子
        
    Returns:
        dict: 包含各种骨架相似度指标的字典
    """
    # 确保形状一致
    if pred_skeleton.shape != gt_skeleton.shape:
        print(f"警告: 骨架形状不匹配 - pred: {pred_skeleton.shape}, gt: {gt_skeleton.shape}")
        # 调整尺寸以匹配
        if len(pred_skeleton.shape) == 2:
            pred_skeleton = cv2.resize(pred_skeleton, (gt_skeleton.shape[1], gt_skeleton.shape[0]))
        
    # 展平为1D数组进行计算
    pred_flat = pred_skeleton.flatten()
    gt_flat = gt_skeleton.flatten()
    
    # 计算基本统计量
    tp = np.sum(pred_flat * gt_flat)  # True Positive
    fp = np.sum(pred_flat * (1 - gt_flat))  # False Positive
    fn = np.sum((1 - pred_flat) * gt_flat)  # False Negative
    tn = np.sum((1 - pred_flat) * (1 - gt_flat))  # True Negative
    
    # 骨架特定的度量
    # 1. Skeleton Dice Coefficient
    skeleton_dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    
    # 2. Skeleton IoU
    skeleton_iou = (tp + smooth) / (tp + fp + fn + smooth)
    
    # 3. Skeleton Precision
    skeleton_precision = (tp + smooth) / (tp + fp + smooth)
    
    # 4. Skeleton Recall (Sensitivity)
    skeleton_recall = (tp + smooth) / (tp + fn + smooth)
    
    # 5. Skeleton Accuracy
    skeleton_accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    
    # 6. Skeleton F1-Score (等同于 Skeleton Dice)
    skeleton_f1 = skeleton_dice
    
    # 7. 骨架完整性度量 (Skeleton Completeness)
    # 衡量有多少真实骨架被正确检测到
    gt_skeleton_pixels = np.sum(gt_flat)
    detected_skeleton_pixels = np.sum(pred_flat * gt_flat)
    skeleton_completeness = (detected_skeleton_pixels + smooth) / (gt_skeleton_pixels + smooth) if gt_skeleton_pixels > 0 else 0.0
    
    # 8. 骨架准确性度量 (Skeleton Correctness)
    # 衡量检测到的骨架中有多少是正确的
    pred_skeleton_pixels = np.sum(pred_flat)
    skeleton_correctness = (detected_skeleton_pixels + smooth) / (pred_skeleton_pixels + smooth) if pred_skeleton_pixels > 0 else 0.0
    
    # 9. 基于距离的骨架相似度 (可选的更高级度量)
    skeleton_distance_score = calculate_skeleton_distance_similarity(pred_skeleton, gt_skeleton)
    
    return {
        'skeleton_dice': float(skeleton_dice),
        'skeleton_iou': float(skeleton_iou),
        'skeleton_precision': float(skeleton_precision),
        'skeleton_recall': float(skeleton_recall),
        'skeleton_accuracy': float(skeleton_accuracy),
        'skeleton_f1': float(skeleton_f1),
        'skeleton_completeness': float(skeleton_completeness),
        'skeleton_correctness': float(skeleton_correctness),
        'skeleton_distance_score': float(skeleton_distance_score),
        'tp': float(tp),
        'fp': float(fp),
        'tn': float(tn),
        'fn': float(fn)
    }

def calculate_skeleton_distance_similarity(pred_skeleton, gt_skeleton, max_distance=10):
    """
    计算基于距离的骨架相似度
    
    Args:
        pred_skeleton: 预测骨架
        gt_skeleton: ground truth骨架
        max_distance: 最大允许距离
        
    Returns:
        float: 距离相似度分数 (0-1)
    """
    try:
        # 找到骨架点的坐标
        pred_points = np.column_stack(np.where(pred_skeleton > 0))
        gt_points = np.column_stack(np.where(gt_skeleton > 0))
        
        if len(pred_points) == 0 or len(gt_points) == 0:
            return 0.0
        
        # 计算每个预测点到最近ground truth点的距离
        distances = []
        for pred_point in pred_points:
            # 计算到所有GT点的距离
            dists_to_gt = np.sqrt(np.sum((gt_points - pred_point) ** 2, axis=1))
            min_dist = np.min(dists_to_gt)
            distances.append(min_dist)
        
        # 计算相似度分数
        distances = np.array(distances)
        # 使用指数衰减函数，距离越小分数越高
        scores = np.exp(-distances / max_distance)
        similarity_score = np.mean(scores)
        
        return similarity_score
        
    except Exception as e:
        print(f"计算距离相似度时出错: {e}")
        return 0.0

def evaluate_model_skeleton(model, test_data, device, batch_size=1):
    """
    使用骨架化方法评估模型
    
    Args:
        model: 训练好的模型
        test_data: 测试数据 (images, masks)
        device: 运行设备
        batch_size: 批次大小
        
    Returns:
        dict: 包含骨架评估指标的字典
    """
    model.eval()
    
    test_images, test_masks = test_data
    
    if len(test_images) == 0:
        print("警告: 没有测试数据")
        return {}
    
    print("开始骨架化评估...")
    
    # 转换为 torch tensors
    test_images_tensor = torch.FloatTensor(test_images).permute(0, 3, 1, 2).to(device)
    
    # 初始化累积指标
    total_skeleton_metrics = {
        'skeleton_dice': 0.0,
        'skeleton_iou': 0.0,
        'skeleton_precision': 0.0,
        'skeleton_recall': 0.0,
        'skeleton_accuracy': 0.0,
        'skeleton_f1': 0.0,
        'skeleton_completeness': 0.0,
        'skeleton_correctness': 0.0,
        'skeleton_distance_score': 0.0,
        'tp': 0.0, 'fp': 0.0, 'tn': 0.0, 'fn': 0.0
    }
    
    # 同时计算传统指标用于比较
    total_traditional_metrics = {
        'dice': 0.0, 'iou': 0.0, 'accuracy': 0.0,
        'precision': 0.0, 'recall': 0.0
    }
    
    num_processed = 0
    
    with torch.no_grad():
        # 使用进度条显示处理进度
        pbar = tqdm(total=len(test_images), desc="骨架化评估", unit="images")
        
        for i in range(0, len(test_images_tensor), batch_size):
            batch_images = test_images_tensor[i:i+batch_size]
            batch_masks = test_masks[i:i+batch_size]
            
            # 获取模型预测
            outputs = model(batch_images)
            pred_probs = torch.sigmoid(outputs)
            pred_binary = (pred_probs > 0.5).float()
            
            # 转换为 numpy 进行骨架化处理
            pred_binary_np = pred_binary.cpu().numpy()
            
            # 处理每个样本
            for j in range(len(batch_images)):
                # 获取预测和真实掩码
                pred_mask = pred_binary_np[j].squeeze()  # 移除通道维度
                gt_mask = batch_masks[j].squeeze()  # 移除通道维度
                
                # 确保是2D数组
                if len(pred_mask.shape) > 2:
                    pred_mask = pred_mask[0]  # 取第一个通道
                if len(gt_mask.shape) > 2:
                    gt_mask = gt_mask[:, :, 0] if gt_mask.shape[2] == 1 else gt_mask[0]
                
                # 生成骨架
                try:
                    pred_skeleton = skeletonize_mask(pred_mask)
                    gt_skeleton = skeletonize_mask(gt_mask)
                    
                    # 计算骨架指标
                    skeleton_metrics = calculate_skeleton_metrics(pred_skeleton, gt_skeleton)
                    
                    # 累积骨架指标
                    for key in total_skeleton_metrics:
                        total_skeleton_metrics[key] += skeleton_metrics[key]
                    
                    # 计算传统指标用于比较
                    pred_tensor = torch.FloatTensor(pred_mask).unsqueeze(0).unsqueeze(0)
                    gt_tensor = torch.FloatTensor(gt_mask).unsqueeze(0).unsqueeze(0)
                    traditional_metrics = calculate_metrics(pred_tensor, gt_tensor)
                    
                    for key in total_traditional_metrics:
                        total_traditional_metrics[key] += traditional_metrics[key]
                    
                    num_processed += 1
                    
                except Exception as e:
                    print(f"处理第 {num_processed} 个样本时出错: {e}")
                    continue
                
                pbar.update(1)
        
        pbar.close()
    
    if num_processed == 0:
        print("错误: 没有成功处理任何样本")
        return {}
    
    # 计算平均指标
    avg_skeleton_metrics = {key: value / num_processed for key, value in total_skeleton_metrics.items()}
    avg_traditional_metrics = {key: value / num_processed for key, value in total_traditional_metrics.items()}
    
    # 合并结果
    final_results = {
        'num_samples': num_processed,
        'skeleton_metrics': avg_skeleton_metrics,
        'traditional_metrics': avg_traditional_metrics
    }
    
    return final_results

def save_skeleton_results(skeleton_results, train_time, filepath):
    """
    保存骨架化评估结果
    
    Args:
        skeleton_results: 骨架评估结果
        train_time: 训练时间
        filepath: 结果保存路径
    """
    if not skeleton_results:
        print("警告: 没有骨架评估结果可保存")
        return
    
    skeleton_metrics = skeleton_results.get('skeleton_metrics', {})
    traditional_metrics = skeleton_results.get('traditional_metrics', {})
    num_samples = skeleton_results.get('num_samples', 0)
    
    # 保存结果到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("骨架化评估结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("基本信息:\n")
        f.write(f"训练时间: {train_time:.2f} 秒\n")
        f.write(f"测试样本数: {num_samples}\n")
        f.write(f"评估方法: 骨架化相似度评估\n\n")
        
        f.write("骨架化评估指标:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Skeleton Dice Coefficient: {skeleton_metrics.get('skeleton_dice', 0):.6f}\n")
        f.write(f"Skeleton IoU: {skeleton_metrics.get('skeleton_iou', 0):.6f}\n")
        f.write(f"Skeleton Precision: {skeleton_metrics.get('skeleton_precision', 0):.6f}\n")
        f.write(f"Skeleton Recall: {skeleton_metrics.get('skeleton_recall', 0):.6f}\n")
        f.write(f"Skeleton Accuracy: {skeleton_metrics.get('skeleton_accuracy', 0):.6f}\n")
        f.write(f"Skeleton F1-Score: {skeleton_metrics.get('skeleton_f1', 0):.6f}\n")
        f.write(f"Skeleton Completeness: {skeleton_metrics.get('skeleton_completeness', 0):.6f}\n")
        f.write(f"Skeleton Correctness: {skeleton_metrics.get('skeleton_correctness', 0):.6f}\n")
        f.write(f"Skeleton Distance Score: {skeleton_metrics.get('skeleton_distance_score', 0):.6f}\n\n")
        
        f.write("传统评估指标 (用于比较):\n")
        f.write("-" * 40 + "\n")
        f.write(f"Traditional Dice: {traditional_metrics.get('dice', 0):.6f}\n")
        f.write(f"Traditional IoU: {traditional_metrics.get('iou', 0):.6f}\n")
        f.write(f"Traditional Precision: {traditional_metrics.get('precision', 0):.6f}\n")
        f.write(f"Traditional Recall: {traditional_metrics.get('recall', 0):.6f}\n")
        f.write(f"Traditional Accuracy: {traditional_metrics.get('accuracy', 0):.6f}\n\n")
        
        f.write("骨架混淆矩阵统计:\n")
        f.write("-" * 40 + "\n")
        f.write(f"True Positive: {skeleton_metrics.get('tp', 0):.2f}\n")
        f.write(f"False Positive: {skeleton_metrics.get('fp', 0):.2f}\n")
        f.write(f"True Negative: {skeleton_metrics.get('tn', 0):.2f}\n")
        f.write(f"False Negative: {skeleton_metrics.get('fn', 0):.2f}\n\n")
        
        f.write("评估说明:\n")
        f.write("-" * 40 + "\n")
        f.write("骨架化评估专注于分割结果的拓扑结构和连通性，\n")
        f.write("特别适用于细长结构如血管、神经纤维等的分割评估。\n")
        f.write("Skeleton Completeness: 衡量真实骨架的检测完整性\n")
        f.write("Skeleton Correctness: 衡量检测骨架的准确性\n")
        f.write("Distance Score: 基于距离的骨架匹配度\n")
    
    # 在控制台也显示结果
    print("\n" + "=" * 60)
    print("骨架化评估完成！")
    print("=" * 60)
    print(f"测试样本数: {num_samples}")
    print("-" * 40)
    print("主要骨架评估指标:")
    print(f"  Skeleton Dice: {skeleton_metrics.get('skeleton_dice', 0):.6f}")
    print(f"  Skeleton IoU: {skeleton_metrics.get('skeleton_iou', 0):.6f}")
    print(f"  Skeleton Completeness: {skeleton_metrics.get('skeleton_completeness', 0):.6f}")
    print(f"  Skeleton Correctness: {skeleton_metrics.get('skeleton_correctness', 0):.6f}")
    print(f"  Distance Score: {skeleton_metrics.get('skeleton_distance_score', 0):.6f}")
    print("-" * 40)
    print("传统指标 (比较参考):")
    print(f"  Traditional Dice: {traditional_metrics.get('dice', 0):.6f}")
    print(f"  Traditional IoU: {traditional_metrics.get('iou', 0):.6f}")
    print("=" * 60)
    
    print(f"详细结果已保存至: {filepath}")

def save_skeleton_results_with_images(model, test_data, train_time, filepath, model_name="Model"):
    """
    增强版的骨架化结果保存函数，同时保存评估指标和骨架化图像
    
    Args:
        model: 训练好的模型
        test_data: 测试数据
        train_time: 训练时间
        filepath: 结果文本文件保存路径
        model_name: 模型名称
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 首先进行骨架化评估
    skeleton_results = evaluate_model_skeleton(model, test_data, device, batch_size=1)
    
    # 保存骨架化评估指标
    save_skeleton_results(skeleton_results, train_time, filepath)
    
    # 保存骨架化图像（可选实现）
    save_skeleton_prediction_images(model, test_data, output_folder="predictions", model_name=f"{model_name}_Skeleton")

def save_skeleton_prediction_images(model, test_data, output_folder="predictions", model_name="Model_Skeleton"):
    """
    保存模型預測結果和骨架化圖像
    
    Args:
        model: 訓練好的模型
        test_data: 測試數據 (images, masks) tuple
        output_folder: 輸出文件夾路徑
        model_name: 模型名稱
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from pathlib import Path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    test_images, test_masks = test_data
    
    # 創建保存目錄
    save_dir = Path(output_folder) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在保存骨架化預測圖像到 {save_dir}")
    
    # 限制保存的圖像數量，避免生成過多文件
    max_images = min(20, len(test_images))
    
    with torch.no_grad():
        for i in tqdm(range(max_images), desc="保存骨架化圖像"):
            # 準備輸入數據
            image = test_images[i]
            mask = test_masks[i]
            
            # 轉換為 tensor 並添加批次維度
            if isinstance(image, np.ndarray):
                image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
            else:
                image_tensor = image.unsqueeze(0).to(device)
            
            # 模型預測
            output = model(image_tensor)
            pred_prob = torch.sigmoid(output)
            pred_mask = (pred_prob > 0.5).float()
            
            # 轉換為 numpy 數組
            image_np = image if isinstance(image, np.ndarray) else image.permute(1, 2, 0).cpu().numpy()
            pred_mask_np = pred_mask.squeeze().cpu().numpy()
            true_mask_np = mask.squeeze() if hasattr(mask, 'squeeze') else mask
            if hasattr(true_mask_np, 'cpu'):
                true_mask_np = true_mask_np.cpu().numpy()
            
            # 確保是2D數組
            if len(pred_mask_np.shape) > 2:
                pred_mask_np = pred_mask_np[0] if pred_mask_np.shape[0] == 1 else pred_mask_np
            if len(true_mask_np.shape) > 2:
                true_mask_np = true_mask_np[:, :, 0] if true_mask_np.shape[2] == 1 else true_mask_np[0]
            
            # 生成骨架
            try:
                pred_skeleton = skeletonize_mask(pred_mask_np)
                true_skeleton = skeletonize_mask(true_mask_np)
                
                # 創建對比圖
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'Sample {i+1} - Skeleton Analysis', fontsize=14, fontweight='bold')
                
                # 第一行：原圖、真實掩碼、預測掩碼
                axes[0, 0].imshow(image_np)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(true_mask_np, cmap='gray', vmin=0, vmax=1)
                axes[0, 1].set_title('Ground Truth Mask')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(pred_mask_np, cmap='gray', vmin=0, vmax=1)
                axes[0, 2].set_title('Predicted Mask')
                axes[0, 2].axis('off')
                
                # 第二行：真實骨架、預測骨架、骨架重疊
                axes[1, 0].imshow(true_skeleton, cmap='gray', vmin=0, vmax=1)
                axes[1, 0].set_title('Ground Truth Skeleton')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(pred_skeleton, cmap='gray', vmin=0, vmax=1)
                axes[1, 1].set_title('Predicted Skeleton')
                axes[1, 1].axis('off')
                
                # 骨架重疊圖
                skeleton_overlay = np.zeros((*true_skeleton.shape, 3))
                skeleton_overlay[true_skeleton > 0] = [1, 0, 0]  # 紅色：真實骨架
                skeleton_overlay[pred_skeleton > 0] += [0, 0, 1]  # 藍色：預測骨架
                # 重疊區域會變成紫色 [1, 0, 1]
                
                axes[1, 2].imshow(skeleton_overlay)
                axes[1, 2].set_title('Skeleton Overlay\n(Red: GT, Blue: Pred, Purple: Match)')
                axes[1, 2].axis('off')
                
                # 計算並顯示骨架指標
                skeleton_metrics = calculate_skeleton_metrics(pred_skeleton, true_skeleton)
                
                # 在圖像上添加指標文字
                metrics_text = f"""Skeleton Metrics:
Dice: {skeleton_metrics['skeleton_dice']:.3f}
IoU: {skeleton_metrics['skeleton_iou']:.3f}
Completeness: {skeleton_metrics['skeleton_completeness']:.3f}
Correctness: {skeleton_metrics['skeleton_correctness']:.3f}
Distance Score: {skeleton_metrics['skeleton_distance_score']:.3f}"""
                
                fig.text(0.02, 0.02, metrics_text, fontsize=9, verticalalignment='bottom',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(save_dir / f'sample_{i+1:03d}_skeleton_analysis.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"處理第 {i+1} 個樣本時出錯: {e}")
                continue
    
    print(f"骨架化圖像保存完成！共保存了 {max_images} 個樣本的分析圖像")