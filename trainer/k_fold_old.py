import torch
import numpy as np
import os
import time
from tqdm import tqdm
import pandas as pd
from data_helpers.Glas import load_data
from model import create_unet, create_lunext, train_model
import copy
from torch.utils.data import DataLoader, TensorDataset

def train_kfold_glas(model_type="LUNeXt", k_fold=5, times=3, folder="./data/Glas", 
                    target_size=(224, 224), epochs=100, batch_size=16, learning_rate=0.001,
                    eval_batch_size=4, custom_loss=None):
    """
    對Glas數據集進行K次K折交叉驗證訓練
    
    Args:
        model_type: 模型類型，可選 "UNet" 或 "LUNeXt"
        k_fold: K折交叉驗證的折數，默認為5
        times: 重複K折交叉驗證的次數，默認為3
        folder: 數據集根目錄路徑
        target_size: 目標圖像大小 (height, width)
        epochs: 訓練輪數
        batch_size: 訓練批次大小
        learning_rate: 學習率
        eval_batch_size: 評估時的批次大小，用於節省記憶體
    
    Returns:
        結果數據框，包含每次交叉驗證的結果
    """# 確保保存結果的目錄存在
    os.makedirs("./predictions", exist_ok=True)
    os.makedirs(f"./predictions/{model_type}", exist_ok=True)
    
    # 結果存儲
    all_results = []
    best_model_path = f"best_{model_type.lower()}_kfold_model.pth"
    best_dice = 0
    best_model = None
    best_test_metrics = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 清理 GPU 記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU 記憶體狀態: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    
    # 使用生成器獲取k折交叉驗證數據
    # 這樣可以避免一次性載入所有交叉驗證數據分割，優化內存使用
    k_fold_gen = load_data(folder=folder, target_size=target_size, k_fold=k_fold, times=times)
    
    # 保存測試數據以便最後的評估
    test_data_for_final = None
    
    # 遍歷每一次交叉驗證的每一折
    for train_data, val_data, test_data, t, fold_idx in k_fold_gen:
        # 保存一份測試數據用於最終評估
        if test_data_for_final is None:
            test_data_for_final = test_data
        print(f"\n----- 第 {t+1} 次交叉驗證, 第 {fold_idx+1}/{k_fold} 折 -----")
        
        # 強制清理 GPU 記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"訓練前 GPU 記憶體狀態: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        
        # 訓練模型
        print(f"訓練 {model_type} 模型...")
        start_time = time.time()
        model = train_model(
            train_data, val_data, model_type=model_type,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            device=device, custom_loss=custom_loss
        )
        train_time = time.time() - start_time
        print(f"訓練完成！耗時: {train_time:.2f} 秒")
        
        # 重設 GPU 的最大記憶體計數器
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # 評估模型
        model.eval()
        val_metrics = evaluate_model(model, val_data, device, batch_size=eval_batch_size)
        
        # 清理記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存結果
        result_record = {
            "Time": t+1,
            "Fold": fold_idx+1,
            "Model": model_type,
            "Dice": val_metrics["dice"],
            "IoU": val_metrics["iou"],
            "Precision": val_metrics["precision"],
            "Recall": val_metrics["recall"],
            "Accuracy": val_metrics["accuracy"],
            "F1": val_metrics["f1"],
            "Training Time": train_time
        }
        all_results.append(result_record)
        
        # 檢查是否為最佳模型
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型! Dice: {best_dice:.4f}")
        
        # 刪除當前模型以釋放記憶體
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 輸出當前記憶體使用情況
            print(f"訓練後 GPU 記憶體狀態: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    
    # 將結果轉換為DataFrame並保存
    results_df = pd.DataFrame(all_results)
    results_file = f"{model_type}_{k_fold}fold_{times}times_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n結果已保存到 {results_file}")
    
    # 統計摘要
    print("\n===== 統計摘要 =====")
    # 只對數值型欄位計算平均值，排除 'Time', 'Fold', 'Model' 等非數值欄位
    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
    # 從數值列中移除 'Time' 和 'Fold'，因為它們是分組欄位，不應參與平均計算
    metric_cols = [col for col in numeric_cols if col not in ['Time', 'Fold']]
    mean_results = results_df.groupby('Time')[metric_cols].mean().reset_index()
    overall_mean = results_df[metric_cols].mean()
    overall_std = results_df[metric_cols].std()
    
    print(f"每次交叉驗證的平均結果:")
    for _, row in mean_results.iterrows():
        print(f"第 {int(row['Time'])} 次: Dice={row['Dice']:.4f}, IoU={row['IoU']:.4f}")
    
    print(f"\n總體平均結果:")
    print(f"Dice: {overall_mean['Dice']:.4f} ± {overall_std['Dice']:.4f}")
    print(f"IoU: {overall_mean['IoU']:.4f} ± {overall_std['IoU']:.4f}")
    print(f"Precision: {overall_mean['Precision']:.4f} ± {overall_std['Precision']:.4f}")
    print(f"Recall: {overall_mean['Recall']:.4f} ± {overall_std['Recall']:.4f}")
    print(f"Accuracy: {overall_mean['Accuracy']:.4f} ± {overall_std['Accuracy']:.4f}")
    print(f"F1: {overall_mean['F1']:.4f} ± {overall_std['F1']:.4f}")
    print(f"平均訓練時間: {overall_mean['Training Time']:.2f} ± {overall_std['Training Time']:.2f} 秒")
    
    # 測試最佳模型在測試數據上的表現
    print("\n===== 使用最佳模型在測試數據上進行評估 =====")
    if best_model is not None and test_data_for_final is not None:
        # 確保最佳模型在正確的設備上
        best_model = best_model.to(device)
        best_metrics = evaluate_model(best_model, test_data_for_final, device, batch_size=max(1, eval_batch_size//2))  # 使用更小的批次大小
        print(f"最佳模型在測試數據上的表現:")
        print(f"Dice: {best_metrics['dice']:.4f}")
        print(f"IoU: {best_metrics['iou']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"F1: {best_metrics['f1']:.4f}")
        
        # 清理記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results_df, best_model_path

def evaluate_model(model, data, device, batch_size=8):
    """
    評估模型在給定數據上的性能
    
    Args:
        model: 訓練好的模型
        data: (images, masks) 元組
        device: 運行設備
        batch_size: 批次大小，用於分批處理以節省記憶體
        
    Returns:
        dict: 包含各種評估指標的字典
    """
    images, masks = data
    
    # 確保模型在評估模式
    model.eval()
    
    # 創建更高效的數據載入器
    # 轉換為 torch tensors
    images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    # 使用 TensorDataset 而不是臨時創建大量張量
    eval_dataset = TensorDataset(images_tensor)
    
    # 較低的 num_workers 避免過多進程消耗資源
    num_workers = 0  # 在評估時減少工作進程以避免記憶體洩漏
    pin_memory = torch.cuda.is_available()
    # 使用較小的 prefetch_factor 減少記憶體使用
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    # 分批處理以節省記憶體
    all_predictions = []
    
    # 清理 GPU 記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 使用混合精度進行推理，速度更快
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        for batch_images in eval_loader:
            # 獲取當前批次
            batch_images = batch_images[0].to(device, non_blocking=True)
            
            # 獲取預測結果
            outputs = model(batch_images)
            outputs = torch.sigmoid(outputs)
            batch_predictions = (outputs > 0.5).float().cpu().numpy()
            
            # 立即將結果轉移到 CPU 並轉為 NumPy 以釋放 GPU 記憶體
            all_predictions.append(batch_predictions)
            
            # 顯式刪除不需要的張量
            del batch_images, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 合併所有預測結果
    predictions = np.concatenate(all_predictions, axis=0)
    # 釋放記憶體
    del all_predictions
    
    # 轉換 masks 為 numpy
    masks_np = masks.squeeze() if masks.ndim > 2 else masks
    predictions_np = predictions.squeeze() if predictions.ndim > 2 else predictions
    
    # 計算評估指標
    metrics = calculate_metrics(masks_np, predictions_np)
    
    # 顯式刪除中間變量
    del predictions, predictions_np, masks_np, images_tensor, eval_dataset
    
    return metrics

def calculate_metrics(y_true, y_pred):
    """
    計算各種評估指標
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        
    Returns:
        dict: 包含各種評估指標的字典
    """
    # 確保輸入是2D
    if y_true.ndim > 2:
        y_true = y_true.squeeze()
    if y_pred.ndim > 2:
        y_pred = y_pred.squeeze()
    
    # 使用布爾運算減少記憶體使用
    true_positive = np.logical_and(y_pred == 1, y_true == 1)
    true_negative = np.logical_and(y_pred == 0, y_true == 0)
    false_positive = np.logical_and(y_pred == 1, y_true == 0)
    false_negative = np.logical_and(y_pred == 0, y_true == 1)
    
    # 計算混淆矩陣相關值
    tp = np.sum(true_positive)
    tn = np.sum(true_negative)
    fp = np.sum(false_positive)
    fn = np.sum(false_negative)
    
    # 釋放記憶體
    del true_positive, true_negative, false_positive, false_negative
    
    # 避免除零錯誤
    eps = 1e-8
    
    # 計算度量
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'f1': float(f1)
    }

def main():
    """
    主函數：執行Glas數據集的K次K折交叉驗證
    """
    # 配置參數
    MODEL_TYPE = "LUNeXt"  # 可選: "UNet" 或 "LUNeXt"
    K_FOLD = 5             # 交叉驗證折數
    TIMES = 3              # 重複次數
    DATASET_FOLDER = "./data/Glas"  # 數據集根目錄
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    
    # 自動調整評估批次大小，根據模型類型和可用內存調整
    if MODEL_TYPE == "LUNeXt":  # LUNeXt 通常需要更多的記憶體
        EVAL_BATCH_SIZE = 4
    else:  # UNet 通常需要較少的記憶體
        EVAL_BATCH_SIZE = 8
    
    print(f"開始 {TIMES} 次 {K_FOLD} 折交叉驗證訓練 - 使用 {MODEL_TYPE} 模型")
    
    # 顯示當前系統記憶體情況（如果可用）
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"系統記憶體: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent}%)")
    except ImportError:
        print("未安裝 psutil 套件，無法顯示系統記憶體使用情況")
    
    if torch.cuda.is_available():
        # GPU資訊
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.memory_reserved(0)/1024**3:.1f} GB 已保留")
    
    try:
        results_df, best_model_path = train_kfold_glas(
            model_type=MODEL_TYPE, 
            k_fold=K_FOLD, 
            times=TIMES, 
            folder=DATASET_FOLDER, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            learning_rate=LEARNING_RATE,
            eval_batch_size=EVAL_BATCH_SIZE
        )
        
        print(f"\n訓練完成！最佳模型已保存為 {best_model_path}")
        print("可以使用此模型進行後續的推理或測試。")
    
    finally:
        # 確保記憶體被釋放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"最終 GPU 記憶體: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
