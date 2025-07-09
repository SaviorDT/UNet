import torch
import numpy as np
import os
import time
from tqdm import tqdm
import pandas as pd
from model import create_unet, create_lunext, train_model
import copy
from torch.utils.data import DataLoader, TensorDataset
import gc
from data_helpers.utils import split_dataset_kfold
from data_helpers.data import load_data

def train_kfold(model_type="UNet", k_fold=5, times=3, dataset="Glas", folder="./data/Glas", 
                    target_size=(224, 224), epochs=100, batch_size=8, learning_rate=0.001,
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
    
    # 清理並重置 GPU 狀態
    if torch.cuda.is_available():
        # 完全重置 CUDA 環境
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        
        # 設置 GPU 記憶體分配器
        try:
            # 嘗試設置較為激進的記憶體釋放策略
            torch.cuda.memory.set_per_process_memory_fraction(0.8)  # 限制使用不超過 80% 的 GPU 記憶體
            print("已設置 GPU 記憶體限制為 80%")
        except:
            print("無法設置 GPU 記憶體限制，使用預設值")
        
        # 顯示記憶體狀態
        print(f"GPU 記憶體狀態: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    
    # 使用生成器獲取k折交叉驗證數據
    # if dataset == "Glas":
    #     from data_helpers.Glas import load_data
    #     origin_train_data, origin_val_data, origin_test_data = load_data(folder=folder, target_size=target_size)
    # elif dataset == "ISIC2018":
    #     from data_helpers.ISIC2018 import load_data
    #     origin_train_data, origin_val_data, origin_test_data = load_data(folder=folder, target_size=target_size)
    # elif dataset == "my_proj1":
    #     from data_helpers.my_proj1 import load_data_with_random_split
    #     origin_train_data, origin_val_data, origin_test_data = load_data_with_random_split(folder=folder, target_size=target_size, train_ratio=0.8, val_ratio=0)
    # else:
    #     raise ValueError(f"未知數據集: {dataset}. 可選值為 'Glas', 'ISIC2018', 'my_proj1'。")
    origin_train_data, origin_val_data, origin_test_data = load_data(dataset_name=dataset, folder=folder, target_size=target_size)
    origin_train_image, origin_train_mask = origin_train_data
    origin_val_image, origin_val_mask = origin_val_data
    
    # 將驗證數據合併到訓練數據中
    combined_train_image = np.concatenate([origin_train_image, origin_val_image], axis=0)
    combined_train_mask = np.concatenate([origin_train_mask, origin_val_mask], axis=0)
    print(f"合併後的訓練數據形狀: 圖像 {combined_train_image.shape}, 掩碼 {combined_train_mask.shape}")
    
    k_fold_gen = split_dataset_kfold(combined_train_image, combined_train_mask, k_fold=k_fold, times=times)
    
    # 保存測試數據以便最後的評估
    test_data_for_final = origin_test_data
    
    # 遍歷每一次交叉驗證的每一折
    for train_data, val_data, t, fold_idx in k_fold_gen:
        print(f"\n----- 第 {t+1} 次交叉驗證, 第 {fold_idx+1}/{k_fold} 折 -----")
        
        # 重啟 Python 垃圾回收
        gc.collect()
        
        # 強制清理 GPU 記憶體並重置統計資訊
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"訓練前 GPU 記憶體狀態: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
            
            # 檢查 GPU 溫度 (如果可能)
            # try:
            #     import pynvml
            #     pynvml.nvmlInit()
            #     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            #     temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            #     print(f"GPU 溫度: {temp}°C")
            # except:
            #     print("無法檢查 GPU 溫度")
        
        # 訓練模型 (添加自動批次大小縮放)
        print(f"訓練 {model_type} 模型...")
        start_time = time.time()
        try:
            model = train_model(
                train_data, val_data, model_type=model_type,
                epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                device=device, custom_loss=custom_loss
            )
        except RuntimeError as e:
            raise e
            # if "out of memory" in str(e):
            #     print(f"GPU 記憶體不足，嘗試減小批次大小...")
            #     # 嘗試減小批次大小重新訓練
            #     reduced_batch = max(1, batch_size // 2)
            #     print(f"使用較小的批次大小: {reduced_batch}")
            #     if torch.cuda.is_available():
            #         torch.cuda.empty_cache()
            #     model = train_model(
            #         train_data, val_data, model_type=model_type,
            #         epochs=epochs, batch_size=reduced_batch, learning_rate=learning_rate,
            #         device=device, custom_loss=custom_loss
            #     )
            # else:
            #     raise e
        
        train_time = time.time() - start_time
        print(f"訓練完成！耗時: {train_time:.2f} 秒")
        
        # 評估模型（先清理記憶體）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 重啟 Python 垃圾回收
        gc.collect()
        
        # 評估模型
        model.eval()
        val_metrics = evaluate_model(model, val_data, device, batch_size=eval_batch_size)
        
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
        
        # 顯式刪除當前模型以釋放記憶體
        del model
        gc.collect()  # 強制垃圾回收
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # 檢查 CUDA 記憶體分配情況
            alloc = torch.cuda.memory_allocated()/1024**3
            max_alloc = torch.cuda.max_memory_allocated()/1024**3
            print(f"訓練後 GPU 記憶體狀態: {alloc:.2f} GB / {max_alloc:.2f} GB")
            # 如果記憶體釋放不完全，嘗試重置 CUDA
            if alloc > 0.1:  # 如果仍有超過 100MB 未釋放
                print("記憶體未完全釋放，嘗試重置 CUDA...")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
    
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
    
    # 轉換為 torch tensors (預處理數據以提高效率)
    try:
        # 如果設備是 cuda，使用更激進的預處理和內存管理
        # 分批處理數據轉換，避免一次性大量數據操作
        total_samples = len(images)
        predictions_list = []
        chunk_size = min(100, total_samples)  # 每次處理最多 100 張圖片
        
        for i in range(0, total_samples, chunk_size):
            end_idx = min(i + chunk_size, total_samples)
            chunk_images = images[i:end_idx]
            
            # 優化: 依序處理數據，避免大量臨時內存分配
            images_tensor = torch.FloatTensor(chunk_images).permute(0, 3, 1, 2)
            eval_dataset = TensorDataset(images_tensor)
            
            # 使用無工作進程的 DataLoader 避免 worker 泄漏
            eval_loader = DataLoader(
                eval_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=0,  # 避免工作進程累積
                pin_memory=torch.cuda.is_available()
            )
            
            # 推理階段，使用混合精度
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                chunk_predictions = []
                
                for batch_images in eval_loader:
                    batch_images = batch_images[0].to(device, non_blocking=True)
                    
                    # 執行推理
                    outputs = model(batch_images)
                    outputs = torch.sigmoid(outputs)
                    batch_predictions = (outputs > 0.5).float().cpu().numpy()
                    
                    # 釋放內存
                    chunk_predictions.append(batch_predictions)
                    del batch_images, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 合併批次結果
            chunk_result = np.concatenate(chunk_predictions, axis=0)
            predictions_list.append(chunk_result)
            
            # 手動清理內存
            del chunk_images, images_tensor, eval_dataset, eval_loader, chunk_predictions
            
            # 強制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 合併所有區塊結果
        predictions = np.concatenate(predictions_list, axis=0)
        
        # 清理內存
        del predictions_list
    
    except Exception as e:
        print(f"分批評估過程中出錯: {e}")
        # print("嘗試使用更簡單的評估方法...")
        
        # # 備用方案: 使用較簡單的方法進行推理
        # images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2)
        
        # # 還是分批處理，但是更簡單的結構
        # eval_dataset = TensorDataset(images_tensor)
        # eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        # all_predictions = []
        # with torch.no_grad():
        #     for batch_images in eval_loader:
        #         batch_images = batch_images[0].to(device)
        #         outputs = model(batch_images)
        #         outputs = torch.sigmoid(outputs)
        #         batch_predictions = (outputs > 0.5).float().cpu().numpy()
        #         all_predictions.append(batch_predictions)
        #         del batch_images, outputs
        
        # predictions = np.concatenate(all_predictions, axis=0)
        # del all_predictions, images_tensor, eval_dataset
    
    # 轉換 masks 為 numpy (必要時進行維度調整)
    masks_np = masks.squeeze() if masks.ndim > 2 else masks
    predictions_np = predictions.squeeze() if predictions.ndim > 2 else predictions
    
    # 計算評估指標
    metrics = calculate_metrics(masks_np, predictions_np)
    
    # 清理剩余中間變量
    del predictions, predictions_np, masks_np
    gc.collect()
    
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