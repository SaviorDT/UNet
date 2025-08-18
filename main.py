import time
import torch
import os
import torch.cuda
from data_helpers.data import load_data
from model import create_unet, train_model, create_lunext, save_results_with_images
from trainer.k_fold import train_kfold

def main():
    """
    主函數 - 支援 UNet 和 LUNeXt 兩種模型的訓練和測試，
    以及 Glas 數據集的 K 次 K 折交叉驗證訓練
    """

    # 配置參數
    MODEL_TYPE = "UNet"  # 可選: "UNet" 或 "LUNeXt"
    MODE = "test"     # 可選: "train", "test", 或 "k_fold"
    TEST_SCORE = "traditional"  # 可選: "traditional"，未來可能新增 skeleton 評分方法
    DATASET = "my_proj2"    # 可選: "ISIC2018" 或 "Glas"
    DATASET_FOLDER = "./data/" + DATASET  # 數據集根目錄
    if DATASET == "ISIC2018":
        TARGET_SIZE = (128, 128)
        BATCH_SIZE = 16
    elif DATASET == "Glas":
        TARGET_SIZE = (224, 224)
        BATCH_SIZE = 8
    elif DATASET == "my_proj1":
        TARGET_SIZE = (640, 480)
        BATCH_SIZE = 2
    elif DATASET == "my_proj2":
        TARGET_SIZE = (540, 360) # my_proj2 會自動將圖片裁切為 16 的倍數，讓 unet 順利運作
        BATCH_SIZE = 2
    EPOCHS = 100
    LEARNING_RATE = 0.00015
    LOSS_TYPE = "cl_dice"  # 可選: None 或 "self_reg", "cl_dice"
    # 交叉驗證參數
    K_FOLD = 5          # K折交叉驗證的折數
    TIMES = 3           # 重複K折交叉驗證的次數
    
    # 顯示CUDA信息
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理記憶體以準備訓練
        current_device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(current_device)
        
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用GPU數量: {torch.cuda.device_count()}")
        print(f"目前GPU型號: {torch.cuda.get_device_name(current_device)}")
        print(f"GPU運算能力: {gpu_properties.major}.{gpu_properties.minor}")
        print(f"GPU記憶體總量: {gpu_properties.total_memory/1024**3:.2f} GB")
        print(f"當前GPU記憶體用量: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"當前GPU記憶體峰值: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        
        # 自動調整批次大小提示
        total_vram_gb = gpu_properties.total_memory / (1024**3)
        if MODEL_TYPE == "LUNeXt" and total_vram_gb < 8:
            print(f"警告: LUNeXt 模型在 {total_vram_gb:.1f}GB VRAM 上可能需要較小的批次大小")
            print(f"建議: 考慮將 BATCH_SIZE 調整為 8 或更低")
        elif MODEL_TYPE == "UNet" and total_vram_gb < 4:
            print(f"警告: UNet 模型在 {total_vram_gb:.1f}GB VRAM 上可能需要較小的批次大小")
            print(f"建議: 考慮將 BATCH_SIZE 調整為 8 或更低")
    
    print(f"模式: {MODE}")
    print(f"模型類型: {MODEL_TYPE}")
    print(f"數據集: {DATASET}")
    print(f"批次大小: {BATCH_SIZE}")
    if MODE == "test":
        print(f"評分方法: {TEST_SCORE}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    if MODE == "k_fold":
        # K-fold 交叉驗證模式
        print(f"\n開始 {TIMES} 次 {K_FOLD} 折交叉驗證訓練 - 使用 {MODEL_TYPE} 模型")
        results_df, best_model_path = train_kfold(
            model_type=MODEL_TYPE, 
            k_fold=K_FOLD, 
            times=TIMES, 
            dataset=DATASET,
            folder=DATASET_FOLDER, 
            target_size=TARGET_SIZE,
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            learning_rate=LEARNING_RATE,
            custom_loss=LOSS_TYPE
        )
        
        print(f"\n交叉驗證訓練完成！最佳模型已保存為 {best_model_path}")
        print("可以使用此模型進行後續的推理或測試。")
        
        # 儲存結果資料框的摘要到文本檔
        summary_file = f"{MODEL_TYPE}_{K_FOLD}fold_{TIMES}times_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"=== {MODEL_TYPE} 模型在 Glas 數據集上的 {TIMES} 次 {K_FOLD} 折交叉驗證結果 ===\n\n")
              # 計算每次交叉驗證的平均結果
            # 只對數值型欄位計算平均值，排除非數值欄位
            numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
            mean_results = results_df.groupby('Time')[numeric_cols].mean()
            # 移除索引列，避免重複插入
            mean_results.reset_index(drop=True, inplace=True)
            f.write("每次交叉驗證的平均結果:\n")
            for _, row in mean_results.iterrows():
                f.write(f"第 {int(row['Time'])} 次: Dice={row['Dice']:.4f}, IoU={row['IoU']:.4f}\n")
            
            # 計算總體平均和標準差
            overall_mean = results_df[numeric_cols].mean()
            overall_std = results_df[numeric_cols].std()
            
            f.write(f"\n總體平均結果:\n")
            f.write(f"Dice: {overall_mean['Dice']:.4f} ± {overall_std['Dice']:.4f}\n")
            f.write(f"IoU: {overall_mean['IoU']:.4f} ± {overall_std['IoU']:.4f}\n")
            f.write(f"Precision: {overall_mean['Precision']:.4f} ± {overall_std['Precision']:.4f}\n")
            f.write(f"Recall: {overall_mean['Recall']:.4f} ± {overall_std['Recall']:.4f}\n")
            f.write(f"Accuracy: {overall_mean['Accuracy']:.4f} ± {overall_std['Accuracy']:.4f}\n")
            f.write(f"F1: {overall_mean['F1']:.4f} ± {overall_std['F1']:.4f}\n")
        
        print(f"摘要結果已保存到 {summary_file}")
        return
    
    # 載入數據
    elif MODE == "train":
        # 訓練模式：載入完整數據集
        train_data, validation_data, test_data = load_data(
            dataset_name=DATASET,
            folder=DATASET_FOLDER, 
            target_size=TARGET_SIZE, 
            test_only=False
        )
        print("數據載入完成 - 包含訓練、驗證和測試集")
    else:
        # 測試模式：僅載入測試數據
        train_data, validation_data, test_data = load_data(
            dataset_name=DATASET,
            folder=DATASET_FOLDER, 
            target_size=TARGET_SIZE,
            test_only=True
        )
        print("數據載入完成 - 僅測試集")
    
    if MODE == "train":
        # 訓練模式
        print(f"\n開始訓練 {MODEL_TYPE} 模型...")
        start_time = time.time()
        if MODEL_TYPE == "UNet":
            model = train_model(
                train_data, validation_data, model_type="UNet",
                epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                custom_loss=LOSS_TYPE
            )
            model_filename = f"UNet_model_best.pth"
            result_filename = f"UNet_result.txt"        
        else:  # LUNeXt
            model = train_model(
                train_data, validation_data, model_type="LUNeXt",
                epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                custom_loss=LOSS_TYPE
            )
            model_filename = f"LUNeXt_model_best.pth"
            result_filename = f"LUNeXt_result.txt"
        
        train_time = time.time() - start_time
        print(f"訓練完成！耗時: {train_time:.2f} 秒")
        
    else:
        # 測試模式：載入預訓練模型
        print(f"\n載入預訓練的 {MODEL_TYPE} 模型...")
        
        # 根據測試資料自動決定輸入通道數
        inferred_in_channels = test_data[0].shape[-1] if len(test_data[0]) > 0 else 3

        if MODEL_TYPE == "UNet":
            model = create_unet(in_channels=inferred_in_channels, out_channels=1)
            model_filename = "UNet_model_best.pth"
            result_filename = "UNet_test_result.txt"
        else:  # LUNeXt
            model = create_lunext(in_channels=inferred_in_channels, out_channels=1)
            model_filename = "LUNeXt_model_best.pth"
            result_filename = "LUNeXt_test_result.txt"
        
        # 檢查模型文件是否存在
        if not os.path.exists(model_filename):
            print(f"錯誤: 找不到模型文件 {model_filename}")
            print("請先訓練模型或確認模型文件路徑正確")
            return
        
        # 載入模型權重
        print(f"載入模型: {model_filename}")
        model.load_state_dict(torch.load(model_filename, map_location=device))
        model.to(device)
        model.eval()
        
        # 模擬訓練時間（測試模式下為0）
        train_time = 0.0
        print("模型載入完成")
    
    # 評估模型並保存結果
    print(f"\n開始評估模型...")
    
    print("使用傳統評分方法進行評估...")
    save_results_with_images(
        model, 
        test_data, 
        train_time, 
        result_filename, 
        model_name=MODEL_TYPE
    )
    print(f"傳統評估完成！結果已保存到 {result_filename}")
    print(f"預測圖像已保存到 predictions/{MODEL_TYPE}/ 資料夾")

if __name__ == "__main__":
    main()