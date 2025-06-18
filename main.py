import time
import torch
import os
from data import *
from model import create_unet, train_model, create_lunext, save_results, save_results_with_images

def main():
    """
    主函數 - 支援 UNet 和 LUNeXt 兩種模型的訓練和測試
    """

      # 配置參數
    MODEL_TYPE = "LUNeXt"  # 可選: "UNet" 或 "LUNeXt"
    MODE = "train"       # 可選: "train" 或 "test"
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    
    print(f"模式: {MODE}")
    print(f"模型類型: {MODEL_TYPE}")
    
    # 載入數據
    if MODE == "train":
        # 訓練模式：載入完整數據集
        train_data, validation_data, test_data = load_data(
            folder="./data/ISIC2018", 
            target_size=(128, 128), 
            test_only=False
        )
        print("數據載入完成 - 包含訓練、驗證和測試集")
    else:
        # 測試模式：僅載入測試數據
        train_data, validation_data, test_data = load_data(
            folder="./data/ISIC2018", 
            target_size=(128, 128), 
            test_only=True
        )
        print("數據載入完成 - 僅測試集")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    if MODE == "train":
        # 訓練模式
        print(f"\n開始訓練 {MODEL_TYPE} 模型...")
        start_time = time.time()
        if MODEL_TYPE == "UNet":
            model = train_model(
                train_data, validation_data, model_type="UNet",
                epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE
            )
            model_filename = f"UNet_model_best.pth"
            result_filename = f"UNet_result.txt"        
        else:  # LUNeXt
            model = train_model(
                train_data, validation_data, model_type="LUNeXt",
                epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE
            )
            model_filename = f"LUNeXt_model_best.pth"
            result_filename = f"LUNeXt_result.txt"
        
        train_time = time.time() - start_time
        print(f"訓練完成！耗時: {train_time:.2f} 秒")
        
    else:
        # 測試模式：載入預訓練模型
        print(f"\n載入預訓練的 {MODEL_TYPE} 模型...")
        
        if MODEL_TYPE == "UNet":
            model = create_unet(in_channels=3, out_channels=1)
            model_filename = "UNet_model_best.pth"
            result_filename = "UNet_test_result.txt"
        else:  # LUNeXt
            model = create_lunext(in_channels=3, out_channels=1)
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
    
    # 使用增強版的結果保存函數，同時保存評估指標和預測圖像
    save_results_with_images(
        model, 
        test_data, 
        train_time, 
        result_filename, 
        model_name=MODEL_TYPE
    )
    
    print(f"評估完成！結果已保存到 {result_filename}")
    print(f"預測圖像已保存到 predictions/{MODEL_TYPE}/ 資料夾")

if __name__ == "__main__":
    main()