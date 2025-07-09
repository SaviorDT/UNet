import os
import numpy as np
from .utils import load_images_generic, load_binary_masks_generic, split_dataset, create_simple_numeric_sort_key

def load_images_from_folder(folder_path, target_size, prefix=None):
    """
    從資料夾中載入所有圖像，可選擇按前綴過濾
    
    Args:
        folder_path: 圖像資料夾路徑
        target_size: 目標圖像大小 (height, width)
        prefix: 文件名前綴過濾 (例如: "train_" 或 "testA_")
    
    Returns:
        numpy array: 形狀為 (num_images, height, width, channels)
    """
    # 定義過濾函數 - 只保留不含 "anno" 的檔案，並根據前綴過濾
    def filter_images(filename):
        no_anno = "anno" not in filename.lower()
        if prefix is not None:
            return no_anno and filename.startswith(prefix)
        return no_anno
    
    return load_images_generic(
        folder_path, 
        target_size, 
        file_extensions=('.bmp',), 
        file_filter=filter_images,
        sort_key=create_simple_numeric_sort_key(),
        color_mode='RGB',
        normalize=True
    )

def load_masks_from_folder(folder_path, target_size, prefix=None):
    """
    從資料夾中載入所有遮罩圖像，可選擇按前綴過濾
    
    Args:
        folder_path: 遮罩資料夾路徑
        target_size: 目標圖像大小 (height, width)
        prefix: 文件名前綴過濾 (例如: "train_" 或 "testA_")
    
    Returns:
        numpy array: 形狀為 (num_images, height, width, 1)
    """
    # 定義過濾函數 - 只保留包含 "anno" 的檔案，並根據前綴過濾
    def filter_masks(filename):
        has_anno = "anno" in filename.lower()
        if prefix is not None:
            return has_anno and filename.startswith(prefix)
        return has_anno
    
    return load_binary_masks_generic(
        folder_path, 
        target_size, 
        file_extensions=('.bmp',), 
        file_filter=filter_masks,
        sort_key=create_simple_numeric_sort_key(),
        threshold=0.003
    )

def load_data(folder: str, target_size=(224, 224), test_only=False):
    """
    載入 Glas 數據集，支持按照文件名前綴區分訓練和測試數據，
    並支持多次K折交叉驗證
    
    Args:
        folder: 數據集根目錄路徑 (例如: "./data/Glas")
        target_size: 目標圖像大小 (height, width)
        test_only: 如果為 True，只載入測試數據，不載入訓練和驗證數據
        k_fold: 交叉驗證的折數 (默認為None表示不使用K折)
        times: 重複交叉驗證的次數 (默認為1)
    
    Returns:
        如果 k_fold 為 None:
            tuple: (train_data, validation_data, test_data)
        否則:
            generator: 生成 (train_data, val_data, test_data, time_idx, fold_idx) 元組
    """
    print(f"載入 Glas 數據...")
    
    # 檢查資料夾是否存在
    if not os.path.exists(folder):
        print(f"警告: {folder} 不存在")
        return (np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([]))
    
    # 載入測試圖像（testA和testB）
    testA_images = load_images_from_folder(folder, target_size, prefix="testA_") 
    testA_masks = load_masks_from_folder(folder, target_size, prefix="testA_")
    testB_images = load_images_from_folder(folder, target_size, prefix="testB_")
    testB_masks = load_masks_from_folder(folder, target_size, prefix="testB_")
    
    # 合併testA和testB作為測試集
    test_images = np.concatenate([testA_images, testB_images]) if len(testA_images) > 0 and len(testB_images) > 0 else \
                  testA_images if len(testA_images) > 0 else testB_images
    test_masks = np.concatenate([testA_masks, testB_masks]) if len(testA_masks) > 0 and len(testB_masks) > 0 else \
                 testA_masks if len(testA_masks) > 0 else testB_masks
    
    # print(f"Glas - 訓練圖像: {len(train_images)}, 測試圖像: {len(test_images)}")
    
    # 確保測試集圖像和遮罩數量一致
    if len(test_images) != len(test_masks):
        print(f"警告: 測試集中圖像和遮罩數量不匹配!")
        min_count = min(len(test_images), len(test_masks))
        test_images = test_images[:min_count]
        test_masks = test_masks[:min_count]
        
    # 將測試數據轉換為元組格式
    test_data = (test_images, test_masks)
    
    # 測試模式
    if test_only:
        train_data = (np.array([]), np.array([]))
        validation_data = (np.array([]), np.array([]))
    else:
        # 載入訓練和測試圖像，根據前綴分開
        train_images = load_images_from_folder(folder, target_size, prefix="train_")
        train_masks = load_masks_from_folder(folder, target_size, prefix="train_")

        # 確保訓練集圖像和遮罩數量一致
        if len(train_images) != len(train_masks):
            print(f"警告: 訓練集中圖像和遮罩數量不匹配!")
            min_count = min(len(train_images), len(train_masks))
            train_images = train_images[:min_count]
            train_masks = train_masks[:min_count]

        # 使用通用分割函數，從訓練集中再分出驗證集
        train_data, validation_data, _ = split_dataset(
            train_images, train_masks, 
            train_ratio=0.9, 
            val_ratio=0.1, 
            test_ratio=0,  # 不再分出測試集，因為已經有獨立的測試集
            shuffle=True
        )
    
    print("數據載入完成!")
    print(f"訓練集: {train_data[0].shape if len(train_data[0]) > 0 else '空'}")
    print(f"驗證集: {validation_data[0].shape if len(validation_data[0]) > 0 else '空'}")
    print(f"測試集: {test_data[0].shape if len(test_data[0]) > 0 else '空'}")
    
    return train_data, validation_data, test_data
