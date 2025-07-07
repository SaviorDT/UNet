import os
import numpy as np
from .utils import load_images_generic, load_binary_masks_generic

def load_images_from_folder(folder_path, target_size):
    """
    從資料夾中載入所有圖像
    
    Args:
        folder_path: 圖像資料夾路徑
        target_size: 目標圖像大小 (height, width)
    
    Returns:
        numpy array: 形狀為 (num_images, height, width, channels)
    """
    return load_images_generic(
        folder_path, 
        target_size, 
        file_extensions=('.jpg', '.jpeg', '.png'), 
        color_mode='RGB',
        normalize=True
    )

def load_masks_from_folder(folder_path, target_size):
    """
    從資料夾中載入所有遮罩圖像
    
    Args:
        folder_path: 遮罩資料夾路徑
        target_size: 目標圖像大小 (height, width)
    
    Returns:
        numpy array: 形狀為 (num_images, height, width, 1)
    """
    return load_binary_masks_generic(
        folder_path, 
        target_size, 
        file_extensions=('.jpg', '.jpeg', '.png'), 
        threshold=0.5
    )

def load_data(folder: str, target_size=(128, 128), test_only=False):
    """
    載入 ISIC2018 數據集
    
    Args:
        folder: 數據集根目錄路徑 (例如: "./data/ISIC2018")
        target_size: 目標圖像大小 (height, width)
        test_only: 如果為 True，只載入測試數據，不載入訓練和驗證數據
    
    Returns:
        tuple: (train_data, validation_data, test_data)
        每個元素包含 (images, masks) 的 tuple
        如果 test_only=True，train_data 和 validation_data 為空
    """
    # 載入各個數據集
    datasets = {}
    
    # 根據 test_only 參數決定要載入哪些數據集
    if test_only:
        splits_to_load = ['test']
        print("僅載入測試數據...")
    else:
        splits_to_load = ['train', 'validation', 'test']
    
    for split in splits_to_load:
        input_folder = os.path.join(folder, split, 'input')
        truth_folder = os.path.join(folder, split, 'truth')
        
        print(f"載入 {split} 數據...")
        
        # 檢查資料夾是否存在
        if not os.path.exists(input_folder):
            print(f"警告: {input_folder} 不存在")
            datasets[split] = (np.array([]), np.array([]))
            continue
        
        if not os.path.exists(truth_folder):
            print(f"警告: {truth_folder} 不存在")
            datasets[split] = (np.array([]), np.array([]))
            continue
        
        # 載入圖像和遮罩
        images = load_images_from_folder(input_folder, target_size)
        masks = load_masks_from_folder(truth_folder, target_size)
        
        print(f"{split} - 圖像數量: {len(images)}, 遮罩數量: {len(masks)}")
        
        # 確保圖像和遮罩數量一致
        if len(images) != len(masks):
            print(f"警告: {split} 中圖像和遮罩數量不匹配!")
            min_count = min(len(images), len(masks))
            images = images[:min_count]
            masks = masks[:min_count]
        
        datasets[split] = (images, masks)
    
    # 返回結果，如果 test_only=True，則 train_data 和 validation_data 為空
    if test_only:
        train_data = (np.array([]), np.array([]))
        validation_data = (np.array([]), np.array([]))
        test_data = datasets.get('test', (np.array([]), np.array([])))
    else:
        train_data = datasets.get('train', (np.array([]), np.array([])))
        validation_data = datasets.get('validation', (np.array([]), np.array([])))
        test_data = datasets.get('test', (np.array([]), np.array([])))
    
    print("數據載入完成!")
    print(f"訓練集: {train_data[0].shape if len(train_data[0]) > 0 else '空'}")
    print(f"驗證集: {validation_data[0].shape if len(validation_data[0]) > 0 else '空'}")
    print(f"測試集: {test_data[0].shape if len(test_data[0]) > 0 else '空'}")
    
    return train_data, validation_data, test_data
