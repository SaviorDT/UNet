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

def load_data(folder: str, target_size=(128, 128), test_only=False, train_ratio=0.7, val_ratio=0.2):
    """
    載入 my_proj1 數據集
    
    Args:
        folder: 數據集根目錄路徑 (例如: "./data/my_proj1")
        target_size: 目標圖像大小 (height, width)
        test_only: 如果為 True，只載入測試數據，不載入訓練和驗證數據
        train_ratio: 訓練集比例 (默認 0.7)
        val_ratio: 驗證集比例 (默認 0.2)，剩餘的作為測試集
    
    Returns:
        tuple: (train_data, validation_data, test_data)
        每個元素包含 (images, masks) 的 tuple
        如果 test_only=True，train_data 和 validation_data 為空
    """
    img_folder = os.path.join(folder, 'img')
    mask_folder = os.path.join(folder, 'masks')
    
    # 檢查資料夾是否存在
    if not os.path.exists(img_folder):
        print(f"錯誤: {img_folder} 不存在")
        return (np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([]))
    
    if not test_only and not os.path.exists(mask_folder):
        print(f"錯誤: {mask_folder} 不存在")
        return (np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([]))
    
    print("載入 my_proj1 數據...")
    
    # 載入所有圖像和遮罩
    images = load_images_from_folder(img_folder, target_size)
    masks = load_masks_from_folder(mask_folder, target_size) if not test_only else np.array([np.zeros((target_size[1], target_size[0], 1))]*len(images))
    
    print(f"圖像數量: {len(images)}, 遮罩數量: {len(masks)}")
    
    # 確保圖像和遮罩數量一致
    if len(images) != len(masks):
        print(f"警告: 圖像和遮罩數量不匹配!")
        min_count = min(len(images), len(masks))
        images = images[:min_count]
        masks = masks[:min_count]
    
    total_samples = len(images)
    
    if test_only:
        # 如果只要測試數據，返回所有數據作為測試集
        train_data = (np.array([]), np.array([]))
        validation_data = (np.array([]), np.array([]))
        test_data = (images, masks)
        print("僅載入測試數據...")
    else:
        # 計算分割點
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        # 為了確保可重現性，我們不隨機打亂，按順序分割
        # 如果需要隨機分割，可以先設置隨機種子再打亂索引
        train_images = images[:train_end]
        train_masks = masks[:train_end]
        
        val_images = images[train_end:val_end]
        val_masks = masks[train_end:val_end]
        
        test_images = images[val_end:]
        test_masks = masks[val_end:]
        
        train_data = (train_images, train_masks)
        validation_data = (val_images, val_masks)
        test_data = (test_images, test_masks)
    
    print("數據載入完成!")
    print(f"訓練集: {train_data[0].shape if len(train_data[0]) > 0 else '空'}")
    print(f"驗證集: {validation_data[0].shape if len(validation_data[0]) > 0 else '空'}")
    print(f"測試集: {test_data[0].shape if len(test_data[0]) > 0 else '空'}")
    
    return train_data, validation_data, test_data

def load_data_with_random_split(folder: str, target_size=(128, 128), test_only=False, 
                               train_ratio=0.7, val_ratio=0.2, random_seed=42):
    """
    載入 my_proj1 數據集並進行隨機分割
    
    Args:
        folder: 數據集根目錄路徑 (例如: "./data/my_proj1")
        target_size: 目標圖像大小 (height, width)
        test_only: 只載入 test 數據
        train_ratio: 訓練集比例 (默認 0.7)
        val_ratio: 驗證集比例 (默認 0.2)，剩餘的作為測試集
        random_seed: 隨機種子，確保可重現性
    
    Returns:
        tuple: (train_data, validation_data, test_data)
        每個元素包含 (images, masks) 的 tuple
    """
    img_folder = os.path.join(folder, 'img')
    mask_folder = os.path.join(folder, 'masks')
    
    # 檢查資料夾是否存在
    if not os.path.exists(img_folder):
        print(f"錯誤: {img_folder} 不存在")
        return (np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([]))
    
    if not test_only and not os.path.exists(mask_folder):
        print(f"錯誤: {mask_folder} 不存在")
        return (np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([]))
    
    print("載入 my_proj1 數據（隨機分割）...")
    
    # 載入所有圖像和遮罩
    images = load_images_from_folder(img_folder, target_size)
    masks = load_masks_from_folder(mask_folder, target_size) if not test_only else np.array([np.zeros((target_size[1], target_size[0], 1))]*len(images))
    
    print(f"圖像數量: {len(images)}, 遮罩數量: {len(masks)}")
    
    # 確保圖像和遮罩數量一致
    if len(images) != len(masks):
        print(f"警告: 圖像和遮罩數量不匹配!")
        min_count = min(len(images), len(masks))
        images = images[:min_count]
        masks = masks[:min_count]
    
    total_samples = len(images)

    # 設置隨機種子確保可重現性
    np.random.seed(random_seed)
    
    # 創建隨機索引
    indices = np.random.permutation(total_samples)
    
    # 計算分割點
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    # 分割索引
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # 根據索引分割數據
    train_images = images[train_indices]
    train_masks = masks[train_indices]
    
    val_images = images[val_indices]
    val_masks = masks[val_indices]
    
    test_images = images[test_indices]
    test_masks = masks[test_indices]
    
    train_data = (train_images, train_masks)
    validation_data = (val_images, val_masks)
    test_data = (test_images, test_masks)
    
    print("數據載入完成!")
    print(f"訓練集: {train_data[0].shape if len(train_data[0]) > 0 else '空'}")
    print(f"驗證集: {validation_data[0].shape if len(validation_data[0]) > 0 else '空'}")
    print(f"測試集: {test_data[0].shape if len(test_data[0]) > 0 else '空'}")
    
    return train_data, validation_data, test_data
