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

def load_data(folder: str, target_size=(128, 128), test_only=False, k_fold=None, times=1):
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
        if k_fold:
            return
        else:
            return (np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([]))
    
    # 載入訓練和測試圖像，根據前綴分開
    train_images = load_images_from_folder(folder, target_size, prefix="train_")
    train_masks = load_masks_from_folder(folder, target_size, prefix="train_")
    
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
    
    print(f"Glas - 訓練圖像: {len(train_images)}, 測試圖像: {len(test_images)}")
    
    # 確保訓練集圖像和遮罩數量一致
    if len(train_images) != len(train_masks):
        print(f"警告: 訓練集中圖像和遮罩數量不匹配!")
        min_count = min(len(train_images), len(train_masks))
        train_images = train_images[:min_count]
        train_masks = train_masks[:min_count]
    
    # 確保測試集圖像和遮罩數量一致
    if len(test_images) != len(test_masks):
        print(f"警告: 測試集中圖像和遮罩數量不匹配!")
        min_count = min(len(test_images), len(test_masks))
        test_images = test_images[:min_count]
        test_masks = test_masks[:min_count]
        
    # 將測試數據轉換為元組格式
    test_data = (test_images, test_masks)
        
    # K折交叉驗證模式
    if k_fold is not None:
        from sklearn.model_selection import KFold
        
        # 定義生成器函數
        def k_fold_generator():
            """生成器函數，每次返回一個交叉驗證分割"""
            for t in range(times):
                # 每次使用不同的隨機種子
                kf = KFold(n_splits=k_fold, shuffle=True, random_state=42+t)
                
                for fold_idx, (train_index, val_index) in enumerate(kf.split(train_images)):
                    # 只對訓練數據集進行交叉驗證分割
                    fold_train_images = train_images[train_index]
                    fold_train_masks = train_masks[train_index]
                    fold_val_images = train_images[val_index]
                    fold_val_masks = train_masks[val_index]
                    
                    train_data_fold = (fold_train_images, fold_train_masks)
                    val_data_fold = (fold_val_images, fold_val_masks)
                    
                    # 生成當前折的數據
                    yield train_data_fold, val_data_fold, test_data, t, fold_idx
        
        print(f"K折交叉驗證數據生成器已準備! 將進行{times}次{k_fold}折交叉驗證")
        return k_fold_generator()
    
    # 常規訓練/測試模式
    elif test_only:
        train_data = (np.array([]), np.array([]))
        validation_data = (np.array([]), np.array([]))
    else:
        # 使用通用分割函數，從訓練集中再分出驗證集
        train_part, validation_data = split_dataset(
            train_images, train_masks, 
            train_ratio=0.9, 
            val_ratio=0.1, 
            test_ratio=0,  # 不再分出測試集，因為已經有獨立的測試集
            shuffle=True
        )
        # 只需要前兩個元素，因為沒有test_data
        train_data = train_part
    
    print("數據載入完成!")
    print(f"訓練集: {train_data[0].shape if len(train_data[0]) > 0 else '空'}")
    print(f"驗證集: {validation_data[0].shape if len(validation_data[0]) > 0 else '空'}")
    print(f"測試集: {test_data[0].shape if len(test_data[0]) > 0 else '空'}")
    
    return train_data, validation_data, test_data

# def apply_augmentations(images, masks):
#     """
#     對圖像和遮罩應用常用的增強操作 - 優化版本
#     使用並行處理和批次處理以提高效率

#     Args:
#         images: numpy array, 圖像數據
#         masks: numpy array, 遮罩數據

#     Returns:
#         augmented_images: 增強後的圖像數據
#         augmented_masks: 增強後的遮罩數據
#     """
#     import albumentations as A
#     from albumentations.core.transforms_interface import ImageOnlyTransform
#     from concurrent.futures import ThreadPoolExecutor
#     import numpy as np
    
#     # 效能優化: 使用更高效的增強操作集
#     augmentation_pipeline = A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         # 使用 Affine 替代 ShiftScaleRotate 以避免警告
#         A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-45, 45), p=0.5),
#         A.GaussianBlur(p=0.2),
#         # 不在這裡應用 Normalize，留到模型訓練階段以避免重複處理
#     ])

#     # 使用多執行緒並行處理
#     max_workers = 4  # 根據CPU核心數調整
#     batch_size = min(32, len(images))  # 批次大小限制
#     results = []
    
#     # 定義並行工作函數
#     def process_batch(start_idx, end_idx):
#         batch_aug_images = []
#         batch_aug_masks = []
#         for idx in range(start_idx, end_idx):
#             img, mask = images[idx], masks[idx]
#             augmented = augmentation_pipeline(image=img, mask=mask)
#             batch_aug_images.append(augmented['image'])
#             batch_aug_masks.append(augmented['mask'])
#         return batch_aug_images, batch_aug_masks
    
#     # 創建批次處理任務
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = []
#         for i in range(0, len(images), batch_size):
#             end_idx = min(i + batch_size, len(images))
#             futures.append(executor.submit(process_batch, i, end_idx))
        
#         print(f"正在並行處理 {len(images)} 張圖像的增強 (使用 {max_workers} 個執行緒)...")
        
#         # 收集結果
#         augmented_images = []
#         augmented_masks = []
#         for future in futures:
#             batch_imgs, batch_masks = future.result()
#             augmented_images.extend(batch_imgs)
#             augmented_masks.extend(batch_masks)
    
#     print(f"增強完成，處理了 {len(augmented_images)} 張圖像")
#     return np.array(augmented_images), np.array(augmented_masks)
