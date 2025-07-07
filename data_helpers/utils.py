import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_images_generic(folder_path, target_size, file_extensions=('.jpg', '.jpeg', '.png'), 
                       file_filter=None, sort_key=None, color_mode='RGB', normalize=True):
    """
    通用圖像載入函數，可用於不同類型的數據集
    
    Args:
        folder_path: 圖像資料夾路徑
        target_size: 目標圖像大小 (width, height)
        file_extensions: 要載入的文件擴展名
        file_filter: 過濾函數，接收文件名作為輸入，返回布爾值決定是否載入該文件
        sort_key: 排序鍵函數，接收文件名作為輸入，返回可用於排序的值。如果為None則使用字母排序
        color_mode: 圖像顏色模式 ('RGB' 或 'L'(灰度))
        normalize: 是否將像素值標準化到 [0, 1]
    
    Returns:
        numpy array: 形狀為 (num_images, height, width, channels)
    """
    images = []
    image_files = []
    
    # 檢查文件夾是否存在
    if not os.path.exists(folder_path):
        print(f"警告: 資料夾 {folder_path} 不存在")
        return np.array([])
    
    # 獲取所有符合條件的圖像文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(file_extensions):
            if file_filter is None or file_filter(filename):
                image_files.append(filename)
      # 排序確保順序一致，使用自定義排序鍵或預設字母排序
    if sort_key is not None:
        image_files.sort(key=sort_key)
    else:
        image_files.sort()
    
    # 使用 tqdm 顯示進度條
    for filename in tqdm(image_files, desc=f"載入圖像", unit="files"):
        img_path = os.path.join(folder_path, filename)
        try:
            # 讀取圖像
            img = Image.open(img_path)
            
            # 轉換顏色模式
            if img.mode != color_mode:
                img = img.convert(color_mode)
            
            # 調整大小
            img = img.resize(target_size)
            
            # 轉換為 numpy 陣列
            img_array = np.array(img, dtype=np.float32)
            
            # 標準化
            if normalize:
                img_array = img_array / 255.0
            
            # 對於灰度圖像，添加通道維度
            if color_mode == 'L':
                img_array = np.expand_dims(img_array, axis=-1)
                
            images.append(img_array)
            
        except Exception as e:
            print(f"無法讀取圖像 {filename}: {e}")
    
    return np.array(images) if images else np.array([])

def load_binary_masks_generic(folder_path, target_size, file_extensions=('.jpg', '.jpeg', '.png'), 
                             file_filter=None, sort_key=None, threshold=0.5):
    """
    通用二值遮罩載入函數
    
    Args:
        folder_path: 遮罩資料夾路徑
        target_size: 目標圖像大小 (width, height)
        file_extensions: 要載入的文件擴展名
        file_filter: 過濾函數，接收文件名作為輸入，返回布爾值決定是否載入該文件
        sort_key: 排序鍵函數，接收文件名作為輸入，返回可用於排序的值。如果為None則使用字母排序
        threshold: 二值化閾值
    
    Returns:
        numpy array: 形狀為 (num_images, height, width, 1)
    """    # 使用通用載入函數，指定為灰度模式
    masks = load_images_generic(
        folder_path, 
        target_size, 
        file_extensions=file_extensions,
        file_filter=file_filter,
        sort_key=sort_key,
        color_mode='L',
        normalize=True
    )
    
    # 如果沒有成功載入任何遮罩，直接返回空陣列
    if masks.size == 0:
        return masks
    
    # 二值化
    masks = (masks > threshold).astype(np.float32)
    
    return masks

def split_dataset(images, masks, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True):
    """
    將數據集分割為訓練、驗證和測試集
    
    Args:
        images: 圖像數組 shape: (num_images, height, width, channels)
        masks: 遮罩數組 shape: (num_images, height, width, 1)
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
        test_ratio: 測試集比例
        shuffle: 是否打亂數據
    
    Returns:
        tuple: (train_data, validation_data, test_data)
        每個元素包含 (images, masks) 的 tuple
    """
    # 確保比例之和為1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例之和必須為1"
    
    n_samples = len(images)
    indices = np.arange(n_samples)
    
    # 打亂數據
    if shuffle:
        np.random.shuffle(indices)
    
    # 計算分割點
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # 分割數據
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_data = (images[train_idx], masks[train_idx])
    validation_data = (images[val_idx], masks[val_idx])
    test_data = (images[test_idx], masks[test_idx])
    
    return train_data, validation_data, test_data

def extract_numeric_sort_key(filename):
    """
    從檔案名稱中提取數字部分用於排序
    例如: 'train_1.bmp' -> (1, 'train_1.bmp')
          'train_10.bmp' -> (10, 'train_10.bmp')
          'train_1_anno.bmp' -> (1, 'train_1_anno.bmp')
    
    Args:
        filename: 檔案名稱
    
    Returns:
        tuple: (數字, 檔案名稱) 用於排序
    """
    import re
    
    # 尋找檔案名稱中的數字
    match = re.search(r'(\d+)', filename)
    if match:
        return (int(match.group(1)), filename)
    else:
        # 如果沒有找到數字，返回一個大數和檔案名稱
        return (float('inf'), filename)

def create_paired_sort_key(base_pattern, anno_pattern='_anno'):
    """
    創建一個排序鍵函數，用於配對的圖像和標註檔案
    確保 train_1.bmp 和 train_1_anno.bmp 排序在一起
    
    Args:
        base_pattern: 基礎檔案的模式 (例如 'train')
        anno_pattern: 標註檔案的標識 (例如 '_anno')
    
    Returns:
        function: 排序鍵函數
    """
    def sort_key(filename):
        import re
        
        # 移除副檔名
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # 檢查是否為標註檔案
        is_anno = anno_pattern in filename
        
        # 提取數字
        match = re.search(r'(\d+)', filename)
        if match:
            number = int(match.group(1))
            # 標註檔案排在原始檔案後面
            return (number, 1 if is_anno else 0, filename)
        else:
            return (float('inf'), 1 if is_anno else 0, filename)
    
    return sort_key

# 使用範例:
# 對於 train_1.bmp, train_1_anno.bmp 這樣的檔案
# sort_key = create_paired_sort_key('train', '_anno')
# image_files.sort(key=sort_key)

def create_simple_numeric_sort_key():
    """
    創建一個簡單的數字排序鍵函數
    
    Returns:
        function: 排序鍵函數，按檔案名稱中的數字排序
    """
    return extract_numeric_sort_key
