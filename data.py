import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_data(folder: str, target_size=(128, 128), test_only=False):
    """
    載入數據集
    
    Args:
        folder: 數據集根目錄路徑 (例如: "./data/ISIC2018")
        target_size: 目標圖像大小 (height, width)
        test_only: 如果為 True，只載入測試數據，不載入訓練和驗證數據
    
    Returns:
        tuple: (train_data, validation_data, test_data)
        每個元素包含 (images, masks) 的 tuple
        如果 test_only=True，train_data 和 validation_data 為空
    """
    
    def load_images_from_folder(folder_path, target_size):
        """
        從資料夾中載入所有圖像
        
        Args:
            folder_path: 圖像資料夾路徑
            target_size: 目標圖像大小 (height, width)
        
        Returns:
            numpy array: 形狀為 (num_images, height, width, channels)
        """
        images = []
        image_files = []
        
        # 獲取所有圖像文件
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(filename)
          # 排序確保順序一致
        image_files.sort()
        
        # 使用 tqdm 顯示進度條
        for filename in tqdm(image_files, desc=f"載入圖像", unit="files"):
            img_path = os.path.join(folder_path, filename)
            try:
                # 讀取圖像
                img = Image.open(img_path)
                
                # 轉換為 RGB (如果是灰度圖或 RGBA)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 調整大小
                img = img.resize(target_size)
                
                # 轉換為 numpy 陣列並正規化到 [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                
            except Exception as e:
                print(f"無法讀取圖像 {filename}: {e}")
        
        return np.array(images)
    
    def load_masks_from_folder(folder_path, target_size):
        """
        從資料夾中載入所有遮罩圖像
        
        Args:
            folder_path: 遮罩資料夾路徑
            target_size: 目標圖像大小 (height, width)
        
        Returns:
            numpy array: 形狀為 (num_images, height, width, 1)
        """
        masks = []
        mask_files = []
        
        # 獲取所有遮罩文件
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                mask_files.append(filename)
          # 排序確保順序一致
        mask_files.sort()
        
        # 使用 tqdm 顯示進度條
        for filename in tqdm(mask_files, desc=f"載入遮罩", unit="files"):
            mask_path = os.path.join(folder_path, filename)
            try:
                # 讀取遮罩圖像
                mask = Image.open(mask_path)
                
                # 轉換為灰度圖
                if mask.mode != 'L':
                    mask = mask.convert('L')
                
                # 調整大小
                mask = mask.resize(target_size)
                
                # 轉換為 numpy 陣列並二值化 (0 或 1)
                mask_array = np.array(mask, dtype=np.float32) / 255.0
                mask_array = (mask_array > 0.5).astype(np.float32)
                
                # 添加通道維度
                mask_array = np.expand_dims(mask_array, axis=-1)
                masks.append(mask_array)
                
            except Exception as e:
                print(f"無法讀取遮罩 {filename}: {e}")
        
        return np.array(masks)
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
