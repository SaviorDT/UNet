import os
import numpy as np
from typing import Tuple, List, Dict, Any, Callable, Optional

# Import dataset-specific implementations
from data_helpers.ISIC2018 import load_data as load_isic2018_data
from data_helpers.Glas import load_data as load_glas_data

def load_data(dataset_name: str, folder: str, target_size=(128, 128), test_only=False):
    """
    通用數據載入函數 - 根據指定的數據集名稱調用相應的載入邏輯
    
    Args:
        dataset_name: 數據集名稱 (例如: "ISIC2018", "Glas")
        folder: 數據集根目錄路徑 (例如: "./data/ISIC2018")
        target_size: 目標圖像大小 (height, width)
        test_only: 如果為 True，只載入測試數據，不載入訓練和驗證數據
    
    Returns:
        tuple: (train_data, validation_data, test_data)
        每個元素包含 (images, masks) 的 tuple
        如果 test_only=True，train_data 和 validation_data 為空
    """
    
    # 根據數據集名稱選擇適當的載入函數
    if dataset_name.lower() == "isic2018":
        return load_isic2018_data(folder, target_size, test_only)
    elif dataset_name.lower() == "glas":
        return load_glas_data(folder, target_size, test_only)
    else:
        raise ValueError(f"不支持的數據集: {dataset_name}")
