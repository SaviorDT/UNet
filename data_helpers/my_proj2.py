import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def _resize_channel(channel_np: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize a single-channel image (H, W) to target_size (W, H) using PIL, return float32 in [0,1].
    """
    # Ensure uint8 for PIL, then back to float32 normalized
    ch = (channel_np * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(ch, mode='L')
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def load_images_from_folder(folder_path: str, target_size: tuple[int, int]):
    """
    載入 my_proj2 的輸入圖像。

    每張原始圖大小為 1080x360（W x H），左右兩半皆為黑白圖。
    流程（先分半，再裁切，確保對齊）：
    1) 讀入整張灰階圖，先分割為左右兩半 (各 w/2 x h)
    2) 針對每一半，將寬裁切為 16 的倍數，高裁切為 16 的倍數（從右下角裁掉多餘部分，保留左上區域）
    3) 將處理後的左半與右半疊為雙通道，得到 shape (H, W, 2)
    4) 若提供 target_size，將其向下對齊至 16 的倍數，再分別 resize 到該尺寸
    5) 正規化到 [0,1]

    為了配合 UNet 多次下採樣，會先從右下角裁切，使最終輸出尺寸滿足：
    - 高度 H 為 16 的倍數
    - 寬度 W（裁切後再對半）為 16 的倍數 => 裁切前整張圖的寬 new_w_full 需為 32 的倍數

    若提供 target_size，會將其向下對齊至 16 的倍數再 resize。

    回傳: numpy array, shape (N, H, W, 2)，其中 H、W 皆為 16 的倍數
    """
    images = []

    if not os.path.exists(folder_path):
        print(f"警告: 資料夾 {folder_path} 不存在")
        return np.array([])

    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    files.sort()

    for filename in tqdm(files, desc="載入 my_proj2 圖像", unit="files"):
        path = os.path.join(folder_path, filename)
        try:
            img = Image.open(path)
            if img.mode != 'L':
                img = img.convert('L')  # 灰階

            w, h = img.size  # PIL size is (W, H)
            if w < 2:
                print(f"跳過異常圖像尺寸: {filename} 大小為 {img.size}")
                continue

            # 先分半
            half_w = w // 2
            left = img.crop((0, 0, half_w, h))
            right = img.crop((half_w, 0, w, h))

            # 針對每半進行裁切（右下角裁掉 -> 保留左上區域），寬/高對齊到 16 的倍數
            def floor_to_multiple(v: int, m: int) -> int:
                return v - (v % m)

            new_half_w = floor_to_multiple(half_w, 16)
            new_h = floor_to_multiple(h, 16)
            if new_half_w <= 0 or new_h <= 0:
                print(f"跳過裁切後半幅尺寸無效: {filename} 半寬 {half_w}, 高 {h}")
                continue

            left = left.crop((0, 0, new_half_w, new_h))
            right = right.crop((0, 0, new_half_w, new_h))

            # 轉 numpy 並正規化
            left_np = np.array(left, dtype=np.float32) / 255.0
            right_np = np.array(right, dtype=np.float32) / 255.0

            # resize（如有必要） - target_size 是 (W, H)
            if target_size is not None:
                # 目標尺寸也向下對齊至 16 的倍數
                tgt_w = target_size[0] - (target_size[0] % 16)
                tgt_h = target_size[1] - (target_size[1] % 16)
                aligned_target = (max(tgt_w, 16), max(tgt_h, 16))
                if (new_half_w, new_h) != aligned_target:
                    left_np = _resize_channel(left_np, aligned_target)
                    right_np = _resize_channel(right_np, aligned_target)

            # 疊成雙通道 -> shape (H, W, 2)
            stacked = np.stack([left_np, right_np], axis=-1)
            images.append(stacked)
        except Exception as e:
            print(f"無法讀取圖像 {filename}: {e}")

    return np.array(images, dtype=np.float32) if images else np.array([])


def load_masks_from_folder(folder_path: str, target_size: tuple[int, int], threshold: float = 0.5):
    """
    載入 my_proj2 的遮罩圖。

    每張原始 mask 大小為 1080x360（W x H）。流程（先分半，再裁切，避免與影像錯位）：
    1) 先僅取右半邊 (w/2 x h)
    2) 將該半邊寬裁切為 16 的倍數、高裁切為 16 的倍數（從右下角裁掉多餘部分，保留左上區域）
    3) 二值化；若提供 target_size，將其向下對齊至 16 的倍數後再 resize。
    回傳 shape (N, H, W, 1)，其中 H、W 皆為 16 的倍數
    """
    masks = []

    if not os.path.exists(folder_path):
        print(f"警告: 資料夾 {folder_path} 不存在")
        return np.array([])

    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    files.sort()

    for filename in tqdm(files, desc="載入 my_proj2 遮罩", unit="files"):
        path = os.path.join(folder_path, filename)
        try:
            img = Image.open(path)
            if img.mode != 'L':
                img = img.convert('L')

            w, h = img.size
            if w < 2:
                print(f"跳過異常遮罩尺寸: {filename} 大小為 {img.size}")
                continue

            # 先取右半
            half_w = w // 2
            right = img.crop((half_w, 0, w, h))

            # 對該半幅做裁切（右下角裁掉 -> 保留左上區域），寬/高對齊到 16 的倍數
            def floor_to_multiple(v: int, m: int) -> int:
                return v - (v % m)

            new_half_w = floor_to_multiple(half_w, 16)
            new_h = floor_to_multiple(h, 16)
            if new_half_w <= 0 or new_h <= 0:
                print(f"跳過裁切後半幅尺寸無效: {filename} 半寬 {half_w}, 高 {h}")
                continue

            right = right.crop((0, 0, new_half_w, new_h))  # 取右半內的左上區域

            mask_np = np.array(right, dtype=np.float32) / 255.0

            # resize（如有必要） - target_size 是 (W, H)
            if target_size is not None:
                tgt_w = target_size[0] - (target_size[0] % 16)
                tgt_h = target_size[1] - (target_size[1] % 16)
                aligned_target = (max(tgt_w, 16), max(tgt_h, 16))
                if (new_half_w, new_h) != aligned_target:
                    mask_np = _resize_channel(mask_np, aligned_target)

            # 二值化，擴通道維度 -> (H, W, 1)
            mask_np = (mask_np > threshold).astype(np.float32)
            mask_np = np.expand_dims(mask_np, axis=-1)
            masks.append(mask_np)
        except Exception as e:
            print(f"無法讀取遮罩 {filename}: {e}")

    return np.array(masks, dtype=np.float32) if masks else np.array([])


def load_data_with_random_split(folder: str, target_size=(540, 360), test_only=False,
                               train_ratio=0.7, val_ratio=0.2, random_seed: int = 42):
    """
    載入 my_proj2 數據集，並進行隨機分割。

    - 影像會被處理為雙通道 (左半、右半) 的灰階堆疊，大小預期為 (540, 360)
    - 遮罩僅使用右半部分，大小 (540, 360)

    回傳: (train_data, validation_data, test_data)，各為 (images, masks)
    """
    img_folder = os.path.join(folder, 'img')
    mask_folder = os.path.join(folder, 'masks')

    # 檢查目錄
    if not os.path.exists(img_folder):
        print(f"錯誤: {img_folder} 不存在")
        return (np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([]))
    if not os.path.exists(mask_folder):
        print(f"錯誤: {mask_folder} 不存在")
        return (np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([]))

    print("載入 my_proj2 數據（隨機分割）...")
    images = load_images_from_folder(img_folder, target_size)
    masks = load_masks_from_folder(mask_folder, target_size)

    print(f"圖像數量: {len(images)}, 遮罩數量: {len(masks)}")
    if len(images) != len(masks):
        print("警告: 圖像和遮罩數量不匹配，將對齊最小數量！")
        n = min(len(images), len(masks))
        images = images[:n]
        masks = masks[:n]

    total = len(images)
    if total == 0:
        print("警告: 未載入到任何數據")
        return (np.array([]), np.array([])), (np.array([]), np.array([])), (np.array([]), np.array([]))

    # 隨機分割
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(total)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    if test_only:
        train_data = (np.array([]), np.array([]))
        val_data = (np.array([]), np.array([]))
        test_data = (images[indices], masks[indices])
        print("僅載入測試數據...")
    else:
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_data = (images[train_idx], masks[train_idx])
        val_data = (images[val_idx], masks[val_idx])
        test_data = (images[test_idx], masks[test_idx])

    print("數據載入完成!")
    print(f"訓練集: {train_data[0].shape if len(train_data[0])>0 else '空'}")
    print(f"驗證集: {val_data[0].shape if len(val_data[0])>0 else '空'}")
    print(f"測試集: {test_data[0].shape if len(test_data[0])>0 else '空'}")

    return train_data, val_data, test_data
