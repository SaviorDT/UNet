import glob
from PIL import Image
import numpy as np
from main import render_points, render_overlay, reduce_points
from uav import uav
from skimage.morphology import skeletonize
from scipy.ndimage import label
import time

"""
    讀取 icp/fundus/ 資料夾中的眼底影像，及 icp/predictions/ 資料夾中的預測影像，
    使用 uav 方法對齊血管點，並將對齊結果以影像形式保存到 icp/match/ 資料夾中
    命名格式：
    icp/fundus/s001_fundus.jpg
    icp/predictions/s001_t00628f00.png
    輸出格式：
    icp/match/s001_t00628f00_overlay.png
    其中 s001 為眼底影像編號，t00628f00 為預測影像編號
"""

def preprocess_image(arr: np.ndarray) -> np.ndarray:
    """
    對輸入的灰階影像陣列進行預處理，包含 skeletonize（消除中間的那顆球）、去除左邊框框的痕跡，以及去除太小的點。

    Args:
        arr (np.ndarray): 輸入的黑白影像陣列，元素應為 0 或 1。

    Returns:
        np.ndarray: 預處理後的二值化影像陣列。
    """
    # skeletonize
    skeleton = skeletonize(arr)
    if skeleton.shape[0] == 640:
        skeleton[125:130, : ] = 0
    elif skeleton.shape[0] == 540:
        skeleton[133:187, 120:169 ] = 0

    # Remove small objects
    min_size = 10
    
    structure = np.ones((3, 3), dtype=np.uint)
    labeled_image, _ = label(skeleton, structure=structure)
    sizes = np.bincount(labeled_image.ravel())
    mask = sizes >= min_size
    mask[0] = 0
    cleaned_image = mask[labeled_image]

    return cleaned_image

def _read_points(path: str, threshold: int = 127, inverse: bool = True, cut: int = 540, three_d: bool = True) -> np.ndarray:
    """
    從眼底影像讀取血管點座標。
    讀取灰階圖，將像素值大於 threshold 的視為血管點（白色），
    如果 inverse=True，則反轉顏色（黑色視為血管點）。
    並回傳這些點的 (x,y) 座標，單位為像素，原點在左下角，x 向右、y 向上。

    Args:
        path (str): 影像檔案路徑。
        threshold (int, optional): 二值化閾值，預設為 127。
        inverse (bool, optional): 是否反轉顏色，預設為 True。
        three_d (bool, optional): 是否回傳 (x,y,1) 形狀的陣列，預設為 True。

    Returns:
        np.ndarray: 血管點的 (x,y) 座標陣列，形狀為 (N, 3)，如果設定為 three_d=True，則回傳 (x,y,1) 形狀的陣列。
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    arr = arr[:, cut:]
    arr = np.flipud(arr).T # 轉置並上下翻轉，使原點在左下角
    arr = np.where(arr > threshold, 1, 0).astype(bool)
    if inverse:
        arr = 1 - arr
        
    arr = preprocess_image(arr)

    points = np.argwhere(arr == True)
    if three_d:
        points = np.insert(points, 2, 1, axis=1)
    return points

def _get_initial_RT():
    suggested_s = np.diag([.48, .48, 1])
    theta = np.deg2rad(5)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float64)

    suggested_R = Rz @ np.diag([-1, 1, 1])
    R = suggested_s @ suggested_R
    t = np.array([415, 15, 0])

    return R, t

def main():
    fundus_name = "s013"
    fundus_path = f"./icp/fundus/{fundus_name}_fundus.jpg"
    image_path_filter = f"./icp/predictions/{fundus_name}_*.png"
    image_path = sorted(glob.glob(image_path_filter))

    fundus_points = _read_points(fundus_path, threshold=127, inverse=True, cut=540, three_d=True)
    cal_fundus_points = reduce_points(fundus_points, factor=1)
    # image_points = []
    render_points(cal_fundus_points, './icp/test_A.png', img_size=(360, 540))

    R, t = _get_initial_RT()

    for idx, path in enumerate(image_path):
        print(f"{idx+1}/{len(image_path)} Processing {path}...")
        pts = _read_points(path, threshold=127, inverse=False, cut=0, three_d=True)
        cal_pts = reduce_points(pts, factor=2)

        if idx == 0:
            render_points(pts, f'./icp/test_B.png', img_size=(480, 640))

        # image_points.append(pts)
        R, t = uav(cal_fundus_points, cal_pts, B_w = 640, B_h = 480, neighbor_threshold=4000, max_it=10, max_tol=1e-2, suggested_Rs=R, suggested_t=t)
        render_overlay(fundus_points, pts @ R + t, np.empty((0, 2)), f"./icp/match/{path.split('/')[-1][:-4]}_overlay.png", img_size=(360, 540))
    
    # render_points(image_points[0], './icp/test_B.png', img_size=(480, 640))
    

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time: {end - start:.4f} seconds")