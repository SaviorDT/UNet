import glob
from PIL import Image
import numpy as np
from main import render_points, render_overlay
from uav import uav

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

def _read_points(path: str, threshold: int = 127, inverse: bool = True, cut: int = 540, three_d: bool = True) -> np.ndarray:
    """
    從眼底影像讀取血管點座標。
    讀取灰階圖，將像素值大於於 threshold 的視為血管點（白色），
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
    if inverse:
        arr = 255 - arr
    points = np.argwhere(arr > threshold)
    if three_d:
        points = np.insert(points, 2, 1, axis=1)
    return points

def _get_initial_RT():
    suggested_s = np.diag([.42, .42, 1])
    theta = np.deg2rad(10)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float64)

    suggested_R = Rz @ np.diag([-1, 1, 1])
    R = suggested_s @ suggested_R
    t = np.array([400, 100, 0])

    return R, t

def main():
    fundus_name = "s001"
    fundus_path = f"./icp/fundus/{fundus_name}_fundus.jpg"
    image_path_filter = f"./icp/predictions/{fundus_name}_*.png"
    image_path = sorted(glob.glob(image_path_filter))

    fundus_points = _read_points(fundus_path, threshold=127, inverse=True, cut=540, three_d=True)
    # image_points = []

    R, t = _get_initial_RT()

    for path in image_path:
        pts = _read_points(path, threshold=127, inverse=False, cut=0, three_d=True)
        # image_points.append(pts)
        R, t = uav(fundus_points, pts, B_w = 640, B_h = 480, neighbor_threshold=4000, max_it=10, max_tol=1e-2, suggested_Rs=R, suggested_t=t)
        render_overlay(fundus_points, pts @ R + t, np.empty((0, 2)), f"./icp/match/{path.split('/')[-1][:-4]}_overlay.png", img_size=(360, 540))
    
    # render_points(fundus_points, './icp/test_A.png', img_size=(360, 540))
    # render_points(image_points[0], './icp/test_B.png', img_size=(480, 640))
    

if __name__ == "__main__":
    main()