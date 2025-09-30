import time
import numpy as np
from typing import Tuple
from uav import uav
import imageio.v2 as imageio

def _write_png(img: np.ndarray, path: str) -> None:
    """Write a 2D uint8 grayscale image to a PNG file using imageio."""
    if img.ndim != 2:
        raise ValueError("_write_png expects a 2D grayscale image")
    imageio.imwrite(path, np.ascontiguousarray(img, dtype=np.uint8))


def generate_points(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Uniform points in a unit square centered at origin
    pts = rng.uniform(-0.5, 0.5, size=(n, 2))
    return pts.astype(np.float64)


def rotate_points(pts: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[c, -s], [s, c]], dtype=pts.dtype)
    return (pts @ R2.T)


def to3(pts2: np.ndarray) -> np.ndarray:
    if pts2.ndim != 2 or pts2.shape[1] not in (2, 3):
        raise ValueError("pts must be (N,2) or (N,3)")
    if pts2.shape[1] == 3:
        return pts2
    return np.hstack([pts2, np.ones((pts2.shape[0], 1), dtype=pts2.dtype)])


def render_points(pts: np.ndarray, path: str, img_size: Tuple[int, int] = (512, 512), pad: float = 0.0) -> None:
    """Render points using raw coordinates without scaling.
    Assumes pts are in pixel units with x-right, y-up. z (if present) is ignored.
    """
    if pts.shape[1] >= 2:
        pts2 = pts[:, :2]
    else:
        raise ValueError("pts must have at least 2 columns for x,y")

    h, w = img_size
    # Round to nearest integer pixel and map y-up to image row index (top=0)
    px = np.rint(pts2[:, 0]).astype(int)
    py_up = np.rint(pts2[:, 1]).astype(int)
    py = (h - 1) - py_up

    # Clip to image bounds
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)

    img = np.zeros((h, w), dtype=np.uint8)
    img[py, px] = 255
    _write_png(img, path)

def _read_points_from_image(path: str, threshold: int = 127) -> tuple[np.ndarray, tuple[int, int]]:
    """Read a PNG (grayscale or RGB/RGBA) and extract white pixels as points.
    Returns (points (N,2) in x-right, y-up coords, (H, W)).
    """
    img = imageio.imread(path)
    if img.ndim == 3:
        # Convert to grayscale by averaging channels
        gray = img.mean(axis=2)
    else:
        gray = img
    gray = gray.astype(np.float32)
    h, w = gray.shape[:2]
    mask = gray > threshold
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError(f"No points found in image: {path}")
    # Convert to y-up coordinates (origin bottom-left)
    pts = np.column_stack([xs, (h - 1) - ys]).astype(np.float64)
    return pts, (h, w)


def render_overlay(pts_red: np.ndarray, pts_green: np.ndarray, pts_blue: np.ndarray, path: str, img_size: Tuple[int, int]) -> None:
    """Render two point sets as an RGB overlay without scaling.
    - pts_red plotted in red channel
    - pts_green plotted in green channel
    - pts_blue plotted in blue channel
    Coordinates are raw pixels in x-right, y-up.
    """
    def to_px_py(pts: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        if pts.shape[1] >= 2:
            p2 = pts[:, :2]
        else:
            raise ValueError("points must have at least 2 columns for x,y")
        px = np.rint(p2[:, 0]).astype(int)
        py_up = np.rint(p2[:, 1]).astype(int)
        py = (h - 1) - py_up
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        return px, py

    h, w = img_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    rx, ry = to_px_py(pts_red, h, w)
    gx, gy = to_px_py(pts_green, h, w)
    bx, by = to_px_py(pts_blue, h, w)
    img[ry, rx, 0] = 255  # Red channel
    img[gy, gx, 1] = 255  # Green channel
    img[by, bx, 2] = 255  # Blue channel
    imageio.imwrite(path, img)

def reduce_points(pts: np.ndarray, factor: int) -> np.ndarray:
    """Reduce the number of points by keeping every 'factor'-th point."""
    if factor <= 0:
        raise ValueError("factor must be a positive integer")
    idx = np.random.choice(len(pts), size=len(pts)//factor, replace=False)
    return pts[idx]

def main():
    # start = time.time()
    # Read points from test_A.png and test_B.png
    A2, (hA, wA) = _read_points_from_image("./icp/test_A.png")
    B2, (hB, wB) = _read_points_from_image("./icp/test_B.png")

    # Lift to 3D with z=1 to use uav
    A3 = to3(A2)
    B3 = to3(B2)

    A3 = reduce_points(A3, factor=8)
    B3 = reduce_points(B3, factor=40)

    # Use a generous neighbor threshold in pixel units (max image dimension)
    neighbor_threshold = 4000

    suggested_s = np.diag([.55, .55, 1])

    theta = np.deg2rad(0)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float64)

    suggested_R = Rz @ np.diag([-1, 1, 1])
    suggested_trans = suggested_s @ suggested_R
    suggested_t = np.array([430, 50, 0])

    # end = time.time()
    # print(f"Data preparation time: {end - start:.4f} seconds")

    # Estimate R, t using uav
    R, t = uav(A3, B3, B_w = wB, B_h = hB, neighbor_threshold=neighbor_threshold, max_it=10, max_tol=1e-2, suggested_Rs=suggested_trans, suggested_t=suggested_t)

    # end2 = time.time()
    # print(f"UAV computation time: {end2 - end:.4f} seconds")

    # 5) Transform B and bring back to 2D for rendering
    B3_aligned = to3(B2) @ R + t
    # B2_aligned = (R @ B2.T).T + t


    # 6) Save images: A.png, B.png, RB_t.png
    render_points(A3, "./icp/A.png",img_size=(hA, wA))
    render_points(B3 @ suggested_trans + suggested_t, "./icp/B.png",img_size=(hB, wB))
    
    render_points(B3_aligned, "./icp/AB.png",img_size=(hA, wA))
    # Overlay A2 (red) and B3_aligned (green)
    render_overlay(A2, to3(B2) @ suggested_trans + suggested_t, np.empty((0, 2)), "./icp/ABO.png", img_size=(hA, wA))
    render_overlay(A2, np.empty((0, 2)), B3_aligned, "./icp/ABN.png", img_size=(hA, wA))

    # Print results
    np.set_printoptions(precision=4, suppress=True)
    print("R =\n", R)
    print("t = ", t)

    # end3 = time.time()
    # print(f"Total time: {end3 - end2:.4f} seconds")

# def match_fundus()


if __name__ == "__main__":
    # start = time.time()
    main()
    # end = time.time()
    # print(f"Total execution time: {end - start:.4f} seconds")
