import itertools
import time
import numpy as np
from typing import Tuple
from scipy.spatial import cKDTree as KDTree

def uav(A_pts_origin: np.ndarray, B_pts_origin: np.ndarray, B_w: float, B_h: float, neighbor_threshold: float, max_it: int, max_tol: float,
        suggested_Rs = np.eye(3, dtype=np.float64), suggested_t = np.zeros((3,), dtype=np.float64)) -> tuple[np.ndarray, np.array]:

    if A_pts_origin.shape[1] == 2:
        A_pts_origin = np.hstack([A_pts_origin, np.ones((A_pts_origin.shape[0], 1), dtype=A_pts_origin.dtype)])
        B_pts_origin = np.hstack([B_pts_origin, np.ones((B_pts_origin.shape[0], 1), dtype=B_pts_origin.dtype)])
    if A_pts_origin.shape[1] != 3:
        raise ValueError("points must have shape (N, 2) or (N, 3)")

    B_pts_origin = B_pts_origin @ suggested_Rs.T + suggested_t
    B_C = np.array([B_w / 2, B_h / 2, 0], dtype=np.float64) @ suggested_Rs.T + suggested_t
    RS_trans, t_trans = _get_transforms(B_C[0], B_C[1])

    # return RS_trans[0] @ suggested_Rs, t_trans[0] + suggested_t

    A_pts = A_pts_origin.copy()
    kd_tree_A = KDTree(A_pts)

    for j in range(len(RS_trans)):
        if j % 100 == 0:
            print("it", j, "of", len(RS_trans))
        # start_time = time.time()

        RS = RS_trans[j]
        t = t_trans[j]
        B_pts = B_pts_origin @ RS.T + t

        paired_A, paired_B = _search_pairs(kd_tree_A, A_pts, B_pts, 10)
        if len(paired_A) < len(A_pts) * .7:
            # print("it", j, "skipped")
            continue

        # time_A = time.time()
        # print(f"Transformation {j+1} took {time_A - start_time:.4f} seconds")

        for i in range(max_it):
            paired_A, paired_B = _search_pairs(kd_tree_A, A_pts, B_pts, neighbor_threshold)
            tmp_R, tmp_t = uav_util(paired_A, paired_B)
            # Check for convergence
            if np.linalg.norm(tmp_R) < max_tol and np.linalg.norm(tmp_t) < max_tol:
                break
            B_pts = (tmp_R @ B_pts.T).T + tmp_t
            RS, t = tmp_R @ RS, tmp_R @ t + tmp_t

        # time_B = time.time()
        # print(f"Iterations {i+1} took {time_B - time_A:.4f} seconds")

        paired_A, paired_B = _search_pairs(kd_tree_A, A_pts, B_pts, 3)

        # time_C = time.time()
        # print(f"Pair search took {time_C - time_B:.4f} seconds")
        # print("it", j, "of", len(RS_trans), ", final score:", len(paired_A), "/", len(A_pts))
        if len(paired_A) >= len(A_pts) * .6:
            return RS @ suggested_Rs, RS @ suggested_t + t.reshape(3)

    # raise ValueError("UAV did not converge")
    return RS @ suggested_Rs, RS @ suggested_t + t.reshape(3)

def uav_util(A_pts: np.ndarray, B_pts: np.ndarray) -> tuple[np.ndarray, np.array]:
    """Compute the UAV metric between two point clouds.

    The UAV metric is defined as the average distance from each point in A
    to its nearest neighbor in B, and vice versa.

    Args:
        A_pts (np.ndarray): First point cloud of shape (N, 3).
        B_pts (np.ndarray): Second point cloud of shape (M, 3).
    """

    mu_A = np.mean(A_pts, axis=0)
    mu_B = np.mean(B_pts, axis=0)

    diff_A = A_pts - mu_A
    diff_B = B_pts - mu_B

    sigma = (diff_A.T @ diff_B) / diff_A.shape[0]
    sigma_B = np.sum(np.sum(diff_B * diff_B, axis=1)) / diff_B.shape[0]


    # Perform Singular Value Decomposition on sigma
    U, D, Vt = np.linalg.svd(sigma, full_matrices=True)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W = np.diag([1, 1, -1])
    else:
        W = np.diag([1, 1, 1])
    R = U @ W @ Vt
    s = np.trace(np.diag(D) @ W) / sigma_B
    R *= s
    t = mu_A - (R @ mu_B.T).T

    return R, t.reshape(3)



def _search_pairs(A_tree: KDTree, A_pts: np.ndarray, B_pts: np.ndarray, neighbor_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    # Handle edge cases
    # if A_pts is None or B_pts is None:
    #     raise ValueError("A_pts and B_pts must not be None")

    # if A_pts.ndim != 2 or B_pts.ndim != 2:
    #     raise ValueError("A_pts and B_pts must be 2D arrays of shape (N, D) and (M, D)")

    # if A_pts.shape[1] != B_pts.shape[1]:
    #     raise ValueError("A_pts and B_pts must have the same dimensionality")

    dim = A_pts.shape[1]

    if A_pts.shape[0] == 0 or B_pts.shape[0] == 0 or neighbor_threshold <= 0:
        raise ValueError("A_pts and B_pts must be non-empty and neighbor_threshold must be positive")

    # # Build KDTree on A points and query nearest for each B point within threshold
    # tree = KDTree(A_pts)
    if A_tree is None:
        raise ValueError("A_tree must not be None")
    dists, indices = A_tree.query(B_pts, k=1, distance_upper_bound=neighbor_threshold)

    # cKDTree returns inf distance (and index == n) for no neighbor within bound
    valid_mask = np.isfinite(dists)

    if not np.any(valid_mask):
        empty = np.empty((0, dim), dtype=A_pts.dtype)
        return empty, empty.copy()

    paired_A = A_pts[indices[valid_mask]]
    paired_B = B_pts[valid_mask]

    return paired_A, paired_B


def _get_transforms(cx, cy) -> tuple[np.ndarray, np.ndarray]:
    candidate_theta = [np.deg2rad(0), np.deg2rad(10), np.deg2rad(-10), np.deg2rad(20), np.deg2rad(-20), np.deg2rad(30), np.deg2rad(-30), np.deg2rad(40), np.deg2rad(-40)]
    candidate_tx = [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 105, -105, 120, -120]
    candidate_ty = [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 105, -105, 120, -120]
    candidate_scale = [1, 1.1, 0.9, 1.2, 0.8]

    # candidate_theta = [np.deg2rad(-10)]
    # candidate_tx = [75]
    # candidate_ty = [0]
    # candidate_scale = [0.85]

    candidate_R = np.empty((len(candidate_theta), 3, 3), dtype=np.float64)
    candidate_s = np.empty((len(candidate_scale), 3, 3), dtype=np.float64)

    result_len = len(candidate_theta) * len(candidate_tx) * len(candidate_ty) * len(candidate_scale)
    result_R = np.empty((result_len, 3, 3), dtype=np.float64)
    result_t = np.empty((result_len, 3), dtype=np.float64)

    for i, theta in enumerate(candidate_theta):
        c, s = np.cos(theta), np.sin(theta)
        candidate_R[i] = np.array([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]], dtype=np.float64)

    for i, s in enumerate(candidate_scale):
        candidate_s[i] = np.diag([s, s, s])

    i = 0
    for R, tx, ty, s in _enumerator_form_nearest(candidate_R, candidate_tx, candidate_ty, candidate_s):
        result_R[i] = R @ s
        trans_c = np.array([cx, cy, 1]) @ result_R[i].T
        result_t[i] = np.array([cx + tx, cy + ty, 0], dtype=np.float64) - trans_c

        i += 1

    return result_R, result_t


def _enumerator_form_nearest(arr1, arr2, arr3, arr4):
    arrays = [arr1, arr2, arr3, arr4]
    # 產生所有 index 組合
    all_indices = list(itertools.product(*(range(len(a)) for a in arrays)))
    # 依 index 總和排序
    all_indices_sorted = sorted(all_indices, key=sum)
    
    for indices in all_indices_sorted:
        yield tuple([arr1[indices[0]], arr2[indices[1]], arr3[indices[2]], arr4[indices[3]]])

if __name__ == "__main__":
    _get_transforms(640, 480)