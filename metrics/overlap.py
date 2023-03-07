import open3d as o3d
import torch
import numpy as np


def get_overlap_ratio(source, target, threshold=0.03):
    """
    We compute overlap ratio from source point cloud to target point cloud
    """
    pcd_tree = o3d.geometry.KDTreeFlann(target)

    match_count = 0
    for i, point in enumerate(source.points):
        [count, _, _] = pcd_tree.search_radius_vector_3d(point, threshold)
        if (count != 0):
            match_count += 1

    overlap_ratio = match_count / len(source.points)
    return overlap_ratio


def sample_interest_points(method, scores, N):
    """
    We can do random sampling, probabilistic sampling, or top-k sampling
    """
    assert method in ['prob', 'topk', 'random']
    n = scores.size(0)
    if n < N:
        choice = np.random.choice(n, N)
    else:
        if method == 'random':
            choice = np.random.permutation(n)[:N]
        elif method == 'topk':
            choice = torch.topk(scores, N, dim=0)[1]
        elif method == 'prob':
            idx = np.arange(n)
            probs = (scores / scores.sum()).numpy().flatten()
            choice = np.random.choice(idx, size=N, replace=False, p=probs)

    return choice
