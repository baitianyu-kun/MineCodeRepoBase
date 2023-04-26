import torch


def pairwise_distance_batch(x, y):
    """ pairwise_distance
    Args:
        x: Input features of source point clouds. Size [B, C, N]
        y: Input features of source point clouds. Size [B, C, M]
    Returns:
        pair_distances: Euclidean distance. Size [B, N, M]
    """
    xx = torch.sum(torch.mul(x, x), 1, keepdim=True)  # [b,1,n]
    yy = torch.sum(torch.mul(y, y), 1, keepdim=True)  # [b,1,n]
    inner = -2 * torch.matmul(x.transpose(2, 1), y)  # [b,n,n]
    pair_distance = xx.transpose(2, 1) + inner + yy  # [b,n,n]
    zeros_matrix = torch.zeros_like(pair_distance, device=x.device)
    pair_distance_square = torch.where(pair_distance > 0.0, pair_distance, zeros_matrix)
    error_mask = torch.le(pair_distance_square, 0.0)
    pair_distances = torch.sqrt(pair_distance_square + error_mask.float() * 1e-16)
    pair_distances = torch.mul(pair_distances, (1.0 - error_mask.float()))
    return pair_distances


if __name__ == '__main__':
    x = torch.rand((2, 3, 4))
    y = torch.rand((2, 3, 4))
    print(pairwise_distance_batch(x, y))
