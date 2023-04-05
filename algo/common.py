import torch


def knn(x, k):
    """ find knn neighbor of x, dgcnn implement version
    Input:
        - x:       [bs, num_channels, num_pts],  input features
        - k:
    Output:
        - idx:     [bs, num_pts, k], the indices of knn neighbors
    """
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def knn2(x, k, ignore_self=False, normalized=False):
    """ find feature space knn neighbor of x, is ignore_self = False and normalized = False
        then it equals to the upper knn(x,k)
    Input:
        - x:       [bs, num_pts, num_channels],  input features
        - k:
        - ignore_self:  True/False, return knn include self or not.
        - normalized:   True/False, if the feature x normalized.
    Output:
        - idx:     [bs, num_pts, k], the indices of knn neighbors
    """
    inner = 2 * torch.matmul(x, x.transpose(2, 1))
    if normalized:
        pairwise_distance = 2 - inner
    else:
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)
        pairwise_distance = xx - inner + xx.transpose(2, 1)

    if ignore_self is False:
        idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (batch_size, num_points, k)
    else:
        idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx


if __name__ == '__main__':
    x = torch.rand((1, 3, 10))
    k = 2
    data1 = knn(x, k)
    data2 = knn2(x.transpose(1, 2), k, ignore_self=True)
    data3 = knn2(x.transpose(1, 2), k, ignore_self=False)
    print(data2)
    print(data3)
