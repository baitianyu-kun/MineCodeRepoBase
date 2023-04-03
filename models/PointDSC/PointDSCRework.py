import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NonlocalSingleBlock(nn.Module):
    def __init__(self, num_channels=128):
        super(NonlocalSingleBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(),
            nn.Conv1d(num_channels // 2, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(),
            nn.Conv1d(num_channels // 2, num_channels, kernel_size=1),
        )
        self.conv_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.conv_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.conv_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels

    def forward(self, cors_feature, beta_attention):
        Q = self.conv_q(cors_feature)
        K = self.conv_k(cors_feature)
        V = self.conv_v(cors_feature)
        weight = torch.bmm(Q.transpose(1, 2), K)
        weight = torch.softmax(weight * beta_attention, dim=-1)
        message = torch.bmm(V, weight)
        message = self.mlp(message)
        # (batch, num_channels, num_cors)
        return cors_feature + message


class NonLocalNetwork(nn.Module):
    def __init__(self, in_dim=6, num_layers=6, num_channels=128):
        super(NonLocalNetwork, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.in_dim = in_dim

        # first layer put the dim = 6 into dim = 128, and next all are dim = 128 into dim = 128
        self.layer0 = nn.Conv1d(in_dim, num_channels, kernel_size=1)

        # mini-point net layer and non local layer
        self.blocks = nn.ModuleDict()
        for i in range(num_layers):
            conv_layer = nn.Sequential(
                nn.Conv1d(num_channels, num_channels, 1),
                nn.BatchNorm1d(num_channels),
                nn.ReLU()
            )
            self.blocks[f'PointCN_layer_{i}'] = conv_layer
            self.blocks[f'NonLocal_layer_{i}'] = NonlocalSingleBlock(num_channels)

    def forward(self, cors_feature, beta_attention):
        cors_feature = self.layer0(cors_feature)
        for i in range(self.num_layers):
            cors_feature = self.blocks[f'PointCN_layer_{i}'](cors_feature)
            cors_feature = self.blocks[f'NonLocal_layer_{i}'](cors_feature, beta_attention)
        return cors_feature


class PointDSCRework(nn.Module):
    def __init__(self, sigma_d=0.1, in_dim=6, num_layers=6, num_channels=128):
        super(PointDSCRework, self).__init__()
        self.seed_ratio = 0.1
        self.num_iterations = 10
        # the k points around seeds
        self.k = 40
        self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True)
        self.sigma_spat = nn.Parameter(torch.Tensor([sigma_d]).float(), requires_grad=False)
        self.feature_encoder = NonLocalNetwork(in_dim, num_layers, num_channels)
        self.seed_confidence = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1)
        )

    def forward(self, cors_pos, src_keypts, tgt_keypts):
        batch_size, num_cors, _ = cors_pos.shape

        # 1. extract feature for each correspondence and calculate M
        # line_distance_src_tgt: torch.Size([2, 10, 10]) -> (bs, num_cors, num_cors)
        line_distance_src_tgt = (torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)) \
                                - (torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1))
        # beta_attention: torch.Size([2, 10, 10])
        beta_attention = torch.clamp(1.0 - line_distance_src_tgt ** 2 / self.sigma_spat ** 2, min=0)
        cors_feature = self.feature_encoder(cors_pos.transpose(1, 2), beta_attention).transpose(1, 2)
        cors_feature_normed = F.normalize(cors_feature, p=2, dim=-1)

        # ====calculate cors_feature_normed similarity matrix====
        M = torch.matmul(cors_feature_normed, cors_feature_normed.transpose(1, 2))
        M = torch.clamp(1 - (1 - M) / self.sigma ** 2, min=0, max=1)
        # M: torch.Size([2, 10, 10]) -> (batch, num_cors, num_cors)
        M[:, torch.arange(M.shape[1]), torch.arange(M.shape[1])] = 0
        # ====calculate cors_feature_normed similarity matrix====

        # 2. estimate initial confidence by MLP, find highly confident and well-distributed points as seeds.
        # seed_confidence: torch.Size([2, 1, 10]) -> torch.Size([2, 10])
        seed_confidence = self.seed_confidence(cors_feature.transpose(1, 2)).squeeze(1)
        # find the most big k seeds' idx use argsort
        seed_idx = torch.argsort(seed_confidence, dim=1, descending=True)[:, 0:int(num_cors * self.seed_ratio)]
        seedwise_trans = self.calculate_seeds_transform(seed_idx, cors_feature_normed, src_keypts, tgt_keypts)
        # TODO CALCULATE INLIER NUMBER FOR EACH HYPOTHESIS

    def calculate_seeds_transform(self, seed_idx, cors_feature, src_keypts, tgt_keypts):
        # the cors_feature has to be normalized
        batch_size, num_cors, num_channels = cors_feature.shape
        k = min(self.k, num_cors - 1)
        knn_idx = self.knn(cors_feature, k, ignore_self=True, normalized=True)
        # find the knn_idx that belongs to the seeds; 取seed所在那一行的knn_idx,得到seed的邻域,最终得到所有seeds邻域
        # seed_idx: torch.Size([2, 1]), 共10个点, 0.1 ratio -> 一个batch取1个seed
        # seed_idx[:, :, None].expand(-1, -1, k).shape: torch.Size([2, 1, 9]); 9 is the k
        # knn_seed_idx: torch.Size([2, 1, 9]); 9 is the k
        knn_seed_idx = torch.gather(knn_idx, dim=1, index=seed_idx[:, :, None].expand(-1, -1, k))

        # ===construct the feature consistency matrix of each correspondence subset.===
        # find the seed knn feature in cors feature
        # knn_seed_feature: torch.Size([2, 9, 128]); 9 is the k
        knn_seed_feature = torch.gather(cors_feature, dim=1,
                                        index=knn_seed_idx.view(batch_size, -1)[:, :, None]
                                        .expand(-1, -1, num_channels))
        # knn_seed_feature: torch.Size([2, 1, 9, 128]) -> (batch, num_seeds, k, num_channels)
        knn_seed_feature = knn_seed_feature.view(batch_size, -1, k, num_channels)
        # knn_M: torch.Size([2, 1, 9, 9])
        knn_M = torch.matmul(knn_seed_feature, knn_seed_feature.transpose(2, 3))
        knn_M = torch.clamp(1.0 - (1.0 - knn_M) / self.sigma ** 2, min=0)
        knn_M = knn_M.view(-1, k, k)
        feature_knn_M = knn_M
        # ===construct the feature consistency matrix of each correspondence subset.===

        # ===construct the spatial consistency matrix of each correspondence subset.===
        # src_knn: torch.Size([2, 1, 9, 3]) -> [bs, num_seeds, k, 3]; 是seed的邻域的点
        src_knn = torch.gather(src_keypts, dim=1,
                               index=knn_seed_idx.view(batch_size, -1)[:, :, None]
                               .expand(-1, -1, 3)).view([batch_size, -1, k, 3])
        tgt_knn = torch.gather(tgt_keypts, dim=1,
                               index=knn_seed_idx.view(batch_size, -1)[:, :, None]
                               .expand(-1, -1, 3)).view([batch_size, -1, k, 3])
        # knn_M: torch.Size([2, 1, 9, 9]) -> (batch, seeds, k, k)
        knn_M = torch.norm(src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :], dim=-1) - \
                torch.norm(tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :], dim=-1)
        knn_M = torch.clamp(1.0 - knn_M ** 2 / self.sigma_spat ** 2, min=0)
        knn_M = knn_M.view(-1, k, k)
        spatial_knn_M = knn_M
        # ===construct the spatial consistency matrix of each correspondence subset.===

        # ===power iteration to get the inlier probability===
        # total_knn_M: torch.Size([2, 9, 9]) -> (batch, k, k); Eq 3
        total_knn_M = feature_knn_M * spatial_knn_M
        # set diagonal of total_knn_M to zero
        total_knn_M[:, torch.arange(total_knn_M.shape[1]), torch.arange(total_knn_M.shape[1])] = 0
        # total_weight: torch.Size([2, 9]) -> (batch, leading_eigenvector)
        # TODO FIND OUT WHY LEADING EIGENVECTOR
        total_weight = self.cal_leading_eigenvector(total_knn_M, method='power')
        # total_weight: torch.Size(torch.Size([2, 1, 9])) -> (batch, seeds, k)
        total_weight = total_weight.view(batch_size, -1, k)
        # 正则化
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)
        # ===power iteration to get the inlier probability===

        # ===calculate seeds wise transforms===
        # src_knn: torch.Size([2, 1, 9, 3]) -> (bs, num_seeds, k, 3)
        # src_knn.view(-1,k,3): torch.Size([2, 9, 3]) -> (batch, k, 3)
        seedwise_trans = self.rigid_transform_3d(src_knn.view(-1, k, 3), tgt_knn.view(-1, k, 3),
                                                 total_weight.view(-1, k))
        # seedwise_trans: torch.Size([2, 1, 4, 4]) -> (batch, seeds, 4, 4)
        seedwise_trans = seedwise_trans.view(batch_size, -1, 4, 4)
        # ===calculate seeds wise transforms===
        return seedwise_trans

    def knn(self, x, k, ignore_self=False, normalized=True):
        """ find feature space knn neighbor of x
        Input:
            - x:       [bs, num_corr, num_channels],  input features
            - k:
            - ignore_self:  True/False, return knn include self or not.
            - normalized:   True/False, if the feature x normalized.
        Output:
            - idx:     [bs, num_corr, k], the indices of knn neighbors
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

    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input:
            - M:      [bs, num_corr, num_corr] the compatibility matrix
            - method: select different method for calculating the learding eigenvector.
        Output:
            - solution: [bs, num_corr] leading eigenvector
        """
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def rigid_transform_3d(self, A, B, weights=None, weight_threshold=0):
        """
        Input:
            - A:       [bs, num_corr, 3], source point cloud
            - B:       [bs, num_corr, 3], target point cloud
            - weights: [bs, num_corr]     weight for each correspondence
            - weight_threshold: float,    clips points with weight below threshold
        Output:
            - R, t
        """
        bs = A.shape[0]
        if weights is None:
            weights = torch.ones_like(A[:, :, 0])
        weights[weights < weight_threshold] = 0
        # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

        # find mean of point cloud
        centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
                torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
        centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
                torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        # construct weight covariance matrix
        Weight = torch.diag_embed(weights)
        H = Am.permute(0, 2, 1) @ Weight @ Bm

        # find rotation
        U, S, Vt = torch.svd(H.cpu())
        U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
        delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
        eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
        eye[:, -1, -1] = delta_UV
        R = Vt @ eye @ U.permute(0, 2, 1)
        t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
        # warp_A = transform(A, integrate_trans(R,t))
        # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
        return self.integrate_trans(R, t)

    def integrate_trans(self, R, t):
        """
        Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
        Input
            - R: [3, 3] or [bs, 3, 3], rotation matrix
            - t: [3, 1] or [bs, 3, 1], translation matrix
        Output
            - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
        """
        if len(R.shape) == 3:
            if isinstance(R, torch.Tensor):
                trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
            else:
                trans = np.eye(4)[None]
            trans[:, :3, :3] = R
            trans[:, :3, 3:4] = t.view([-1, 3, 1])
        else:
            if isinstance(R, torch.Tensor):
                trans = torch.eye(4).to(R.device)
            else:
                trans = np.eye(4)
            trans[:3, :3] = R
            trans[:3, 3:4] = t
        return trans


if __name__ == '__main__':
    # cors_feature = torch.rand((2, 1024, 6))
    # beta_attention = torch.rand((2, 1024, 1024))
    # model = NonLocalNetwork()
    # model(cors_feature.transpose(1, 2), beta_attention)

    cors_pos = torch.rand((2, 10, 6))
    src_keypts = torch.rand((2, 10, 3))
    tgt_keypts = torch.rand((2, 10, 3))
    model = PointDSCRework()
    model(cors_pos, src_keypts, tgt_keypts)
