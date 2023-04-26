from code_project1.rie_loss import *
from code_project1.rie_base import PointNet, Pointer, get_knn_index, Discriminator, feature_extractor, \
    get_keypoints
from utils.cors_utils import pairwise_distance_batch
from algo.svd import compute_rigid_transform2
import se_math.se3 as se3


class SVDHead(nn.Module):
    def __init__(self,
                 n_keypoints,
                 nn_margin,
                 dim):
        super(SVDHead, self).__init__()
        self.num_keypoints = n_keypoints
        self.weight_function = Discriminator(dim=dim)
        self.fuse = Pointer()
        self.nn_margin = nn_margin

    def forward(self, *input):
        """
        Args:
            src: Source point clouds. Size (B, 3, N)
            tgt: target point clouds. Size (B, 3, M)
            src_embedding: Features of source point clouds. Size (B, C, N)
            tgt_embedding: Features of target point clouds. Size (B, C, M)
            src_idx: Nearest neighbor indices. Size [B * N * k]
            k: Number of nearest neighbors.
            src_knn: Coordinates of nearest neighbors. Size [B, N, K, 3]
            i: i-th iteration.
            tgt_knn: Coordinates of nearest neighbors. Size [B, M, K, 3]
            src_idx1: Nearest neighbor indices. Size [B * N * k]
            idx2:  Nearest neighbor indices. Size [B, M, k]
            k1: Number of nearest neighbors.
        Returns:
            R/t: rigid transformation.
            src_keypoints, tgt_keypoints: Selected keypoints of source and target point clouds. Size (B, 3, num_keypoint)
            src_keypoints_knn, tgt_keypoints_knn: KNN of keypoints. Size [b, 3, num_kepoints, k]
            loss_scl: Spatial Consistency loss.
        """
        src = input[0]
        tgt = input[1]
        src_embedding = input[2]
        tgt_embedding = input[3]
        src_idx = input[4]
        k = input[5]
        src_knn = input[6]  # [b, n, k, 3]
        i = input[7]
        tgt_knn = input[8]  # [b, n, k, 3]
        src_idx1 = input[9]  # [b * n * k1]
        idx2 = input[10]  # [b, m, k1]
        k1 = input[11]

        batch_size, num_dims_src, num_points = src.size()
        batch_size, _, num_points_tgt = tgt.size()
        batch_size, _, num_points = src_embedding.size()

        ########################## Matching Map Refinement Module ##########################
        # 计算distance_map
        distance_map = pairwise_distance_batch(src_embedding, tgt_embedding)  # [b, n, m]
        # point-wise matching map
        scores = torch.softmax(-distance_map, dim=2)  # [b, n, m]  Eq. (1)

        # neighborhood-wise matching map
        src_knn_scores = scores.view(batch_size * num_points, -1)[src_idx1, :]

        src_knn_scores = src_knn_scores.view(batch_size, num_points, k1, num_points)  # [b, n, k, m]
        # 通过点的周围点计算其neighborhood-wise score -> Eq. (2)
        src_knn_scores = torch.gather(
            src_knn_scores.view(batch_size * num_points, k1, num_points),
            dim=2, index=idx2.view(batch_size, 1, num_points * k1).repeat(1, num_points, 1) \
                             .view(batch_size * num_points, num_points * k1)[:, None, :].repeat(1, k1, 1).long()) \
                             .view(batch_size, num_points, k1, num_points, k1)[:, :, 1:, :, 1:] \
                             .sum(-1).sum(2) / (k1 - 1)  # Eq. (2)

        # The hyper-parameter α (nn_margin) controls the influence of the neighborhood consensus.
        src_knn_scores = self.nn_margin - src_knn_scores
        refined_distance_map = torch.exp(src_knn_scores) * distance_map
        refined_matching_map = torch.softmax(-refined_distance_map, dim=2)  # [b, n, m] Eq. (3)
        # pseudo correspondences of source point clouds (pseudo target point clouds)
        src_corr = torch.matmul(tgt, refined_matching_map.transpose(2, 1).contiguous())  # [b,3,n] Eq. (4)

        ############################## Inlier Evaluation Module ##############################
        # neighborhoods of pseudo target point clouds
        src_knn_corr = src_corr.transpose(2, 1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_knn_corr = src_knn_corr.view(batch_size, num_points, k, num_dims_src)  # [b, n, k, 3]

        # edge features of the pseudo target neighborhoods and the source neighborhoods
        # eij = pi - pj
        # 这里没有提到论文中使用的edge feature的注意力,直接使用edge_feature来进行计算weight,没有使用softmax
        knn_distance = src_corr.transpose(2, 1).contiguous().unsqueeze(2) - src_knn_corr  # [b, n, k, 3]
        src_knn_distance = src.transpose(2, 1).contiguous().unsqueeze(2) - src_knn  # [b, n, k, 3]

        # inlier confidence
        weight = self.weight_function(knn_distance, src_knn_distance)  # [b, 1, n] # Eq. (7)

        # compute rigid transformation
        G = compute_rigid_transform2(src.transpose(1, 2), src_corr.transpose(1, 2), weight.view(src.shape[0], -1))
        R, t = se3.decompose_trans(G)

        ########################### Preparation for the Loss Function #########################
        # choose k keypoints with highest weights
        src_topk_idx, src_keypoints, tgt_keypoints = get_keypoints(src, src_corr, weight, self.num_keypoints)

        # spatial consistency loss 
        idx_tgt_corr = torch.argmax(refined_matching_map, dim=-1).int()  # [b, n]
        identity = torch.eye(num_points_tgt).to(src.device).unsqueeze(0).repeat(batch_size, 1, 1)  # [b, m, m]
        # one_hot_number = pointnet2_utils.gather_operation(identity, idx_tgt_corr) # [b, m, n]

        one_hot_number = torch.gather(identity, dim=2,
                                      index=idx_tgt_corr[:, None, :].repeat(1, identity.shape[1], 1).long())

        src_keypoints_idx = src_topk_idx.repeat(1, num_points_tgt, 1)  # [b, m, num_keypoints]
        keypoints_one_hot = torch.gather(one_hot_number, dim=2, index=src_keypoints_idx).transpose(2, 1).reshape(
            batch_size * self.num_keypoints, num_points_tgt)
        # [b, m, num_keypoints] - [b, num_keypoints, m] - [b * num_keypoints, m]
        predicted_keypoints_scores = torch.gather(refined_matching_map.transpose(2, 1), dim=2,
                                                  index=src_keypoints_idx).transpose(2, 1).reshape(
            batch_size * self.num_keypoints, num_points_tgt)
        # 计算spatial consistency loss
        loss_scl = (-torch.log(predicted_keypoints_scores + 1e-15) * keypoints_one_hot).sum(1).mean()

        # neighorhood information
        src_keypoints_idx2 = src_topk_idx.unsqueeze(-1).repeat(1, 3, 1, k)  # [b, 3, num_keypoints, k]
        tgt_keypoints_knn = torch.gather(knn_distance.permute(0, 3, 1, 2), dim=2,
                                         index=src_keypoints_idx2)  # [b, 3, num_kepoints, k]

        src_transformed = se3.transform_torch(se3.integrate_trans(R, t), src.transpose(1, 2)).transpose(1, 2)

        src_transformed_knn_corr = src_transformed.transpose(2, 1).contiguous().view(batch_size * num_points, -1)[
                                   src_idx, :]
        src_transformed_knn_corr = src_transformed_knn_corr.view(batch_size, num_points, k,
                                                                 num_dims_src)  # [b, n, k, 3]

        knn_distance2 = src_transformed.transpose(2, 1).contiguous().unsqueeze(
            2) - src_transformed_knn_corr  # [b, n, k, 3]
        src_keypoints_knn = torch.gather(knn_distance2.permute(0, 3, 1, 2), dim=2,
                                         index=src_keypoints_idx2)  # [b, 3, num_kepoints, k]
        return R, t.view(batch_size, 3), src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, loss_scl


class RIENET(nn.Module):
    def __init__(self,
                 n_keypoints=256,
                 nn_margin=0.7,
                 n_iters=1,
                 loss_margin=0.01,
                 list_k1=[5, 5, 5],
                 list_k2=[5, 5, 5],
                 emb_dims=256,
                 dim=8,
                 ):
        super(RIENET, self).__init__()
        # 首先使用DGCNN来提取特征
        self.emb_nn = feature_extractor(emb_dims)
        self.single_point_embed = PointNet()
        self.forwards = SVDHead(n_keypoints=n_keypoints, nn_margin=nn_margin, dim=dim)
        self.iter = n_iters
        self.loss = LossFunction(loss_margin=loss_margin)
        self.list_k1 = list_k1
        self.list_k2 = list_k2

    def forward(self, *input):
        """ 
            feature extraction.
            Args:
                src = input[0]: Source point clouds. Size [B, 3, N]
                tgt = input[1]: Target point clouds. Size [B, 3, N]
            Returns:
                rotation_ab_pred: Size [B, 3, 3]
                translation_ab_pred: Size [B, 3]
                global_alignment_loss
                consensus_loss
                spatial_consistency_loss
        """

        src = input[0]
        tgt = input[1]
        batch_size, _, _ = src.size()
        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        global_alignment_loss, consensus_loss, spatial_consistency_loss = 0.0, 0.0, 0.0

        for i in range(self.iter):
            # i为迭代次数->list_k1 and list_k2为每次迭代取的KNN中的K
            # 首先使用DGCNN提取Source和Target特征
            src_embedding, src_idx, src_knn, _ = self.emb_nn(src, self.list_k1[i])
            tgt_embedding, _, tgt_knn, _ = self.emb_nn(tgt, self.list_k1[i])
            # 返回Source和Target的Nearest neighbor indices.
            src_idx1, _ = get_knn_index(src, self.list_k2[i])
            _, tgt_idx = get_knn_index(tgt, self.list_k2[i])

            rotation_ab_pred_i, translation_ab_pred_i, src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, spatial_consistency_loss_i \
                = self.forwards(src, tgt, src_embedding, tgt_embedding, src_idx, self.list_k1[i], src_knn, i, tgt_knn, \
                                src_idx1, tgt_idx, self.list_k2[i])
            # R^-1 * R
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            # R^-1 * t + t^-1
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i

            src = se3.transform_torch(se3.integrate_trans(rotation_ab_pred_i, translation_ab_pred_i),
                                      src.transpose(1, 2)).transpose(1, 2)

            neighborhood_consensus_loss_i, global_alignment_loss_i = self.loss(src_keypoints, tgt_keypoints, \
                                                                               rotation_ab_pred_i,
                                                                               translation_ab_pred_i, src_keypoints_knn,
                                                                               tgt_keypoints_knn, self.list_k2[i], src,
                                                                               tgt)

            global_alignment_loss += global_alignment_loss_i
            consensus_loss += neighborhood_consensus_loss_i
            spatial_consistency_loss += spatial_consistency_loss_i

        return rotation_ab_pred, translation_ab_pred, global_alignment_loss, consensus_loss, spatial_consistency_loss


# if __name__ == '__main__':
#     src = torch.rand((2, 3, 1024)).cuda()
#     tgt = torch.rand((2, 3, 1024)).cuda()
#     rie = RIENET().cuda()
#     rotation_ab_pred, translation_ab_pred, global_alignment_loss, consensus_loss, spatial_consistency_loss=rie(src, tgt)
#     print(translation_ab_pred)