import torch
import torch.nn as nn
import torch.nn.functional as F

def estimate_center_distance(self, points, k=16):
    """
    Compute the distance between each point and the centroid of its k-NN neighborhood.

    :param points: Input point cloud of shape [B, N, 3].
    :param k: Number of neighbors used for centroid estimation.
    :return: A tensor of shape [B, N] containing per-point distances to the local centroid.
    """

    B, N, _ = points.shape
    center_dist = torch.zeros(B, N, device=points.device)

    for b in range(B):
        for i in range(N):
            dist = torch.norm(points[b] - points[b, i], dim=1)
            knn_idx = torch.topk(dist, k=k+1, largest=False)[1][1:]
            knn_pts = points[b, knn_idx]  # Shape: [k, 3]
            center = knn_pts.mean(dim=0)
            center_dist[b, i] = torch.norm(points[b, i] - center)

    return center_dist

def divide_point_cloud_by_center_distance(self, points, k=20, contour_num=512, flat_num=512):
    """
    Divide the point cloud into contour and flat subsets based on centroid-distance computed from k-NN neighborhoods.

    :param points: Input point cloud [B, N, 3] or [B, N, 6] (xyz used).
    :param k: Number of nearest neighbors used for centroid distance estimation.
    :param contour_num: Number of points selected as contour points (largest distances).
    :param flat_num: Number of points selected as flat points (smallest distances).

    :return:
        flat_sets:     Flat point subsets, shape [B, flat_num, 3]
        contour_sets:  Contour point subsets, shape [B, contour_num, 3]
        flat_indices:  Indices of flat points in the original point cloud, shape [B, flat_num]
        contour_indices: Indices of contour points, shape [B, contour_num]
        center_dist:   Per-point centroid distances, shape [B, N]
    """

    pts = points[:, :, :3]
    center_dist = self.estimate_center_distance(pts, k)
    B, N, _ = pts.shape

    contour_sets, flat_sets = [], []
    contour_indices_list, flat_indices_list = [], []

    for b in range(B):
        sorted_indices = torch.argsort(center_dist[b], descending=True)

        contour_indices = sorted_indices[:contour_num]
        flat_indices = sorted_indices[-flat_num:]

        contour_sets.append(pts[b, contour_indices])
        flat_sets.append(pts[b, flat_indices])

        contour_indices_list.append(contour_indices)
        flat_indices_list.append(flat_indices)

    contour_sets = torch.stack(contour_sets, dim=0)
    flat_sets = torch.stack(flat_sets, dim=0)
    contour_indices_list = torch.stack(contour_indices_list, dim=0)
    flat_indices_list = torch.stack(flat_indices_list, dim=0)

    return flat_sets, contour_sets, flat_indices_list, contour_indices_list, center_dist

def get_optimal_direction(self, point, gradient, knn_graph):
    """
    Compute the optimal perturbation direction based on the local neighborhood.

    :param point: Coordinates of the current point, shape [3].
    :param gradient: Gradient vector at the current point, shape [1, 3].
    :param knn_graph: Binary KNN indicator vector of shape [N], where non-zero entries mark neighboring points.
    :return: A unit perturbation direction of shape [1, 3].
    """

    neighbor_indices = torch.nonzero(knn_graph).squeeze()

    if len(neighbor_indices) == 0:
        return gradient

    neighbor_directions = self.points[0, neighbor_indices, :] - point
    neighbor_directions = F.normalize(neighbor_directions, p=2, dim=1)

    gradient = gradient.squeeze()                      # [1, 3] -> [3]
    cos_angles = torch.matmul(neighbor_directions, gradient)

    valid_mask = cos_angles > 0
    if not valid_mask.any():
        return gradient.unsqueeze(0)                   # return [1, 3]

    valid_directions = neighbor_directions[valid_mask]
    valid_cos_angles = cos_angles[valid_mask]

    best_idx = torch.argmax(valid_cos_angles)
    
    return valid_directions[best_idx].unsqueeze(0)     # shape [1, 3]
