# author: David Colomer Matachana
# GitHub username: acse-dc1823

import torch
import math
import heapq


def compute_dynamic_top_k_avg_precision(dist_matrix, labels, max_k, device):
    """
    Calculate the top-k average precision for each sample, with dynamic
    adjustment based on class size,and return the mean of these
    values excluding cases where dynamic k is zero. Dynamic k is introduced
    because some classes just have a few exemplars per image, fewer than
    max_k. In these cases, the typical top k average precision would be
    capped at a maximum of num_exemplars_in_class / k, lower than 1.
    Calculated each batch. Obviously, the larger the dataset, the lower
    the precision, as the chance of finding the correct match is lower.
    Metric is much stricter than top k match rate. 
    It gives higher precision to those having matches
    closer to the anchor, as they'll count for all the k comparisons.

    :param dist_matrix: A 2D PyTorch tensor where dist_matrix[i, j]
        is the distance from sample i to sample j.
    :param labels: A 1D PyTorch tensor with class labels for each sample.
    :param max_k: The maximum k for calculating average precision.
    :return: The mean average precision across all valid samples.
    """
    dist_matrix = dist_matrix.to(device)
    labels = labels.to(device)
    num_samples = dist_matrix.size(0)
    avg_precisions = torch.zeros(num_samples, device=device)
    valid_counts = 0

    # Calculate the count of each class minus one (for self-comparison)
    class_counts = torch.bincount(labels) - 1

    for i in range(num_samples):
        # Get current sample's class
        current_class = labels[i]

        # Determine dynamic k based on class size and max_k
        dynamic_k = min(class_counts[current_class].item(), max_k)

        if dynamic_k > 0:
            # Set distance to itself to infinity to ignore it
            dists = dist_matrix[i].clone()
            dists[i] = float("inf")

            # Find the top dynamic k smallest distances.
            _, top_k_indices = torch.topk(dists, dynamic_k, largest=False, sorted=True)

            # Get the labels of the top k closest samples
            top_k_labels = labels[top_k_indices]

            # True positives at each k position
            true_positives = (top_k_labels == current_class).float()

            # Cum sum of true positives to calculate precision at each cut-off
            cum_true_positives = true_positives.cumsum(0)

            # Ranks (1-based) for each of the top k
            ranks = torch.arange(1, dynamic_k + 1, device=device).float()
            # Precision at each k
            precision_at_k = cum_true_positives / ranks

            # Average of precisions at each k
            avg_precisions[i] = precision_at_k.mean()
            valid_counts += 1

    # Compute the mean of valid average precisions
    if valid_counts > 0:
        mean_avg_precision = avg_precisions.sum() / valid_counts
    else:
        mean_avg_precision = torch.tensor(0.0)  # In case all classes have only one sample

    return mean_avg_precision


def compute_class_distance_ratio(dist_matrix, labels, device):
    """
    Calculate the ratio of the average intra-class distance to the average
    inter-class distance. A lower value indicates that the model is learning
    correctly to tell the difference between different classes. 

    :param dist_matrix: A 2D PyTorch tensor where dist_matrix[i, j]
        is the distance from sample i to sample j.
    :param labels: A 1D PyTorch tensor with class labels for each sample.
    :return: The mean ratio of intra-class to inter-class distances.
    """
    dist_matrix = dist_matrix.to(device)
    labels = labels.to(device)
    num_samples = dist_matrix.size(0)
    ratios = []

    for i in range(num_samples):
        # Intra-class distances (masking other classes)
        intra_mask = labels == labels[i]
        intra_mask[i] = False  # Exclude self distance
        intra_distances = dist_matrix[i, intra_mask]

        # Inter-class distances (masking the same class)
        inter_mask = labels != labels[i]
        inter_distances = dist_matrix[i, inter_mask]

        if intra_distances.numel() > 0 and inter_distances.numel() > 0:
            mean_intra_distance = intra_distances.mean()
            mean_inter_distance = inter_distances.mean()

            # Ratio of intra-class to inter-class distances
            ratio = mean_intra_distance / mean_inter_distance
            ratios.append(ratio)

    # Compute the mean of all ratios if any valid ratios were calculated
    if ratios:
        mean_ratio = torch.tensor(ratios, device=device).mean()
    else:
        mean_ratio = torch.tensor(float("nan"), device=device)

    return mean_ratio


def compute_top_k_rank_match_detection(dist_matrix, labels, max_k, device):
    """
    Calculate the Top-k Rank match detection: Calculates the ratio of images
    that contain a match in the top-k most similar ranks. Return
    the accuracy for each k from 1 to max_k. Used for figure 8 in paper.

    :param dist_matrix: A 2D PyTorch tensor where dist_matrix[i, j]
        is the distance from sample i to sample j.
    :param labels: A 1D PyTorch tensor with class labels for each sample.
    :param max_k: The maximum k for calculating rank accuracy.
    :param device: The device (CPU or GPU) to perform computations.
    :return: mean match rates for each k in ascending order.
    """
    dist_matrix = dist_matrix.to(device)
    labels = labels.to(device)
    num_samples = dist_matrix.size(0)

    # For storing individual accuracies per k to visualize the distribution
    all_accuracies = torch.zeros((num_samples, max_k), device=device)
    mask = torch.zeros(
        num_samples, dtype=torch.bool, device=device
    )  # Mask to track processed samples

    # Count each label in the dataset excluding the sample itself
    class_counts = torch.bincount(labels) - 1

    for i in range(num_samples):
        # Only compute if there are other sampler with the same label
        if class_counts[labels[i]] > 0:
            mask[i] = True
            dists = dist_matrix[i].clone()
            # Ignore self in distance matrix
            dists[i] = float("inf")

            # Get indices of the elements sorted by closest distance
            _, indices = torch.sort(dists)

            # Get the labels of the sorted elements
            sorted_labels = labels[indices]

            # Check for matches in top k elements
            matches = (sorted_labels[:max_k] == labels[i]).unsqueeze(0)

            # Record results for each k
            for k in range(max_k):
                all_accuracies[i, k] = matches[:, : k + 1].any(dim=1).float()

    # Only consider rows in `all_accuracies` where `mask` is True
    accuracies = all_accuracies[mask].mean(dim=0)

    return accuracies


def compute_average_angular_separation(dist_mat, targets):
    """
    Compute average angular separation for same class and different class pairs,
    and store the top 10 smallest angular separations. Use the cosine distance
    matrix for this, taking the arccos (expensive operation, hence why this is
    only computed for test set) for this. It then masks the correct and different
    class pairs and computes the average angular separation for each.

    Useful for understanding how the embedding space behaves in high dimensional space.
    In a good embedding space, the same class pairs should have significantly smaller
    angular separations than different class pairs. Different class pairs should be
    close to orthogonal.


    Args:
    dist_mat (torch.Tensor): Cosine distance matrix
    targets (torch.Tensor): Class labels for each sample

    Returns:
    tuple: (avg_same_class_angle, avg_diff_class_angle, top_10_smallest_angles)
    """
    # Convert cosine distance to angle (in degrees)
    angle_mat = torch.acos(1 - dist_mat) * (180 / math.pi)

    n = targets.size(0)
    mask_same = targets.unsqueeze(1) == targets.unsqueeze(0)
    mask_diff = ~mask_same

    # Remove self-comparisons
    mask_same.fill_diagonal_(False)

    # Compute average angles
    avg_same_class_angle = angle_mat[mask_same].mean().item()
    avg_diff_class_angle = angle_mat[mask_diff].mean().item()

    # Find top 10 smallest angular separations
    triu_indices = torch.triu_indices(n, n, offset=1)
    upper_triangle = angle_mat[triu_indices[0], triu_indices[1]]
    top_10_smallest = heapq.nsmallest(10, upper_triangle.tolist())
    top_10_smallest = [round(angle, 2) for angle in top_10_smallest]

    return avg_same_class_angle, avg_diff_class_angle, top_10_smallest
