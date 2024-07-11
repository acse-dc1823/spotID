import torch


def compute_dynamic_k_avg_precision(dist_matrix, labels, max_k):
    """
    Calculate the top-k average precision for each sample, with dynamic
    adjustment based on class size,and return the mean of these
    values excluding cases where dynamic k is zero.

    :param dist_matrix: A 2D PyTorch tensor where dist_matrix[i, j]
        is the distance from sample i to sample j.
    :param labels: A 1D PyTorch tensor with class labels for each sample.
    :param max_k: The maximum k for calculating average precision.
    :return: The mean average precision across all valid samples.
    """
    num_samples = dist_matrix.size(0)
    avg_precisions = torch.zeros(num_samples)
    valid_counts = 0

    # Calculate the count of each class minus one (for self-comparison)
    class_counts = torch.bincount(labels) - 1
    print("labels size", labels.size())
    print("matrix size", dist_matrix.size())
    print("labels", labels)
    print("class counts size", class_counts.size())
    print("class_counts", class_counts)

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
            dists1, top_k_indices = torch.topk(dists, dynamic_k, largest=False,
                                               sorted=True)
            print("dists1", dists1)
            print("top k indices", top_k_indices)

            # Get the labels of the top k closest samples
            top_k_labels = labels[top_k_indices]
            print("top k labels", top_k_labels)

            # True positives at each k position
            true_positives = (top_k_labels == current_class).float()
            print("true positives", true_positives)

            # Cum sum of true positives to calculate precision at each cut-off
            cum_true_positives = true_positives.cumsum(0)

            # Ranks (1-based) for each of the top k
            ranks = torch.arange(1, dynamic_k + 1).float()

            # Precision at each k
            precision_at_k = cum_true_positives / ranks

            # Average of precisions at each k
            avg_precisions[i] = precision_at_k.mean()
            valid_counts += 1

    # Compute the mean of valid average precisions
    if valid_counts > 0:
        mean_avg_precision = avg_precisions.sum() / valid_counts
    else:
        mean_avg_precision = torch.tensor(
            0.0
        )  # In case all classes have only one sample

    return mean_avg_precision
