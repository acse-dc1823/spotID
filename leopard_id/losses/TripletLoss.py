import torch
import torch.nn as nn
import numpy as np


def euclidean_dist(x, y):
    """
    Compute the Euclidean distance matrix between two sets of vectors.
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    # TODO: Ensure this is correct
    dist_squared = torch.pow(x - y, 2).sum(2)
    return torch.sqrt(torch.clamp(dist_squared, min=1e-11))


class TripletLoss(nn.Module):
    def __init__(self, margin=2.0, verbose=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        self.verbose = verbose

    def forward(self, features, labels):
        """
        Args:
            features: feature matrix with shape (batch_size, features_dim)
            labels: ground truth numerical labels with shape (batch_size)
            from dataset.label_to_index
        """
        dist_mat = euclidean_dist(features, features)
        batch_size = features.size(0)
        if self.verbose:
            print("batch size:", batch_size)

        # Initialize triplet loss
        triplet_loss = 0.0

        # Iterate over each anchor, select all positive and random negative
        counter = 0
        for i in range(batch_size):
            all_pos_indices = (
                (labels == labels[i]).nonzero(as_tuple=False).view(-1)
            )

            # Only consider one-way pairs, also removing anchor itself
            pos_indices = all_pos_indices[all_pos_indices > i]
            neg_indices = (
                (labels != labels[i]).nonzero(as_tuple=False).view(-1)
            )

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue  # No valid triplets

            # Choose all positives and one random negative for each positive
            neg_idx = np.random.choice(neg_indices.cpu().numpy())
            neg_dist = dist_mat[i, neg_idx]

            for pos_idx in pos_indices:
                pos_dist = dist_mat[i, pos_idx]
                # Calculate the triplet loss, accumulate
                loss = self.ranking_loss(
                    neg_dist.unsqueeze(0),
                    pos_dist.unsqueeze(0),
                    torch.tensor([1.0], device=features.device),
                )
                triplet_loss += loss
                counter += 1

        if counter > 0:
            # Average loss over the batch
            triplet_loss = triplet_loss / counter
        else:
            triplet_loss = torch.tensor(0.0, device=features.device)

        if self.verbose:
            print(
                "total number of positive-positive",
                "with random negative pairs is:",
                counter,
            )
        return triplet_loss


# if __name__ == "__main__":
#     # torch.manual_seed(42)  # Set the random seed for reproducibility
#     features = torch.rand(10, 512)  # 10 samples, 512-dimensional features
#     labels = torch.tensor([1, 1, 2, 3, 2, 2, 3, 1, 3, 2])  # Sample labels
#     triplet_loss_func = TripletLoss()
#     loss = triplet_loss_func(features, labels)
#     print("Triplet Loss:", loss.item())
