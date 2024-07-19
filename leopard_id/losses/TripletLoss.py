import torch
import torch.nn as nn
import numpy as np

import logging

logging.basicConfig(
    filename="logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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
    def __init__(self, margin=0.20, verbose=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        self.verbose = verbose

    def forward(self, features, labels, epoch=0):
        """
        Args:
            features: feature matrix with shape (batch_size, features_dim)
            labels: ground truth numerical labels with shape (batch_size)
            from dataset.label_to_index
            epoch: epoch we currently are at. Used for semi-hard negative
            mining.
        """
        dist_mat = euclidean_dist(features, features)
        batch_size = features.size(0)


        # if self.verbose:
        #     logging.info(f"batch size: {batch_size}")

        
        # if self.verbose:
        #     logging.info(f"batch size: {batch_size}")

        triplet_loss = 0.0
        counter = 0
        for i in range(batch_size):
            pos_indices = (labels == labels[i]).nonzero(as_tuple=False).view(-1)
            pos_indices = pos_indices[pos_indices > i]  # Exclude the anchor itself
            neg_indices = (labels != labels[i]).nonzero(as_tuple=False).view(-1)

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue  # No valid triplets
            
            # Iterate over all positive pairs for the anchor
            for pos_idx in pos_indices:
                pos_dist = dist_mat[i, pos_idx]
                # Only implement semi hard mining after a few epochs.
                # Otherwise, learning stagnates.
                if epoch > 3:
                    # Semi-hard negative mining: negatives harder than the current positive but within margin
                    semi_hard_negatives = neg_indices[(dist_mat[i, neg_indices] > pos_dist) & (dist_mat[i, neg_indices] < pos_dist + self.margin)]
                    if len(semi_hard_negatives) == 0:
                        continue  # Skip if no semi-hard negatives are found

                    # Inverse distance weighting for semi-hard negatives selection
                    neg_distances = dist_mat[i, semi_hard_negatives]
                    weights = 1.0 / neg_distances
                    # Normalize weights to sum to 1
                    weights = weights / torch.sum(weights)

                    # Weighted random choice of negative indices
                    neg_idx = np.random.choice(semi_hard_negatives.cpu().detach().numpy(), p=weights.cpu().detach().numpy())

                else:
                    # Random choice from all valid negatives (completely random selection)
                    neg_idx = np.random.choice(neg_indices.cpu().numpy(), 1)[0]

                neg_dist = dist_mat[i, neg_idx]

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
            logging.info(f"labels: {labels}")
            logging.info("No valid triplets, triplet loss is thus 0.")
            triplet_loss = torch.tensor(0.0, device=features.device)

        return triplet_loss


# if __name__ == "__main__":
#     # torch.manual_seed(42)  # Set the random seed for reproducibility
#     features = torch.rand(10, 512)  # 10 samples, 512-dimensional features
#     labels = torch.tensor([1, 1, 2, 3, 2, 2, 3, 1, 3, 2])  # Sample labels
#     triplet_loss_func = TripletLoss()
#     loss = triplet_loss_func(features, labels)
#     print("Triplet Loss:", loss.item())
