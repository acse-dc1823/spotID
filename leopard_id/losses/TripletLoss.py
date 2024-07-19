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
        """
        dist_mat = euclidean_dist(features, features)
        batch_size = features.size(0)

        # if self.verbose:
        #     logging.info(f"batch size: {batch_size}")

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

            # Iterate over all positive pairs for the anchor
            for pos_idx in pos_indices:
                pos_dist = dist_mat[i, pos_idx]
                
                # Only implement semi hard mining after a few epochs.
                # Otherwise, learning stagnates.
                if epoch > 3:
                    # Mask for semi-hard negatives: further away than the positive
                    # but less than the anchor-positive distance plus margin
                    semi_hard_negatives = (dist_mat[i, neg_indices] > pos_dist) & (
                        dist_mat[i, neg_indices] < pos_dist + self.margin
                    )
                    if semi_hard_negatives.any():
                        # Selecting the hardest semi-hard negative
                        hardest_negative_idx = neg_indices[semi_hard_negatives][
                            torch.argmin(
                                dist_mat[i, neg_indices[semi_hard_negatives]]
                            )
                        ]
                        neg_dist = dist_mat[i, hardest_negative_idx]
                        counter += 1

                else:
                    neg_idx = np.random.choice(neg_indices.cpu().numpy())
                    neg_dist = dist_mat[i, neg_idx]
                    counter += 1
                    # Calculate the triplet loss for the selected triplet
                loss = self.ranking_loss(
                    pos_dist.unsqueeze(0),
                    neg_dist.unsqueeze(0),
                    torch.tensor([1.0], device=features.device),
                )
                triplet_loss += loss

        if counter > 0:
            # Average loss over the batch
            triplet_loss = triplet_loss / counter
        else:
            logging.info(f"labels: {labels}")
            logging.info("No valid triplets, triplet loss is thus 0.")
            triplet_loss = torch.tensor(0.0, device=features.device)

        # if self.verbose:
        #     logging.info(f"total number of positive-positive"
        #                  f"with random negative pairs is: {counter}")
        return triplet_loss


# if __name__ == "__main__":
#     # torch.manual_seed(42)  # Set the random seed for reproducibility
#     features = torch.rand(10, 512)  # 10 samples, 512-dimensional features
#     labels = torch.tensor([1, 1, 2, 3, 2, 2, 3, 1, 3, 2])  # Sample labels
#     triplet_loss_func = TripletLoss()
#     loss = triplet_loss_func(features, labels)
#     print("Triplet Loss:", loss.item())
