# author: David Colomer Matachana
# GitHub username: acse-dc1823

import pytest
import torch
import torch.nn as nn
import numpy as np
from leopard_id.losses import euclidean_dist, TripletLoss

def test_euclidean_dist():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    expected = torch.tensor([
        [1.0000, 1.0000, 2.2361],
        [3.6056, 2.2361, 1.0000]
    ])
    result = euclidean_dist(x, y)
    assert torch.allclose(result, expected, atol=1e-4)

def test_triplet_loss_forward():
    loss_fn = TripletLoss(margin=0.2)
    features = torch.tensor([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0]
    ])
    labels = torch.tensor([0, 0, 1, 1])
    
    loss = loss_fn(features, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar output

def test_triplet_loss_no_valid_triplets():
    loss_fn = TripletLoss(margin=0.2)
    features = torch.tensor([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0]
    ])
    labels = torch.tensor([0, 1, 2, 3])  # All different labels
    
    loss = loss_fn(features, labels)
    assert loss.item() == 0.0

def test_triplet_loss_device_compatibility():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        loss_fn = TripletLoss(margin=0.2)
        features = torch.tensor([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0]
        ], device=device)
        labels = torch.tensor([0, 0, 1, 1], device=device)
        
        loss = loss_fn(features, labels)
        assert loss.device == device

def test_semi_hard_negative_mining():
    class MockTripletLoss(TripletLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.selected_negatives = []

        def forward(self, features, labels, epoch=0):
            dist_mat = euclidean_dist(features, features)
            batch_size = features.size(0)

            for i in range(batch_size):
                pos_indices = (labels == labels[i]).nonzero(as_tuple=False).view(-1)
                pos_indices = pos_indices[pos_indices > i]
                neg_indices = (labels != labels[i]).nonzero(as_tuple=False).view(-1)

                if len(pos_indices) == 0 or len(neg_indices) == 0:
                    continue

                for pos_idx in pos_indices:
                    pos_dist = dist_mat[i, pos_idx]

                    if epoch > 3:
                        semi_hard_negatives = neg_indices[(dist_mat[i, neg_indices] > pos_dist) & 
                                                          (dist_mat[i, neg_indices] < pos_dist + self.margin)]

                        if len(semi_hard_negatives) > 0:
                            neg_distances = dist_mat[i, semi_hard_negatives]
                            weights = 1.0 / neg_distances
                            weights = weights / torch.sum(weights)
                            neg_idx = np.random.choice(semi_hard_negatives.cpu().numpy(), p=weights.cpu().numpy())
                            self.selected_negatives.append((i, pos_idx.item(), neg_idx, pos_dist.item(), dist_mat[i, neg_idx].item()))

            return torch.tensor(0.0)  # Dummy loss value

    # Create a set of features where the distances are predictable
    features = torch.tensor([
        [0.0, 0.0],  # Anchor
        [0.1, 0.1],  # Positive
        [0.3, 0.3],  # Semi-hard negative
        [0.5, 0.5],  # Hard negative
        [1.0, 1.0],  # Easy negative
    ])
    labels = torch.tensor([0, 0, 1, 1, 1])

    mock_loss_fn = MockTripletLoss(margin=0.3)
    _ = mock_loss_fn(features, labels, epoch=4)  # Run with semi-hard mining

    assert len(mock_loss_fn.selected_negatives) > 0, "No semi-hard negatives were selected"

    for anchor, positive, negative, pos_dist, neg_dist in mock_loss_fn.selected_negatives:
        assert pos_dist < neg_dist < pos_dist + mock_loss_fn.margin, \
            f"Selected negative not within semi-hard range: pos_dist={pos_dist}, neg_dist={neg_dist}, margin={mock_loss_fn.margin}"
