# author: David Colomer Matachana
# GitHub username: acse-dc1823
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
import os

# Calculate the directory containing your module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
module_dir = os.path.join(parent_dir, 'leopard_id')  # Path to the leopard_id directory

sys.path.insert(0, module_dir)

from leopard_id.engine import evaluate_epoch_test
from leopard_id.metrics import (
    compute_dynamic_top_k_avg_precision,
    compute_class_distance_ratio,
    compute_top_k_rank_match_detection,
)

import pytest


class DeterministicDummyModel(nn.Module):
    def __init__(self, output_dim=256):
        super(DeterministicDummyModel, self).__init__()
        self.output_dim = output_dim
        # Define a simple linear layer with predefined weights and no biases
        self.linear = nn.Linear(3 * 224 * 224, output_dim, bias=False)
        
        # Initialize weights to a fixed value (e.g., all ones)
        nn.init.constant_(self.linear.weight, 0.01)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        # Apply the linear transformation
        x = self.linear(x)
        return x


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_evaluation_consistency():
    """
    Ensures that the evaluation metrics are consistent when using different batch sizes,
    as we made a change to the evaluation function to aggregate outputs across batches.
    """
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_k = 5  # Set according to your needs

    # Create a dummy dataset
    num_samples = 100
    X = torch.randn(num_samples, 3, 224, 224)  # Example dimension for image data
    y = torch.randint(0, 2, (num_samples,))  # Binary classification
    dataset = TensorDataset(X, y)

    # Instantiate the model
    model = DeterministicDummyModel()
    model.to(device)

    # DataLoader with full dataset
    full_loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    full_metrics = evaluate_epoch_test(model, full_loader, device, max_k)

    # DataLoader with smaller batches
    batch_size = 10
    batch_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch_metrics = evaluate_epoch_test(model, batch_loader, device, max_k)

    # Assert the metrics are the same
    assert full_metrics[0] == pytest.approx(batch_metrics[0]), "Precision mismatch"
    assert full_metrics[1] == pytest.approx(batch_metrics[1]), "Class distance ratio mismatch"
    assert full_metrics[2] == pytest.approx(batch_metrics[2]), "Match rate mismatch"

def test_compute_dynamic_top_k_avg_precision():
    """
    tests dynamic top k average precision calculation. Tests all functionality. Firstly,
    that dynamic top k is followed (as we are defining classes with less tha top k samples).
    We also test that samples with just one sample are not included in the calculation.
    Finally, we test that the mean average precision is calculated correctly.
    """
    device = torch.device('cpu')
    dist_matrix = torch.tensor([
        [0.0, 0.1, 0.2, 0.3, 0.4],
        [0.1, 0.0, 0.5, 0.6, 0.7],
        [0.2, 0.5, 0.0, 0.8, 0.9],
        [0.3, 0.6, 0.8, 0.0, 1.0],
        [0.4, 0.7, 0.9, 1.0, 0.0]
    ])
    labels = torch.tensor([0, 0, 1, 1, 2])
    max_k = 3

    result = compute_dynamic_top_k_avg_precision(dist_matrix, labels, max_k, device)
    
    # Expected result:
    # Sample 0: AP = (1/1 + 2/2) / 2 = 1
    # Sample 1: AP = (1/1 + 2/2) / 2 = 1
    # Sample 2: AP = 0/1 = 0
    # Sample 3: AP = 0/1 = 0
    # Sample 4: No matches, not included
    # Mean AP = (1 + 1 + 0 + 0) / 4 = 1
    
    assert result == pytest.approx(0.5)


def test_compute_top_k_rank_match_detection():
    """
    Ensures top k rank match detection is calculated correctly. Includes test that singleton
    classes are skipped over.
    """
    device = torch.device('cpu')
    dist_matrix = torch.tensor([
        [0.0, 0.1, 0.2, 0.3, 0.4],
        [0.1, 0.0, 0.5, 0.6, 0.7],
        [0.2, 0.5, 0.0, 0.8, 0.9],
        [0.3, 0.8, 0.6, 0.0, 1.0],
        [0.4, 0.7, 0.9, 1.0, 0.0]
    ])
    labels = torch.tensor([0, 0, 1, 1, 2])
    max_k = 3

    result = compute_top_k_rank_match_detection(dist_matrix, labels, max_k, device)
    
    # Expected result:
    # k=1: [1, 1, 0, 0] = 0.5
    # k=2: [1, 1, 0, 1] = 0.75
    # k=3: [1, 1, 1, 1] = 1.0
    
    expected = torch.tensor([0.5, 0.75, 1.0])
    assert torch.allclose(result, expected)

