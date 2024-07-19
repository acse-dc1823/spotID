import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
import os

# Calculate the directory containing your module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Assumes the tests folder is at the same level as leopard_id
module_dir = os.path.join(parent_dir, 'leopard_id')  # Path to the leopard_id directory

# Add it to sys.path for resolving the modules
sys.path.insert(0, module_dir)

from leopard_id.engine import evaluate_epoch_test

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
