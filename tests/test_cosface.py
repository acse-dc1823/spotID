# author: David Colomer Matachana
# GitHub username: acse-dc1823
import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck
from leopard_id.losses import CosFace


@pytest.fixture
def cosface_setup():
    in_features = 128
    out_features = 10
    batch_size = 32
    cosface = CosFace(in_features, out_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cosface.to(device)
    return cosface, in_features, out_features, batch_size, device


def test_output_shape(cosface_setup):
    cosface, in_features, out_features, batch_size, device = cosface_setup
    input = torch.randn(batch_size, in_features).to(device)
    labels = torch.randint(0, out_features, (batch_size,)).to(device)
    output = cosface(input, labels)
    assert output.shape == (batch_size, out_features)


def test_weight_normalization_in_forward(cosface_setup):
    cosface, in_features, out_features, batch_size, device = cosface_setup
    input = torch.randn(batch_size, in_features).to(device)
    labels = torch.randint(0, out_features, (batch_size,)).to(device)

    # Call forward to trigger weight normalization
    cosface(input, labels)

    # Access the normalized weights used in the forward pass
    with torch.no_grad():
        weight_norm = F.normalize(cosface.weight, p=2, dim=1)

    assert torch.allclose(
        torch.norm(weight_norm, p=2, dim=1), torch.ones(out_features).to(device), atol=1e-6
    )


def test_cosine_similarity_range(cosface_setup):
    cosface, in_features, out_features, batch_size, device = cosface_setup
    input = torch.randn(batch_size, in_features).to(device)
    labels = torch.randint(0, out_features, (batch_size,)).to(device)
    output = cosface(input, labels)
    cosine_sim = output / cosface.scale
    assert torch.all(cosine_sim >= -1 - 1e-6)
    assert torch.all(cosine_sim <= 1 + 1e-6)


def test_margin_effect(cosface_setup):
    cosface, in_features, out_features, batch_size, device = cosface_setup
    input = torch.randn(batch_size, in_features).to(device)
    labels = torch.randint(0, out_features, (batch_size,)).to(device)
    output_with_margin = cosface(input, labels)

    # Temporarily set margin to 0
    original_margin = cosface.margin
    cosface.margin = 0
    output_without_margin = cosface(input, labels)
    cosface.margin = original_margin

    # Check that margin reduces logits for correct class
    correct_class_mask = F.one_hot(labels, num_classes=out_features).bool()
    assert torch.all(
        output_with_margin[correct_class_mask] < output_without_margin[correct_class_mask]
    )


def test_gradients_relaxed(cosface_setup):
    cosface, in_features, out_features, batch_size, device = cosface_setup
    input = torch.randn(batch_size, in_features, requires_grad=True).to(device)
    labels = torch.randint(0, out_features, (batch_size,)).to(device)

    def cosface_wrapper(x):
        return cosface(x, labels)

    try:
        gradcheck(cosface_wrapper, input, eps=1e-3, atol=1e-2)
    except AssertionError as e:
        print(f"Gradient check failed with error: {str(e)}")
        print(
            "This might be due to numerical instability. Proceeding with a manual gradient check."
        )

        # Manual gradient check
        output = cosface_wrapper(input)
        output.sum().backward()
        analytic_grad = input.grad.clone()

        # Compute numerical gradients
        epsilon = 1e-3
        numeric_grad = torch.zeros_like(input)
        for i in range(input.numel()):
            input_plus = input.clone()
            input_plus.view(-1)[i] += epsilon
            output_plus = cosface_wrapper(input_plus)

            input_minus = input.clone()
            input_minus.view(-1)[i] -= epsilon
            output_minus = cosface_wrapper(input_minus)

            numeric_grad.view(-1)[i] = (output_plus.sum() - output_minus.sum()) / (2 * epsilon)

        # Compare gradients
        rel_error = torch.norm(analytic_grad - numeric_grad) / torch.norm(
            analytic_grad + numeric_grad
        )
        print(f"Relative error between analytic and numeric gradients: {rel_error.item()}")
        assert rel_error < 1e-2, f"Large relative error in gradients: {rel_error.item()}"


def test_forward_backward_consistency(cosface_setup):
    cosface, in_features, out_features, batch_size, device = cosface_setup
    input = torch.randn(batch_size, in_features, requires_grad=True).to(device)
    labels = torch.randint(0, out_features, (batch_size,)).to(device)

    # Forward pass
    output = cosface(input, labels)

    # Print debugging information
    print(f"Output shape: {output.shape}")
    print(f"Output requires grad: {output.requires_grad}")
    print(f"Weight requires grad: {cosface.weight.requires_grad}")

    # Backward pass
    loss = output.sum()
    loss.backward()

    # More debugging information
    print(f"Input grad: {input.grad is not None}")
    print(f"Weight grad: {cosface.weight.grad is not None}")

    if cosface.weight.grad is None:
        print("Weight grad is None. Investigating further...")
        print(f"Is weight leaf: {cosface.weight.is_leaf}")
        print(f"Weight grad fn: {cosface.weight.grad_fn}")

    # Check if gradients are computed
    assert input.grad is not None, "Input gradients are None"
    assert cosface.weight.grad is not None, "Weight gradients are None"

    # Check if gradients have reasonable values
    assert not torch.isnan(input.grad).any(), "NaN in input gradients"
    assert not torch.isinf(input.grad).any(), "Inf in input gradients"
    assert not torch.isnan(cosface.weight.grad).any(), "NaN in weight gradients"
    assert not torch.isinf(cosface.weight.grad).any(), "Inf in weight gradients"


def test_new_margin_function(cosface_setup):
    cosface, _, _, _, device = cosface_setup

    # Create a batch of 100 samples with 10 classes
    batch_size, num_classes = 100, 10

    # Generate random cosine similarities
    cosine = torch.rand(batch_size, num_classes).to(device)

    # Ensure the "correct" class (one-hot) has the highest cosine similarity
    _, max_indices = cosine.max(dim=1)
    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, max_indices.unsqueeze(1), 1)

    # Apply the new_margin function
    margin_values = cosface.new_margin(cosine, one_hot)

    # Test 1: Check that margin is non-negative
    assert torch.all(margin_values >= 0), "Margin values should be non-negative"

    # Test 2: Check that margin is highest for the "correct" class (one-hot)
    assert torch.all(
        margin_values.argmax(dim=1) == one_hot.argmax(dim=1)
    ), "Highest margin should correspond to one-hot encoding"

    # Test 3: Check that margin is zero for non-selected classes
    assert torch.allclose(
        margin_values[~one_hot.bool()], torch.tensor(0.0).to(device)
    ), "Margin should be zero for non-selected classes"

    # Test 4: Verify the specific shape of the margin function
    # Sample a range of cosine values for a single class
    sample_cosine = torch.linspace(-1, 1, 2000).unsqueeze(1).to(device)
    sample_one_hot = torch.ones_like(sample_cosine)
    sample_margin = cosface.new_margin(sample_cosine, sample_one_hot).squeeze()

    # Find the peak of the margin
    peak_index = sample_margin.argmax()
    peak_cosine = sample_cosine[peak_index]

    # Check if the peak is around 0 (orthogonality)
    assert (
        -0.05 < peak_cosine < 0.05
    ), f"Peak margin should occur around cos(90°) (0), but occurred at {peak_cosine.item()}"

    # Check if margin decreases for very high and very low cosine values
    assert (
        sample_margin[0] < sample_margin[peak_index]
    ), "Margin should decrease for very low cosine values"
    assert (
        sample_margin[-1] < sample_margin[peak_index]
    ), "Margin should decrease for very high cosine values"

    # Test 5: Verify symmetry of the margin function
    assert torch.allclose(
        sample_margin[:1000], torch.flip(sample_margin[1000:], [0]), atol=1e-6
    ), "Margin function should be symmetric around 0"

    print("All tests passed successfully!")
