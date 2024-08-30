# author: David Colomer Matachana
# GitHub username: acse-dc1823

import pytest
import torch
import timm
from leopard_id.model import CustomResNet, CustomEfficientNet, EmbeddingNetwork, Normalize

@pytest.fixture
def sample_input():
    return torch.randn(1, 5, 512, 256)

def test_custom_resnet():
    original_model = timm.create_model('resnet18', pretrained=False)
    custom_model = CustomResNet(original_model, num_input_channels=5)
    
    # Test forward pass
    x = torch.randn(1, 5, 512, 256)
    output = custom_model(x)
    
    assert output.shape == (1, 1000)
    assert custom_model.conv1.in_channels == 5

def test_custom_efficientnet():
    original_model = timm.create_model('tf_efficientnetv2_b2', pretrained=False)
    custom_model = CustomEfficientNet(original_model, num_input_channels=5)
    
    # Test forward pass
    x = torch.randn(1, 5, 512, 256)
    output = custom_model(x)
    
    assert output.shape == (1, 1000)
    assert custom_model.conv_stem.in_channels == 5

def test_triplet_network_resnet():
    model = EmbeddingNetwork(backbone_model="resnet18", num_dims=256, 
                           input_channels=5)
    
    # Test forward pass
    x = torch.randn(1, 5, 512, 256)
    output = model(x)
    
    assert output.shape == (1, 256)
    assert torch.allclose(torch.norm(output, p=2, dim=1), 
                          torch.tensor([model.s]))

def test_triplet_network_efficientnet():
    model = EmbeddingNetwork(backbone_model="tf_efficientnetv2_b2", num_dims=256,
                           input_channels=5)
    
    # Test forward pass
    x = torch.randn(1, 5, 512, 256)
    output = model(x)
    
    assert output.shape == (1, 256)
    assert torch.allclose(torch.norm(output, p=2, dim=1), torch.tensor([model.s]))

def test_normalize_layer():
    normalize = Normalize(p=2, dim=1)
    x = torch.randn(10, 5)
    output = normalize(x)
    
    assert torch.allclose(torch.norm(output, p=2, dim=1), torch.ones(10))

def test_custom_resnet_weight_initialization(sample_input):
    original_model = timm.create_model('resnet18', pretrained=False)
    custom_model = CustomResNet(original_model, num_input_channels=5)
    
    # Check if the first 3 channels are identical to the original
    assert torch.allclose(custom_model.conv1.weight[:, :3, :, :], 
                          original_model.conv1.weight)
    
    # Check if the additional channels are copies of the first channel
    assert torch.allclose(custom_model.conv1.weight[:, 3, :, :],
                          original_model.conv1.weight[:, 0, :, :])
    assert torch.allclose(custom_model.conv1.weight[:, 4, :, :],
                          original_model.conv1.weight[:, 0, :, :])

def test_custom_efficientnet_weight_initialization(sample_input):
    original_model = timm.create_model('tf_efficientnetv2_b2', pretrained=False)
    custom_model = CustomEfficientNet(original_model, num_input_channels=5)
    
    # Check if the first 3 channels are identical to the original
    assert torch.allclose(custom_model.conv_stem.weight[:, :3, :, :], 
                          original_model.conv_stem.weight)
    
    # Check if the additional channels are copies of the first channel
    assert torch.allclose(custom_model.conv_stem.weight[:, 3, :, :],
                          original_model.conv_stem.weight[:, 0, :, :])
    assert torch.allclose(custom_model.conv_stem.weight[:, 4, :, :],
                          original_model.conv_stem.weight[:, 0, :, :])

def test_triplet_network_output_scaling():
    model = EmbeddingNetwork(backbone_model="resnet18", num_dims=256,
                             input_channels=3, s=10.0)
    x = torch.randn(1, 3, 512, 256)
    output = model(x)
    
    # Check if the norm of the output is equal to the scaling factor
    assert torch.allclose(torch.norm(output), torch.tensor([10.0]),
                          atol=1e-6)
