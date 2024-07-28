import torch.nn as nn
import timm
import torch

from copy import deepcopy

class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return nn.functional.normalize(x, p=self.p, dim=self.dim)


class CustomEfficientNet(nn.Module):
    def __init__(self, original_model, num_input_channels):
        super(CustomEfficientNet, self).__init__()
        # Deep copy the original model to modify
        original_model = deepcopy(original_model)

        # Modify the first layer to handle different input channels
        out_channels = original_model.conv_stem.out_channels
        kernel_size = original_model.conv_stem.kernel_size
        stride = original_model.conv_stem.stride
        padding = original_model.conv_stem.padding
        bias = original_model.conv_stem.bias is not None

        # Create a new conv_stem with updated in_channels
        self.conv_stem = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        # Initialize the new conv_stem weights
        self._initialize_weights(self.conv_stem, original_model.conv_stem, num_input_channels)

        # Transfer all other modules from the original model to this new model
        for name, module in original_model.named_children():
            if name != 'conv_stem':
                setattr(self, name, module)

    def _initialize_weights(self, new_layer, original_layer, num_input_channels):
        extra_channels = num_input_channels - 3
        with torch.no_grad():
            # Copy weights for the first 3 channels from the original layer
            new_layer.weight.data[:, :3, :, :] = original_layer.weight.data[:, :3, :, :].clone()
            # Initialize weights for additional channels by repeating the first channel's weights
            for i in range(extra_channels):
                new_layer.weight.data[:, 3+i, :, :] = original_layer.weight.data[:, 0, :, :].clone()
            # Copy the bias if it exists
            if original_layer.bias is not None:
                new_layer.bias.data = original_layer.bias.data.clone()

    def forward(self, x):
        # Manually handle the forward pass for each layer
        x = self.conv_stem(x)
        for name, module in self.named_children():
            if name != 'conv_stem':  # Skip conv_stem since it's already applied
                x = module(x)
        return x

class TripletNetwork(nn.Module):
    def __init__(self, backbone_model="efficientnet_b0", num_dims=256, input_channels=3, s=16.0):
        super(TripletNetwork, self).__init__()
        self.s = s
        print("num input channels: ", input_channels)

        if input_channels == 3:
            # Load the pre-trained model directly if there are 3 input channels
            self.final_backbone = timm.create_model(backbone_model, pretrained=True, features_only=False)
        else:
            # Use a custom modification if there are not 3 input channels
            original_model = timm.create_model(backbone_model, pretrained=True, features_only=False)
            self.final_backbone = CustomEfficientNet(original_model, num_input_channels=input_channels)

        # Determine the number of features from the backbone's last layer
        final_in_features = self.final_backbone.classifier.out_features

        # Define a new embedding layer
        self.embedding_layer = nn.Linear(final_in_features, num_dims)

        # Add normalization layer
        self.normalization = nn.BatchNorm1d(num_dims)  # Replaced Normalize() with nn.BatchNorm1d for simplicity

    def forward(self, x):
        # Forward pass through the backbone model
        features = self.final_backbone(x)

        # Pass the output of the backbone's final layer to the embedding layer
        embeddings = self.embedding_layer(features)

        # Normalize the embeddings
        embeddings_normalized = self.normalization(embeddings)

        # Apply scaling
        embeddings_scaled = self.s * embeddings_normalized
        
        return embeddings_scaled