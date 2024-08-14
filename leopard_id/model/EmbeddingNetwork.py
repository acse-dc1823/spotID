import torch.nn as nn
import timm
import torch

import logging

from copy import deepcopy

logging.basicConfig(filename='logs.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return nn.functional.normalize(x, p=self.p, dim=self.dim)


class CustomResNet(nn.Module):
    def __init__(self, original_model, num_input_channels):
        super(CustomResNet, self).__init__()
        assert (num_input_channels > 3) or (num_input_channels == 1), "CustomResNet should only be used when there are more than 3 input channels."

        # Create a new first layer with the adjusted number of input channels
        self.conv1 = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=original_model.conv1.out_channels,
            kernel_size=original_model.conv1.kernel_size,
            stride=original_model.conv1.stride,
            padding=original_model.conv1.padding,
            bias=(original_model.conv1.bias is not None)
        )

        # Initialize the new first layer's weights based on the original first layer
        self._initialize_weights(original_model.conv1, num_input_channels)

        # Assign all other components of the original model directly to this modified model
        for name, module in original_model.named_children():
            if name != 'conv1':  # Skip replacing the first conv layer
                setattr(self, name, module)

    def _initialize_weights(self, original_first_layer, num_input_channels):
        with torch.no_grad():
            if num_input_channels == 1:
                # For 1-channel input, use the mean of the original weights across the channel dimension
                self.conv1.weight.data = original_first_layer.weight.data.mean(dim=1, keepdim=True)
            else:
                # Copy weights for the first 3 channels
                self.conv1.weight[:, :3, :, :] = original_first_layer.weight.data.clone()
                # Initialize weights for additional channels by copying the first channel's weights
                for i in range(num_input_channels - 3):
                    self.conv1.weight[:, 3+i, :, :] = original_first_layer.weight.data[:, 0, :, :].clone()
            if original_first_layer.bias is not None:
                self.conv1.bias.data = original_first_layer.bias.data.clone()

    def forward(self, x):
        # Use the modified first layer and then proceed with the original layers
        x = self.conv1(x)
        # Continue with the rest of the original model's forward pass
        for module in list(self.children())[1:]:  # Skip the first layer which is already applied
            x = module(x)
        return x


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
        with torch.no_grad():
            if num_input_channels == 1:
                # For 1-channel input, use the mean of the original weights across the channel dimension
                new_layer.weight.data = original_layer.weight.data.mean(dim=1, keepdim=True)
            else:
                # Copy weights for the first 3 channels from the original layer
                new_layer.weight.data[:, :3, :, :] = original_layer.weight.data[:, :3, :, :].clone()
                # Initialize weights for additional channels by repeating the first channel's weights
                for i in range(num_input_channels - 3):
                    new_layer.weight.data[:, 3+i, :, :] = original_layer.weight.data[:, 0, :, :].clone()
            if original_layer.bias is not None:
                new_layer.bias.data = original_layer.bias.data.clone()

    def forward(self, x):
        # Manually handle the forward pass for each layer
        x = self.conv_stem(x)
        for name, module in self.named_children():
            if name != 'conv_stem':  # Skip conv_stem since it's already applied
                x = module(x)
        return x


class EmbeddingNetwork(nn.Module):
    def __init__(self, backbone_model="tf_efficientnetv2_b2", num_dims=256, input_channels=3, s=64.0):
        super(EmbeddingNetwork, self).__init__()
        self.s = s
        print("num input channels: ", input_channels)
        if input_channels == 3:
            # Load the pre-trained model directly if there are 3 input channels
            self.final_backbone = timm.create_model(backbone_model, pretrained=True, features_only=False)
        else:
            # Use a custom modification if there are not 3 input channels
            original_model = timm.create_model(backbone_model, pretrained=True, features_only=False)
            if backbone_model == "tf_efficientnetv2_b2":
                logging.info("creating custom efficientnet")
                self.final_backbone = CustomEfficientNet(original_model, num_input_channels=input_channels)
                final_in_features = self.final_backbone.classifier.out_features
            elif backbone_model == "resnet18":
                logging.info("creating custom resnet")
                self.final_backbone = CustomResNet(original_model, num_input_channels=input_channels)
                final_in_features = self.final_backbone.fc.out_features
            else:
                print("Backbone model should be either resnet18 or tf_efficientnetv2_b2")
                raise ValueError
        
        if backbone_model == "tf_efficientnetv2_b2":
            final_in_features = self.final_backbone.classifier.out_features
        else:
            final_in_features = self.final_backbone.fc.out_features

        # Define a new embedding layer

        self.embedding_layer = nn.Linear(final_in_features, num_dims)

        # Add normalization layer
        self.normalization = Normalize()

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
