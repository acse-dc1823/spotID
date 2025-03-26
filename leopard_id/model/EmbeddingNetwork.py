# author: David Colomer Matachana
# GitHub username: acse-dc1823

import torch.nn as nn
import timm
import torch
import os

import logging

from copy import deepcopy

logging.basicConfig(
    filename="logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return nn.functional.normalize(x, p=self.p, dim=self.dim)


class CustomResNet(nn.Module):
    """
    A custom ResNet model that supports a variable number of input channels.

    This class modifies a pre-trained ResNet model to work with input tensors
    that have more than 3 channels or exactly 1 channel.

    Attributes:
        conv1 (nn.Conv2d): The modified first convolutional layer.
    """
    def __init__(self, original_model, num_input_channels):
        """
        Initialize the CustomResNet module.

        Args:
            original_model (nn.Module): The original pre-trained ResNet model.
            num_input_channels (int): The number of input channels for the new model.

        Raises:
            AssertionError: If num_input_channels is not > 3 or == 1.
        """
        super(CustomResNet, self).__init__()
        assert (num_input_channels > 3) or (
            num_input_channels == 1
        ), "CustomResNet should only be used when there are more than 3 input channels."

        # Create a new first layer with the adjusted number of input channels
        self.conv1 = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=original_model.conv1.out_channels,
            kernel_size=original_model.conv1.kernel_size,
            stride=original_model.conv1.stride,
            padding=original_model.conv1.padding,
            bias=(original_model.conv1.bias is not None),
        )

        # Initialize the new first layer's weights based on the original first layer
        self._initialize_weights(original_model.conv1, num_input_channels)

        # Assign all other components of the original model directly to this modified model
        for name, module in original_model.named_children():
            if name != "conv1":  # Skip replacing the first conv layer
                setattr(self, name, module)

    def _initialize_weights(self, original_first_layer, num_input_channels):
        """
        Initialize the weights of the modified first convolutional layer.

        This method copies weights from the original layer and initializes
        additional channels when necessary.

        Args:
            original_first_layer (nn.Conv2d): The first convolutional layer of the original model.
            num_input_channels (int): The number of input channels for the new model.
        """
        with torch.no_grad():
            if num_input_channels == 1:
                # For 1-channel input, use the mean of the original weights across the channel dimension
                self.conv1.weight.data = original_first_layer.weight.data.mean(dim=1, keepdim=True)
            else:
                # Copy weights for the first 3 channels
                self.conv1.weight[:, :3, :, :] = original_first_layer.weight.data.clone()
                # Initialize weights for additional channels by copying the first channel's weights
                for i in range(num_input_channels - 3):
                    self.conv1.weight[:, 3 + i, :, :] = original_first_layer.weight.data[
                        :, 0, :, :
                    ].clone()
            if original_first_layer.bias is not None:
                self.conv1.bias.data = original_first_layer.bias.data.clone()

    def forward(self, x):
        """
        Defines the forward pass of the CustomResNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass through the network.
        """
        # Use the modified first layer and then proceed with the original layers
        x = self.conv1(x)
        # Continue with the rest of the original model's forward pass
        for module in list(self.children())[1:]:  # Skip the first layer which is already applied
            x = module(x)
        return x


class CustomEfficientNet(nn.Module):
    """
    A custom EfficientNet model that supports a variable number of input channels.

    This class modifies a pre-trained EfficientNet model to work with input tensors
    that have a different number of channels than the original model.

    Attributes:
        conv_stem (nn.Conv2d): The modified first convolutional layer (stem).
    """
    def __init__(self, original_model, num_input_channels):
        """
        Initialize the CustomEfficientNet module.

        Args:
            original_model (nn.Module): The original pre-trained EfficientNet model.
            num_input_channels (int): The number of input channels for the new model.
        """
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
            bias=bias,
        )

        # Initialize the new conv_stem weights
        self._initialize_weights(self.conv_stem, original_model.conv_stem, num_input_channels)

        # Transfer all other modules from the original model to this new model
        for name, module in original_model.named_children():
            if name != "conv_stem":
                setattr(self, name, module)

    def _initialize_weights(self, new_layer, original_layer, num_input_channels):
        """
        Initialize the weights of the modified first convolutional layer (stem).

        This method copies weights from the original layer and initializes
        additional channels when necessary.

        Args:
            new_layer (nn.Conv2d): The new convolutional stem layer.
            original_layer (nn.Conv2d): The original convolutional stem layer.
            num_input_channels (int): The number of input channels for the new model.
        """
        with torch.no_grad():
            if num_input_channels == 1:
                # For 1-channel input, use the mean of the original weights across the channel dimension
                new_layer.weight.data = original_layer.weight.data.mean(dim=1, keepdim=True)
            else:
                # Copy weights for the first 3 channels from the original layer
                new_layer.weight.data[:, :3, :, :] = original_layer.weight.data[:, :3, :, :].clone()
                # Initialize weights for additional channels by repeating the first channel's weights
                for i in range(num_input_channels - 3):
                    new_layer.weight.data[:, 3 + i, :, :] = original_layer.weight.data[
                        :, 0, :, :
                    ].clone()
            if original_layer.bias is not None:
                new_layer.bias.data = original_layer.bias.data.clone()

    def forward(self, x):
        """
        Defines the forward pass of the CustomEfficientNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass through the network.
        """
        # Manually handle the forward pass for each layer
        x = self.conv_stem(x)
        for name, module in self.named_children():
            if name != "conv_stem":  # Skip conv_stem since it's already applied
                x = module(x)
        return x


class EmbeddingNetwork(nn.Module):
    """
    An embedding network that uses a backbone model to generate embeddings.

    This network can use different backbone models (EfficientNet v2 b2 and v2 b3,
    ResNet18) and supports a variable number of input channels. It produces normalized
    and scaled embeddings.

    Attributes:
        s (float): Scaling factor for the embeddings.
        final_backbone (nn.Module): The backbone network used for feature extraction.
        embedding_layer (nn.Linear): Linear layer that produces the final embeddings.
        normalization (Normalize): Normalization layer for the embeddings.
    """
    def __init__(
        self, backbone_model="tf_efficientnetv2_b2", num_dims=256, input_channels=3, s=64.0
    ):
        """
        Initialize the EmbeddingNetwork.

        Args:
            backbone_model (str, optional): Name of the backbone model to use. Defaults to
            "tf_efficientnetv2_b2".
            num_dims (int, optional): Dimensionality of the output embeddings. Defaults to 256.
            input_channels (int, optional): Number of input channels. Defaults to 3.
            s (float, optional): Scaling factor for the embeddings. Defaults to 64.0.

        Raises:
            ValueError: If an unsupported backbone model is specified.
        """
        super(EmbeddingNetwork, self).__init__()
        self.s = s
        print("num input channels: ", input_channels)
        if input_channels == 3:
            # Load the pre-trained model with local weights if available
            weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", f"{backbone_model}_pretrained.pth")
            pretrained = False  # Don't download from HuggingFace
            
            self.final_backbone = timm.create_model(
                backbone_model, pretrained=pretrained, features_only=False
            )
            if os.path.exists(weights_path):
                self.final_backbone.load_state_dict(torch.load(weights_path, map_location='cpu'))
                logging.info(f"Loaded local pretrained weights from {weights_path}")
            else:
                logging.warning(f"Local pretrained weights not found at {weights_path}, using random initialization")
        else:
            # Use a custom modification if there are not 3 input channels
            weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", f"{backbone_model}_pretrained.pth")
            pretrained = False  # Don't download from HuggingFace
            
            original_model = timm.create_model(backbone_model, pretrained=pretrained, features_only=False)
            if os.path.exists(weights_path):
                original_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                logging.info(f"Loaded local pretrained weights from {weights_path}")
            else:
                logging.warning(f"Local pretrained weights not found at {weights_path}, using random initialization")
            if backbone_model == "tf_efficientnetv2_b2" or backbone_model == "tf_efficientnetv2_b3":
                logging.info("creating custom efficientnet")
                self.final_backbone = CustomEfficientNet(
                    original_model, num_input_channels=input_channels
                )
                final_in_features = self.final_backbone.classifier.out_features
            elif backbone_model == "resnet18":
                logging.info("creating custom resnet")
                self.final_backbone = CustomResNet(
                    original_model, num_input_channels=input_channels
                )
                final_in_features = self.final_backbone.fc.out_features
            else:
                print("Backbone model should be either resnet18 or tf_efficientnetv2_b2")
                raise ValueError

        if backbone_model == "tf_efficientnetv2_b2" or backbone_model == "tf_efficientnetv2_b3":
            final_in_features = self.final_backbone.classifier.out_features
        else:
            final_in_features = self.final_backbone.fc.out_features

        # Define a new embedding layer

        self.embedding_layer = nn.Linear(final_in_features, num_dims)

        # Add normalization layer
        self.normalization = Normalize()

    def forward(self, x):
        """
        Defines the forward pass of the EmbeddingNetwork.

        This method passes the input through the backbone network, embedding layer,
        normalization, and applies scaling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Scaled, normalized embedding tensor.
        """
        # Forward pass through the backbone model
        features = self.final_backbone(x)

        # Pass the output of the backbone's final layer to the embedding layer
        embeddings = self.embedding_layer(features)

        # Normalize the embeddings
        embeddings_normalized = self.normalization(embeddings)
        # Apply scaling
        embeddings_scaled = self.s * embeddings_normalized

        return embeddings_scaled
