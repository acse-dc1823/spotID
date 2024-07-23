import torch.nn as nn
import timm
import torch


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
        assert num_input_channels > 3, "CustomResNet should only be used when there are more than 3 input channels."

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
        # Extend the original weights if more input channels are used
        extra_channels = num_input_channels - 3
        with torch.no_grad():
            # Copy weights for the first 3 channels
            self.conv1.weight[:, :3, :, :] = original_first_layer.weight.data.clone()
            # Initialize weights for additional channels by copying the first channel's weights
            for i in range(extra_channels):
                self.conv1.weight[:, 3+i, :, :] = original_first_layer.weight.data[:, 0, :, :].clone()
            if original_first_layer.bias is not None:
                self.conv1.bias.data = original_first_layer.bias.data.clone()

    def forward(self, x):
        # Use the modified first layer and then proceed with the original layers
        x = self.conv1(x)
        # Continue with the rest of the original model's forward pass
        # You must skip this part in forward and directly use the remaining forward definition from the original model if defined elsewhere
        for module in list(self.children())[1:]:  # Skip the first layer which is already applied
            x = module(x)
        return x


class TripletNetwork(nn.Module):
    def __init__(self, backbone_model="resnet50", num_dims=256, input_channels=3):
        super(TripletNetwork, self).__init__()
        if input_channels == 3:
            # Load the pre-trained model directly if there are 3 input channels
            self.final_backbone = timm.create_model(backbone_model, pretrained=True, features_only=False)
        else:
            # Use a custom modification if there are not 3 input channels
            original_model = timm.create_model(backbone_model, pretrained=True, features_only=False)
            self.final_backbone = CustomResNet(original_model, num_input_channels=input_channels)

        # Determine the number of features from the backbone's last layer
        if hasattr(self.final_backbone, "classifier"):
            final_in_features = self.final_backbone.classifier.in_features
        elif hasattr(self.final_backbone, "fc"):
            final_in_features = self.final_backbone.fc.out_features
        else:
            raise NotImplementedError("Backbone model must end with a recognizable classifier or fc layer.")

        # Define a new embedding layer
        self.embedding_layer = nn.Linear(final_in_features, num_dims)

        # Add normalization layer
        self.normalization = nn.LayerNorm(num_dims)

    def forward(self, x):
        # Forward pass through the backbone model
        features = self.final_backbone(x)

        # Pass the output of the backbone's final layer to the embedding layer
        embeddings = self.embedding_layer(features)

        # Normalize the embeddings
        embeddings = self.normalization(embeddings)
        return embeddings