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
    def __init__(self, original_model, num_input_channels=3):
        super(CustomResNet, self).__init__()

        # Retrieve the original first convolutional layer
        original_first_layer = original_model.conv1

        # Create a new convolutional layer to replace the original first layer
        # This layer will use the same parameters except for potentially different input channels
        modified_first_conv = nn.Conv2d(
            num_input_channels,
            original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=(original_first_layer.bias is not None)
        )

        # Initialize the weights and biases for the new convolutional layer
        with torch.no_grad():
            if num_input_channels > 3:
                # If there are more input channels than the original, extend the original weights
                extra_channels = num_input_channels - 3
                # Copy the original weights for the first 3 channels
                modified_first_conv.weight[:, :3, :, :] = original_first_layer.weight.data.clone()
                # Replicate the first channel's weights for the additional channels
                for i in range(extra_channels):
                    modified_first_conv.weight[:, 3+i, :, :] = original_first_layer.weight.data[:, 0, :, :].clone()
            else:
                # If the input channels are 3 or fewer, just use the original weights directly
                modified_first_conv.weight.data[:num_input_channels, :, :, :] = original_first_layer.weight.data[:num_input_channels, :, :, :]

            # Handle the bias if it exists
            if original_first_layer.bias is not None:
                modified_first_conv.bias.data = original_first_layer.bias.data.clone()

        # Assign the modified convolutional layer to the original model
        original_model.conv1 = modified_first_conv

        # Assign the modified model to the backbone attribute
        self.backbone = original_model

    def forward(self, x):
        return self.backbone(x)


class TripletNetwork(nn.Module):
    def __init__(self, backbone_model="resnet50", num_dims=256, input_channels=3):
        super(TripletNetwork, self).__init__()
        # Load the pre-trained model from timm
        original_model = timm.create_model(backbone_model,
                                           pretrained=True,
                                           features_only=False)

        # Integrate the modified first convolution layer into the pre-trained model
        self.backbone = CustomResNet(original_model,
                                     num_input_channels=input_channels)

        # Determine the number of features from the backbone's last layer
        if hasattr(self.backbone.backbone, "classifier"):
            final_in_features = self.backbone.backbone.classifier.out_features
        elif hasattr(self.backbone.backbone, "fc"):
            final_in_features = self.backbone.backbone.fc.out_features
            print(final_in_features)
        else:
            raise NotImplementedError("Backbone model must end with a recognizable classifier or fc layer.")

        # Define a new embedding layer
        self.embedding_layer = nn.Linear(final_in_features, num_dims)

        # Add normalization layer
        self.normalization = Normalize()

    def forward(self, x):
        # Forward pass through the backbone model
        features = self.backbone(x)

        # Pass the output of the backbone's final layer to the embedding layer
        embeddings = self.embedding_layer(features)

        # Normalize the embeddings
        embeddings = self.normalization(embeddings)
        return embeddings
