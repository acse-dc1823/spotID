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

        # Modify the first convolutional layer to have the required number of input channels
        self.conv1 = nn.Conv2d(
            num_input_channels,
            original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=(original_first_layer.bias is not None)
        )

        # Initialize the weights for the modified conv1 layer with Parameter wrapping
        with torch.no_grad():
            # Handle cases where the number of input channels is different
            if num_input_channels > 3:
                # Initialize the extra channel weights (replicating the weights from the first channel as an example)
                extra_channels = num_input_channels - 3
                # Copy the original weights for the first 3 channels
                self.conv1.weight[:, :3, :, :] = original_first_layer.weight.clone()
                # Replicate the first channel weights for the additional channels
                for i in range(extra_channels):
                    self.conv1.weight[:, 3+i, :, :] = original_first_layer.weight[:, 0, :, :].clone()
            else:
                self.conv1.weight[:, :num_input_channels, :, :] = original_first_layer.weight[:, :num_input_channels, :, :].clone()

            # If the original layer has a bias, clone that as well
            if original_first_layer.bias is not None:
                self.conv1.bias = nn.Parameter(original_first_layer.bias.clone())

        # Ensure the weights and biases are wrapped as nn.Parameter
        self.conv1.weight = nn.Parameter(self.conv1.weight)
        if self.conv1.bias is not None:
            self.conv1.bias = nn.Parameter(self.conv1.bias)

        # Replace the first layer in the original model with the modified layer
        original_model.conv1 = self.conv1
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
