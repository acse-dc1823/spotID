import torch.nn as nn
import timm


class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return nn.functional.normalize(x, p=self.p, dim=self.dim)


class TripletNetwork(nn.Module):
    def __init__(self, backbone_model="resnet50", num_dims=256):
        super(TripletNetwork, self).__init__()
        # Load the pre-trained model from timm
        self.backbone = timm.create_model(
            backbone_model, pretrained=True, features_only=False
        )

        # Determine the number of features from the backbone's last layer
        if hasattr(self.backbone, "classifier"):
            final_in_features = self.backbone.classifier.in_features
        elif hasattr(self.backbone, "fc"):
            final_in_features = self.backbone.fc.out_features
        else:
            raise NotImplementedError(
                "Backbone model must end with a",
                "recognizable classifier or fc layer."
            )

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
