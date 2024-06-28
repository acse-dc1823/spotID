import torch.nn as nn
import timm


class TripletNetwork(nn.Module):
    def __init__(self, backbone_model='resnet50', num_dims=256):
        super(TripletNetwork, self).__init__()
        # Load the pre-trained model from timm,
        # keeping the entire original model intact
        self.backbone = timm.create_model(backbone_model, pretrained=True,
                                          features_only=False)
        
        # Check if the backbone model ends with a classifier or fc layer and get the number of in_features
        if hasattr(self.backbone, 'classifier'):
            final_in_features = self.backbone.classifier.in_features
        elif hasattr(self.backbone, 'fc'):
            final_in_features = self.backbone.fc.out_features
        else:
            raise NotImplementedError("Backbone model must end with a recognizable classifier or fc layer.")
        
        # Define a new embedding layer that takes the output of the backbone's classifier
        self.embedding_layer = nn.Linear(final_in_features, num_dims)

    def forward(self, x):
        # Forward pass through the backbone model
        features = self.backbone(x)
        
        # Pass the output of the backbone's final layer to the embedding layer
        embeddings = self.embedding_layer(features)
        return embeddings