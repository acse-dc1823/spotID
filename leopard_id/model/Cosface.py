import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def cosine_dist(x, y):
    """
    Compute the cosine distance matrix between two sets of vectors.
    """
    # Normalize x and y along the feature dimension (dim=1)
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)

    # Compute cosine similarity
    cosine_sim = torch.mm(x_norm, y_norm.t())

    # Convert similarity to distance
    cosine_dist = 1 - cosine_sim
    cosine_dist = torch.clamp(cosine_dist, min=0)

    return cosine_dist


class CosFace(nn.Module):
    """
    Linear layer with normalized weights. Sets up logits to compute the 
    CosFace loss as in the paper below.
    https://arxiv.org/pdf/1801.09414
    """
    def __init__(self, in_features, out_features, scale=32.0, margin=0.3, m2=0.3):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.m2 = m2
        # Initialize the weights for the fc layer from embeddings to num classes
        # No biases as per paper. requires_grad=True
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input, labels, epoch=None):
        """
        For each exemplar, we need to calculate the cosine similarity between
        the feature vector (the embedding) and the weight vector associated
        to its correct class (of size embedding). In vector form, calculate
        W_i^T * x_i, the result of the neural network with no bias at the
        correct class. Since we are speeding up the training process, we will
        calculate it for an entire batch of exemplars, hence we use the linear
        transformation for the multiplication of weights with the batch input.

        We then apply the margin, but only to the correct class. With the
        modified logits, we can call CrossEntropyLoss safely to calculate the
        loss.
        """
        # Normalize the weights for each row, they have to have norm 1 for the formula
        with torch.no_grad():
            weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity using normalized feature vectors. W^T * x
        cosine = F.linear(F.normalize(input, p=2, dim=1), weight_norm)

        # Apply the margin to the correct class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        # Calculate h(theta_y_i) and apply it to the correct class
        cosine_correct = cosine * one_hot
        h_theta_yi = self.margin * (1 - cosine_correct.pow(2))
        
        # Calculate g(theta_j) and apply it to the incorrect classes
        g_theta_j = self.m2 * cosine.pow(2) * (1 - one_hot)
        
        # Adjust the cosine similarity
        modified_cosine = cosine - h_theta_yi + g_theta_j
        
        # Scale the logits
        logits = modified_cosine * self.scale

        return logits
