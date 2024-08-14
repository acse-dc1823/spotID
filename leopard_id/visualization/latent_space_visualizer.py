import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader import LeopardDataset, LeopardBatchSampler
from model import EmbeddingNetwork

import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def create_latent_space(encoder, data_loader, device):
    """
    Extract embeddings representing rich features of leopards and
    their corresponding labels from the model.
    """
    encoder.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            embeddings = encoder(images)
            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return embeddings, labels


def plot_embeddings(embeddings, labels,
                    title='2D Visualization of Embeddings'):
    """
    Plot embeddings using t-SNE. We reduce the dimensions of the latent space
    from N to 2 through dimensionality reduction algorithm. Of course, we
    are losing a lot of information, so this should really only be taken as an
    approximation and a visual queue of how the model is performing and
    separating labels.
    """
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels,
                          cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Label')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title(title)

    plt.savefig(os.path.join(project_root, "visualization", "outputs",
                             "embeddings.png"))


def main_executor_visualization(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model from path.
    # TODO: Check if this works.
    if model is None:
        path_model = os.path.join(project_root, "weights", "leopard-id.pth")
        model = EmbeddingNetwork()
        model.load_state_dict(torch.load(path_model))
    model.eval()
    model.to(device)

    # Initialize the dataset and DataLoader
    root_dir = os.path.join(project_root, "data",
                            "crop_output_subset_2")
    val_dataset = LeopardDataset(
        root_dir=root_dir,
        transform=transforms.Compose([
            transforms.Resize((512, 256)),
            transforms.ToTensor()
        ])
    )
    val_sampler = LeopardBatchSampler(val_dataset, batch_size=len(val_dataset))
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    # Generate embeddings
    embeddings, labels = create_latent_space(model, val_loader, device)

    # Convert tensors to NumPy for t-SNE visualization
    embeddings_np = embeddings.numpy()
    labels_np = labels.numpy()

    # Visualize embeddings
    plot_embeddings(embeddings_np, labels_np)
