# author: David Colomer Matachana
# GitHub username: acse-dc1823

import torch
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataloader import LeopardDataset, LeopardBatchSampler, create_transforms
from model import EmbeddingNetwork

import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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


def plot_embeddings_3d(embeddings, labels, title="3D Visualization of Embeddings"):
    """
    Plot embeddings using t-SNE in 3D. We reduce the dimensions of the latent space
    from N to 3 through dimensionality reduction algorithm.
    """
    tsne = TSNE(n_components=3, random_state=0)
    embeddings_3d = tsne.fit_transform(embeddings)

    # Create a trace for the 3D scatter plot
    trace = go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode="markers",
        marker=dict(size=5, color=labels, colorscale="Viridis", opacity=0.8),
        text=[f"Label: {label}" for label in labels],
        hoverinfo="text",
    )

    # Create the layout for the plot
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title="t-SNE Feature 1",
            yaxis_title="t-SNE Feature 2",
            zaxis_title="t-SNE Feature 3",
        ),
        margin=dict(r=0, b=0, l=0, t=40),
    )

    # Create the figure and add the trace
    fig = go.Figure(data=[trace], layout=layout)

    # Save the plot as an interactive HTML file
    output_path = os.path.join(project_root, "visualization", "outputs", "embeddings_3d.html")
    fig.write_html(output_path)
    print(f"3D visualization saved to {output_path}")


def main_executor_visualization(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model from path.
    if model is None:
        path_model = os.path.join(project_root, "weights", "leopard-id.pth")
        model = EmbeddingNetwork()
        model.load_state_dict(torch.load(path_model))
    model.eval()
    model.to(device)

    # Initialize the dataset and DataLoader
    root_dir = os.path.join(project_root, "data", "minimum_train_data_cropped")
    mask_dir = os.path.join(project_root, "data", "minimum_train_data_binary")

    val_transforms = create_transforms(
        resize_width=512,
        resize_height=256,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        mean_binary_mask=[0.456],
        std_binary_mask=[0.225],
        apply_dropout=False,
        apply_augmentations=False,
        mask_only=False,
    )

    val_dataset = LeopardDataset(
        root_dir=root_dir,
        transform=val_transforms,
        mask_dir=mask_dir,
        skip_singleton_classes=True,
        mask_only=False,
    )
    val_sampler = LeopardBatchSampler(val_dataset, batch_size=len(val_dataset))
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    # Generate embeddings
    embeddings, labels = create_latent_space(model, val_loader, device)

    # Convert tensors to NumPy for t-SNE visualization
    embeddings_np = embeddings.numpy()
    labels_np = labels.numpy()

    # Visualize embeddings in 3D
    plot_embeddings_3d(embeddings_np, labels_np)


if __name__ == "__main__":
    main_executor_visualization()
