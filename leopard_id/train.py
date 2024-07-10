import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import TripletNetwork
from dataloader import LeopardDataset, LeopardBatchSampler, ResizeTransform
from engine import train_model
from visualization import main_executor_visualization

import os
project_root = os.path.dirname(os.path.abspath(__file__))


def setup_data_loader():
    root_dir = os.path.join(project_root, "data",
                            "crop_output_subset")
    # check root_dir is valid directory
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"Directory {root_dir} does not exist.")
    else:
        print(f"Found directory {root_dir}")

    test_dataset = LeopardDataset(
        root_dir=root_dir,
        transform=transforms.Compose(
            [
                ResizeTransform(width=512, height=256),  # Resize images
                transforms.ToTensor(),  # Convert images to tensor
            ]
        ),
    )

    test_sampler = LeopardBatchSampler(test_dataset, batch_size=32)
    return DataLoader(test_dataset, batch_sampler=test_sampler)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = setup_data_loader()

    model = TripletNetwork(backbone_model="resnet18").to(device)
    resnet_model = train_model(
        model, test_loader, None, lr=1e-3, epochs=2, device=device
    )
    # save model
    save_path = os.path.join(project_root, "weights", "leopard-id.pth")
    torch.save(resnet_model.state_dict(), save_path)
    main_executor_visualization(resnet_model)


if __name__ == "__main__":
    main()
