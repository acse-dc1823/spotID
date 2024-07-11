import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import TripletNetwork
from dataloader import LeopardDataset, LeopardBatchSampler, ResizeTransform
from engine import train_model
from visualization import main_executor_visualization

import os
project_root = os.path.dirname(os.path.abspath(__file__))


def setup_data_loader(verbose):
    root_dir = os.path.join(project_root, "data",
                            "crop_output")
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

    # TODO: Change verbose here
    test_sampler = LeopardBatchSampler(test_dataset, batch_size=32,
                                       verbose=False)
    return DataLoader(test_dataset, batch_sampler=test_sampler)


def main():
    verbose = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    test_loader = setup_data_loader(verbose=verbose)

    model = TripletNetwork(backbone_model="resnet50").to(device)
    resnet_model = train_model(
        model, test_loader, None, lr=1e-3, epochs=30, device=device,
        verbose=verbose
    )
    # save model
    save_path = os.path.join(project_root, "weights", "leopard-id.pth")
    torch.save(resnet_model.state_dict(), save_path)
    main_executor_visualization(resnet_model)


if __name__ == "__main__":
    main()
