import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary

from model import TripletNetwork
from dataloader import LeopardDataset, LeopardBatchSampler, ResizeTransform
from engine import train_model
from visualization import main_executor_visualization

import os

project_root = os.path.dirname(os.path.abspath(__file__))


def setup_data_loader(verbose):
    root_dir_train = os.path.join(project_root, "data", "minimum_usable_dataset")
    root_dir_test = os.path.join(project_root, "data",
                                 "minimum_usable_dataset_2")
    # check root_dir is valid directory
    if not os.path.isdir(root_dir_train):
        raise NotADirectoryError(f"Directory {root_dir_train} does not exist.")
    else:
        print(f"Found directory {root_dir_train}")

    train_dataset = LeopardDataset(
        root_dir=root_dir_train,
        transform=transforms.Compose(
            [
                ResizeTransform(width=512, height=256),  # Resize images
                transforms.ToTensor(),  # Convert images to tensor
            ]
        ),
    )

    test_dataset = LeopardDataset(
        root_dir=root_dir_test,
        transform=transforms.Compose(
            [
                ResizeTransform(width=512, height=256),  # Resize images
                transforms.ToTensor(),  # Convert images to tensor
            ]
        ),
    )

    # TODO: Change verbose here
    train_sampler = LeopardBatchSampler(
        train_dataset, batch_size=32, verbose=False
    )
    # DataLoader for test should have batch size equal to the number of images
    return DataLoader(train_dataset, batch_sampler=train_sampler), DataLoader(
        test_dataset, batch_size=len(test_dataset)
    )


def main():
    verbose = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, test_loader = setup_data_loader(verbose=verbose)

    model = TripletNetwork(backbone_model="resnet18").to(device)
    print(summary(model, (3, 256, 512)))
    resnet_model = train_model(
        model,
        train_loader,
        test_loader,
        lr=1e-3,
        epochs=40,
        device=device,
        verbose=verbose,
    )
    # save model
    save_path = os.path.join(project_root, "weights", "leopard-id.pth")
    torch.save(resnet_model.state_dict(), save_path)
    main_executor_visualization(resnet_model)


if __name__ == "__main__":
    main()
