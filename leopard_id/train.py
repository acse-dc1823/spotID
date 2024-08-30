# author: David Colomer Matachana
# GitHub username: acse-dc1823

import json
import torch
import os
import logging
import argparse

from torchsummary import summary

from model import EmbeddingNetwork
from dataloader import setup_data_loader
from engine import train_model
from visualization import main_executor_visualization

logging.basicConfig(
    filename="logs.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

project_root = os.path.dirname(os.path.abspath(__file__))


def main(config_file="config.json"):
    config_path = os.path.join(project_root, config_file)

    with open(config_path, "r") as f:
        config = json.load(f)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    train_loader, test_loader = setup_data_loader(config, project_root)

    num_input_channels = next(iter(train_loader))[0].shape[1]
    logging.info(f"Number of input channels for the model: {num_input_channels}")

    model = EmbeddingNetwork(
        backbone_model=config["backbone_model"],
        num_dims=config["number_embedding_dimensions"],
        input_channels=num_input_channels,
    ).to(device)
    print(model)
    print(summary(model, (num_input_channels, config["resize_height"], config["resize_width"])))

    model = train_model(
        model,
        train_loader,
        test_loader,
        device=device,
        config=config,
        num_input_channels=num_input_channels,
        project_root=project_root,
    )
    main_executor_visualization(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the network training with an optional configuration file."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the configuration JSON file. Default is 'config.json'.",
    )
    args = parser.parse_args()
    main(args.config_file)
