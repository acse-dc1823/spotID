import torch
import torch.optim as optim

# To launch TensorBoard, run tensorboard --logdir runs
from torch.utils.tensorboard import SummaryWriter
import time
from losses import TripletLoss, euclidean_dist
from metrics import (
    compute_dynamic_k_avg_precision,
    compute_class_distance_ratio,
)


def train_epoch(model, data_loader, optimizer, criterion, device):
    """
    Train the model for one epoch and return the average loss.
    """
    model.train()
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # loss is already averaged per comparison
        total_loss += loss.item()

    # Average loss per batch
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate_epoch(model, data_loader, device, max_k=5, verbose=False):
    """
    Evaluate the model and return the average precision and class distance
    ratio.
    """
    model.eval()
    total_precision = 0
    total_class_distance_ratio = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # Distance matrix computation
            start_time = time.time()
            dist_mat = euclidean_dist(outputs, outputs)
            time_taken_dist = time.time() - start_time

            # Precision calculation
            start_time_precision = time.time()
            batch_precision = compute_dynamic_k_avg_precision(
                dist_mat, targets, max_k, device
            )
            time_taken_precision = time.time() - start_time_precision

            # Class distance ratio calculation
            start_time_ratio = time.time()
            class_distance_ratio = compute_class_distance_ratio(
                dist_mat, targets, device
            )
            time_taken_ratio = time.time() - start_time_ratio

            total_precision += batch_precision
            total_class_distance_ratio += class_distance_ratio

            if verbose:
                print(
                    f"Time taken to calculate class distance:"
                    f"{time_taken_dist:.2f} s"
                )
                print(
                    f"Time taken to calculate precision: "
                    f"{time_taken_precision:.2f} s"
                )
                print(
                    f"Time taken to calculate class distance ratio"
                    f"ratio: {time_taken_ratio:.2f} s"
                )

    average_precision = total_precision / len(data_loader)
    average_class_distance_ratio = total_class_distance_ratio / len(
        data_loader
    )

    return average_precision, average_class_distance_ratio


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    criterion,
    config
):
    """
    Train and evaluate the model, focusing only on the last added
    embedding layer.
    """
    # Freeze all pretrained layers
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Only parameters of the embedding layer are trainable
    optimizer = optim.Adam(model.embedding_layer.parameters(),
                           lr=config["learning_rate"])

    criterion = TripletLoss(verbose=config["verbose"])
    writer = SummaryWriter()  # TensorBoard summary writer initialized here

    model.to(device)

    hparams = {
        "lr": config["learning_rate"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "device": str(device),
        "margin": criterion.margin if hasattr(criterion, "margin") else -1,
        "backbone_model": config["backbone_model"],
        "max_k": config["max_k"],
    }

    print("Training the model... with hyperparameters:")
    print(hparams)
    for epoch in range(config["epochs"]):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_precision, train_class_distance_ratio = evaluate_epoch(
            model, train_loader, device, max_k=config["max_k"]
        )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Precision/Train", train_precision, epoch)
        writer.add_scalar(
            "Class Distance Ratio/Train", train_class_distance_ratio, epoch
        )

        if test_loader is not None:
            test_precision, test_class_distance_ratio = evaluate_epoch(
                model, test_loader, device, verbose=True, max_k=config["max_k"]
            )
            writer.add_scalar("Precision/Test", test_precision, epoch)
            writer.add_scalar(
                "Class Distance Ratio/Test", test_class_distance_ratio, epoch
            )

        elapsed_time = time.time() - start_time
        epochs = config["epochs"]
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Train Precision: {train_precision:.4f} - "
            f"Train Class Distance Ratio: {train_class_distance_ratio:.4f} - "
            f"Time: {elapsed_time:.2f} s"
        )
        if test_loader is not None:
            print(
                f"Test Precision: {test_precision:.4f} - "
                f"Test Class Distance Ratio: {test_class_distance_ratio:.4f}"
            )

    final_metrics = {
        "final_train_precision": train_precision,
        "final_test_precision": (
            test_precision if test_loader is not None else None
        ),
    }
    writer.add_hparams(hparam_dict=hparams, metric_dict=final_metrics)

    writer.close()
    return model
