import torch
import torch.optim as optim

# To launch TensorBoard, run tensorboard --logdir runs
from torch.utils.tensorboard import SummaryWriter
import time
from losses import TripletLoss, euclidean_dist
from metrics import compute_dynamic_k_avg_precision

import time

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
    Evaluate the model and return the average precision.
    """
    model.eval()
    total_precision = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            start_time = time.time()
            dist_mat = euclidean_dist(outputs, outputs)
            time_taken_dist = time.time() - start_time

            if verbose:
                print("Time taken to compute distance matrix {:.2f} s".format(time_taken_dist))

            start_time_precision = time.time()
            batch_precision = compute_dynamic_k_avg_precision(dist_mat, targets, max_k, device)
            time_taken_precision = time.time() - start_time_precision 

            if verbose:
                print("Time taken to calculate precision {:.2f} s".format(time_taken_precision))

            total_precision += batch_precision

    average_precision = total_precision / len(data_loader)

    return average_precision


def train_model(model, train_loader, test_loader, lr, epochs, device, verbose, criterion, backbone_model, max_k):
    """
    Train and evaluate the model, focusing only on the last added
    embedding layer.
    """
    # Freeze all pretrained layers
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Only parameters of the embedding layer are trainable
    optimizer = optim.Adam(model.embedding_layer.parameters(), lr=lr)

    criterion = TripletLoss(verbose=verbose)
    writer = SummaryWriter()  # TensorBoard summary writer initialized here

    model.to(device)
    
    hparams = {
        'lr': lr,
        'epochs': epochs,
        'batch_size': train_loader.batch_size if hasattr(train_loader, 'batch_size') else None,
        'device': device,
        'margin': criterion.margin if hasattr(criterion, 'margin') else None,
        'backbone_model': backbone_model,
        'max_k': max_k
    }

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_precision = evaluate_epoch(model, train_loader, device, max_k=max_k)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Precision/Train", train_precision, epoch)

        if test_loader is not None:
            test_precision = evaluate_epoch(model, test_loader, device, verbose=True, max_k=max_k)
            # Log the training and validation loss for TensorBoard
            writer.add_scalar("Precision/Test", test_precision, epoch)

        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Train Precision: {train_precision:.4f} - "
            f"Time: {elapsed_time:.2f} s"
        )
        if test_loader is not None:
            print(f"Test Precision: {test_precision:.4f}")

    final_metrics = {
        'final_train_precision': train_precision,
        'final_test_precision': test_precision if test_loader is not None else None}
    writer.add_hparams(hparam_dict=hparams, metric_dict=final_metrics)

    writer.close()
    return model
