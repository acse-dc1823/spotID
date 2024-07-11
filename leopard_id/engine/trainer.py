import torch
import torch.optim as optim

# To launch TensorBoard, run tensorboard --logdir runs
from torch.utils.tensorboard import SummaryWriter
import time
from losses import TripletLoss, euclidean_dist
from metrics import compute_dynamic_k_avg_precision


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


def evaluate_epoch(model, data_loader, device, max_k=5):
    """
    Evaluate the model and return the average precision and average loss.
    """
    model.eval()
    total_precision = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            dist_mat = euclidean_dist(outputs, outputs)
            batch_precision = compute_dynamic_k_avg_precision(dist_mat,
                                                              targets, max_k)

            total_precision += batch_precision

    average_precision = total_precision / len(data_loader)

    return average_precision


def train_model(model, train_loader, test_loader, lr, epochs, device, verbose):
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

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_precision = evaluate_epoch(model, train_loader, device)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Precision/Train", train_precision, epoch)

        if test_loader is not None:
            test_precision = evaluate_epoch(model, test_loader, device)
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

    writer.close()
    return model
