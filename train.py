import torch
import torch.nn as nn


def train(model, train_loader, val_loader, epochs=70, lr=1e-4, device="cuda"):
    """
    Train function for BiLSTM / HuBERT emotion recognition models.

    Args:
        model: PyTorch model
        train_loader: training DataLoader
        val_loader: validation DataLoader
        epochs: number of epochs
        lr: learning rate
        device: cuda or cpu

    Returns:
        trained model,
        train_losses,
        val_losses,
        val_accuracies
    """

    model = model.to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for data, label in train_loader:

            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Loss
            loss = loss_fn(output, label)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        train_losses.append(avg_train_loss)

        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for data, label in val_loader:

                data = data.to(device)
                label = label.to(device)

                # Forward pass
                output = model(data)

                # Validation loss
                loss = loss_fn(output, label)

                val_loss += loss.item()

                # Accuracy
                _, predicted = torch.max(output, 1)

                correct += (predicted == label).sum().item()

                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)

        accuracy = correct / total

        val_losses.append(avg_val_loss)

        val_accuracies.append(accuracy)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Accuracy: {accuracy:.4f}"
        )

    return model, train_losses, val_losses, val_accuracies