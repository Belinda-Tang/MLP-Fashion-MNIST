import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data.loader import get_dataloader
from model.net import FashionNet

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 64
CKPT_PATH = './checkpoints/fashion_mlp.pth'


def evaluate(model, loader):
    """Calculate the model's accuracy on the given DataLoader."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            correct += (model(imgs).argmax(dim=1) == lbls).sum().item()
            total += lbls.size(0)
    return correct / total


def plot_curve(train_losses, test_accuracies):
    """Plot train loss (left y-axis) and test accuracy (right y-axis)."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left y-axis: train loss
    color_loss = '#E76F51'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', color=color_loss, fontsize=12)
    ax1.plot(epochs, train_losses, color=color_loss, linewidth=2, marker='o')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    # Right y-axis: test accuracy
    color_acc = '#2A9D8F'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy', color=color_acc, fontsize=12)
    ax2.plot(epochs, test_accuracies, color=color_acc, linewidth=2, marker='s')
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(0.7, 1.0)

    plt.title('MLP Training Curve - Fashion-MNIST', fontsize=14)
    fig.tight_layout()
    plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training curve saved to: training_curve.png")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data
    train_loader, test_loader = get_dataloader(BATCH_SIZE)

    # Model, loss function, optimizer
    model = FashionNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Record loss and accuracy for plotting
    train_losses = []
    test_accuracies = []

    # Training
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # clear old gradients
            optimizer.zero_grad()
            # forward pass
            pred = model(imgs)
            # compute loss
            loss = criterion(pred, labels)
            # backward pass
            loss.backward()
            # update parameters
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        train_losses.append(average_loss)

        # Evaluate on test set
        acc = evaluate(model, test_loader)
        test_accuracies.append(acc)
        print(f"Epoch [{epoch + 1:>2d}/{EPOCHS}]  "
              f"Train Loss: {average_loss:.4f}  "
              f"Test Accuracy: {acc * 100:.2f}%")

    print(f"Final test accuracy: {acc * 100:.2f}%")

    # Save model weights
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"Model weights saved to: {CKPT_PATH}")

    # Plot training curve
    plot_curve(train_losses, test_accuracies)


if __name__ == "__main__":
    main()