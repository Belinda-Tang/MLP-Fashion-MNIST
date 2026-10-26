import torch

from data.loader import get_dataloader
from model.net import FashionNet

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

CKPT_PATH = './checkpoints/fashion_mlp.pth'   # must match the path in train.py
NUM_SAMPLES = 5


def main():
    # Choose device: Apple GPU (mps) if available, otherwise CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load the trained model
    model = FashionNet().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print(f"Loaded weights from: {CKPT_PATH}\n")

    #first 5 images in the test set
    _, test_loader = get_dataloader(64)
    images, labels = next(iter(test_loader))
    images, labels = images[:NUM_SAMPLES].to(device), labels[:NUM_SAMPLES]

    # Predict
    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()

    # Print results
    print(f"{'No.':<5} {'True Label':<15} {'Predicted':<15} {'Correct'}")
    print("-" * 45)
    for i in range(NUM_SAMPLES):
        true_name = CLASS_NAMES[labels[i].item()]
        pred_name = CLASS_NAMES[preds[i].item()]
        correct = "✓" if labels[i] == preds[i] else "✗"
        print(f"{i + 1:<5} {true_name:<15} {pred_name:<15} {correct}")

    num_correct = (labels == preds).sum().item()
    print(f"\nCorrect: {num_correct}/{NUM_SAMPLES}")


if __name__ == "__main__":
    main()