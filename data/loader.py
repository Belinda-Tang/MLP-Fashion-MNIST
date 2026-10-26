from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
def get_dataloader(batch_size=64):
    """
    return train_loader, test_loader
    :param batch_size:
    :return:
    """
    # convert to standardized tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # download dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader