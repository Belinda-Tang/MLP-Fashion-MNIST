import torch
import torchvision

print("PyTorch 版本:", torch.__version__)
print("torchvision 版本:", torchvision.__version__)
print("MPS (Apple GPU) 可用:", torch.backends.mps.is_available())

x = torch.rand(2, 3)
print(x)