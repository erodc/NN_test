import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from resnet import ResNet18
import cv2 as cv

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

mnist_test = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

img, target = mnist_test[0]
img = img.numpy().transpose(1, 2, 0)
print(target)
while True:
    cv.imshow('test', img)
    if cv.waitKey() & 0xFF == 27:
        cv.destroyAllWindows()
        break
