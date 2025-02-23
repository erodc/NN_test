# VGGNet16-D
# 代码结构可参考，运行有报错，以后有时间再查问题

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
# from torch.cuda.amp import GradScaler, autocast

import matplotlib.pyplot as plt


class VGG(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super(VGG, self).__init__()
        self.features = self._make_layers(input_channels)
        self.classifier = self._make_classifier(num_classes)

    def _make_layers(self, input_channels):
        layers = []
        layers += self._conv_block(input_channels, 64)
        layers += self._conv_block(64, 128)
        layers += self._conv_block(128, 256)
        layers += self._conv_block(256, 512)

        return nn.Sequential(*layers)

    def _conv_block(self, in_channels, out_channels):
        block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        return block

    def _make_classifier(self, num_classes):
        return nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def get_data_loader(batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST('./data', train=True,
                                   download=True, transform=transform)
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers)


def initialize_model(device, num_classes=10):
    model = VGG(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


def train_epoch(model, train_loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with (tqdm(train_loader, desc='Training', unit='batch', ncols=100) as pbar):
        for data, target in pbar:
            data, target = data.to(device, non_blocking=True), \
                target.to(device, non_blocking=True)
            optimizer.zero_grad()

            # from torch.cuda.amp import GradScaler, autocast
            # with autocast():
            #     output = model(data)
            #     loss = criterion(output, target)
            #
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predict = torch.max(output, 1)
            total += target.size(0)
            correct += (predict == target).sum().item()

            pbar.set_postfix(loss=running_loss / (total // len(data)),
                             accuracy=100 * correct // total)

            return running_loss / len(train_loader), 100 * correct / total


def train_model(num_epochs=5, batch_size=64, lr=0.001):
    train_loader = get_data_loader(batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer, criterion = initialize_model(device)
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        epoch_loss, epoch_accuracy = train_epoch(model, train_loader, device,
                                                 criterion, optimizer)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
              f'Accuracy: {epoch_accuracy:.2f}%')
        save_model(model)


def save_model(model, filepath='vggnet_minst.pth'):
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')


def load_model(model_path='vggnet_minst.pth', num_classes=10):
    model = VGG(num_classes)
    model.load_state_dict(torch.load(model_path))
    return model


def get_test_loader(batch_size=64, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    images, labels, preds = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if len(images) < 6:
                batch_size = data.size(0)
                for i in range(min(6 - len(images), batch_size)):
                    images.append(data[i].cpu())
                    labels.append(target[i].cpu())
                    preds.append(predicted[i].cpu())

    accuracy = 100 * correct / total
    return accuracy, images, labels, preds


def display_images(images, labels, preds):
    fig, axes = plt.subplot(2, 3, figsize=(10, 6))
    axes = axes.ravel()

    for i in range(6):
        axes[i].imshow(images[i][0].squeeze(), cmap='gray')
        axes[i].set_title(f'True: {labels[i].item()}, pred: {preds[i].item()}')
        axes[i].axis('off')

    plt.show()


if __name__ == '__main__':
    train_model()
