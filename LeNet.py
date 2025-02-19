import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.subsample1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.subsample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.L1 = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(120, 84)
        self.L3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.subsample1(out)
        out = self.layer2(out)
        out = self.subsample2(out)
        out = out.view(-1, 400)
        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)
        out = self.relu(out)
        out = self.L3(out)
        return out


if __name__ == '__main__':
    train_dataset = datasets.MNIST(root='./data',
                                   train=True,
                                   transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=(0.1307,),
                                                            std=(0.3081,)),
                                   ]),
                                   download=True)

    test_dataset = datasets.MNIST(root='./data',
                                  train=False,
                                  transform=transforms.Compose([
                                      transforms.Resize((32, 32)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.1325,),
                                                           std=(0.3105,)),
                                  ]),
                                  download=True)

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5(10).to(device)
    cost = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = cost(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 400 == 0:
                print('Epoch [{}/{}], step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'
          .format(100 * correct / total))

