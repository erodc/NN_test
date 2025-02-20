import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data


def load_data_fashion_mnist(batch_size, resize=None, root='./data'):
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True,
                                                    download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=True,
                                                   download=True, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


# input 244*244*3
# o = floor((n + 2*p -f) / s) + 1
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(5 * 5 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        result = self.classifier(x)
        return result


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    net = AlexNet(10)
    net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 5

    net.train()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = \
            0.0, 0.0, 0, 0, time.time()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))

    net.eval()
    acc_sum = 0
    n = 0
    with torch.no_grad():
        for x, y in test_iter:
            acc_sum += (net(x.to(device)).argmax(dim=1) ==
                        y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    print('test acc %.3f' % (acc_sum / n))
