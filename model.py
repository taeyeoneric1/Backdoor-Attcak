from torch.nn import Module
from torch import nn
import torch

class mnist_Model(Module):
    def __init__(self):
        super(mnist_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

class cifar_Model(Module):
    def __init__(self):
        super(cifar_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


class Attack_classifier(nn.Module):
    def __init__(self, in_features, out_features=2):
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=out_features)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x

class Conv(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._encoder_cifar = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self._classifier_cifar = torch.nn.Sequential(
            torch.nn.Linear(800, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 10),
            torch.nn.Softmax(dim=1)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight, gain=1.414)

    def forward(self, x: torch.Tensor):
        x = self._encoder_cifar(x)
        x = x.view(-1, 800)
        x = self._classifier_cifar(x)

        return x