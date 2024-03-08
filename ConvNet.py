import torch.nn as nn
import torch.nn.functional as F

class SignalConvNet2(nn.Module):

    def __init__(self):
        super(SignalConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (3, 3))
        self.conv3 = nn.Conv2d(16, 16, (3, 3))
        self.fc1 = nn.Linear(16 * 157, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        print(x.shape)
        x = self.pool(F.relu(self.conv3(x)))  # -> n, 16, 5, 5
        print(x.shape)
        x = x.view(-1, 16 * 157)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)  # -> n, 7
        return x

class SignalConvNet(nn.Module):
    def __init__(self):
        super(SignalConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.bn = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 3 * 63, 120, bias=True)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(120, 84, bias=True)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.fc3 = nn.Linear(84, 7, bias=True)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        # -> n, 1, 640, 16
        #print(x.shape)
        x = self.pool(F.relu(self.bn(self.conv1(x))))  # -> n, 6, 3, 63
       # print(x.shape)
        x = x.view(-1, 6 * 3 * 63)            # -> n, 1134
      #  print(x.shape)
        x = F.relu(self.fc1(x))               # -> n, 84
     #   print(x.shape)
        x = F.relu(self.fc2(x))               # -> n, 120
        x = self.fc3(x)                      # -> n, 7
     #   print(x.shape)
        return x


class SignalConvNet4(nn.Module):
    def __init__(self):
        super(SignalConvNet4, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3), nn.MaxPool2d(2, 2), nn.ReLU())
        self.fc1 = nn.Linear(6 * 3 * 63, 120)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc3 = nn.Linear(120, 7)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):

        #print(x.shape)
        x = self.conv1(x)  # -> n, 6, 3, 63
       # print(x.shape)
        x = x.view(x.size(0), -1)
      #  print(x.shape)
        x = F.relu(self.fc1(x))
     #   print(x.shape)
        x = self.fc3(x)
     #   print(x.shape)
        return F.log_softmax(x, dim=1)
