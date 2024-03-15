import torch
import torch.nn as nn
import torch.nn.functional as F
import enum

class Finger(enum.Enum):
    R1 = 1
    R15 = 2
    DIGIT = 3
    MINI = 4

class RGB2NormNetR1(nn.Module):
    def __init__(self):
        super(RGB2NormNetR1, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 2)
        self.drop_layer = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc2(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc3(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc4(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc5(x))
        x = F.tanh(self.fc6(x))
        x = self.fc7(x)
        return x


''' nn architecture for r1.5 and mini '''
class RGB2NormNetR15(nn.Module):
    def __init__(self):
        super(RGB2NormNetR15, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,2)
        self.drop_layer = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        x = self.fc4(x)
        return x
