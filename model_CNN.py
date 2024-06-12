import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class CNN_Model(nn.Module):

    def __init__(self):

        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(3, 7))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3,3))
        self.fc1 = nn.Linear(1120, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 1)
        self.flat = nn.Flatten()

    def forward(self, x, istraining=False, minibatch=64):

        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 1))
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 1))

        x = F.dropout(self.flat(x),training=istraining)

        # python detector.py train/ output.json --plot_lvl 0 --training
        # Fully connected layers with ReLU activations

        x = F.dropout(F.relu(self.fc1(x)),training=istraining)  # Output: (256)

        x =  F.dropout(F.relu(self.fc2(x)),training=istraining)  # Output: (120)

        # Sigmoid output layer
        x = self.fc3(x)  # Output: (1)

        x = torch.sigmoid(x)  # Output: (1)

        
        return x