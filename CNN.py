import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, height, width, conv_layers, num_channels, dense_nodes, dropout):

        super(Net, self).__init__()

        self.conv_layers = conv_layers
        self.num_channels = num_channels
        self.dense_nodes = dense_nodes
        self.dropout = dropout
        # Input 1 channel(b/c black and white) with 32 filters each with kernel_size=3 (3*3)
        self.conv1 = nn.Conv2d(1, 8 * num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8 * num_channels, 8 * num_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8 * num_channels * 2, 8 * num_channels * 2 * 2, kernel_size=3, stride=1, padding=1)

        # Normalize outputs
        self.batch_norm1 = nn.BatchNorm2d(8 * num_channels)
        self.batch_norm2 = nn.BatchNorm2d(8 * num_channels * 2)
        self.batch_norm3 = nn.BatchNorm2d(8 * num_channels * 2 * 2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout2d(p=dropout)


        # Fully connected layer
        self.fc1 = nn.Linear(8 * num_channels * height * width, 2**(dense_nodes+4))
        if self.conv_layers > 1:
          self.fc1 = nn.Linear(8 * num_channels * 2 * height * width, 2**(dense_nodes+4))
        if self.conv_layers > 2:
          self.fc1 = nn.Linear(8 * num_channels * 2 * 2 * height * width, 2**(dense_nodes+4))

        self.fc2 = nn.Linear(2**(dense_nodes+4), 2)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout(x)

        if self.conv_layers > 1:
            x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
            x = self.dropout(x)

        if self.conv_layers > 2:
            x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
