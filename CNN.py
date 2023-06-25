import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, height, width, params):

        super(Net, self).__init__()

        self.conv_layers = params['conv_layers']
        self.num_channels = params['num_channels']
        self.dense_nodes = params['dense_nodes']
        self.dropout = params['dropout']
        # Input 1 channel(b/c black and white) with 32 filters each with kernel_size=3 (3*3)
        self.conv1 = nn.Conv2d(1, 8 * params['num_channels'], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8 * params['num_channels'], 8 * params['num_channels'] * 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8 * params['num_channels'] * 2, 8 * params['num_channels'] * 2 * 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(8 * params['num_channels'] * 2 * 2, 8 * params['num_channels'] * 2 * 2 * 2, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(8 * params['num_channels'] * 2 * 2 * 2, 8 * params['num_channels'] * 2 * 2 * 2 * 2,
                               kernel_size=3, stride=1, padding=1)

        # Normalize outputs
        self.batch_norm1 = nn.BatchNorm2d(8 * params['num_channels'])
        self.batch_norm2 = nn.BatchNorm2d(8 * params['num_channels'] * 2)
        self.batch_norm3 = nn.BatchNorm2d(8 * params['num_channels'] * 2 * 2)
        self.batch_norm4 = nn.BatchNorm2d(8 * params['num_channels'] * 2 * 2 * 2)
        self.batch_norm5 = nn.BatchNorm2d(8 * params['num_channels'] * 2 * 2 * 2 * 2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout2d(p=params['dropout'])


        # Fully connected layer
        self.fc1 = nn.Linear(8 * params['num_channels'] * int(height/2) * int(width/2), 2**(params['dense_nodes']+4))
        if self.conv_layers > 1:
          self.fc1 = nn.Linear(8 * params['num_channels'] * 2 * int(height/4) * int(width/4), 2**(params['dense_nodes']+4))
        if self.conv_layers > 2:
          self.fc1 = nn.Linear(8 * params['num_channels'] * 2 * 2 * int(height/8) * int(width/8), 2**(params['dense_nodes']+4))
        if self.conv_layers > 3:
            self.fc1 = nn.Linear(8 * params['num_channels'] * 2 * 2 * 2 * int(height / 16) * int(width / 16), 2 ** (params['dense_nodes'] + 4))
        if self.conv_layers > 4:
            self.fc1 = nn.Linear(8 * params['num_channels'] * 2 * 2 * 2 * 2 * int(height / 32) * int(width / 32),
                                 2 ** (params['dense_nodes'] + 4))

        self.fc2 = nn.Linear(2**(params['dense_nodes']+4), 2)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout(x)

        if self.conv_layers > 1:
            x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
            x = self.dropout(x)

        if self.conv_layers > 2:
            x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
            x = self.dropout(x)
        if self.conv_layers > 3:
            x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
            x = self.dropout(x)
        if self.conv_layers > 4:
            x = self.pool(F.relu(self.batch_norm5(self.conv5(x))))
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
