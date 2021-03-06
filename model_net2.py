import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        ## Define all the layers of this CNN, the only requirements are:
        ## This network takes in a square (224 x 224), RGB image as input
        ## num_classes output channels/feature maps

        # 1 - input image channel (RGB), 32 output channels/feature maps, 4x4 square convolution kernel
        # 2x2 max pooling with 10% droupout
        # ConvOut: (32, 221, 221) <-- (W-F+2p)/s+1 = (224 - 4)/1 + 1
        # PoolOut: (32, 110, 110) <-- W/s
        self.conv1 = nn.Conv2d(3, 32, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p = 0.1)

        # 2 - 64 output channels/feature maps, 3x3 square convolution kernel
        # 2x2 max pooling with 20% droupout
        # ConvOut: (64, 108, 108) <-- (W-F+2p)/s+1 = (110 - 3)/1 + 1
        # PoolOut: (64, 54, 54) <-- W/s
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p = 0.2)

        # 3 - 128 output channels/feature maps, 2x2 square convolution kernel
        # 2x2 max pooling with 30% droupout
        # ConvOut: (128, 53, 53) <-- (W-F+2p)/s+1 = (54 - 2)/1 + 1
        # PoolOut: (128, 26, 26) <-- W/s
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p = 0.3)

        # 4 - 256 output channels/feature maps, 3x3 square convolution kernel
        # 2x2 max pooling with 30% droupout
        # ConvOut: (256, 24, 24) <-- (W-F+2p)/s+1 = (24 - 3)/1 + 1
        # PoolOut: (256, 12, 12) <-- W/s
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p = 0.4)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256*12*12, 1000)
        self.dropout5 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(1000, 1000)
        self.dropout6 = nn.Dropout(p = 0.6)
        self.fc3 = nn.Linear(1000, 250)
        self.dropout7 = nn.Dropout(p = 0.7)
        self.fc4 = nn.Linear(250, num_classes)


    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # convolutions
        x = self.dropout1(self.pool1(torch.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(torch.relu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(torch.relu(self.bn4(self.conv4(x)))))

        #flatten
        x = x.view(x.size(0), -1)

        #fully connected
        x = self.dropout5(self.fc1(x))
        x = self.dropout6(self.fc2(x))
        x = self.dropout7(self.fc3(x))
        x = self.fc4(x)
        output = x
        return output


model_class_name = Net
