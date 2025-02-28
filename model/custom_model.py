import torch.nn as nn
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

architecture_config = [
    #Tuple: (kernel_size, number of filters, strides, padding)
    (7, 64, 2, 3),
    #"M" = Max Pool Layer
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    #List: [(tuple), (tuple), how many times to repeat]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
    #Doesnt include fc layers
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        CNN Block with batch normalization and activation function
        """
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class CustomYolo(nn.Module):
    def __init__(self, in_channels=4, S=7, C=2):
        super(CustomYolo, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.S = S 
        self.C = C
        self.conv_layer = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs()
    
    def create_conv_layers(self, architecture):
        layers = []
        #first input channel    
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0] #frist CNNBlock layer
                conv2 = x[1] #Second CNNBlock layer
                repeats = x[2] #Repetation
                for _ in range(repeats):
                    layers += [CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers += [CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]
                    
        return nn.Sequential(*layers)
    
    def create_fcs(self):

        return nn.Sequential(nn.Flatten(), 
                             nn.Linear(1024 * self.S * self.S, 496), 
                             nn.Dropout(0.2), 
                             nn.LeakyReLU(0.1), 
                             nn.Linear(496, self.S * self.S * (self.C + 5)))
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.fcs(self.conv_layer(x))
        pred_cls = x[...,:self.C]
        pred_obj_box = torch.sigmoid(x[...,self.C:])
        x = torch.cat((pred_cls, pred_obj_box), dim=-1)

        #shape = [batch_size, -1, classes + conf + xc + yc + w + h]
        return x.reshape(batch_size,-1,self.C+5)

if __name__ == "__main__":
    input_size = 448
    num_classes = 5
    inp = torch.randn(2, 4, input_size, input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomYolo().to(device)
    model.train()
    out = model(inp.to(device))
    print(out.shape)
