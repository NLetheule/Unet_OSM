import torch
import torch.nn as nn

class NetworkSample(nn.Module):
    def __init__(self, DA=False):
        super(NetworkSample, self).__init__()
        

    def forward(self, x):
        #Do something with x:
        #e.g. output = f(x)
        #return output
        return None
    
class Hypercolumns(nn.Module):
    def __init__(self):
        super(Hypercolumns, self).__init__()
        
        #Define the convolution layers
        self.conv1 = nn.Conv2d(3, 64, 7)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 5)
        
        #Pooling layer and activation:
        self.pool = nn.MaxPool2d(3)
        self.activation = nn.Tanh()
        
        #The interpolation. Note that it will upsample any input size to 512*512
        self.interpol = nn.Upsample(size=(256,256), mode='bilinear', align_corners=True)
        
        #And the final MLP layers. Note that we use 1*1 convolution for that.
        self.MLP1 = nn.Conv2d((256+128+64+64+3), 128, 1)
        self.MLP2 = nn.Conv2d(128, 128, 1)
        self.MLP3 = nn.Conv2d(128, 8, 1)

    def forward(self, x):
        #Go through the feature extractors
        x1 = self.pool(self.activation(self.conv1(x)))
#         print(x1.shape)
        x2 = self.pool(self.activation(self.conv2(x1)))
#         print(x2.shape)
        x3 = self.pool(self.activation(self.conv3(x2)))
#         print(x3.shape)
        x4 = self.activation(self.conv4(x3))
#         print(x4.shape)
        
        #Construct hypercolumn
        features = torch.cat((x, self.interpol(x1), self.interpol(x2), self.interpol(x3), self.interpol(x4)),1)
        
        #MLP
        hc1 = self.activation(self.MLP1(features))
        hc2 = self.activation(self.MLP2(hc1))
        hc3 = self.MLP3(hc2)
        
        return hc3