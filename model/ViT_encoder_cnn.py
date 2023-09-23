import torchvision as tv
import torch.nn as nn
import torch

class Encoder(nn.Module):
    
    '''
    Convolutional encoder network
    '''
    
    def __init__(self, num_channels = 1, pretrained = True):
        super().__init__()
        
        if pretrained:
            weights = 'ResNet18_Weights.DEFAULT'
        else:
            weights = None
            
        architecture = tv.models.resnet18(weights = weights)
        architecture = list(architecture.children())
        
        if num_channels == 1:
            w = architecture[0].weight
            architecture[0] = nn.Conv2d(3, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)
            architecture[0].weight = nn.Parameter(torch.mean(w, dim = 1, keepdim = True))
            self.model = nn.Sequential(*architecture[:-2])
        else:
            self.model = nn.Sequential(*architecture[:-2])
            
    def forward(self, x):
        for param in self.model.parameters():
            param.requires_grad = True
        features = self.model(x)
        features = features.view((*features.shape[0:2], -1))
        return features
