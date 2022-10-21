import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
        pass
    def forward(self, x):
        return x.view(x.size(0), -1)

class BranchConv(nn.Module):
    def __init__(self, input, n_class, avgpool=True, n_fc=1, width=256):
        super(BranchConv, self).__init__()
        in_channel = input.size(1)
        net = []
        if avgpool:
            net += [nn.Conv2d(in_channel, width, kernel_size=3, padding=1),
                    nn.BatchNorm2d(width), 
                    nn.ReLU(inplace=True)]
            net += [nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.BatchNorm2d(width), 
                    nn.ReLU(inplace=True)]
            net += [nn.AvgPool2d(kernel_size=input.size(2))]
        net += [View()]
        
        if avgpool:
            in_dim = width # input.size(1)
        else:
            in_dim = int(input.numel() / input.size(0))
        
        if n_fc == 1:
            net += [nn.Linear(in_dim, n_class)]
        elif n_fc > 1:
            net += [nn.Linear(in_dim, width),
                    nn.ReLU()]
            for _ in range(n_fc - 2):
                net += [nn.Linear(width, width),
                        nn.ReLU()]
            net += [nn.Linear(width, n_class)]
        else:
            raise NotImplementedError
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        return self.net(x)