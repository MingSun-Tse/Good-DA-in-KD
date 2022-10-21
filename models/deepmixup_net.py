import torch
import torch.nn as nn


class AvgPool2d(nn.Module):
    def __init__(self):
        super(AvgPool2d, self).__init__()
    def forward(self, x):
        x = nn.functional.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))
        x = x.view(x.shape[0], -1)
        return x

class RandNet(nn.Module):
    def __init__(self, chl, upscale):
        super(RandNet, self).__init__()
        self.chl = chl
        self.upscale = upscale
    def forward(self, x):
        N, C, H, W = x.shape # # [N, 2C, H, W]
        h, w = H // self.upscale, W // self.upscale
        return torch.rand([N, self.chl, h, w]).cuda()
        
class DeepMixupNet(nn.Module):
    def __init__(self, mode, depth, width, depth_3x3=2, upscale=1, layerwise_width=''):
        super(DeepMixupNet, self).__init__()
        Ws = [int(w) for w in layerwise_width.split(',')] if layerwise_width else [width] * (depth - 1)

        net, in_channels = [], 6 # 2 RGB images, so 6 channels
        # body
        for d in range(depth - 1):
            ks = 3 if d < depth_3x3 else 1
            net += [nn.Conv2d(in_channels, Ws[d], kernel_size=ks, padding=ks//2),
                    nn.BatchNorm2d(Ws[d]), 
                    nn.LeakyReLU(inplace=True)]
            in_channels = Ws[d]
        
        # tail (last layer)
        d = depth - 1
        ks = 3 if d < depth_3x3 else 1
        if mode in ['channels', 'spatial']:
            out_channels = 3 if mode == 'channels' else 1
            net += [nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=ks//2),
                    nn.BatchNorm2d(out_channels),
                    nn.Sigmoid()]
        
        elif mode in ['single']:
            net += [AvgPool2d(),
                    nn.Linear(in_channels, 1),
                    nn.Sigmoid()]
        
        elif mode in ['upsample_channel3', 'upsample_channel1']:
            chl = int(mode[-1])
            net += [nn.AvgPool2d(kernel_size=upscale, stride=upscale),
                    nn.Conv2d(in_channels, chl, kernel_size=ks, padding=ks//2),
                    nn.Sigmoid(),
                    nn.Upsample(scale_factor=upscale)]
        
        elif mode in ['rand_upsample_channel3', 'rand_upsample_channel1']:
            chl = int(mode[-1])
            net = [RandNet(chl, upscale),
                    nn.Upsample(scale_factor=upscale)]
        
        else:
            raise NotImplementedError

        self.net = nn.Sequential(*net)
    
    def forward(self, img1, img2, forward_scale=0):
        # img1, img2 shape: [N, C, H, W]
        x = torch.cat([img1, img2], dim=1) # [N, 2C, H, W]
        
        # **********************************************************
        alpha = self.net(x) # [N, C, H, W] or [N, 1, H, W] or [N, 1]
        if len(alpha.shape) == 2: # [N, 1]
            alpha = alpha[..., None, None]
        if alpha.shape[1] == 1: # [N, 1, H, W]
            alpha = alpha.expand_as(img1) # [N, C, H, W]
        # **********************************************************
        
        if forward_scale > 0:
            alpha = forward_scale * (alpha - 0.5) + 0.5 
        
        return img1 * alpha + img2 * (1 - alpha), alpha


class DeepMixupNet_V2(nn.Module):
    def __init__(self, mode, depth, width, depth_3x3=2, upscale=1, layerwise_width=''):
        super(DeepMixupNet_V2, self).__init__()
        Ws = [int(w) for w in layerwise_width.split(',')] if layerwise_width else [width] * (depth - 1)

        net, in_channels = [], 6 # 2 RGB images, so 6 channels
        # body
        for d in range(depth - 1):
            ks = 3 if d < depth_3x3 else 1
            net += [nn.Conv2d(in_channels, Ws[d], kernel_size=ks, padding=ks//2),
                    nn.BatchNorm2d(Ws[d]), 
                    nn.LeakyReLU(inplace=True)]
            in_channels = Ws[d]
        
        # tail (last layer)
        d = depth - 1
        ks = 3 if d < depth_3x3 else 1
        net += [nn.Conv2d(in_channels, 3, kernel_size=ks, padding=ks//2)]
        self.net = nn.Sequential(*net)
    
    def forward(self, img1, img2, forward_scale=0):
        # img1, img2 shape: [N, C, H, W]
        x = torch.cat([img1, img2], dim=1) # [N, 2C, H, W]
        return self.net(x), None