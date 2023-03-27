import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
def crop(enc_ftrs, x):
    _, _, H, W = x.shape
    enc_ftrs   = transforms.CenterCrop([H, W])(enc_ftrs)
    return enc_ftrs

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
        self.bn2 = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(1,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        enc_fmap = []
        for block in self.enc_blocks:
            x = block(x)
            enc_fmap.append(x)
            x = self.pool(x)
        return enc_fmap



class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 

    def forward(self, x, encoder_features):
        dec_fmap = []
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
            dec_fmap.append(x)
        return dec_fmap
