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
    
# Below is our attempt to make a DMF Network for 2D images
class Conv2d_Block(nn.Module):
    def __init__(self,num_in,num_out,kernel_size=1,stride=1,g=1,padding=None,norm=None):
        super(Conv2d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_in, num_out, kernel_size=kernel_size, padding=padding,stride=stride, groups=g, bias=False)

    def forward(self, x): # BN + Relu + Conv
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h
    
class DilatedConv2DBlock(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=(1,1), stride=1, g=1, d=(1,1), norm=None):
        super(DilatedConv2DBlock, self).__init__()
        assert isinstance(kernel_size,tuple) and isinstance(d,tuple)

        padding = tuple(
            [(ks-1)//2 *dd for ks, dd in zip(kernel_size, d)]
        )

        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_in,num_out,kernel_size=kernel_size,padding=padding,stride=stride,groups=g,dilation=d,bias=False)

    def forward(self, x):
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h
    
class MFunit(nn.Module):
    def __init__(self, num_in, num_out, g=1, stride=1, d=(1,1),norm=None):
        """  
        :param num_in: number of input channels
        :param num_out: number of output channels
        :param g: groups of group conv.
        :param stride: 1 or 2
        :param d: tuple, d[0] for the first 3x1 conv while d[1] for the 1x3 conv
        """
        super(MFunit, self).__init__()
        
        num_mid = num_in if num_in <= num_out else num_out
        self.conv1x1_in1 = Conv2d_Block(num_in,num_in//4,kernel_size=1,stride=1,norm=norm)
        self.conv1x1_in2 = Conv2d_Block(num_in//4,num_mid,kernel_size=1,stride=1,norm=norm)
        self.conv3x3_m1 = DilatedConv2DBlock(num_mid,num_out,kernel_size=(3,1),stride=stride,g=g,d=(d[0],d[0]),norm=norm) # dilated
        self.conv3x3_m2 = DilatedConv2DBlock(num_out,num_out,kernel_size=(1,3),stride=1,g=g,d=(d[1],d[1]),norm=norm)
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(1,3,3),stride=1,g=g,d=(1,d[1],d[1]),norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1_shortcut = Conv2d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0,norm=norm)
            if stride == 2:
                # if MF block with stride=2, 2x2x2
                self.conv2x2_shortcut = Conv2d_Block(num_in, num_out, kernel_size=2, stride=2,padding=0, norm=norm) # params
        
    def forward(self, x):
        x1 = self.conv1x1_in1(x)
        x2 = self.conv1x1_in2(x1)
        x3 = self.conv3x3_m1(x2)
        x4 = self.conv3x3_m2(x3)

        shortcut = x

        if hasattr(self,'conv1x1_shortcut'):
            shortcut = self.conv1x1_shortcut(shortcut)
        if hasattr(self,'conv2x2_shortcut'):
            shortcut = self.conv2x2_shortcut(shortcut)

        return x4 + shortcut

class DMFUnit(nn.Module):
    # weighred add
    def __init__(self, num_in, num_out, g=1, stride=1,norm=None,dilation=None):
        super(DMFUnit, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))

        num_mid = num_in if num_in <= num_out else num_out

        self.conv1x1_in1 = Conv2d_Block(num_in, num_in // 4, kernel_size=1, stride=1, norm=norm)
        self.conv1x1_in2 = Conv2d_Block(num_in // 4,num_mid,kernel_size=1, stride=1, norm=norm)

        self.conv3x3_m1 = nn.ModuleList()
        if dilation == None:
            dilation = [1,2]
        for i in range(2):
            self.conv3x3_m1.append(
                DilatedConv2DBlock(num_mid,num_out, kernel_size=(3, 3), stride=stride, g=g, d=(dilation[i],dilation[i]),norm=norm)
            )

        # It has not Dilated operation
        self.conv3x3_m2 = DilatedConv2DBlock(num_out, num_out, kernel_size=(3, 3), stride=(1,1), g=g,d=(1,1), norm=norm)
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(1, 3, 3), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1_shortcut = Conv2d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                self.conv2x2_shortcut = Conv2d_Block(num_in, num_out, kernel_size=2, stride=2, padding=0, norm=norm)


    def forward(self, x):
        x1 = self.conv1x1_in1(x)
        x2 = self.conv1x1_in2(x1)
        x3 = self.weight1*self.conv3x3_m1[0](x2) + self.weight2*self.conv3x3_m1[1](x2)
        x4 = self.conv3x3_m2(x3)
        shortcut = x
        if hasattr(self, 'conv1x1_shortcut'):
            shortcut = self.conv1x1_shortcut(shortcut)
        if hasattr(self, 'conv2x2_shortcut'):
            shortcut = self.conv2x2_shortcut(shortcut)
        return x4 + shortcut
