import torch
from torch import nn
from modelBlocks import Encoder, Decoder, MFunit, DMFUnit
import torch.nn.functional as F
from torchsummary import summary
class UNet2D(nn.Module):
    def __init__(self, enc_chs=(1,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), 
                 num_class=1, retain_dim=False, out_sz=(256,256), return_fmap=False):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.Fmap = return_fmap
        self.out_sz = out_sz
        
    def forward(self, x):
        enc_fmaps = self.encoder(x)
        dec_fmaps      = self.decoder(enc_fmaps[::-1][0], enc_fmaps[::-1][1:])
        out      = self.head(dec_fmaps[-1])
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        if self.Fmap:
            return enc_fmaps+dec_fmaps+[out]
        return out
    
# Below is our attempt to make a DMF Network for 2D images
    
class MFNet(nn.Module): #
    
    def __init__(self, c=1,n=32,channels=128,groups = 16,norm='bn', num_classes=4):
        super(MFNet, self).__init__()

        # Entry flow
        self.encoder_block1 = nn.Conv2d( c, n, kernel_size=3, padding=1, stride=2, bias=False)# H//2
        self.encoder_block2 = nn.Sequential(
            MFunit(n, channels, g=groups, stride=2, norm=norm),# H//4 down
            MFunit(channels, channels, g=groups, stride=1, norm=norm),
            MFunit(channels, channels, g=groups, stride=1, norm=norm)
        )
        #
        self.encoder_block3 = nn.Sequential(
            MFunit(channels, channels*2, g=groups, stride=2, norm=norm), # H//8
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm),
            MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm)
        )

        self.encoder_block4 = nn.Sequential(# H//8,channels*4
            MFunit(channels*2, channels*3, g=groups, stride=2, norm=norm), # H//16
            MFunit(channels*3, channels*3, g=groups, stride=1, norm=norm),
            MFunit(channels*3, channels*2, g=groups, stride=1, norm=norm),
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # H//8
        self.decoder_block1 = MFunit(channels*2+channels*2, channels*2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # H//4
        self.decoder_block2 = MFunit(channels*2 + channels, channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # H//2
        self.decoder_block3 = MFunit(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # H
        self.seg = nn.Conv2d(n, num_classes, kernel_size=1, padding=0,stride=1,bias=False)

        self.softmax = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight) #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_block1(x)# H//2 down
        x2 = self.encoder_block2(x1)# H//4 down
        x3 = self.encoder_block3(x2)# H//8 down
        x4 = self.encoder_block4(x3) # H//16
        # Decoder
        y1 = self.upsample1(x4)# H//8
        y1 = torch.cat([x3,y1],dim=1)
        y1 = self.decoder_block1(y1)

        y2 = self.upsample2(y1)# H//4
        y2 = torch.cat([x2,y2],dim=1)
        y2 = self.decoder_block2(y2)

        y3 = self.upsample3(y2)# H//2
        y3 = torch.cat([x1,y3],dim=1)
        y3 = self.decoder_block3(y3)
        y4 = self.upsample4(y3)
        
        y4 = self.seg(y4)
        
        if hasattr(self,'softmax'):
            y4 = self.softmax(y4)
        return y4
    
class DMFNet(MFNet): # softmax
    def __init__(self, c=1,n=32,channels=128, groups=16,norm='bn', num_classes=4):
        super(DMFNet, self).__init__(c,n,channels,groups, norm, num_classes)

        self.encoder_block2 = nn.Sequential(
            DMFUnit(n, channels, g=groups, stride=2, norm=norm,dilation=[1,2]),# H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2]), # Dilated Conv 2
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2])
        )

        self.encoder_block3 = nn.Sequential(
            DMFUnit(channels, channels*2, g=groups, stride=2, norm=norm,dilation=[1,2]), # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2]),# Dilated Conv 2
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2])
        )
        
if __name__ == "__main__":
    
    im = torch.rand((1,1,256,256))
    model = UNet2D()
    ex = model(im)
    summary(model, (1,256,256))
    
