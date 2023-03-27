import torch
from torch import nn
from modelBlocks import Encoder, Decoder
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

if __name__ == "__main__":
    
    im = torch.rand((1,1,256,256))
    model = UNet2D()
    ex = model(im)
    summary(model, (1,256,256))
    
