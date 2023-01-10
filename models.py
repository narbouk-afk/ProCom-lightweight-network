import torch
from torch import nn
from modelBlocks import First3D, Encoder3D, Decoder3D, Last3D, Center3D, pad_to_shape


def dualConv(in_ch, out_ch):
    conv = nn.Sequential(
    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding="same"),
    nn.BatchNorm3d(out_ch),
    nn.ReLU(inplace=True),
    nn.Conv3d(out_ch, out_ch, kernel_size=3, padding="same"),
    nn.BatchNorm2d(out_ch),
    nn.ReLU(inplace=True)
    )
    return conv

#def cropAndMerge(x):   


class MyUnet3D(nn.Module):

    def __init__(self, n_out_layers):
        super(MyUnet3D, self).__init__()

        self.enc_conv1 = dualConv(3,64)
        self.enc_conv2 = dualConv(64,128)
        self.enc_conv3 = dualConv(128,256)
        self.enc_conv4 = dualConv(256,512)
        self.enc_conv5 = dualConv(512,1024)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2)
        self.dec_conv1 = dualConv(1024, 512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2)
        self.dec_conv2 = dualConv(512, 25)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2)
        self.dec_conv3 = dualConv(256, 12)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2,)
        self.dec_conv4 = dualConv(128, 64)
        
        self.out = nn.Conv2d(64, n_out_layers, kernel_size=1)

    def forward(self, image):
        print(image.shape)
        x1 = self.enc_conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.enc_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.enc_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.enc_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.enc_conv5(x8)
        
        x = self.upconv1(x9)
        x = self.dec_conv1(torch.cat([x,x7], dim=1))
        
        x = self.upconv2(x)
        x = self.dec_conv2(torch.cat([x,x5], dim=1))
        
        x = self.upconv3(x)
        x = self.dec_conv3(torch.cat([x,x3], dim=1))
        
        x = self.upconv4(x)
        x = self.dec_conv4(torch.cat([x,x1], dim=1))
        
        x = self.out(x)
        
        print("Image passed through the network")
        return x
    
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depths=(64, 128, 256, 512, 1024)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(UNet3D, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First3D(in_channels, conv_depths[0], conv_depths[0]))
        encoder_layers.extend([Encoder3D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder3D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i])
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last3D(conv_depths[1], conv_depths[0], out_channels))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center3D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1])]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec

if __name__ == "__main__":
    from torchsummary import summary
    im = torch.rand((1,1,30,100,100))
    model = UNet3D(in_channels=1, out_channels=3)
    ex = model(im)
    summary(model, (1,30,100,100))
    
