import torch
import torch.nn as nn 
import torchsummary
import layers 
import numpy as np 
import torch.nn.functional as F 
# import layers

class Unet3(nn.Module):
    def __init__(self, in_channels, out_channels, nker, use_resize=False) -> None:
        super(Unet3, self).__init__()
         # encoder
         
        self.encoder1 = layers.Down(in_channels, nker * 1)
        self.encoder2 = layers.Down(nker * 1, nker * 2)
        self.encoder3 = layers.Down(nker * 2, nker * 4)

        self.neck = layers.CBDR2d(nker * 4, nker * 8)
        
        self.decoder3 = layers.Up(nker * 8, nker * 4, use_resize=use_resize) 
        self.decoder2 = layers.Up(nker * 4, nker * 2, use_resize=use_resize) 
        self.decoder1 = layers.Up(nker * 2, nker * 1, use_resize=use_resize) 

        self.final = nn.Conv2d(nker * 1, out_channels, 3, 1, 1)
    
    def forward(self, x):
        x, residual1 = self.encoder1(x)
        x, residual2 = self.encoder2(x)
        x, residual3 = self.encoder3(x)

        x = self.neck(x)

        x = self.decoder3(x, residual3)
        x = self.decoder2(x, residual2)
        x = self.decoder1(x, residual1)

        x = self.final(x)

        return x

class UnetWithAttentionMultiOut(nn.Module):
    def __init__(self, in_channels, out_channels, nker, attention_mode='CBAM', relu=0.0, use_resize=False):
        super(UnetWithAttentionMultiOut, self).__init__()
         # encoder
         
        self.encoder1 = layers.DownWithAttention(in_channels, nker * 1, attention_mode=attention_mode, relu=relu )
        self.encoder2 = layers.DownWithAttention(nker * 1, nker * 2, attention_mode=attention_mode, relu=relu)
        self.encoder3 = layers.DownWithAttention(nker * 2, nker * 4, attention_mode=attention_mode, relu=relu)
        self.encoder4 = layers.DownWithAttention(nker * 4, nker * 8, attention_mode=attention_mode, relu=relu)
        self.encoder5 = layers.DownWithAttention(nker * 8, nker * 16, attention_mode=attention_mode, relu=relu)

        self.neck = layers.CBDAR2d(nker * 16, nker * 32 , use_attention=True, attention_mode=attention_mode, relu=relu)
        
        self.decoder5 = layers.UpWithAttention(nker * 32, nker * 16, attention_mode=attention_mode, relu=relu, use_resize=use_resize) 
        self.decoder4 = layers.UpWithAttention(nker * 16, nker * 8, attention_mode=attention_mode, relu=relu, use_resize=use_resize) 
        self.decoder3 = layers.UpWithAttention(nker * 8, nker * 4, attention_mode=attention_mode, relu=relu, use_resize=use_resize) 
        self.decoder2 = layers.UpWithAttention(nker * 4, nker * 2, attention_mode=attention_mode, relu=relu, use_resize=use_resize) 
        self.decoder1 = layers.UpWithAttention(nker * 2, nker * 1, attention_mode=attention_mode, relu=relu, use_resize=use_resize) 

        self.final = nn.Conv2d(nker * 1, out_channels, 3, 1, 1)

        self.decoder3_out = nn.Conv2d(nker * 4, out_channels, 3, 1, 1)
        self.decoder2_out = nn.Conv2d(nker * 2, out_channels, 3, 1, 1)
    
    
    def forward(self, x):
        e1, residual1 = self.encoder1(x)
        e2, residual2 = self.encoder2(e1)
        e3, residual3 = self.encoder3(e2)
        e4, residual4 = self.encoder4(e3)
        e5, residual5 = self.encoder5(e4)

        neck = self.neck(e5)

        d5 = self.decoder5(neck, residual5)
        d4 = self.decoder4(d5, residual4)
        d3 = self.decoder3(d4, residual3)
        d2 = self.decoder2(d3, residual2)
        d1 = self.decoder1(d2, residual1)

        final = self.final(d1)

        return final, self.decoder2_out(d2), self.decoder3_out(d3)
    
class Pix2PixDiscriminator(nn.Module):    
    def __init__(self, in_channels=3, out_channels=1, nkr=64, use_sigmoid=True):
        super(Pix2PixDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=nkr, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=nkr, out_channels=nkr * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nkr * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=nkr * 2, out_channels=nkr * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nkr * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=nkr * 4, out_channels=nkr * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nkr * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=nkr * 8, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() if use_sigmoid else nn.Identity()
        )

    def forward(self, x):
        x = self.main(x)        
        return x


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

## 네트워크 weights 초기화 하기
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    
    

if __name__=='__main__':
    model = Pix2PixDiscriminator(3, 256, 256, attention_mode='RCBAM').to('cuda')
    print(model)
    model(torch.randn(1,3,256,256).to('cuda'))
    torchsummary.summary(model=model, input_size=(3, 256,256))