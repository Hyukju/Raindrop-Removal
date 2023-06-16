import torch 
import torch.nn as nn 
import torch.nn.functional as F 

def resize(A, size, mode='padding'):
    if mode == 'interpolation':
        out = F.interpolate(A, size)
    if mode == 'padding':
        rows_a, cols_a = A.shape[-2:]
        rows_b, cols_b = size
        p4d = (0, cols_b - cols_a, 0, rows_b - rows_a)
        out = F.pad(A, p4d, 'replicate')
    return out

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm='batch', use_drop_out=False, relu=0.0):
        super(Down, self).__init__()
        self.layer1 = CBDR2d(in_channels, out_channels, kernel_size, stride, padding, bias, norm, use_drop_out, relu)
        self.layer2 = CBDR2d(out_channels, out_channels, kernel_size, stride, padding, bias, norm, use_drop_out, relu)
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.layer1(x)
        residual = self.layer2(x)
        x = self.maxpool(residual)
        return x, residual 

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm='batch', use_drop_out=False, relu=0.0, use_resize=False):
        super(Up, self).__init__()
        self.use_resize=use_resize
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.layer1 = CBDR2d(in_channels, out_channels, kernel_size, stride, padding, bias, norm, use_drop_out, relu)
        self.layer2 = CBDR2d(out_channels, out_channels, kernel_size, stride, padding, bias, norm, use_drop_out, relu)
    
    def resize(self, A, size):
        return F.interpolate(A, size)

    def forward(self, x, residual):
        x = self.up_conv(x)
        if self.use_resize:
            x = resize(x, residual.shape[-2:])
        x = torch.cat([x, residual], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class CBDR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm='batch', use_drop_out=False, relu=0.0):
        super(CBDR2d, self).__init__()
        sequence = [] 
        sequence += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        
        if norm == 'batch':
            sequence += [nn.BatchNorm2d(num_features=out_channels)]
        elif norm == 'instance':
            sequence += [nn.InstanceNorm2d(num_features=out_channels)]

        if use_drop_out:
            sequence += [nn.Dropout2d(0.5)]
        
        if relu is not None:
            sequence += [nn.ReLU() if relu == 0.0 else nn.LeakyReLU(relu)]
        
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x) 

    
class CBDAR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm='batch', use_drop_out=False, use_attention=False, attention_mode='CBAM', relu=0.0):
        super(CBDAR2d, self).__init__()
        # assert attention_mode in ['CBAM', 'RCBAM', 'proposed', 'None']

        self.norm = norm 
        self.use_drop_out = use_drop_out
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        
        if attention_mode == 'RCBAM':
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bnorm = nn.BatchNorm2d(num_features=out_channels)
        self.inorm = nn.InstanceNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.ca = ChannelAttention(in_channels=out_channels, ratio=16)
        self.sa = SpatialAttention(kernel_size=7)
        self.activation = nn.ReLU() if relu == 0.0 else nn.LeakyReLU(relu)

    def forward(self, x):
        residual = x
        x = self.conv(x)

        if self.norm == 'batch':
            x = self.bnorm(x)
        elif self.norm == 'instance':
            x = self.inorm(x)
        
        if self.use_drop_out:
            x = self.dropout(x)
        
        if self.use_attention:
            if self.attention_mode == 'CBAM':
                x = self.ca(x) * x 
                x = self.sa(x) * x # 
            elif self.attention_mode == 'RCBAM':
                x = self.ca(x) * x 
                x = self.sa(x) * x # 
                x = x + self.residual_conv(residual)
                
        x = self.activation(x)

        return x


#######################################################################################

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // self.ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_channels // self.ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DownWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm='batch', use_drop_out=False, attention_mode='CBAM',relu=0.0):
        super(DownWithAttention, self).__init__()
        self.layer1 = CBDAR2d(in_channels, out_channels, kernel_size, stride, padding, bias, norm, use_drop_out, use_attention=False, relu=relu)
        self.layer2 = CBDAR2d(out_channels, out_channels, kernel_size, stride, padding, bias, norm, use_drop_out, use_attention=True, attention_mode=attention_mode, relu=relu)
        
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.layer1(x)
        residual = self.layer2(x)
        x = self.maxpool(residual)
        return x, residual 

class UpWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm='batch', use_drop_out=False, attention_mode='CBAM', relu=0.0, use_resize=False):
        super(UpWithAttention, self).__init__()
        self.use_resize = use_resize
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.layer1 = CBDAR2d(in_channels, out_channels, kernel_size, stride, padding, bias, norm, use_drop_out, use_attention=False, relu=relu)
        self.layer2 = CBDAR2d(out_channels, out_channels, kernel_size, stride, padding, bias, norm, use_drop_out, use_attention=True,  attention_mode=attention_mode, relu=relu)

    def forward(self, x, residual):
        x = self.up_conv(x)
        if self.use_resize:
            x = resize(x, residual.shape[-2:])
        x = torch.cat([x, residual], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x