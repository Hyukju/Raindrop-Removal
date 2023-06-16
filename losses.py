import torch
import torch.nn as nn 
from torchvision.models import vgg19 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        # 입력 경계의 반사를 사용하여 상/하/좌/우에 입력 텐서를 추가로 채웁니다.
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # shape : (xh, xw) -> (xh + 2, xw + 2)
        x = self.refl(x) 
        # shape : (yh, yw) -> (yh + 2, yw + 2)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # Loss function
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean()

class VGGLoss(nn.Module):
    def __init__(self, layer=36):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:layer].eval().to(DEVICE)
        self.loss_fn = nn.MSELoss()
        # RGB format
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)
       

        for param in self.vgg.parameters():
            param.requires_grad = False 

    def normalize(self, x):
        x = x * 0.5 + 0.5 
        mean = self.vgg_mean.view(-1, 1, 1)
        std = self.vgg_std.view(-1, 1, 1)
        return (x - mean) / std

    def forward(self, pred, target):
        vgg_pred_features =  self.vgg(self.normalize(pred))
        vgg_target_features = self.vgg(self.normalize(target))
        return self.loss_fn(vgg_pred_features, vgg_target_features)