
import torch
import torch.nn as nn
import torch.optim as optim
import networks
import losses
import utils
import numpy as np 
from torchvision.transforms import Resize
from model.base_model import BaseModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model(BaseModel):
    def __init__(self, phase, in_channels=3, out_channels=3, nker=64, lr=0.0002) -> None:
        assert phase in ['train', 'test']

        self.use_resize = False if phase=='trian' else True
        self.generator_mask = networks.Unet3(3, 1, nker, use_resize=self.use_resize).to(DEVICE)
        self.generator = networks.UnetWithAttentionMultiOut(in_channels + 1, out_channels, nker, attention_mode='RCBAM', relu=0.2, use_resize=self.use_resize).to(DEVICE)
        self.discriminator = networks.Pix2PixDiscriminator().to(DEVICE)

        # 가중치 초기화
        networks.init_weights(self.generator, init_type='normal', init_gain=0.02)
        networks.init_weights(self.generator_mask, init_type='normal', init_gain=0.02)
        networks.init_weights(self.discriminator, init_type='normal', init_gain=0.02)

        self.loss_MSE_fn = nn.MSELoss()  
        self.loss_SSIM_fn = losses.SSIM()
        self.loss_VGG_fn = losses.VGGLoss()  
        self.loss_D_fn = nn.BCELoss()
        self.loss_MASK_fn = nn.MSELoss() 

        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optim_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optim_GM = optim.Adam(self.generator_mask.parameters(), lr=lr, betas=(0.5, 0.999))
        self.phase = phase
        
        # 모델의 가중치 
        self.model_weights = self.get_model_weights()

    def get_model_weights(self):
        # load 또는 save 시 필요한 model 과 optimizer의 정보를 딕셔너리 형태로 저장
        model_weights = {
            'generator':self.generator.state_dict(),
            'generator_mask':self.generator_mask.state_dict(),
            'discriminator':self.discriminator.state_dict(),
            'optim_D':self.optim_D.state_dict(),
            'optim_G':self.optim_G.state_dict(),
            'optim_GM':self.optim_GM.state_dict()
            }
        return model_weights

    def set_model_weights(self):
        # load 후 기존의 학습된 weights로 갱신
        self.generator.load_state_dict(self.model_weights['generator'])
        self.generator_mask.load_state_dict(self.model_weights['generator_mask'])
        self.discriminator.load_state_dict(self.model_weights['discriminator'])
        self.optim_D.load_state_dict(self.model_weights['optim_D'])
        self.optim_G.load_state_dict(self.model_weights['optim_G'])
        self.optim_GM.load_state_dict(self.model_weights['optim_GM'])

    def set_inputs(self, data):
        # 학습에 필요한 입력 및 레이블 영상 불러오기
        self.rain = data['input'].to(DEVICE) 
        self.clean = data['label'].to(DEVICE) 
        self.mask = data['mask'].to(DEVICE)
    
    '''
    학습 및 테스트 관련 
    '''
    def forward(self):
        self.fake_mask = self.generator_mask(self.rain)
        self.fake_clean_list = self.generator(torch.cat((self.rain, self.fake_mask.detach()), dim=1))
        self.fake_clean = self.fake_clean_list[0]

    def backward_D(self):
        networks.set_requires_grad(self.discriminator, True)
       
        pred_fake_clean = self.discriminator(self.fake_clean.detach())
        label_fake_clean = torch.zeros_like(pred_fake_clean)
        loss_fake_clean = self.loss_D_fn(pred_fake_clean, label_fake_clean)

        pred_real_clean = self.discriminator(self.clean)
        label_real_clean = torch.ones_like(pred_real_clean)
        loss_real_clean = self.loss_D_fn(pred_real_clean, label_real_clean)     

        self.loss_d = (loss_fake_clean + loss_real_clean) * 0.5 
        
        if self.phase == 'train':
            self.loss_d.backward()

    def multiscale_loss(self, y_pred_list, y_true):
        loss = 0 
        for y_pred in y_pred_list:
            if y_pred.shape[-2:] != y_true.shape[-2:]:
                transform = Resize(y_pred.shape[-2:])
                resized_y_true = transform(y_true)
                loss += self.loss_MSE_fn(y_pred, resized_y_true)
            else:
                loss += self.loss_MSE_fn(y_pred, y_true)
            

        return loss / len(y_pred_list)
    
    def backward_G(self):
        networks.set_requires_grad(self.discriminator, False)

        pred_fake_clean = self.discriminator(self.fake_clean)
        label_real_clean = torch.ones_like(pred_fake_clean)
        
        self.loss_g_gan = self.loss_D_fn(pred_fake_clean, label_real_clean)
        
        self.loss_MUTISCALE = self.multiscale_loss(self.fake_clean_list, self.clean)
        self.loss_SSIM = self.loss_SSIM_fn(self.fake_clean, self.clean)
        self.loss_VGG = self.loss_VGG_fn(self.fake_clean, self.clean) 
        self.loss_g = self.loss_g_gan + self.loss_MUTISCALE + self.loss_SSIM + self.loss_VGG

        if self.phase == 'train':
            self.loss_g.backward()

    def backward_GM(self):
        self.loss_mask = self.loss_MASK_fn(self.fake_mask, self.mask)
        
        if self.phase == 'train':
            self.loss_mask.backward()


    def train_on_batch(self):
        # batch 기준 학습 수행         
        self.discriminator.train()
        self.generator.train()
        self.generator_mask.train()

        self.forward()

        self.optim_GM.zero_grad()
        self.backward_GM()
        self.optim_GM.step()

        self.optim_D.zero_grad()
        self.backward_D()
        self.optim_D.step()

        self.optim_G.zero_grad()
        self.backward_G()
        self.optim_G.step()

    def test_on_batch(self):
        # test 또는 validation을 batch 기준으로 수행
        self.generator.eval()
        with torch.no_grad():
            self.forward()

    def test_one_image(self, input):
        # test 시 영상 한 장에 대한 결과 확인 용
        self.generator.eval()
        self.generator_mask.eval()
        with torch.no_grad():
            mask  = self.generator_mask(input.to(DEVICE))
            output, _, _ = self.generator(torch.cat((input.to(DEVICE), mask), dim=1))

        return {
            'input':utils.tensor2numpy(input),
            'output':utils.tensor2numpy(output),
            'mask':utils.tensor2numpy(mask),
        }

    '''
    학습 및 테스트 결과 영상 및 로스 저장 관련 
    '''    
    def get_outputs(self):
        # 결과 영상을 저장하기 위하여 딕셔너리 형태로 만듦
        return {
            'input': utils.tensor2numpy(self.rain),
            'label': utils.tensor2numpy(self.clean), 
            'mask': utils.tensor2numpy(self.mask), 
            'output': utils.tensor2numpy(self.fake_clean),
            'output mask': utils.tensor2numpy(self.fake_mask),
            }

    def get_losses(self):   
        # 현재 loss를 출력하기 위하여 딕셔너리 형태로 만듦   
        # .item()은 batch 평균 loss를 나타냄
        return {
            'LOSS D': self.loss_d.item(),
            'LOSS G': self.loss_g.item(),
            'LOSS GAN': self.loss_g_gan.item(),
            'LOSS MULTSCALE': self.loss_MUTISCALE.item(),
            'LOSS SSIM': self.loss_SSIM.item(),
            'LOSS VGG': self.loss_VGG.item(),
            'LOSS MASK': self.loss_mask.item()
            }
