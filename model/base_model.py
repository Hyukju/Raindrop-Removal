import torch
import utils
from abc import abstractmethod

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseModel():
    def __init__(self, opts) -> None:
        
        '''
        self.opts = opts
        self.model = network.Unet(opts.in_channels, opts.out_channels, opts.nker).to(DEVICE)
        self.loss_fn = nn.L1Loss()        
        self.optim = optim.Adam(self.model.parameters(), opts.lr=0.002, betas=(0.5, 0.999))
        self.phase = opts.phase
        
        # 모델의 가중치 
        self.model_weights = self.get_model_weights()

        self.inputs = None 
        self.labels = None 
        self.outputs = None
        self.loss = None 
        '''
        pass 


    @abstractmethod
    def get_model_weights(self):
        '''
        # load 또는 save 시 필요한 model 과 optimizer의 정보를 딕셔너리 형태로 저장
        model_weights = {
            'model':self.model.state_dict(),
            'optim':self.optim.state_dict()
            }
        return model_weights
        '''
        model_weights = {}
        return model_weights
       
   
    @abstractmethod
    def set_model_weights(self):
        '''
        # load 후 기존의 학습된 weights로 갱신
        self.model.load_state_dict(self.model_weights['model'])
        self.optim.load_state_dict(self.model_weights['optim'])
   
        '''
        pass
      
    @abstractmethod
    def set_inputs(self, data):
        '''
        # 학습에 필요한 입력 및 레이블 영상 불러오기
        self.inputs = data['input'].to(DEVICE) 
        self.labels = data['label'].to(DEVICE) 
        '''
        pass
      
    
    '''
    학습 및 테스트 관련 
    '''    
    @abstractmethod
    def train_on_batch(self):
        '''
        # batch 기준 학습 수행 
        self.model.train()
        self.optim.zero_grad()
        self.backward()
        self.optim.step()
        '''
        pass
      
    '''
    학습 및 테스트 결과 영상 및 로스 저장 관련 
    '''
    @abstractmethod    
    def get_outputs(self):
        '''
        # 결과 영상을 저장하기 위하여 딕셔너리 형태로 만듦
        return {
            'inputs': utils.tensor2numpy(self.inputs),
            'labels': utils.tensor2numpy(self.labels), 
            'outputs': utils.tensor2numpy(self.outputs)
            }
        '''
        outputs = {}
        return outputs
       
    @abstractmethod
    def get_losses(self):   
        '''
        # 현재 loss를 출력하기 위하여 딕셔너리 형태로 만듦   
        # .item()은 batch 평균 loss를 나타냄
        return {'LOSS': self.loss.item()}    
        '''
        losses = {}
        return losses
       
    
    '''
    수정하지 말 것!!
    '''
    def save(self, ckpt_dir, epoch):
        # 학습 가중 치 저장 
        utils.save_weights(ckpt_dir, self.get_model_weights(), epoch)
        
    def load(self, ckpt_dir, epoch=None):
        # 학습된 모델의 가중치 불러오기
        st_epoch, weights = utils.load_weights(ckpt_dir, epoch)
        # st_epoch > 1 때만 가중치 업데이트
        if st_epoch > 1:
           self.model_weights = weights
           self.set_model_weights()
        return st_epoch


    
     