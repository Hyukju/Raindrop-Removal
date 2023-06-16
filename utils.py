import os 
import cv2
import csv
import torch
import math 
import numpy as np 
from datetime import datetime

IMG_EXT = ['.bmp', '.jpg', '.png', '.tif']

def get_image_file_list(img_dir):
    file_list = os.listdir(img_dir)
    img_file_list = [f for f in file_list if os.path.splitext(f)[-1] in IMG_EXT]
    return img_file_list

def print_args(args):
    print('-------------------------------------------')
    for k, v in args.__dict__.items(): 
        print(f'{k}: {v}') 
    print('-------------------------------------------')
        

def save_log(save_dir, phase, args):
    with open(os.path.join(save_dir, f'{phase}_log.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write(f'{k}: {v}\n')
        f.write(f'datetime: {datetime.now()}')

#---------------------------------------------------
#
#                Save model weight 
#
#---------------------------------------------------

def save_weights(ckpt_dir, model_weights, epoch):
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(model_weights, 
                os.path.join(ckpt_dir, f'model_epoch{epoch}.pth' ))

def load_weights(ckpt_dir, epoch=None):
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list = [f for f in ckpt_list if f.endswith('pth')]
        
    if len(ckpt_list) == 0:
        epoch = 1
        model_weights = None
    else:
        if epoch == None:
            ckpt_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            epoch = int(ckpt_list[-1].split('epoch')[1].split('.pth')[0])
            model_weights = torch.load(os.path.join(ckpt_dir, ckpt_list[-1]))
        else:
            model_weights = torch.load(os.path.join(ckpt_dir, f'model_epoch{epoch}.pth'))
    return epoch, model_weights

#---------------------------------------------------
#
#                  LOSS 
#
#---------------------------------------------------

def print_losses(epoch, max_epochs, batch, max_batchs, losses_list, title='TRAIN', mode='last'):
    '''
    epoch: current epoch
    max_epochs: maximum epochs
    batch: current batch
    max_batchs: maximu batchs
    losses_list: [lossess_dict], 딕셔너리 구조를 리스트 형태로 쌓은 형태 
    title: 출력 시 현재 상태를 나타나게 하기 위한 구문 
    mode: 'last' or 'mean', 현재 최신 출력(last)을 나타낼지 아니면 
                            list 내부 평균 값('mean')을 나타낼지 선택
    '''
    disp_losses = process_losses(mode, losses_list)

    if mode=='last':
        print(f'{title}: EPOCH {epoch}/{max_epochs} | BATCH {batch}/{max_batchs}', end=' | ')
    if mode == 'mean':
        print(f'{title}: EPOCH {epoch}/{max_epochs}', end=' | ')

    for k, v in disp_losses.items():
        print(f'{k}: {v:0.4f}', end= ' | ')
    print('')

def process_losses(mode, losses_list):
    assert mode in ['last', 'mean']

    disp_losses = dict()

    if mode=='last':
        disp_losses = losses_list[-1]
        
    if mode == 'mean':
        for losses in losses_list:
            for k, v in losses.items():
                if k in list(disp_losses.keys()):
                    disp_losses[k].append(v)
                else:
                    disp_losses[k] = [v]
        for k, v in disp_losses.items():
            disp_losses[k] = np.mean(v)

    return disp_losses

def save_losses(save_dir, epoch, losses_list, mode='last'):
    disp_losses = process_losses(mode, losses_list)
    titels = list(disp_losses.keys())
    losses = list(disp_losses.values())
    
    # 헤더 생성 유무 
    make_header = False
    if not os.path.isfile(os.path.join(save_dir, 'loss.csv')):
        make_header = True
    
    with open(os.path.join(save_dir, 'loss.csv'), 'a', newline='')  as f:
        wr = csv.writer(f)
        if make_header:
            wr.writerow(['epoch'] + titels)
        wr.writerow([epoch] + losses)

#---------------------------------------------------
#
#                Save result image
#
#---------------------------------------------------

def save_outputs(save_dir, filename, outputs, max_display=5):
    '''
    save_dir : path save dir
    filename : save filename
    outputs  : structure of save image (dictionary) 
               outputs = {
                'inputs': image list # [?, height, width, channl]
                'outputs' : image list # [?, height, width, channl]
               }
    max_display : number of image for saving
    '''
    os.makedirs(save_dir, exist_ok=True)
    
    titles = list(outputs.keys())
    num_images = outputs[titles[0]].shape[0]
    # outputs의 영상 숫자가 max_display 보다 적을 경우 outputs 수 만큼만 출력
    max_display = max_display if max_display < num_images else num_images
    
    # 영상 사이 간격 
    margin = 20
    for i in range(max_display):
        for j, title in enumerate(outputs.keys()):
            img = outputs[title][i].copy()
            # rgb2bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # [rows, cols, 1] --> [rows, cols, 3] 
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            # 이미지를 수평으로 연결
            if j == 0:
                hstack_image = img
                hstack_line = np.zeros_like(hstack_image)
                hstack_line = hstack_line[:,:margin,:]
            else:
                hstack_image = np.hstack([hstack_image, hstack_line, img])
        
        # 이미지를 수직으로 연결
        if i == 0:
            vstack_image = hstack_image
            vstack_line = np.zeros_like(vstack_image)
            vstack_line = vstack_line[:margin,...]
        else:
            vstack_image = np.vstack([vstack_image, vstack_line, hstack_image])
    
    # 이미지 상단 제목 이미지 생성
    cols = img.shape[1]
    title_image = np.zeros_like(hstack_image)
    title_image = title_image[:50,...]

    for i in range(len(titles)):
        title_image = cv2.putText(img=title_image, text=titles[i], org=((cols + margin) * i,30),fontFace=2, fontScale=1, color=(1,1,1), thickness=2)     
    
    # 영상 제목 이미지와 결과 이미지를 수직으로 연결
    vstack_image = np.vstack([title_image, vstack_image])

    # 이미지의 범위 [0,1]를 [0,255]로 변경 후 저장
    cv2.imwrite(os.path.join(save_dir, filename), vstack_image * 255)
     

def tensor2numpy(tensor, mean=0.5, std=0.5):
    arr =  tensor.detach().cpu().numpy().transpose(0,2,3,1)
    arr = arr * std + mean 
    arr = np.clip(arr, 0 ,1)
    return  arr

def numpy2tensor(nump, mean=0.5, std=0.5):
    if nump.dtype == 'uint8':
        nump = (nump/255.)
    nump = (nump - mean) / std
    if nump.ndim == 3:
        nump = nump[np.newaxis,...,]
    tensor = torch.from_numpy(nump.transpose(0,3,1,2).astype('float32'))
    return tensor

def expand_size(img, size):
    rows, cols = img.shape[:2]
    nrwos, ncols = math.ceil(rows/size) * size, math.ceil(cols/size) * size
    output = np.zeros((nrwos, ncols, 3), dtype=img.dtype)
    output[:rows, :cols, :] = img
    return output

def restore_size(img, rows, cols):
    return img[:,:rows, :cols, :]

    

