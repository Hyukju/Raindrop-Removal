# Raindrop-Removal Image Translation Using Target-Mask Network with Attention Module

### Hyuk-Ju Kwon, Sung-Hak Lee*
### Paper (https://www.mdpi.com/2227-7390/11/15/3318)

---
## Abstract
Abstract: Image processing plays a crucial role in improving the performance of models in various
fields such as autonomous driving, surveillance cameras, and multimedia. However, capturing ideal
images under favorable lighting conditions is not always feasible, particularly in challenging weather
conditions such as rain, fog, or snow, which can impede object recognition. This study aims to
address this issue by focusing on generating clean images by restoring raindrop-deteriorated images.
Our proposed model comprises a raindrop-mask network and a raindrop-removal network. The
raindrop-mask network is based on U-Net architecture, which learns the location, shape, and brightness of raindrops. The rain-removal network is a generative adversarial network based on U-Net and
comprises two attention modules: the raindrop-mask module and the residual convolution block
module. These modules are employed to locate raindrop areas and restore the affected regions. Multiple loss functions are utilized to enhance model performance. The image-quality assessment metrics
of proposed method, such as SSIM, PSNR, CEIQ, NIQE, FID, and LPIPS scores, are 0.832, 26.165,
3.351, 2.224, 20.837, and 0.059, respectively. Comparative evaluations against state-of-the-art models
demonstrate the superiority of our proposed model based on qualitative and quantitative results.

### Our Model
<img src="src\main.jpg">

### Our Results
<img src="src\0_rain.png">
<img src="src\15_rain.png">

## 

## Prerequisites:
1. Python 3.8.14
2. Pytorch 1.9.1 

## Dataset 

The whole dataset can be find in ATTGAN author pages(https://github.com/rui1996/DeRaindrop)

## Our Weight and Result Images

Download here: 
1. Weight: model_epoch1800.pth (https://drive.google.com/file/d/1tWIDgMldgRdRjWkg3i2-2v-GXIv9Rip_/view?usp=drive_link)
2. Result Images (https://drive.google.com/file/d/1mPT7ioCxQUDKV8xoAUN0ym87NL-Rjm2o/view?usp=drive_link)

## Directory Structure

The structure should be:
```bash
|────dataset
|    |──train
|    |  |──data
|    |  └──gt
|    |──test_a
|    |  |──data
|    |  └──gt
|    └──teat_b
|       |──data
|       └──gt
|────checkpoint
|    └──proposed/model_epoch1800.pth
|────model
|    |─ ...
|────result
|
|─layer.py
|─losses.py
|─networks.py
|─test.py
|─train.py
└─utils.py

```
## Training
```
python train.py
```
## Test
```
python test.py 
```
## Cite 
```
Kwon, H.-J.; Lee, S.-H. Raindrop-Removal Image Translation Using Target-Mask Network with Attention Module. Mathematics 2023, 11, 3318. https://doi.org/10.3390/math11153318
```



