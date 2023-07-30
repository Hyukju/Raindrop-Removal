## Raindrop-Removal Image Translation Using Target-Mask Network with Attention Module
Hyuk-Ju Kwon, Sung-Hak Lee*

Paper (https://www.mdpi.com/2227-7390/11/15/3318)

---
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



