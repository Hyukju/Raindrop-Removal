## Raindrop-Removal
---
## Prerequisites:
1. Python 3.8.14
2. Pytorch 1.9.1 

## Dataset 

The whole dataset can be find in ATTGAN author pages(https://github.com/rui1996/DeRaindrop)

## Our Weight and Result Images

Download here:

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
|    └──proposed/model_epoch1500.pth
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

