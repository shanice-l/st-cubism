# ST_Cubsim

This repo provides the PyTorch implementation of the work:

*Learning from Temporal Spatial Cubism for Cross-Dataset Skeleton-based Action Recognition*, Yansong Tang*, Xingyu Liu*, Xumin Yu, Danyang Zhang, Jiwen Lu &#9993 and Jie Zhou (accepted by ACM TOMM) [[Paper](https://dl.acm.org/doi/10.1145/3472722)]

## Prerequisites

Our code is based on **Python3.5**. There are a few dependencies to run the code in the following:

- Python (>=3.5)
- PyTorch (0.4.0)
- [torchnet](https://github.com/pytorch/tnt)
- Visdom
- Other version info about some Python packages can be found in `requirements.txt`

## Info

### Datasets

We conduct experiments on NTU &harr; PKU, NTU &harr; kinetics, PKU &harr; kinetics, ORGBD &harr; MSRDA3D, and NTU &harr; SBU.

### Methods

We show the source code of ST-cubsim in this repo, and the code of our compared methods DANN, JAN, CDAN, TA3N, BSP, and GINs.

### Paired action categories

We present the 51 paired action categories between PKU-MMD and NTU RGB+D in *[paired_actions.png](https://github.com/shanice-l/st-cubism/blob/master/paied_actions.png)*, and the 12 paired categories between kinetics and NTU RGB+D in *[paired_actions_between_nk.png](https://github.com/shanice-l/st-cubism/blob/master/paired_actions_between_nk.png)*.

## Results

The experimental results can be referred to *Learning from Temporal Spatial Cubism for Cross-Dataset Skeleton-based Action Recognition*.

## Citation

If you find this repo useful in your research, please consider citing:
```
@article{tang_2022_sda,
author   = {Tang, Yansong and Liu, Xingyu and Yu, Xumin and Zhang, Danyang and Lu, Jiwen and Zhou, Jie},
title    = {Learning from Temporal Spatial Cubism for Cross-Dataset Skeleton-Based Action Recognition},
year     = {2022},
journal  = {ACM Transactions on Multimedia Computing, Communications, and Applications (ACM TOMM)},
url      = {https://doi.org/10.1145/3472722},
doi      = {10.1145/3472722},
}
```
