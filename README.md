# ST_Cubsim

*Learning from Temporal Spatial Cubism for Cross-Dataset Skeleton-based Action Recognition*, Yansong Tang, Xingyu Liu, Xumin Yu, Danyang Zhang, Jiwen Lu and Jie Zhou (in submission)

## Prerequisites

Our code is based on **Python3.5**. There are a few dependencies to run the code in the following:

- Python (>=3.5)
- PyTorch (0.4.0)
- [torchnet](https://github.com/pytorch/tnt)
- Visdom
- Other version info about some Python packages can be found in `requirements.txt`

## Info

### Datasets

We conduct experiments on NTU *↔* PKU, NTU *↔* kinetics, PKU *↔* kinetics, ORGBD *→*MSRDA3D, and NTU *→* SBU.

### Methods

We show the code of ST-cubsim in this repo, and our compared methods MMD, DANN, JAN, CDAN, BSP, TA3N, and GINs.

### Paired action categories

We present the 51 paired action categories between PKU-MMD and NTU RGB+D in *paired_actions.png* .

## Results

The experimental results can be referred to *Learning from Temporal Spatial Cubism for Cross-Dataset Skeleton-based Action Recognition*.
