#  GNN-based Few-Shot Transfer Learning for Device-free Indoor Localization

## Introduction:
This is implementation of few-shot learning based on graph neural network for device-free CSI indoor localization. The main idea is using few-shot learning to transfer the localizing system to different domain with only few data.

This code is based on
- https://github.com/louis2889184/gnn_few_shot_cifar100
- https://github.com/hazdzz/ChebyNet
- https://github.com/sadbb/few-shot-fgnn

## Dataset:
precollected CSI data from two different scenario

## Conception:
### system setup
![](https://i.imgur.com/ic2Jexu.png)
### model structure
![](https://i.imgur.com/deRplI7.png)

## Execution:
Train for GNN with k-shot

`python main.py --model GNN --shot k`

Train for Attentive GNN with k-shot and $\beta=n$ (0.7)

`python main.py --model Attentive_GNN --shot k --beta n` 

Train for EGNN with k-shot and reg = n (0.01)

`python main.py --model EGNN --shot k --reg n`

Train for ChebyNet with k-shot and Kg = n (3)

`python main.py --model ChebyNet --shot k --Kg n`

If you are unable to use GPU, please use argument --device cpu

Where the value in brackets is default.

## Experiement Result:
|                                |       1-shot      |       5-shot      |       10-shot     |
|              :---:             |       :---:       |       :---:       |       :---:       |
| **pretrain**                   | 27.37% | 53.07% | 69.61% |
| **GNN**                        | 35.63% | 75.82% | 86.91% | 
| **Attentive_GNN**              | 38.81% | 76.28% | 88.31% | 
| **EGNN**                       | 38.11% | 77.27% | 88.11% |
| **ChebyNet**                   | 39.64% | 76.69% | 87.19% | 

## Dependencies:

- python==3.9.7
- torch==1.9.0+cu102
- numpy==1.21.1
- sklearn==0.24.2

## Contact:
Bing-Jia Chen, b07901088@ntu.edu.tw
