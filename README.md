# Learning Independent Causal Mechanisms for Sepsis

Learn independent mechanisms inverting data points from a transformed distribution back to their original distribution in an unsupervised way.

Based on [Parascandolo et al 2018](https://arxiv.org/abs/1712.00961)

## How to run

1. Requirements

Python 3.6

torch
torchvision
tensorboardX

2. Training

python main.py 
   --num_experts {number of mechanisms to train}
   --input_size {size of input}
