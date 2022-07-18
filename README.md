# UniVCL

This is the repository for Unifying Vision Unsupervised Contrastive Learning from a Graph Perspective
This repo is mainly copied from the MSF (https://github.com/UMBCvision/MSF).

# Requirements
torch==1.8.1

torchvision==0.9.0

# Usage
To perform self-supervised training on ImageNet, run:

sh graphs_simple/train_gpu8_batchsize_256_ep100_im1000_msf_dropedge_bsz.sh

To perform linear evaluation, run:
sh linear_eval_tune_im1000.sh
