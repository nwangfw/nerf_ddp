# nerf_ddp


This repo try to increase NeRF training by applying Pytorch DDP.

There are two implementations.


(1) tiny_nerf_ddp.py modified code from https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX and add Pytorch Data Parallel.

The author's repo is https://github.com/krrish94/nerf-pytorch


launch command if you have one machine with 8 GPUs
```
python tiny_nerf_pytorch_ddp.py -n 1 -g 8 -i 0
```

