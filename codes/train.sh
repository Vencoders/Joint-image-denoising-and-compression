#!/bin/bash

/root/anaconda3/envs/joint/bin/python /opt/data/private/xyx/DenoiseCompression/CompressAI/codes/train.py OMP_NUM_THREADS=4 -opt ./conf/train/multiscale-decomp_sidd_mse_q6.yml

