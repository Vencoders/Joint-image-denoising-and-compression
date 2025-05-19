# JOINT END-TO-END IMAGE COMPRESSION AND DENOISING

This repository contains the implementation for the paper "JOINT END-TO-END IMAGE COMPRESSION AND DENOISING: LEVERAGING CONTRASTIVE LEARNING AND MULTI-SCALE SELF-ONNS"（IEEE ICIP 2024）.

## Paper
The paper is available on (https://arxiv.org/abs/2402.05582). 

## **Environment**

* NVIDIA-SMI 550.90.07
  
* CUDA Version: 12.4
  
* GPU: GeForce RTX 4090

**Dependencies：**

* pip install -e .
* pip install pyyaml
* pip install opencv-python
* pip install tensorboard
* pip install imagesize
* pip install image_slicer
* pip install h5py
* pip install .

## Datasets

We utilize the Flicker 2W dataset for training and validation. All trained models are evaluated on the Kodak and CLIC datasets, which are commonly used for image processing methods.

## Training
```bash
cd codes
OMP_NUM_THREADS=4 python train.py -opt ./conf/train/<xxx>.yml
```

## Testing
```bash
cd codes
OMP_NUM_THREADS=4 python test.py -opt ./conf/test/<xxx>.yml
```

## Project information
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No [101022466], and from the NSF-Business Finland Center for Big Learning (CBL), Advanced Machine Learning for Industrial Applications (AMaLIA) under Grant 97/31/2023.


## Author

Yuxin Xie {202212210053@nuist.edu.cn}

Li Yu {li.yu@nuist.edu.cn}

Farhad Pakdaman {farhad.pakdaman@tuni.fi)}

Moncef Gabbouj {moncef.gabbouj@tuni.fi}
