# CoTracker
## Installation
Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

## Install CoTracker and dependencies:
```
git clone git@github.com:fairinternal/cotracker co-tracker
cd co-tracker
pip install -e .
pip install opencv-python einops timm matplotlib moviepy

```


## Download model weights:
```
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth
wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_12.pth
wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_8_wind_16.pth
cd ..
```

## Run a demo on a CO3D apple:
```
python demo.py --grid_size 10
```

## Run the example notebook:
```
notebooks/demo.ipynb
```


## Evaluation
```
pip install hydra-core==1.1.0 mediapy tensorboard 
```
```
python ./cotracker/evaluation/evaluate.py --config-name eval_badja exp_dir=./eval_outputs dataset_root=./
```