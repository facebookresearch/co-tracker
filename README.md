# CoTracker

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**; **[University of Oxford, VGG](https://www.robots.ox.ac.uk/~vgg/)**

[Nikita Karaev](https://nikitakaraevv.github.io/), [Ignacio Rocco](https://www.irocco.info/), [Benjamin Graham](https://ai.facebook.com/people/benjamin-graham/), [Natalia Neverova](https://nneverova.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), [Christian Rupprecht](https://chrirupp.github.io/)

[[`Paper`]()] [[`Project`](https://co-tracker.github.io/)] [[`BibTeX`](#citing-cotracker)]

<!-- ![nikita-reading](https://user-images.githubusercontent.com/37815420/236242052-e72d5605-1ab2-426c-ae8d-5c8a86d5252c.gif) -->

**DynamicStereo** is a transformer-based architecture for temporally consistent depth estimation from stereo videos. It has been trained on a combination of two datasets: [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and **Dynamic Replica** that we present below.


## Installation
Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

## Install CoTracker and dependencies:
```
git clone git@github.com:fairinternal/cotracker co-tracker
cd co-tracker
pip install -e .
pip install opencv-python einops timm matplotlib moviepy flow_vis
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


## License


## Citing CoTracker
If you find this repository useful, please consider giving it a star ⭐ and citing it:
```
@article{karaev2023cotracker,
  title={CoTracker: It is Better to Track Together},
  author={Nikita Karaev and Ignacio Rocco and Benjamin Graham and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  journal={arxiv},
  year={2023}
}
```