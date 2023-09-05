# CoTracker: It is Better to Track Together

**[Meta AI Research, GenAI](https://ai.facebook.com/research/)**; **[University of Oxford, VGG](https://www.robots.ox.ac.uk/~vgg/)**

[Nikita Karaev](https://nikitakaraevv.github.io/), [Ignacio Rocco](https://www.irocco.info/), [Benjamin Graham](https://ai.facebook.com/people/benjamin-graham/), [Natalia Neverova](https://nneverova.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), [Christian Rupprecht](https://chrirupp.github.io/)

[[`Paper`](https://arxiv.org/abs/2307.07635)] [[`Project`](https://co-tracker.github.io/)] [[`BibTeX`](#citing-cotracker)]

<a target="_blank" href="https://colab.research.google.com/github/facebookresearch/co-tracker/blob/main/notebooks/demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a href="https://huggingface.co/spaces/facebook/cotracker">
  <img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>

<img width="500" src="./assets/bmx-bumps.gif" />

**CoTracker** is a fast transformer-based model that can track any point in a video. It brings to tracking some of the benefits of Optical Flow.
 
CoTracker can track:
- **Every pixel** in a video
- Points sampled on a regular grid on any video frame 
- Manually selected points

Try these tracking modes for yourself with our [Colab demo](https://colab.research.google.com/github/facebookresearch/co-tracker/blob/master/notebooks/demo.ipynb) or in the [Hugging Face Space](https://huggingface.co/spaces/facebook/cotracker).

### Update: September 5, 2023 
📣 You can now run our Gradio demo [locally](./gradio_demo/app.py)!

## Installation Instructions
Ensure you have both PyTorch and TorchVision installed on your system. Follow the instructions [here](https://pytorch.org/get-started/locally/) for the installation. We strongly recommend installing both PyTorch and TorchVision with CUDA support.

### Pretrained models via PyTorch Hub
The easiest way to use CoTracker is to load a pretrained model from torch.hub:
```
pip install einops timm tqdm
```
```
import torch
import timm
import einops
import tqdm

cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker_w8")
```
Another option is to install it from this gihub repo. That's the best way if you need to run our demo or evaluate / train CoTracker:
### Steps to Install CoTracker and its dependencies:
```
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
pip install -e .
pip install opencv-python einops timm matplotlib moviepy flow_vis 
```


### Download Model Weights:
```
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth
wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_12.pth
wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_8_wind_16.pth
cd ..
```


## Usage:
We offer a number of ways to interact with CoTracker:
1. A demo is available in the [`facebook/cotracker` Hugging Face Space](https://huggingface.co/spaces/facebook/cotracker).
2. You can run the extended demo in Colab:
[Colab notebook](https://colab.research.google.com/github/facebookresearch/co-tracker/blob/master/notebooks/demo.ipynb)
3. You can use the gradio demo locally by running [`python -m gradio_demo.app`](./gradio_demo/app.py) after installing the required packages: ```pip install -r gradio_demo/requirements.txt```.
4. You can play with CoTracker by running the Jupyter notebook located at [`notebooks/demo.ipynb`](./notebooks/demo.ipynb) locally (if you have a GPU).
5. Finally, you can run a local demo with 10*10 points sampled on a grid on the first frame of a video:
```
python demo.py --grid_size 10
```

## Evaluation
To reproduce the results presented in the paper, download the following datasets:
- [TAP-Vid](https://github.com/deepmind/tapnet)
- [BADJA](https://github.com/benjiebob/BADJA)
- [ZJU-Mocap (FastCapture)](https://arxiv.org/abs/2303.11898)

And install the necessary dependencies:
```
pip install hydra-core==1.1.0 mediapy 
```
Then, execute the following command to evaluate on BADJA:
```
python ./cotracker/evaluation/evaluate.py --config-name eval_badja exp_dir=./eval_outputs dataset_root=your/badja/path
```
By default, evaluation will be slow since it is done for one target point at a time, which ensures robustness and fairness, as described in the paper.

## Training
To train the CoTracker as described in our paper, you first need to generate annotations for [Google Kubric](https://github.com/google-research/kubric) MOVI-f dataset.  Instructions for annotation generation can be found [here](https://github.com/deepmind/tapnet).

Once you have the annotated dataset, you need to make sure you followed the steps for evaluation setup and install the training dependencies:
```
pip install pytorch_lightning==1.6.0 tensorboard
```
Now you can launch training on Kubric. Our model was trained for 50000 iterations on 32 GPUs (4 nodes with 8 GPUs).
Modify *dataset_root* and *ckpt_path* accordingly before running this command:
```
python train.py --batch_size 1 --num_workers 28 \
--num_steps 50000 --ckpt_path ./ --dataset_root ./datasets --model_name cotracker \
--save_freq 200 --sequence_len 24 --eval_datasets tapvid_davis_first badja \
--traj_per_sample 256 --sliding_window_len 8 --updateformer_space_depth 6 --updateformer_time_depth 6 \
--save_every_n_epoch 10 --evaluate_every_n_epoch 10 --model_stride 4
```

## License
The majority of CoTracker is licensed under CC-BY-NC, however portions of the project are available under separate license terms: Particle Video Revisited is licensed under the MIT license, TAP-Vid is licensed under the Apache 2.0 license.

## Acknowledgments
We would like to thank [PIPs](https://github.com/aharley/pips) and [TAP-Vid](https://github.com/deepmind/tapnet) for publicly releasing their code and data. We also want to thank [Luke Melas-Kyriazi](https://lukemelas.github.io/) for proofreading the paper, [Jianyuan Wang](https://jytime.github.io/), [Roman Shapovalov](https://shapovalov.ro/) and [Adam W. Harley](https://adamharley.com/) for the insightful discussions.

## Citing CoTracker
If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:
```
@article{karaev2023cotracker,
  title={CoTracker: It is Better to Track Together},
  author={Nikita Karaev and Ignacio Rocco and Benjamin Graham and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  journal={arXiv:2307.07635},
  year={2023}
}
```
