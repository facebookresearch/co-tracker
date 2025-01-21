# CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos

**[Meta AI Research, GenAI](https://ai.facebook.com/research/)**; **[University of Oxford, VGG](https://www.robots.ox.ac.uk/~vgg/)**

[Nikita Karaev](https://nikitakaraevv.github.io/), [Iurii Makarov](https://linkedin.com/in/lvoursl), [Jianyuan Wang](https://jytime.github.io/), [Ignacio Rocco](https://www.irocco.info/), [Benjamin Graham](https://ai.facebook.com/people/benjamin-graham/), [Natalia Neverova](https://nneverova.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), [Christian Rupprecht](https://chrirupp.github.io/)

### [Project Page](https://cotracker3.github.io/) | [Paper #1](https://arxiv.org/abs/2307.07635) | [Paper #2](https://arxiv.org/abs/2410.11831) |  [X Thread](https://twitter.com/n_karaev/status/1742638906355470772) | [BibTeX](#citing-cotracker)

<a target="_blank" href="https://colab.research.google.com/github/facebookresearch/co-tracker/blob/main/notebooks/demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a href="https://huggingface.co/spaces/facebook/cotracker">
  <img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>

<img width="1100" src="./assets/teaser.png" />

**CoTracker** is a fast transformer-based model that can track any point in a video. It brings to tracking some of the benefits of Optical Flow.

CoTracker can track:

- **Any pixel** in a video
- A **quasi-dense** set of pixels together
- Points can be manually selected or sampled on a grid in any video frame

Try these tracking modes for yourself with our [Colab demo](https://colab.research.google.com/github/facebookresearch/co-tracker/blob/master/notebooks/demo.ipynb) or in the [Hugging Face Space 🤗](https://huggingface.co/spaces/facebook/cotracker).

**Updates:**

- [January 21, 2025] 📦 Kubric Dataset used for CoTracker3 now available! This dataset contains **6,000 high-resolution sequences** (512×512px, 120 frames) with slight camera motion, rendered using the Kubric engine. Check it out on [Hugging Face Dataset](https://huggingface.co/datasets/facebook/CoTracker3_Kubric).

- [October 15, 2024] 📣 We're releasing CoTracker3! State-of-the-art point tracking with a lightweight architecture trained with 1000x less data than previous top-performing models. Code for baseline models and the pseudo-labeling pipeline are available in the repo, as well as model checkpoints. Check out our [paper](https://arxiv.org/abs/2410.11831) for more details.

- [September 25, 2024]  CoTracker2.1 is now available! This model has better performance on TAP-Vid benchmarks and follows the architecture of the original CoTracker. Try it out!

- [June 14, 2024]  We have released the code for [VGGSfM](https://github.com/facebookresearch/vggsfm), a model for recovering camera poses and 3D structure from any image sequences based on point tracking! VGGSfM is the first fully differentiable SfM framework that unlocks scalability and outperforms conventional SfM methods on standard benchmarks. 

- [December 27, 2023]  CoTracker2 is now available! It can now track many more (up to **265*265**!) points jointly and it has a cleaner and more memory-efficient implementation. It also supports online processing. See the [updated paper](https://arxiv.org/abs/2307.07635) for more details. The old version remains available [here](https://github.com/facebookresearch/co-tracker/tree/8d364031971f6b3efec945dd15c468a183e58212).

- [September 5, 2023] You can now run our Gradio demo [locally](./gradio_demo/app.py).

## Quick start
The easiest way to use CoTracker is to load a pretrained model from `torch.hub`:

### Offline mode: 
```pip install imageio[ffmpeg]```, then:
```python
import torch
# Download the video
url = 'https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4'

import imageio.v3 as iio
frames = iio.imread(url, plugin="FFMPEG")  # plugin="pyav"

device = 'cuda'
grid_size = 10
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

# Run Offline CoTracker:
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1
```
### Online mode: 
```python
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)

# Run Online CoTracker, the same model with a different API:
# Initialize online processing
cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)  

# Process the video
for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
    pred_tracks, pred_visibility = cotracker(
        video_chunk=video[:, ind : ind + cotracker.step * 2]
    )  # B T N 2,  B T N 1
```
Online processing is more memory-efficient and allows for the processing of longer videos. However, in the example provided above, the video length is known! See [the online demo](./online_demo.py) for an example of tracking from an online stream with an unknown video length.

### Visualize predicted tracks: 
After [installing](#installation-instructions) CoTracker, you can visualize tracks with:
```python
from cotracker.utils.visualizer import Visualizer

vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)
```

We offer a number of other ways to interact with CoTracker:

1. Interactive Gradio demo:
   - A demo is available in the [`facebook/cotracker` Hugging Face Space 🤗](https://huggingface.co/spaces/facebook/cotracker).
   - You can use the gradio demo locally by running [`python -m gradio_demo.app`](./gradio_demo/app.py) after installing the required packages: `pip install -r gradio_demo/requirements.txt`.
2. Jupyter notebook:
   - You can run the notebook in
   [Google Colab](https://colab.research.google.com/github/facebookresearch/co-tracker/blob/master/notebooks/demo.ipynb).
   - Or explore the notebook located at [`notebooks/demo.ipynb`](./notebooks/demo.ipynb). 
2. You can [install](#installation-instructions) CoTracker _locally_ and then:
   - Run an *offline* demo with 10 ⨉ 10 points sampled on a grid on the first frame of a video (results will be saved to `./saved_videos/demo.mp4`)):

     ```bash
     python demo.py --grid_size 10
     ```
    - Run an *online* demo:

      ```bash
      python online_demo.py
      ```

A GPU is strongly recommended for using CoTracker locally.

<img width="500" src="./assets/bmx-bumps.gif" />


## Installation Instructions
You can use a Pretrained Model via PyTorch Hub, as described above, or install CoTracker from this GitHub repo.
This is the best way if you need to run our local demo or evaluate/train CoTracker.

Ensure you have both _PyTorch_ and _TorchVision_ installed on your system. Follow the instructions [here](https://pytorch.org/get-started/locally/) for the installation.
We strongly recommend installing both PyTorch and TorchVision with CUDA support, although for small tasks CoTracker can be run on CPU.




### Install a Development Version

```bash
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
```

You can manually download all CoTracker3 checkpoints (baseline and scaled models, as well as single and sliding window architectures) from the links below and place them in the `checkpoints` folder as follows:

```bash
mkdir -p checkpoints
cd checkpoints
# download the online (multi window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
# download the offline (single window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
cd ..
```
You can also download CoTracker3 checkpoints trained only on Kubric:
```bash
# download the online (sliding window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/baseline_online.pth
# download the offline (single window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/baseline_offline.pth
```
For old checkpoints, see [this section](#previous-version).

## Evaluation

To reproduce the results presented in the paper, download the following datasets:

- [TAP-Vid](https://github.com/deepmind/tapnet)
- [Dynamic Replica](https://dynamic-stereo.github.io/)

And install the necessary dependencies:

```bash
pip install hydra-core==1.1.0 mediapy
```

Then, execute the following command to evaluate the online model on TAP-Vid DAVIS:

```bash
python ./cotracker/evaluation/evaluate.py --config-name eval_tapvid_davis_first exp_dir=./eval_outputs dataset_root=your/tapvid/path
```
And the offline model:
```bash
python ./cotracker/evaluation/evaluate.py --config-name eval_tapvid_davis_first exp_dir=./eval_outputs dataset_root=/fsx-repligen/shared/datasets/tapvid offline_model=True window_len=60 checkpoint=./checkpoints/scaled_offline.pth
```
We run evaluations jointly on all the target points at a time for faster inference. With such evaluations, the numbers are similar to those presented in the paper. If you want to reproduce the exact numbers from the paper, add the flag `single_point=True`. 

These are the numbers that you should be able to reproduce using the released checkpoint and the current version of the codebase:
|  | Kinetics, $\delta_\text{avg}^\text{vis}$ | DAVIS, $\delta_\text{avg}^\text{vis}$ |  RoboTAP, $\delta_\text{avg}^\text{vis}$ | RGB-S, $\delta_\text{avg}^\text{vis}$| 
| :---: |:---: | :---: | :---: | :---: |
| CoTracker2, 27.12.23 | 61.8 | 74.6 | 69.6 | 73.4 | 
| CoTracker2.1, 25.09.24 | 63 | 76.1 | 70.6 | 79.6 | 
| CoTracker3 offline, 15.10.24 | 67.8 | **76.9** | 78.0 | **85.0** | 
| CoTracker3 online, 15.10.24 | **68.3** | 76.7 | **78.8** | 82.7  | 


## Training

### Baseline
To train the CoTracker as described in our paper, you first need to generate annotations for [Google Kubric](https://github.com/google-research/kubric) MOVI-f dataset.
Instructions for annotation generation can be found [here](https://github.com/deepmind/tapnet).
You can also find a discussion on dataset generation in [this issue](https://github.com/facebookresearch/co-tracker/issues/8).

Once you have the annotated dataset, you need to make sure you followed the steps for evaluation setup and install the training dependencies:

```bash
pip install pip==24.0
pip install pytorch_lightning==1.6.0 tensorboard opencv-python
```

Now you can launch training on Kubric.
Our model was trained for 50000 iterations on 32 GPUs (4 nodes with 8 GPUs). 
Modify _dataset_root_ and _ckpt_path_ accordingly before running this command. For training on 4 nodes, add `--num_nodes 4`. 

Here is an example of how to launch training of the online model on Kubric:
```bash
 python train_on_kubric.py --batch_size 1 --num_steps 50000 \
 --ckpt_path ./ --model_name cotracker_three --save_freq 200 --sequence_len 64 \
  --eval_datasets tapvid_davis_first tapvid_stacking --traj_per_sample 384 \
  --sliding_window_len 16 --train_datasets kubric --save_every_n_epoch 5 \
  --evaluate_every_n_epoch 5 --model_stride 4 --dataset_root ${path_to_your_dataset} \
   --num_nodes 4 --num_virtual_tracks 64 --mixed_precision --corr_radius 3 \ 
   --wdecay 0.0005 --linear_layer_for_vis_conf --validate_at_start --add_huber_loss
```

Training the offline model on Kubric:
```bash
python train_on_kubric.py --batch_size 1 --num_steps 50000 \
 --ckpt_path ./ --model_name cotracker_three --save_freq 200 --sequence_len 60 \
 --eval_datasets tapvid_davis_first tapvid_stacking --traj_per_sample 512 \
 --sliding_window_len 60 --train_datasets kubric --save_every_n_epoch 5 \
 --evaluate_every_n_epoch 5 --model_stride 4 --dataset_root ${path_to_your_dataset} \
 --num_nodes 4 --num_virtual_tracks 64 --mixed_precision --offline_model \
 --random_frame_rate --query_sampling_method random --corr_radius 3 \
 --wdecay 0.0005 --random_seq_len --linear_layer_for_vis_conf \
 --validate_at_start --add_huber_loss
```

### Fine-tuning with pseudo labels
In order to launch training with pseudo-labelling, you need to collect your own dataset of real videos. There is a sample class available in [`cotracker/datasets/real_dataset.py`](./cotracker/datasets/real_dataset.py) with keyword-based filtering that we used for training. Your class should implement loading a video and storing it in the `CoTrackerData` class as a field, while pseudo labels will be generated in `train_on_real_data.py`.

You should have an existing Kubric-trained model for fine-tuning with pseudo labels. Here is an example of how you can launch fine-tuning of the online model:
```bash
python ./train_on_real_data.py --batch_size 1 --num_steps 15000 \
 --ckpt_path ./ --model_name cotracker_three --save_freq 200 --sequence_len 64 \
 --eval_datasets tapvid_stacking tapvid_davis_first --traj_per_sample 384 \
 --save_every_n_epoch 15 --evaluate_every_n_epoch 15 --model_stride 4 \
 --dataset_root ${path_to_your_dataset} --num_nodes 4 --real_data_splits 0 \
 --num_virtual_tracks 64 --mixed_precision --random_frame_rate \
 --restore_ckpt ./checkpoints/baseline_online.pth \
 --lr 0.00005 --real_data_filter_sift --validate_at_start \
 --sliding_window_len 16 --limit_samples 15000

```
And the offline model:
```bash
python train_on_real_data.py --batch_size 1 --num_steps 15000 \
 --ckpt_path ./ --model_name cotracker_three --save_freq 200 --sequence_len 80 \
 --eval_datasets tapvid_stacking tapvid_davis_first --traj_per_sample 384 --save_every_n_epoch 15 \
 --evaluate_every_n_epoch 15 --model_stride 4 --dataset_root ${path_to_your_dataset} \
  --num_nodes 4 --real_data_splits 0 --num_virtual_tracks 64 --mixed_precision \
  --random_frame_rate --restore_ckpt ./checkpoints/baseline_offline.pth --lr 0.00005 \
  --real_data_filter_sift --validate_at_start --offline_model --limit_samples 15000
```



## Development

### Building the documentation

To build CoTracker documentation, first install the dependencies:

```bash
pip install sphinx
pip install sphinxcontrib-bibtex
```

Then you can use this command to generate the documentation in the `docs/_build/html` folder:

```bash
make -C docs html
```


## Previous versions
### CoTracker v2
You could use CoTracker v2 with torch.hub in both offline and online modes.
#### Offline mode: 
```pip install imageio[ffmpeg]```, then:
```python
import torch
# Download the video
url = 'https://github.com/facebookresearch/co-tracker/blob/main/assets/apple.mp4'

import imageio.v3 as iio
frames = iio.imread(url, plugin="FFMPEG")  # plugin="pyav"

device = 'cuda'
grid_size = 10
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

# Run Offline CoTracker:
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1
```
#### Online mode: 
```python
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(device)

# Run Online CoTracker, the same model with a different API:
# Initialize online processing
cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)  

# Process the video
for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
    pred_tracks, pred_visibility = cotracker(
        video_chunk=video[:, ind : ind + cotracker.step * 2]
    )  # B T N 2,  B T N 1
```

Checkpoint for v2 could be downloaded with the following command:
```bash
wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth
```

### CoTracker v1
It is directly available via pytorch hub:
```python
import torch
import einops
import timm
import tqdm

cotracker = torch.hub.load("facebookresearch/co-tracker:v1.0", "cotracker_w8")
```
The old version of the code is available [here](https://github.com/facebookresearch/co-tracker/tree/8d364031971f6b3efec945dd15c468a183e58212).
You can also download the corresponding checkpoints:
```bash
wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth
wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_12.pth
wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_8_wind_16.pth
```

## License

The majority of CoTracker is licensed under CC-BY-NC, however portions of the project are available under separate license terms: Particle Video Revisited is licensed under the MIT license, TAP-Vid and LocoTrack are licensed under the Apache 2.0 license.

## Acknowledgments

We would like to thank [PIPs](https://github.com/aharley/pips), [TAP-Vid](https://github.com/deepmind/tapnet), [LocoTrack](https://github.com/cvlab-kaist/locotrack) for publicly releasing their code and data. We also want to thank [Luke Melas-Kyriazi](https://lukemelas.github.io/) for proofreading the paper, [Jianyuan Wang](https://jytime.github.io/), [Roman Shapovalov](https://shapovalov.ro/) and [Adam W. Harley](https://adamharley.com/) for the insightful discussions.

## Citing CoTracker

If you find our repository useful, please consider giving it a star ⭐ and citing our research papers in your work:
```bibtex
@inproceedings{karaev23cotracker,
  title     = {CoTracker: It is Better to Track Together},
  author    = {Nikita Karaev and Ignacio Rocco and Benjamin Graham and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  booktitle = {Proc. {ECCV}},
  year      = {2024}
}
```
```bibtex
@inproceedings{karaev24cotracker3,
  title     = {CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos},
  author    = {Nikita Karaev and Iurii Makarov and Jianyuan Wang and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  booktitle = {Proc. {arXiv:2410.11831}},
  year      = {2024}
}
```
