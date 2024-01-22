# CoTracker: It is Better to Track Together

**[Meta AI Research, GenAI](https://ai.facebook.com/research/)**; **[University of Oxford, VGG](https://www.robots.ox.ac.uk/~vgg/)**

[Nikita Karaev](https://nikitakaraevv.github.io/), [Ignacio Rocco](https://www.irocco.info/), [Benjamin Graham](https://ai.facebook.com/people/benjamin-graham/), [Natalia Neverova](https://nneverova.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), [Christian Rupprecht](https://chrirupp.github.io/)

### [Project Page](https://co-tracker.github.io/) | [Paper](https://arxiv.org/abs/2307.07635) |  [X Thread](https://twitter.com/n_karaev/status/1742638906355470772) | [BibTeX](#citing-cotracker)

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

- [December 27, 2023] 📣 CoTracker2 is now available! It can now track many more (up to **265*265**!) points jointly and it has a cleaner and more memory-efficient implementation. It also supports online processing. See the [updated paper](https://arxiv.org/abs/2307.07635) for more details. The old version remains available [here](https://github.com/facebookresearch/co-tracker/tree/8d364031971f6b3efec945dd15c468a183e58212).

- [September 5, 2023] 📣 You can now run our Gradio demo [locally](./gradio_demo/app.py)!

## Quick start
The easiest way to use CoTracker is to load a pretrained model from `torch.hub`:

### Offline mode: 
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
### Online mode: 
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
Online processing is more memory-efficient and allows for the processing of longer videos. However, in the example provided above, the video length is known! See [the online demo](./online_demo.py) for an example of tracking from an online stream with an unknown video length.

### Visualize predicted tracks: 
```pip install matplotlib```, then:
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

You can manually download the CoTracker2 checkpoint from the links below and place it in the `checkpoints` folder as follows:

```bash
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth
cd ..
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

Then, execute the following command to evaluate on TAP-Vid DAVIS:

```bash
python ./cotracker/evaluation/evaluate.py --config-name eval_tapvid_davis_first exp_dir=./eval_outputs dataset_root=your/tapvid/path
```

By default, evaluation will be slow since it is done for one target point at a time, which ensures robustness and fairness, as described in the paper.

We have fixed some bugs and retrained the model after updating the paper. These are the numbers that you should be able to reproduce using the released checkpoint and the current version of the codebase:
|  | DAVIS First, AJ | DAVIS First, $\delta_\text{avg}^\text{vis}$ | DAVIS First, OA | DAVIS Strided, AJ | DAVIS Strided, $\delta_\text{avg}^\text{vis}$ | DAVIS Strided, OA | DR, $\delta_\text{avg}$| DR, $\delta_\text{avg}^\text{vis}$| DR, $\delta_\text{avg}^\text{occ}$|
| :---: |:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CoTracker2, 27.12.23 | 60.9 | 75.4 | 88.4 | 65.1 | 79.0 | 89.4 | 61.4 | 68.4 | 38.2


## Training

To train the CoTracker as described in our paper, you first need to generate annotations for [Google Kubric](https://github.com/google-research/kubric) MOVI-f dataset.
Instructions for annotation generation can be found [here](https://github.com/deepmind/tapnet).
You can also find a discussion on dataset generation in [this issue](https://github.com/facebookresearch/co-tracker/issues/8).

Once you have the annotated dataset, you need to make sure you followed the steps for evaluation setup and install the training dependencies:

```bash
pip install pytorch_lightning==1.6.0 tensorboard
```

Now you can launch training on Kubric.
Our model was trained for 50000 iterations on 32 GPUs (4 nodes with 8 GPUs). 
Modify _dataset_root_ and _ckpt_path_ accordingly before running this command. For training on 4 nodes, add `--num_nodes 4`.

```bash
python train.py --batch_size 1 \
--num_steps 50000 --ckpt_path ./ --dataset_root ./datasets --model_name cotracker \
--save_freq 200 --sequence_len 24 --eval_datasets dynamic_replica tapvid_davis_first \
--traj_per_sample 768 --sliding_window_len 8 \
--num_virtual_tracks 64 --model_stride 4
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


## Previous version
You can use CoTracker v1 directly via pytorch hub:
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

The majority of CoTracker is licensed under CC-BY-NC, however portions of the project are available under separate license terms: Particle Video Revisited is licensed under the MIT license, TAP-Vid is licensed under the Apache 2.0 license.

## Acknowledgments

We would like to thank [PIPs](https://github.com/aharley/pips) and [TAP-Vid](https://github.com/deepmind/tapnet) for publicly releasing their code and data. We also want to thank [Luke Melas-Kyriazi](https://lukemelas.github.io/) for proofreading the paper, [Jianyuan Wang](https://jytime.github.io/), [Roman Shapovalov](https://shapovalov.ro/) and [Adam W. Harley](https://adamharley.com/) for the insightful discussions.

## Citing CoTracker

If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:

```bibtex
@article{karaev2023cotracker,
  title={CoTracker: It is Better to Track Together},
  author={Nikita Karaev and Ignacio Rocco and Benjamin Graham and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  journal={arXiv:2307.07635},
  year={2023}
}
```
