<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/serve-tapnet/assets/115161827/967a413a-afb9-4051-afe7-ff740bea1bf5" />
  
# # CoTracker object tracking

<p align="center">
  <a href="#Original-work">Original-work</a> •
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Demo">Demo</a> 
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/co-tracker/supervisely_integration/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/co-tracker/supervisely_integration/serve)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/co-tracker/supervisely_integration/serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/co-tracker/supervisely_integration/serve.png)](https://supervise.ly)

</div>

# Original work

Original work is available here: [**paper**](https://arXiv:2307.07635); [**code**](https://github.com/facebookresearch/co-tracker); [**project**](https://co-tracker.github.io/)

> This architecture is based on several ideas from the optical flow and tracking literature, and combines them in a new, flexible and powerful design. It is based on a transformer network that models the correlation of different points in time via specialised attention layers.
> 
> The transformer is designed to update iteratively an estimate of several trajectories. It can be applied in a sliding-window manner to very long videos, for which we engineer an unrolled training loop. It compares favourably against state-of-the-art point tracking methods, both in terms of efficiency and accuracy. 

<img src="https://github.com/supervisely-ecosystem/co-tracker/assets/119248312/0710ae69-3140-42e4-a3d0-f3cddf08bfa1" />

### Points on a uniform grid

> We track points sampled on a regular grid starting from the initial video frame. The colors represent the object (magenta) and the background (cyan).

https://user-images.githubusercontent.com/119248312/d4cbc02e-fd74-4492-b0c0-6dc476df1677.mp4

### Individual points

> We track the same queried point with different methods and visualize its trajectory using color encoding based on time. The red cross (❌) indicates the ground truth point coordinates.

https://user-images.githubusercontent.com/119248312/01cdf2d4-4816-4ec5-a8ec-fe4610839792.mp4

# Overview

This app is an integration of CoTracker model, which is a NN-assisted interactive object tracking model. CoTracker is a fast transformer-based model that can track any point in a video. It brings to tracking some of the benefits of Optical Flow.

#### This application allows you to track the following 4 types of figures:

- Point

- Polyline

- Polygon

- Keypoints

# How to Run

0. Run the application from Ecosystem

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/co-tracker/supervisely_integration/serve" src="https://github.com/supervisely-ecosystem/MixFormer/assets/119248312/e74e2bd9-f915-48b1-bb97-ee808326dff5" width="500px" style='padding-bottom: 20px'/> 

1. Open Video Labeling interface

2. Configure tracking settings

3. Press `Track` button

4. After finishing working with the app, stop the app session manually in the `App sessions` tab

https://user-images.githubusercontent.com/115161827/237701946-2ef6b5bd-3473-4df6-bf7d-33539377c429.mp4

#### You can also use this app to track keypoints. This app can track keypoints graph of any shape and number of points.

1. Open your video project, select suitable frame and click on "Screenshot" button in the upper right corner:

https://user-images.githubusercontent.com/91027877/238152827-1a6fcc7b-7d68-4168-86af-7406d6255d9c.mp4

2. Create keypoints class based on your screenshot:

https://user-images.githubusercontent.com/91027877/238153794-43870be8-37bd-434a-bdf7-536da5267602.mp4

3. Go back to video, set your recently created keypoints graph on target object, select number of frames to be tracked and click on "Track" button:

https://user-images.githubusercontent.com/91027877/238153954-6364579b-2dff-49c4-b4da-35d4ea0e9ce9.mp4

You can change visualization settings of your keypoints graph in right sidebar:

https://user-images.githubusercontent.com/91027877/238154341-ed9acea5-2693-421d-a673-a6f4ab8f515a.mp4

# Demo


