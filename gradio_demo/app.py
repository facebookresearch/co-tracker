import os
import torch
import timm
import einops
import tqdm
import cv2
import gradio as gr

from cotracker.utils.visualizer import Visualizer, read_video_from_path


def cotracker_demo(
    input_video, 
    grid_size: int = 10, 
    grid_query_frame: int = 0, 
    backward_tracking: bool = False,
    tracks_leave_trace: bool = False
    ):
    load_video = read_video_from_path(input_video)

    grid_query_frame = min(len(load_video)-1, grid_query_frame)
    load_video = torch.from_numpy(load_video).permute(0, 3, 1, 2)[None].float()


    model = torch.hub.load("facebookresearch/co-tracker", "cotracker_w8")
    if torch.cuda.is_available():
        model = model.cuda()
        load_video = load_video.cuda()
    pred_tracks, pred_visibility = model(
        load_video, 
        grid_size=grid_size, 
        grid_query_frame=grid_query_frame, 
        backward_tracking=backward_tracking
        )
    linewidth = 2
    if grid_size < 10:
        linewidth = 4
    elif grid_size < 20:
        linewidth = 3
        
    vis = Visualizer(
        save_dir=os.path.join(os.path.dirname(__file__), "results"),
        grayscale=False,
        pad_value=100,
        fps=10,
        linewidth=linewidth,
        show_first_frame=5,
        tracks_leave_trace= -1 if tracks_leave_trace else 0,
    )
    import time

    def current_milli_time():
        return round(time.time() * 1000)

    filename = str(current_milli_time())
    vis.visualize(
        load_video,
        tracks=pred_tracks, 
        visibility=pred_visibility,
        filename=filename,
        query_frame=grid_query_frame,
        )
    return os.path.join(
        os.path.dirname(__file__), "results", f"{filename}_pred_track.mp4"
    )


app = gr.Interface(
    title = "🎨 CoTracker: It is Better to Track Together",
    description = "<div style='text-align: left;'> \
    <p>Welcome to <a href='http://co-tracker.github.io' target='_blank'>CoTracker</a>! This space demonstrates point (pixel) tracking in videos. \
    Points are sampled on a regular grid and are tracked jointly. </p> \
    <p> To get started, simply upload your <b>.mp4</b> video in landscape orientation or click on one of the example videos to load them. The shorter the video, the faster the processing. We recommend submitting short videos of length <b>2-7 seconds</b>.</p> \
    <ul style='display: inline-block; text-align: left;'> \
        <li>The total number of grid points is the square of <b>Grid Size</b>.</li> \
        <li>To specify the starting frame for tracking, adjust <b>Grid Query Frame</b>. Tracks will be visualized only after the selected frame.</li> \
        <li>Use <b>Backward Tracking</b> to track points from the selected frame in both directions.</li> \
        <li>Check <b>Visualize Track Traces</b> to visualize traces of all the tracked points. </li> \
    </ul> \
    <p style='text-align: left'>For more details, check out our <a href='https://github.com/facebookresearch/co-tracker' target='_blank'>GitHub Repo</a> ⭐</p> \
    </div>",
           
    fn=cotracker_demo,
    inputs=[
        gr.Video(type="file", label="Input video", interactive=True),
        gr.Slider(minimum=1, maximum=30, step=1, value=10, label="Grid Size"),
        gr.Slider(minimum=0, maximum=30, step=1, default=0, label="Grid Query Frame"),
        gr.Checkbox(label="Backward Tracking"),
        gr.Checkbox(label="Visualize Track Traces"),
    ],
    outputs=gr.Video(label="Video with predicted tracks"),
    examples=[
        [ "./assets/apple.mp4", 20, 0, False, False ],
        [ "./assets/apple.mp4", 10, 30, True, False ],
    ],
    cache_examples=False
)
app.launch(share=False)
