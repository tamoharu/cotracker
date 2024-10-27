import torch

from cotracker.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def run():
    video_path = "./samples/example.mp4"
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    window_len = 60
    model = CoTrackerPredictor(
        checkpoint='./checkpoints/scaled_online.pth',
        v2=False,
        offline=True,
        window_len=window_len,
    )
    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)

    pred_tracks, pred_visibility = model(
        video,
        grid_size=10,
        grid_query_frame=0,
        # segm_mask=segm_mask
    )
    print("computed")

    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=0,
    )

if __name__ == "__main__":
    run()