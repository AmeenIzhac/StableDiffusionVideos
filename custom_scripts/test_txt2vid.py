import sys
from txt2video import *

#Placeholder "main" code
cfg_path = 'configs/stable-diffusion/v1-inference.yaml'
optimized_cfg_path = 'optimizedSD/v1-inference.yaml'
ckpt_path = 'models/ldm/stable-diffusion-v1/model.ckpt'
model_state = load_model(optimized_cfg_path, ckpt_path, optimized=True)

path_args = PathArgs()

image_args = ImageArgs()
video_args = VideoArgs()

video_args.prompts = [
    "a nice little path with trees, winter season, artstation",
    "a nice little path with trees, spring season, artstation",
    "a nice little path with trees, summer season, artstation",
    "a nice little path with trees, autumn season, artstation",
    "a nice little path with trees, winter season, artstation",
    ]

video_args.fps = 20
video_args.strength = 0.40
video_args.frames = 200
image_args.steps = 60
image_args.W = 512
image_args.H = 512
image_args.scale = 10
video_args.zoom = 1.008
video_args.x = -3
video_args.upscale = True
video_args.color_match = True
video_args.video_name = "the_4_seasons_2"

generate_video(image_args, video_args, path_args, model_state)
