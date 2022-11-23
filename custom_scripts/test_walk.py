import sys
import os
from txt2video import *
sys.path.append('stable-diffusion-2/optimizedSD')

#Placeholder "main" code
cfg_path = 'stable-diffusion-2/configs/stable-diffusion/v1-inference.yaml'
optimized_cfg_path = 'stable-diffusion-2/optimizedSD/v1-inference.yaml'
ckpt_path = 'stable-diffusion-2/models/ldm/stable-diffusion-v1/model.ckpt'
model_state = load_model(optimized_cfg_path, ckpt_path, optimized=True)

path_args = PathArgs()
video_path = 'outputs/videos'

image_args = ImageArgs()
video_args = VideoArgs()

video_args.prompts = ["a picture of a cat"]

video_args.fps = 20
video_args.frames = 60
image_args.steps = 35
image_args.W = 512
image_args.H = 512
image_args.scale = 10
video_args.upscale = True
video_args.color_match = True
video_args.video_name = "first_latent_walk.mp4"

path_args.video_path = os.path.join(video_path, video_args.video_name)
generate_walk_video(image_args, video_args, path_args, model_state, latent_walk=True)

