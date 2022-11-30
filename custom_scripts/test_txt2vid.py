import sys
import os
from txt2video import *
sys.path.append('stable-diffusion-2/optimizedSD')

#Placeholder "main" code
cfg_path = 'stable-diffusion-2/configs/stable-diffusion/v1-inference.yaml'
optimized_cfg_path = 'stable-diffusion-2/optimizedSD/v1-inference.yaml'
ckpt_path = 'stable-diffusion-2/models/ldm/stable-diffusion-v1/model.ckpt'
model_state = load_model(optimized_cfg_path, ckpt_path, optimized=True)

image_args = ImageArgs()
video_args = VideoArgs()

video_args.prompts = ["an alien spaceship in japanese traditional artstyle, vibrant colors, 4k"]

video_args.fps = 20
video_args.strength = 0.37
video_args.frames = 120
image_args.steps = 50
image_args.W = 768
image_args.H = 512
image_args.scale = 10
video_args.zoom = 1.02
video_args.x = 3.0
video_args.upscale = True
video_args.color_match = True

path_args = PathArgs()
video_path = 'outputs/videos'

video_args.video_name = "japanese_trad_spaceship.mp4"
path_args.video_path = os.path.join(video_path, video_args.video_name)

generate_video(image_args, video_args, path_args, model_state)
