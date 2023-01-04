import sys
import os
from txt2video import *
from sd_video_utils import *
sys.path.append('stable-diffusion-2/optimizedSD')
sys.path.append('ECCV2022-RIFE')

image_args = ImageArgs()
video_args = VideoArgs()
path_args = PathArgs()
video_path = 'outputs/videos'

model_state = load_model(path_args, optimized=True)

video_args.strength = 0.44
image_args.steps = 40
image_args.W = 768
image_args.H = 512
image_args.scale = 10
video_args.zoom = 1.06
video_args.x = -3.0
video_args.y = -2.0
video_args.upscale = True
video_args.color_match = True

video_args.prompts = ["a surrealist cubist painting about a child dream, cartoony sun and moon smiling, comforting colors, 19th century child art vibe, artstation 4k"]

video_args.fps = 15
video_args.frames = 10
video_args.interp_exp = 3
video_args.video_name = "child_dream_5.mp4"
path_args.video_path = os.path.join(video_path, video_args.video_name)
path_args.rife_path = './ECCV2022-RIFE/train_log'
generate_video(image_args, video_args, path_args, model_state)