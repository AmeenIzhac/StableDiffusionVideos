import sys
import os
from txt2video import *
from sd_video_utils import *

image_args = ImageArgs()
video_args = VideoArgs()
path_args = PathArgs()
video_path = 'outputs/demo_media'
path_args.image_path = 'outputs/demo_media'
path_args.rife_path = './ECCV2022-RIFE/train_log'

model_state = load_model(path_args, optimized=True)

image_args.steps = 60
image_args.W = 512
image_args.H = 512
image_args.scale = 10
video_args.upscale = True
video_args.color_match = True
video_args.interp_exp = 2
video_args.fps = 30
video_args.strength = 0.40


video_args.prompts = ["a 3d render of the green grass hills in scotland", "a digital art about timesquare in the future, greg rutkowski", "a comfortable child room with teddy bears and toys, artstation"]
video_args.frames = 30
video_args.video_name = "multiple_colors.mp4"
path_args.video_path = os.path.join(video_path, video_args.video_name)
generate_video(image_args, video_args, path_args, model_state)
