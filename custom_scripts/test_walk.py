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

#video_args.prompts = ["a very old 19th century phone", "an old 1930s phone", "a fix 1980s phone", "a 1990s portable phone", "a 2000s nokia cellphone", "a modern iphone"]
video_args.prompts = ["a savana landscape, in africa, with wild animals, beautiful photography, artstation 4k", 
                        "a jungle landscape, in africa, rainforest with wild animals, beautiful photography, artstation 4k"]


image_args.steps = 40
image_args.W = 512
image_args.H = 512
image_args.scale = 10
video_args.upscale = True
video_args.seed = 76
video_args.frames = 60

video_args.fps = 60
video_args.inter_frames = 10
video_args.video_name = "africa_many_interp.mp4"
path_args.video_path = os.path.join(video_path, video_args.video_name)
generate_walk_video(image_args, video_args, path_args, model_state, n_noises=3)
