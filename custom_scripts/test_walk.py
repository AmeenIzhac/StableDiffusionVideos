import sys
import os
from txt2video import *
sys.path.append('stable-diffusion-2/optimizedSD')

image_args = ImageArgs()
video_args = VideoArgs()
path_args = PathArgs()
video_path = 'outputs/videos'

model_state = load_model(path_args, optimized=True)

#video_args.prompts = ["a very old 19th century phone", "an old 1930s phone", "a fix 1980s phone", "a 1990s portable phone", "a 2000s nokia cellphone", "a modern iphone"]
video_args.prompts = ["a savana landscape, in africa, with wild animals, beautiful photography, artstation 4k", 
                        "a jungle landscape, in africa, rainforest with wild animals, beautiful photography, artstation 4k"]


image_args.steps = 40
image_args.W = 768
image_args.H = 512
image_args.scale = 10
video_args.upscale = True

video_args.frames = 30
video_args.fps = 24
video_args.exp = 2
video_args.video_name = "africa_many_interp.mp4"
path_args.video_path = os.path.join(video_path, video_args.video_name)
generate_walk_video(image_args, video_args, path_args, model_state)
