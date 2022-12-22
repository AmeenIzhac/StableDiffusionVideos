import sys
import os
from txt2video import *
from sd_video_utils import *
sys.path.append('stable-diffusion-2/optimizedSD')

#Placeholder "main" code
cfg_path = './stable-diffusion-2/configs/stable-diffusion/v1-inference.yaml'
optimized_cfg_path = './stable-diffusion-2/optimizedSD/v1-inference.yaml'
ckpt_path = './stable-diffusion-2/models/ldm/stable-diffusion-v1/model.ckpt'

image_args = ImageArgs()
video_args = VideoArgs()

video_args.prompts = ["a sad monet painting about a rainy street, high quality", "a very colorful and vibrant kandinsky painting, masterpiece", "the world of rainbows, rainbows everywhere", "black and white screenshot of a film noir depicting a man in a trench smoking a cigar", "a beautiful house decorated for christmas, artstation"]

video_args.fps = 20
video_args.strength = 0.40
video_args.frames = 400
image_args.steps = 50
image_args.W = 768
image_args.H = 512
image_args.scale = 10
video_args.zoom = 1.008
video_args.x = 6.0
video_args.y = 3.0
video_args.upscale = True
video_args.color_match = True

path_args = PathArgs()
video_path = 'outputs/videos'

video_args.video_name = "several_color_worlds.mp4"
path_args.video_path = os.path.join(video_path, video_args.video_name)

model_state = load_model(optimized_cfg_path, ckpt_path, optimized=True)

generate_video(image_args, video_args, path_args, model_state)
