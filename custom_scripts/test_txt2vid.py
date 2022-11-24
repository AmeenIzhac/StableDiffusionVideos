import sys
import os
from txt2video import *
sys.path.append('stable-diffusion-2/optimizedSD')

#Placeholder "main" code
cfg_path = 'configs/stable-diffusion/v1-inference.yaml'
optimized_cfg_path = 'optimizedSD/v1-inference.yaml'
ckpt_path = 'models/ldm/stable-diffusion-v1/model.ckpt'
model_state = load_model(optimized_cfg_path, ckpt_path, optimized=True)

image_args = ImageArgs()
video_args = VideoArgs()

video_args.prompts = [
    "a photography of a person, detailed portait, 4k",
    "a close photography of human skin, detailed",
    "a biological picture of animal cells, skin cells, detailed, microscopic, 4k",
    "a picture of molecules, proteins, dna, genom, biology molecules, carbon, microscopic, 4k",
    "a picture of atoms, electrons rotating around the nucleus, quantum physics, 4k",
    ]

video_args.fps = 20
video_args.strength = 0.36
video_args.frames = 300
image_args.steps = 80
image_args.W = 512
image_args.H = 512
image_args.scale = 10
video_args.zoom = 10 ** (len(video_args.prompts) / video_args.frames)
video_args.upscale = True
video_args.color_match = True

path_args = PathArgs()
video_path = 'outputs/videos'

video_args.video_name = "micro_zoom6.mp4"
path_args.video_path = os.path.join(video_path, video_args.video_name)
generate_video(image_args, video_args, path_args, model_state)

video_args.video_name = "micro_zoom7.mp4"
path_args.video_path = os.path.join(video_path, video_args.video_name)
generate_video(image_args, video_args, path_args, model_state)

video_args.video_name = "micro_zoom8.mp4"
path_args.video_path = os.path.join(video_path, video_args.video_name)
generate_video(image_args, video_args, path_args, model_state)

video_args.video_name = "micro_zoom9.mp4"
path_args.video_path = os.path.join(video_path, video_args.video_name)
generate_video(image_args, video_args, path_args, model_state)
