from refactor_txt2video import *

#Placeholder "main" code
cfg_path = 'configs/stable-diffusion/v1-inference.yaml'
optimized_cfg_path = 'optimizedSD/v1-inference.yaml'
ckpt_path = 'models/ldm/stable-diffusion-v1/model.ckpt'
model_state = ModelState()
load_model(model_state, optimized_cfg_path, ckpt_path, optimized=True)

image_args = ImageArgs()
video_args = VideoArgs()

video_args.prompts = ["underwater ocean scene, with fishes, coral, seeweed, algae, octopus, sharks, water, 4k colorful"]
video_args.fps = 20
video_args.strength = 0.38
video_args.zoom = 1.01
video_args.x = -4
video_args.y = -2
video_args.frames = 50
image_args.steps = 40
image_args.W = 512
image_args.H = 512
image_args.scale = 10
video_args.upscale = False
image_args.seed = 25

for sampler in ['dpm_2_ancestral', 'dpm_2', 'euler_ancestral', 'euler', 'heun', 'lms']:
    video_args.sampler = sampler
    video_args.video_name = f"underwater_{sampler}"
    generate_video(image_args, video_args, model_state)