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

video_args.prompts = ["a picture of a cat",
                        "a picture of a dog",
                            "a picture of a lion",
                                "a picture of giraffe",
                                    "a picture of a bear",
                                        "a picture of a tiger",
                                            "a picture of a hippopothamus",
                                                "a picture of buffalo",
                                                    "a picture of an eagle"]

video_args.fps = 20
video_args.frames = 400
image_args.steps = 35
image_args.W = 512
image_args.H = 512
image_args.scale = 10
video_args.upscale = True
video_args.video_name = "animals.mp4"

path_args.video_path = os.path.join(video_path, video_args.video_name)
generateInitFrame(image_args, video_args, path_args, model_state, n=4)
