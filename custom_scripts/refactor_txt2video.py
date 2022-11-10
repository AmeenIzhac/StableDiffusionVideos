#lib imports
import os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything
from torch import autocast
import random

#repo imports
from ldm.util import instantiate_from_config
from sd_video_utils import *
from kdiffusion import KDiffusionSampler



#TODO : find out how to properly comment in python



class ImageArgs:
    steps = 50
    H = 512
    W = 512
    scale = 8.0
    eta = 0.0
    C = 4
    f = 8

class VideoArgs:
    prompts = ["A cartoon drawing of a sun wearing sunglasses"]
    strength = 0.4
    frames = 60
    fps = 15
    x = 0.0
    y = 0.0
    zoom = 1.0
    angle = 0.0
    color_match = True
    seed = -1
    #sampler = euler_ancestral  


class ModelState:
    model = None
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def load_model(model_state, config_path, ckpt_path):
    config = OmegaConf.load(f"{config_path}")
    model = load_model_from_config(config, f"{ckpt_path}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model_state.model = model


def compute_current_prompt(C, index, frames) :
    return C[0] #TODO obviously do better


#previous image processing (color coherency, noise, encoding)
def process_previous_image(modelFS, previous_sample, xform, color_match=True, color_sample=None, noise=0.03):
    previous_img = sample_to_cv2(previous_sample)
    previous_img = cv2.warpPerspective(
        previous_img,
        xform,
        (previous_img.shape[1], previous_img.shape[0]),
        borderMode=cv2.BORDER_WRAP
    )
    if color_match:
        assert color_sample is not None
        previous_img = maintain_colors(previous_img, color_sample, (i % 2) == 0)
    previous_sample = sample_from_cv2(previous_img)
    previous_noised = add_noise(previous_sample, noise).half().to(device)
    return modelFS.get_first_stage_encoding(modelFS.encode_first_stage(previous_noised))


def send_to_upscale(x_new, output_path):
    #for now we'll just do a mere save
    x_new_clamp = torch.clamp((x_new + 1.0) / 2.0, min=0.0, max=1.0)

    for x_sample in x_new_clamp:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        Image.fromarray(x_sample.astype(np.uint8)).save(output_path)



def generate_image (
    c,
    x = None,
    uc = None,     #TODO see if I can get rid of this one
    ia = None,
    ms = None
) :
    if x is None:
        shape = [ia.C, ia.H // ia.f, ia.W // ia.f]
        x = torch.randn(shape, device=ms.device)
    samples_ddim, _ = sampler.sample(S=ia.steps, conditioning=c, batch_size=1, shape=x[0].shape, verbose=False, unconditional_guidance_scale=ia.scale,
                                             unconditional_conditioning=uc, eta=ia.eta, x_T=x) #TODO see if I need a callback
    return samples_ddim




def generate_video (
    image_args,
    video_args,
    model_state
) -> str :
    #outline : compute embeddings, generate first image, send it to upscale (in parallel), then loop, do processings, compute interpolated prompt, call generate_image, send it to upscale
    
    #Negative seed means random seed
    seed = video_args.seed
    if seed < 0 :
        seed = random.randint(0, 10 ** 6)
    seed_everything(seed)
    
    sampler = KDiffusionSampler(model_state.model,'euler_ancestral') # TODO either we assume we receive a sampler, or it is reinstantiated at each call if we allow sampler choice

    #
    # Place here DIR variables if necessary #TODO probably should receive them from the server caller, cleaner to do that way than to have implicit convention
    # #
    #Idea : have an video or smtg folder, in it have a subdirectory for each user, and in that put images, and simply video, and then maybe a guest directory
    test_dir = 'test_outputs'
    os.makedirs(test_dir, exist_ok=True)

    #
    # Assertions about consistency of arguments
    # #
    n_prompts = len(video_args.prompts)
    assert n_prompts > 0
    assert video_args.frames >= n_prompts


    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model_state.model.ema_scope():

                # 
                # Generate the text embeddings with the language model
                # #
                uc = model_state.model.get_learned_conditioning([""])
                C = []
                for prompt in video_args.prompt:
                    C.append(model_state.model.get_learned_conditioning([prompt])) #look if it can be done in parallel by handing a list of prompts



                #=====================FIRST_IMAGE_GENERATION=========================#
                # 
                # Generate the first image
                # #
                first_sample = generate_image(C[0], image_args, model_state)

                #save it to disk
                save_sample(first_sample, os.join(test_dir, f"{0:05}.png")) #TODO probably send to upscale instead




                #=====================SUBSEQUENT_IMAGES_GENERATION=========================#
                xform = make_xform_2d(  image_args.W, image_args.H, video_args.x, 
                                        video_args.y, video_args.angle, video_args.zoom )

                t_enc = int(video_args.strength * image_args.ddim_steps)
                previous_sample = first_sample
                color_sample = sample_to_cv2(previous_sample).copy()

                #maybe some code to get the proper sampler

                for i in trange(video_args.frames-1, desc="Generating frames"):

                    #seeding
                    seed += 1
                    seed_everything(seed)

                    #get the prompt interpolation point for the current image 
                    c = compute_current_prompt(C, i+1, video_args.frames)

                    previous_latent = process_previous_image(previous_sample, xform, 
                                            video_args.no_color_match, color_sample)

                    x_new = generate_image(c, image_args, model_state) 
                    save_sample(x_new, f"{(i+1):05}.png")
                    previous_sample = x_new

            
                #==========COMPILING=THE=VIDEO==========
                if(video_args.video_name == None):
                    video_count = 0 #len(os.listdir(video_path))
                    video_name = f"video{video_count}"
                else:
                    video_name = video_args.video_name
                sample_regex = os.path.join(test_dir, "%05d.png")
                command = f"ffmpeg -r {video_args.fps} -start_number {0} -i {sample_regex} -c:v libx264 -r 30 -pix_fmt yuv420p {os.path.join(test_dir, opt.video_name)}.mp4"   #TODO do that stuff                         
                os.system(command)

    return





#Placeholder "main" code
cfg_path = 'configs/stable-diffusion/v1-inference.yaml'
ckpt_path = 'models/ldm/stable-diffusion-v1/model.ckpt'
model_state = ModelState()
load_model(model_state, cfg_path, ckpt_path)

image_args = ImageArgs()
video_args = VideoArgs()

generate_video(image_args, video_args, model_state)