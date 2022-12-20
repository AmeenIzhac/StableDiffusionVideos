#lib imports
import os
import cv2
import torch
import numpy as np
import math
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything
from torch import autocast
import random
from concurrent.futures import ThreadPoolExecutor

#repo imports
from sd_video_utils import *
from kdiffusion import KDiffusionSampler
from inference_realesrgan import *
from ldm.util import instantiate_from_config

import sys
sys.path.append('../stable-diffusion-2/optimizedSD')


#TODO : find out how to properly comment in python

# Generatng the initial frames:
#  1. Get(/generateInitFrames?numFrames=x)
#       - clear the temp directory
#       - Server generates 4 frames, stores the torchs in a dict, files in temp directory
#       - returns code:200 upon completion
#  2. x * Get(/getFrame?index=x)
#       - each get request returns the frame x and it's respective id
#  3.  finally user selects a frame on the client side and sends a post request with the id of the frame
#      - server then gets the torch from the id and uses it to generate a video, returns video to client

class ImageArgs:
    def __init__(self):
        self.steps = 50
        self.H = 512
        self.W = 512
        self.scale = 8.0
        self.eta = 0.0
        self.C = 4
        self.f = 8


class VideoArgs:
    def __init__(self):
        self.prompts = ["A cartoon drawing of a sun wearing sunglasses"]
        self.strength = 0.4
        self.frames = 60
        self.fps = 15
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0
        self.angle = 0.0
        self.color_match = True
        self.seed = -1
        self.video_name = None
        self.sampler = 'dpm_2'
        self.upscale = True
        self.init_sample = None


class ModelState:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sampler = None
        self.FS = None
        self.CS = None
        self.upsampler = None


class PathArgs:
    def __init__(self):
        self.image_path = 'outputs/images'
        self.video_path = 'outputs/videos'


class FloatWrapper:
    x = 0.0


def load_model(config_path, ckpt_path, optimized=False):
    model_state = ModelState()

    model_state.upsampler = load_ESRGAN_model(model_name='RealESRGAN_x2plus')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not optimized:
        config = OmegaConf.load(f"{config_path}")
        _, model = load_model_from_config(ckpt_path, config, return_only_sd=False)
        model = model.to(device)
        model_state.model = model
        model_state.CS = model
        model_state.FS = model

    else:
        sd, _ = load_model_from_config(f"{ckpt_path}", return_only_sd=True)
        li, lo = [], []
        for key, value in sd.items():
            sp = key.split(".")
            if (sp[0]) == "model":
                if "input_blocks" in sp:
                    li.append(key)
                elif "middle_block" in sp:
                    li.append(key)
                elif "time_embed" in sp:
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            sd["model1." + key[6:]] = sd.pop(key)
        for key in lo:
            sd["model2." + key[6:]] = sd.pop(key)

        config = OmegaConf.load(f"{config_path}")

        model = instantiate_from_config(config.modelUNet)
        _, _ = model.load_state_dict(sd, strict=False)
        model.eval()
        model.unet_bs = 1
        model.cdevice = device
        model.half()
        model.to(device)
        model.turbo = True

        modelCS = instantiate_from_config(config.modelCondStage)
        _, _ = modelCS.load_state_dict(sd, strict=False)
        modelCS.eval()
        modelCS.cond_stage_model.device = device
        modelCS.half()

        modelFS = instantiate_from_config(config.modelFirstStage)
        _, _ = modelFS.load_state_dict(sd, strict=False)
        modelFS.eval()
        modelFS.half()
        del sd
        model_state.FS = modelFS
        model_state.CS = modelCS
        model_state.model = model

    return model_state


def tensor_multi_step_interpolation(C, image_index, F, prompt_frames, k=4.0) :
    C_s = len(C)
    if C_s == 1:
        return C[0]
    
    r = image_index / (F - 1)
    if(r >= 1):
        return C[C_s - 1]

    i1 = int(r * (C_s-1))
    i2 = int(r * (C_s-1)) + 1
    c1 = C[i1]
    c2 = C[i2]
    ip_ratio = (F - 1) / (C_s - 1)
    f1 = i1 * ip_ratio
    f2 = i2 * ip_ratio
    t = (image_index - f1) / (f2 - f1)

    #k controls the stiffness of the sigmoid, essentially controlling the transition speed between 2 prompts
    s = 1 / (1 + (1/t - 1) ** k) if t > 0.0 else 0.0

    c = slerp(s, c1, c2)

    return c



#previous image processing (color coherency, noise, encoding)
def process_previous_image(ms, previous_sample, xform, color_match=True, color_sample=None, noise=0.03, hsv=False):
    previous_img = sample_to_cv2(previous_sample)
    previous_img = cv2.warpPerspective(
        previous_img,
        xform,
        (previous_img.shape[1], previous_img.shape[0]),
        borderMode=cv2.BORDER_WRAP
    )
    if color_match:
        assert color_sample is not None
        previous_img = maintain_colors(previous_img, color_sample, hsv=hsv)
    previous_sample = sample_from_cv2(previous_img)

    previous_noised = add_noise(previous_sample, noise).half().to(ms.device)
    return ms.FS.get_first_stage_encoding(ms.FS.encode_first_stage(previous_noised))


def save_image(x_new, output_path, model_state, upscale=True):
    x_new_clamp = torch.clamp((x_new + 1.0) / 2.0, min=0.0, max=1.0)

    for x_sample in x_new_clamp:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        if upscale:
            executeRealESRGAN(x_sample.astype(np.uint8), output_path, model_state.upsampler)
        else:
            Image.fromarray(x_sample.astype(np.uint8)).save(output_path)

def generate_image (
    c = None,
    x = None,
    uc = None,
    ia = None,
    ms = None,
    t_enc = None,
    batch_size = 1
) :
    if x is None: #doing txt2img with noise to be generated
        shape = [batch_size, ia.C, ia.H // ia.f, ia.W // ia.f]
        x = torch.randn(shape, device=ms.device)
        samples, _ = ms.sampler.sample(S=ia.steps, conditioning=c, unconditional_guidance_scale=ia.scale,
                                unconditional_conditioning=uc, x_T=x)
    else:
        if t_enc is None: #doing noise walk
            samples, _ = ms.sampler.sample(S=ia.steps, conditioning=c, unconditional_guidance_scale=ia.scale,
                                unconditional_conditioning=uc, x_T=x)
        else: #doing img2img
            #maybe adapt for batch size here if it's of any relevance : just a copy-cat operation ig
            samples, _ = ms.sampler.sample(S=ia.steps, conditioning=c, unconditional_guidance_scale=ia.scale,
                                    unconditional_conditioning=uc, x_T=x, img2img=True, t_enc=t_enc)

    return ms.FS.decode_first_stage(samples)

def init_video_gen(video_args, model_state, path_args):
    #Negative seed means random seed
    seed = video_args.seed
    if seed < 0 :
        seed = random.randint(0, 10 ** 6)
    seed_everything(seed)
    
    model_state.sampler = KDiffusionSampler(model_state.model, video_args.sampler)

    #
    # Place here DIR variables if necessary
    # #
    os.makedirs(path_args.image_path, exist_ok=True)
    os.makedirs(os.path.dirname(path_args.video_path), exist_ok=True)
    base_count = len(os.listdir(path_args.image_path))
    start_number = base_count

    def frame_path(frame_number, ):
        return os.path.join(path_args.image_path, f"{frame_number:05}.png")

    #
    # Assertions about consistency of arguments
    # #
    n_prompts = len(video_args.prompts)
    assert n_prompts > 0
    assert video_args.frames >= n_prompts

    return seed, base_count, start_number


def generate_embeddings(prompts, model_state):
    # 
    # Generate the text embeddings with the language model
    # #
    model_state.CS.to(model_state.device)
    uc = model_state.CS.get_learned_conditioning([""])
    C = []
    for prompt in prompts:
        C.append(model_state.CS.get_learned_conditioning([prompt])) #TODO look if it can be done in parallel by handing a list of prompts
    if model_state.device != "cpu":
        mem = torch.cuda.memory_allocated() / 1e6
        model_state.CS.to("cpu")
        while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(0.1)
    
    return C, uc


def compile_video(video_args, path_args, base_count):
    if(video_args.video_name == None):
        video_count = len(os.listdir(path_args.video_path))
        video_name = f"video{video_count}"
    else:
        video_name = video_args.video_name
    sample_regex = os.path.join(path_args.image_path, "%05d.png")
    command = f"ffmpeg -r {video_args.fps} -start_number {base_count} -i {sample_regex} -c:v libx264 -r 30 -pix_fmt yuv420p {path_args.video_path}"                       
    os.system(command)


def frame_path(frame_number, path_args):
    return os.path.join(path_args.image_path, f"{frame_number:05}.png")


def generate_video (
    image_args,
    video_args,
    path_args,
    model_state,
    progress_var = None
) -> str :
    #outline : compute embeddings, generate first image, send it to upscale (in parallel), then loop, do processings, compute interpolated prompt, call generate_image, send it to upscale
    
    print("Generating walk video with prompts : ", video_args.prompts)

    seed, base_count, start_number = init_video_gen(video_args, model_state, path_args)
    if progress_var is not None : progress_var.x = 0.0

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):

            C, uc = generate_embeddings(video_args.prompts, model_state)

            model_state.FS.to(model_state.device)

            # Init thread pool
            num_workers = 1
            pool = ThreadPoolExecutor(num_workers)

            #=====================FIRST_IMAGE_GENERATION=========================#
            # 
            # Generate the first image (if it wasn't given as input)
            # #
            first_sample = generate_image(c=C[0], uc=uc, ia=image_args, ms=model_state) if video_args.init_sample is None else video_args.init_sample

            #save it to disk
            pool.submit(save_image, first_sample, frame_path(base_count, path_args), model_state, upscale=video_args.upscale)
            if progress_var is not None:
                progress_var.x = 1 / video_args.frames

            #=====================SUBSEQUENT_IMAGES_GENERATION=========================#
            xform = make_xform_2d(  image_args.W, image_args.H, video_args.x, 
                                    video_args.y, video_args.angle, video_args.zoom )

            t_enc = int(video_args.strength * image_args.steps)
            previous_sample = first_sample
            #color_sample = sample_to_cv2(previous_sample).copy()

            C_s = len(C)

            color_samples = []
            color_sample.append(cv2.cvtColor(sample_to_cv2(first_sample), cv2.COLOR_RGB2HSV)) #put hsv or something here
            if C_s > 1 and color_match:
                for i in range(1, C_s):
                    #generate new sample here
                    color_sample = generate_image(c=C[i], uc=uc, ia=image_args, ms=model_state)
                    color_samples.append(cv2.cvtColor(sample_to_cv2(previous_sample), cv2.COLOR_RGB2HSV))

            prompt_frames = np.arange(C_s) * ( (video_args.frames - 1) / (C_s - 1)) if C_s > 1 else None
            
            for i in trange(video_args.frames-1, desc="Generating frames"):

                #seeding
                seed += 1
                seed_everything(seed)

                #get the prompt interpolation point for the current image 
                c = tensor_multi_step_interpolation(C, i+1, video_args.frames, prompt_frames, k=1.0)

                if color_match : color_sample = tensor_multi_step_interpolation(color_samples, i+1, video_args.frames, prompt_frames, k=1.0)
                
                previous_latent = process_previous_image(model_state, previous_sample, xform, 
                                        video_args.color_match, color_sample if color_match else None, hsv= ((i % 2) == 0))

                x_new = generate_image(c=c, x=previous_latent, uc=uc, ia=image_args, ms=model_state, t_enc=t_enc) 
                pool.submit(save_image, x_new, frame_path(base_count + i + 1, path_args), model_state, upscale=video_args.upscale)
                previous_sample = x_new

                if progress_var is not None:
                    progress_var.x = (i+2) / video_args.frames

            #Free some video memory by pushing the auto-encoder to RAM 
            model_state.FS.to("cpu")

            # Wait for upscaling/saving to finish
            pool.shutdown()

            compile_video(video_args, path_args, base_count)


    return


def generate_walk_video(    
    image_args,
    video_args,
    path_args,
    model_state,
    n_noises=1,
    progress_var = None
    ):    

    seed, base_count, start_number = init_video_gen(video_args, model_state, path_args)
    if progress_var is not None : 
        progress_var.x = 0

    print("Generating walk video with prompts : ", video_args.prompts)

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):

            C, uc = generate_embeddings(video_args.prompts, model_state)

            model_state.FS.to(model_state.device)

            # Init thread pool
            num_workers = 1
            pool = ThreadPoolExecutor(num_workers)

            shape = [1, image_args.C, image_args.H // image_args.f, image_args.W // image_args.f]
            Noises = []
            for i in range(n_noises):
                Noises.append(torch.randn(shape, device=model_state.device))
            
            C_s = len(C)
            prompt_frames = np.arange(C_s) * ( (video_args.frames - 1) / (C_s - 1))

            #=====================IMAGES_GENERATION=========================#
            for i in trange(video_args.frames, desc="Generating interpolation steps"):
                x = tensor_multi_step_interpolation(Noises, i, video_args.frames, prompt_frames, k=1.0)
                sample = generate_image(c=tensor_multi_step_interpolation(C, i, video_args.frames, prompt_frames, k=1.0), x=x, uc=uc, ia=image_args, ms=model_state)
                #save it to disk
                pool.submit(save_image, sample, frame_path(base_count+i, path_args), model_state, upscale=video_args.upscale)

                if progress_var is not None:
                    progress_var.x = (i+1) / video_args.frames

            #Free some video memory by pushing the auto-encoder to RAM 
            model_state.FS.to("cpu")

            # Wait for upscaling/saving to finish
            pool.shutdown()

            compile_video(video_args, path_args, base_count)

    return

def generate_initial_images(image_args, video_args, image_dir, model_state, count=4) :
    #I should really factorise this method and generate_video TODO
    seed = video_args.seed
    if seed < 0:
        seed = random.randint(0, 10 ** 6)

def generateInitFrame(image_args, video_args, path_args, model_state, n=4) :
    model_state.sampler = KDiffusionSampler(model_state.model, video_args.sampler)


    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            C, uc = generate_embeddings([video_args.prompts[0]], model_state)

            model_state.FS.to(model_state.device)

            seeds = [random.randint(0, 10 ** 6) for i in range(n)]

    for i,sample in enumerate(samples):
        save_image(sample, output_path=os.join(image_dir, f"{frame_number:05}.png"), model_state=model_state, upscale=False)
    samples_list = torch.split(samples, 1)

    
    return samples_list

