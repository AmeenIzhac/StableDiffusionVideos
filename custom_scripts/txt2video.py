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
from inference_rife import motion_interpolation, load_RIFE_model

import sys
sys.path.append('stable-diffusion-2/optimizedSD')



# Generatng the initial frames:
#  1. Get(/generateInitFrames?numFrames=x)
#       - clear the temp directory
#       - Server generates 4 frames, stores the torchs in a dict, files in temp directory
#       - returns code:200 upon completion
#  2. x * Get(/getFrame?index=x)
#       - each get request returns the frame x and it's respective id
#  3.  finally user selects a frame on the client side and sends a post request with the id of the frame
#      - server then gets the torch from the id and uses it to generate a video, returns video to client

class Constants:
    color_sample_resolution = 448

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
        self.inter_frames = 0
        self.interp_exp = 0
        self.fps = 15
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0
        self.angle = 0.0
        self.color_match = True
        self.several_color_match = True
        self.seed = -1
        self.video_name = None
        self.sampler = 'dpm_2'
        self.upscale = True
        self.init_sample = None
        self.correct = True


class ModelState:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sampler = None
        self.FS = None
        self.CS = None
        self.upsampler = None
        self.rife_model = None


class PathArgs:
    def __init__(self):
        self.image_path = 'outputs/images'
        self.video_path = 'outputs/videos'
        self.cfg_path = './stable-diffusion-2/configs/stable-diffusion/v1-inference.yaml'
        self.optimized_cfg_path = './stable-diffusion-2/optimizedSD/v1-inference.yaml'
        self.ckpt_path = './model_weights/stable-diffusion/model.ckpt'
        self.rife_path = './ECCV2022-RIFE/train_log'


class FloatWrapper:
    x = 0.0


def load_model(path_args, optimized=False):
    model_state = ModelState()

    model_state.upsampler = load_ESRGAN_model(model_name='RealESRGAN_x2plus')
    load_RIFE_model(model_state, path_args.rife_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not optimized:
        config = OmegaConf.load(f"{path_args.cfg_path}")
        _, model = load_model_from_config(path_args.ckpt_path, config, return_only_sd=False)
        model = model.to(device)
        model_state.model = model
        model_state.CS = model
        model_state.FS = model

    else:
        sd, _ = load_model_from_config(f"{path_args.ckpt_path}", return_only_sd=True)
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

        config = OmegaConf.load(f"{path_args.optimized_cfg_path}")

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


def extremise(t, k):
    #k controls the stiffness of the sigmoid, essentially controlling the transition speed between 2 prompts
    return 1 / (1 + (1/t - 1) ** k) if t > 0.0 else 0.0

def tensor_multi_step_interpolation(C, image_index, F, prompt_frames, k=4.0, is_slerp=True) :
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

    s = extremise(t, k)

    if not is_slerp : print(f'type of the array : {C[0].dtype}')
    c = slerp(s, c1, c2) if is_slerp else lerp(s, c1, c2).astype(np.uint8)

    return c



#previous image processing (color coherency, noise, encoding)
def process_previous_image(ms, previous_sample, xform, color_match=True, color_sample=None, noise=0.03, hsv=False, encode=True):
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
    return ms.FS.get_first_stage_encoding(ms.FS.encode_first_stage(previous_noised)) if encode else previous_noised


def save_image(x_new, output_path, model_state, upscale=True):
    x_new_clamp = torch.clamp((x_new + 1.0) / 2.0, min=0.0, max=1.0)

    for x_sample in x_new_clamp:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        if upscale:
            executeRealESRGAN(x_sample.astype(np.uint8), output_path, model_state.upsampler)
        else:
            Image.fromarray(x_sample.astype(np.uint8)).save(output_path)

def frame_interp_images(ms, image1, image2, inter_frames):
    latent1 = ms.FS.get_first_stage_encoding(ms.FS.encode_first_stage(image1)) #encode first stage
    latent2 = ms.FS.get_first_stage_encoding(ms.FS.encode_first_stage(image2))

    return autoencoder_frame_interp(ms, latent1, latent2, inter_frames)


def autoencoder_frame_interp(ms, latent1, latent2, inter_frames):
    interpolated = []
    for i in range(1, inter_frames + 1):
        t = i / (inter_frames + 1)
        latent = slerp(t, latent1, latent2) # maybe lerp?
        interpolated.append(ms.FS.decode_first_stage(latent)) #decode

    return interpolated

def cheap_image_args():
    ia = ImageArgs()
    ia.steps = 20
    ia.H = Constants.color_sample_resolution
    ia.W = Constants.color_sample_resolution
    ia.scale = 8.0

    return ia
    

def generate_image (
    c = None,
    x = None,
    uc = None,
    ia = None,
    ms = None,
    t_enc = None,
    batch_size = 1,
    decode = True
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

    return ms.FS.decode_first_stage(samples) if decode else samples

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


def compile_video(video_args, path_args, base_count, model_state):
    if(video_args.video_name == None):
        video_count = len(os.listdir(path_args.video_path))
        video_name = f"video{video_count}"
    else:
        video_name = video_args.video_name

    if video_args.interp_exp > 0: #we perform motion interpolation
        #TODO make the interp_exp a interp_factor parameter, ensure it is a multiple of 2 and do some log to get it (probably bit shift)
        motion_interpolation(path_args.image_path, path_args.video_path, video_args.fps, frames_count=video_args.frames, exp=video_args.interp_exp, starting_frame=base_count, ms=model_state, scale=1.0, codec='avc1') #TODO add the feature to start at some image
    else:
        sample_regex = os.path.join(path_args.image_path, "%05d.png")
        command = f"ffmpeg -r {video_args.fps} -start_number {base_count} -i {sample_regex} -c:v libx264 -r 30 -pix_fmt yuv420p {path_args.video_path}"                       
        os.system(command)


def frame_path(frame_number, path_args):
    return os.path.join(path_args.image_path, f"{frame_number:05}.png")

def move_FS_UN_to_gpu(model_state):
    model_state.model.to(model_state.device) #move the UNet model to gpu
    model_state.FS.to(model_state.device) #move the autoencoder to gpu
    model_state.model.cuda()
    model_state.FS.cuda()

def move_FS_UN_to_cpu(model_state):
    model_state.FS.to("cpu")
    model_state.model.to("cpu")


def generate_video (
    image_args,
    video_args,
    path_args,
    model_state,
    progress_var = None
) -> str :
    #outline : compute embeddings, generate first image, send it to upscale (in parallel), then loop, do processings, compute interpolated prompt, call generate_image, send it to upscale
    
    print("Generating img2img video with prompts : ", video_args.prompts)

    seed, base_count, start_number = init_video_gen(video_args, model_state, path_args)
    if progress_var is not None : progress_var.x = 0.0

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):

            C, uc = generate_embeddings(video_args.prompts, model_state)

            move_FS_UN_to_gpu(model_state)

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
            xform = make_xform_2d(  image_args.W, image_args.H, -video_args.x, 
                                    video_args.y, video_args.angle, video_args.zoom )


            t_enc = int(video_args.strength * image_args.steps)
            previous_sample = first_sample
            #color_sample = sample_to_cv2(previous_sample).copy()

            C_s = len(C)

            interpolate_colors = video_args.several_color_match
            color_samples = []
            color_samples.append(cv2.resize(sample_to_cv2(first_sample), (Constants.color_sample_resolution,Constants.color_sample_resolution))) #put hsv or something here
            if C_s > 1 and video_args.color_match and interpolate_colors:
                cia = cheap_image_args()
                for i in range(1, C_s):
                    #generate new sample here
                    color_sample = generate_image(c=C[i], uc=uc, ia=cia, ms=model_state) #TODO make it fast (small steps and resolution, fast sampler)
                    color_samples.append(sample_to_cv2(color_sample))
                    pool.submit(save_image, color_sample, f"../color_sample_{base_count+i}.png", model_state, upscale=False)

            previous_color_sample = color_samples[0]

            prompt_frames = np.arange(C_s) * ( (video_args.frames - 1) / (C_s - 1)) if C_s > 1 else None
            
            for i in trange(1, video_args.frames, desc="Generating frames"):

                #seeding
                seed += 1
                seed_everything(seed)

                #get the prompt interpolation point for the current image 
                c = tensor_multi_step_interpolation(C, i, video_args.frames, prompt_frames, k=2.0)

                if video_args.color_match : 
                    color_sample = tensor_multi_step_interpolation(color_samples, i, video_args.frames, prompt_frames, k=4.0, is_slerp=False)
                
                previous_latent = process_previous_image(model_state, previous_sample, xform, 
                                        video_args.color_match, color_sample if video_args.color_match else None, hsv= (((i-1) % 2) == 0))

                new_latent = generate_image(c=c, x=previous_latent, uc=uc, ia=image_args, ms=model_state, t_enc=t_enc, decode=False) 
                x_new = model_state.FS.decode_first_stage(new_latent)
                if video_args.correct:
                    for k, image in enumerate(autoencoder_frame_interp(model_state, previous_latent, new_latent, video_args.inter_frames)):
                        correct_factor = (k + 1 - video_args.inter_frames) / video_args.inter_frames
                        xform_backward = make_xform_2d(
                            image_args.W, image_args.H, 
                            video_args.x * correct_factor, video_args.y * correct_factor,
                            video_args.angle * correct_factor, (video_args.zoom - 1) * correct_factor + 1
                        )
                        if video_args.color_match: correct_color_sample = lerp( k / (video_args.inter_frames+1) , previous_color_sample, color_sample).astype(np.uint8)
                        corrected = process_previous_image(model_state, image, xform_backward, video_args.color_match,
                            correct_color_sample if video_args.color_match else None, noise=0, hsv=True, encode=False) #correct image
                        pool.submit(save_image, corrected, frame_path(base_count + (i-1)*(video_args.inter_frames+1) + k+1, path_args), model_state, upscale=video_args.upscale) #save image
                else:
                    for k, image in enumerate(frame_interp_images(model_state, previous_sample, x_new, video_args.inter_frames)):
                        pool.submit(save_image, image, frame_path(base_count + (i-1)*(video_args.inter_frames+1) + k+1, path_args), model_state, upscale=video_args.upscale) #save image
                
                pool.submit(save_image, x_new, frame_path(base_count + i * (video_args.inter_frames + 1), path_args), model_state, upscale=video_args.upscale)
                previous_sample = x_new
                if video_args.color_match: previous_color_sample = color_sample 

                if progress_var is not None:
                    progress_var.x = (i+1) / video_args.frames

            #Free some video memory by pushing the auto-encoder and the UNet to RAM 
            move_FS_UN_to_cpu

            # Wait for upscaling/saving to finish
            pool.shutdown()

            compile_video(video_args, path_args, base_count, model_state)


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

            move_FS_UN_to_gpu(model_state)

            # Init thread pool
            num_workers = video_args.inter_frames + 1
            pool = ThreadPoolExecutor(num_workers)

            shape = [1, image_args.C, image_args.H // image_args.f, image_args.W // image_args.f]
            Noises = []
            for i in range(n_noises):
                Noises.append(torch.randn(shape, device=model_state.device))
            
            C_s = len(C)
            prompt_frames = np.arange(C_s) * ( (video_args.frames - 1) / (C_s - 1)) if C_s > 1 else None
            
            previous_sample = None
            #=====================IMAGES_GENERATION=========================#
            for i in trange(video_args.frames, desc="Generating interpolation steps"):
                x = tensor_multi_step_interpolation(Noises, i, video_args.frames, prompt_frames, k=1.0)
                sample = generate_image(c=tensor_multi_step_interpolation(C, i, video_args.frames, prompt_frames, k=1.0), x=x, uc=uc, ia=image_args, ms=model_state, decode=False)
                #save it to disk
                if previous_sample is not None:
                    for k, image in enumerate(autoencoder_frame_interp(model_state, previous_sample, sample, video_args.inter_frames)):
                        pool.submit(save_image, decoded, frame_path(base_count + (i-1)*(video_args.inter_frames+1) + k, path_args), model_state, upscale=video_args.upscale)
                
                decoded = model_state.FS.decode_first_stage(sample)
                pool.submit(save_image, decoded, frame_path(base_count + i * (video_args.inter_frames + 1), path_args), model_state, upscale=video_args.upscale)

                previous_sample = sample
                if progress_var is not None:
                    progress_var.x = (i+1) / video_args.frames

            #Free some video memory by pushing the auto-encoder to RAM 
            move_FS_UN_to_cpu(model_state)

            # Wait for upscaling/saving to finish
            pool.shutdown()

            compile_video(video_args, path_args, base_count, model_state)

    return

def generateInitFrame(image_args, video_args, path_args, model_state, n=4) :
    model_state.sampler = KDiffusionSampler(model_state.model, video_args.sampler)


    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            C, uc = generate_embeddings([video_args.prompts[0]], model_state)

            model_state.FS.to(model_state.device)

            seeds = [random.randint(0, 10 ** 6) for i in range(n)]

    for i,sample in enumerate(samples):
        save_image(sample, output_path=os.join(image_dir, f"{video_args.seed}.png"), model_state=model_state, upscale=False)
    samples_list = torch.split(samples, 1)

    
    return samples_list

