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

#repo imports
from sd_video_utils import *
from kdiffusion import KDiffusionSampler
from inference_realesrgan import *
from ldm.util import instantiate_from_config

#questions with raaif:
#implement the server properly (with loading the model once)
#the path question
#the prompt list question
#the init image format question
#the init image saving question




#TODO : find out how to properly comment in python



class ImageArgs:
    steps = 50
    H = 512
    W = 512
    scale = 10.0
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
    video_name = None
    sampler = 'dpm2'  
    upscale = True
    init_sample = None

class ModelState:
    model = None
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sampler = None
    FS = None
    CS = None
    upsampler = None

class PathArgs:
    image_path = 'outputs/images'
    video_path = 'outputs/videos'



def load_model(model_state, config_path, ckpt_path, optimized=False):
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


def compute_current_prompt(C, image_index, F, v=15.0) :
    C_s = len(C)
    if C_s == 1:
        return C[0]
    
    prompt_frames = np.arange(C_s) * ( (F-1) / (C_s - 1)) #TODO cache this
    #if len(C) = 2, F = 100, then prompt i will have prompt_frame = i * 99/1
    #as expected prompt 0 will be prompt_frame 0, and prompt 1 will get frame 99.
    r = image_index / (F - 1)
    if(r >= 1):
        return C[C_s - 1]

    i1 = int(r * (C_s-1))
    i2 = int(r * (C_s-1)) + 1
    c1 = C[i1]
    c2 = C[i2]
    ip_ratio = (F - 1) / (C_s - 1)
    f1 = i1 * ip_ratio #prompt_frames[i1]
    f2 = i2 * ip_ratio
    t = (image_index - f1) / (f2 - f1)
    s = 1 / (1 + math.exp(-v * (t - 0.5))) #v controls the stiffness of the sigmoid, essentially controlling the transition speed between 2 prompts
    c = slerp(s, c1, c2)
    return c



#previous image processing (color coherency, noise, encoding)
def process_previous_image(ms, previous_sample, xform, color_match=True, color_sample=None, noise=0.03, hsv=False): #TODO : look if that can be done in parallel (analyze runtime first to see if it's worth it)
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
        #x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> c w h')
        if upscale:
            executeRealESRGAN(x_sample.astype(np.uint8), output_path, model_state.upsampler)
        else:
            Image.fromarray(x_sample.astype(np.uint8)).save(output_path)



def generate_image (
    c = None,
    x = None,
    uc = None,     #TODO see if I can get rid of this one
    ia = None,
    ms = None,
    t_enc = None,
    batch_size = 1
) :
    if x is None:
        shape = [batch_size, ia.C, ia.H // ia.f, ia.W // ia.f]
        x = torch.randn(shape, device=ms.device)
        samples, _ = ms.sampler.sample(S=ia.steps, conditioning=c, unconditional_guidance_scale=ia.scale,
                                unconditional_conditioning=uc, x_T=x)
    else:
        #maybe adapt for batch size here if it's of any relevance : just a copy-cat operation ig
        samples, _ = ms.sampler.sample(S=ia.steps, conditioning=c, unconditional_guidance_scale=ia.scale,
                                unconditional_conditioning=uc, x_T=x, img2img=True, t_enc=t_enc)
                    
    os.system('nvidia-smi --query-gpu=memory.used --format=csv')
    return ms.FS.decode_first_stage(samples)


def generate_video (
    image_args,
    video_args,
    model_state,
    path_args
) -> str :
    #outline : compute embeddings, generate first image, send it to upscale (in parallel), then loop, do processings, compute interpolated prompt, call generate_image, send it to upscale
    
    #Negative seed means random seed
    seed = video_args.seed
    if seed < 0 :
        seed = random.randint(0, 10 ** 6)
    seed_everything(seed)
    
    model_state.sampler = KDiffusionSampler(model_state.model, video_args.sampler) # TODO either we assume we receive a sampler, or it is reinstantiated at each call if we allow sampler choice

    #
    # Place here DIR variables if necessary #TODO probably should receive them from the server caller, cleaner to do that way than to have implicit convention
    # #
    #Idea : have an video or smtg folder, in it have a subdirectory for each user, and in that put images, and simply video, and then maybe a guest directory
    os.makedirs(path_args.image_path, exist_ok=True)
    os.makedirs(path_args.video_path, exist_ok=True)
    base_count = len(os.listdir(path_args.image_path))
    start_number = base_count

    #
    # Assertions about consistency of arguments
    # #
    n_prompts = len(video_args.prompts)
    assert n_prompts > 0
    assert video_args.frames >= n_prompts


    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            #with model_state.model.ema_scope():

            # 
            # Generate the text embeddings with the language model
            # #
            model_state.CS.to(model_state.device)
            uc = model_state.CS.get_learned_conditioning([""])
            C = []
            for prompt in video_args.prompts:
                C.append(model_state.CS.get_learned_conditioning([prompt])) #look if it can be done in parallel by handing a list of prompts
            if model_state.device != "cpu":
                mem = torch.cuda.memory_allocated() / 1e6
                model_state.CS.to("cpu")
                while torch.cuda.memory_allocated() / 1e6 >= mem:
                    time.sleep(1)

            model_state.FS.to(model_state.device)


            #=====================FIRST_IMAGE_GENERATION=========================#
            # 
            # Generate the first image (if it wasn't given as input)
            # #
            first_sample = generate_image(c=C[0], uc=uc, ia=image_args, ms=model_state) if video_args.init_sample is None else video_args.init_sample

            #save it to disk
            save_image(first_sample, os.path.join(path_args.image_path, f"{base_count:05}.png"), model_state, upscale=video_args.upscale)




            #=====================SUBSEQUENT_IMAGES_GENERATION=========================#
            xform = make_xform_2d(  image_args.W, image_args.H, video_args.x, 
                                    video_args.y, video_args.angle, video_args.zoom )

            t_enc = int(video_args.strength * image_args.steps)
            previous_sample = first_sample
            color_sample = sample_to_cv2(previous_sample).copy()

            #maybe some code to get the proper sampler

            for i in trange(video_args.frames-1, desc="Generating frames"):

                #seeding
                seed += 1
                seed_everything(seed)

                #get the prompt interpolation point for the current image 
                c = compute_current_prompt(C, i+1, video_args.frames)

                previous_latent = process_previous_image(model_state, previous_sample, xform, 
                                        video_args.color_match, color_sample, hsv= ((i % 2) == 0))

                x_new = generate_image(c=c, x=previous_latent, uc=uc, ia=image_args, ms=model_state, t_enc=t_enc) 
                save_image(x_new, os.path.join(path_args.image_path, f"{(base_count + i + 1):05}.png"), model_state, upscale=video_args.upscale)
                previous_sample = x_new


            #==========COMPILING=THE=VIDEO==========
            if(video_args.video_name == None):
                video_count = len(os.listdir(path_args.video_path))
                video_name = f"video{video_count}"
            else:
                video_name = video_args.video_name
            sample_regex = os.path.join(path_args.image_path, "%05d.png")
            command = f"ffmpeg -r {video_args.fps} -start_number {base_count} -i {sample_regex} -c:v libx264 -r 30 -pix_fmt yuv420p {os.path.join(path_args.video_path, video_name)}.mp4"   #TODO do that stuff                         
            os.system(command)

            #TODO : do smtg like put the FS model to cpu at the end

    return

def generate_initial_images(image_args, video_args, model_state, count=4) :
    #I should really factorise this method and generate_video TODO
    seed = video_args.seed
    if seed < 0:
        seed = random.randint(0, 10 ** 6)

    model_state.sampler = KDiffusionSampler(model_state.model, video_args.sampler)

    model_state.CS.to(model_state.device)
    uc = model_state.CS.get_learned_conditioning([""])
    prompt = video_args.prompts[0]
    c = model_state.CS.get_learned_conditioning([prompt])
    mem = torch.cuda.memory_allocated() / 1e6
    model_state.CS.to("cpu")
    while torch.cuda.memory_allocated() / 1e6 >= mem:
            time.sleep(1)
            
    model_state.FS.to(model_state.device)

    samples = generate_images(c=c, uc=uc, ia=image_args, ms=model_state, batch_size=count)

    for sample in samples:
        #do here the image-saving, returning
        images = None
    
    return images, seed

