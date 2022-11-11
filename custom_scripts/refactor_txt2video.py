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
from sd_video_utils import *
from kdiffusion import KDiffusionSampler

#fix that omg
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config




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
    video_name = None
    #sampler = euler_ancestral  

class ModelState:
    model = None
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sampler = None
    FS = None
    CS = None



def load_model(model_state, config_path, ckpt_path, optimized=False):
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


def compute_current_prompt(C, index, frames) :
    return C[0] #TODO obviously do better


#previous image processing (color coherency, noise, encoding)
def process_previous_image(modelFS, previous_sample, xform, color_match=True, color_sample=None, noise=0.03, hsv=False):
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
    previous_noised = add_noise(previous_sample, noise).half().to(model_state.device)
    return modelFS.get_first_stage_encoding(modelFS.encode_first_stage(previous_noised))


def send_to_upscale(x_new, output_path):
    #for now we'll just do a mere save
    x_new_clamp = torch.clamp((x_new + 1.0) / 2.0, min=0.0, max=1.0)

    for x_sample in x_new_clamp:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        Image.fromarray(x_sample.astype(np.uint8)).save(output_path)



def generate_image (
    c = None,
    x = None,
    uc = None,     #TODO see if I can get rid of this one
    ia = None,
    ms = None,
    t_enc = None
) :
    if x is None:
        shape = [1, ia.C, ia.H // ia.f, ia.W // ia.f]
        x = torch.randn(shape, device=ms.device)
        samples_ddim, _ = ms.sampler.sample(S=ia.steps, conditioning=c, unconditional_guidance_scale=ia.scale,
                                unconditional_conditioning=uc, x_T=x)
    else:
        samples_ddim, _ = ms.sampler.sample(S=ia.steps, conditioning=c, unconditional_guidance_scale=ia.scale,
                                unconditional_conditioning=uc, x_T=x, img2img=True, t_enc=t_enc)
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
    
    model_state.sampler = KDiffusionSampler(model_state.model,'euler') # TODO either we assume we receive a sampler, or it is reinstantiated at each call if we allow sampler choice

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
            # Generate the first image
            # #
            first_latent = generate_image(c=C[0], uc=uc, ia=image_args, ms=model_state)
            first_sample = model_state.FS.decode_first_stage(first_latent) #to move in other function

            #save it to disk
            send_to_upscale(first_sample, os.path.join(test_dir, f"{0:05}.png"))




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

                previous_latent = process_previous_image(model_state.FS, previous_sample, xform, 
                                        video_args.color_match, color_sample, hsv= ((i % 2) == 0))

                new_latent = generate_image(c=c, x=previous_latent, uc=uc, ia=image_args, ms=model_state, t_enc=t_enc) 
                x_new = model_state.FS.decode_first_stage(new_latent)
                send_to_upscale(x_new, os.path.join(test_dir, f"{(i+1):05}.png"))
                previous_sample = x_new


            #==========COMPILING=THE=VIDEO==========
            if(video_args.video_name == None):
                video_count = 0 #len(os.listdir(video_path))
                video_name = f"video{video_count}"
            else:
                video_name = video_args.video_name
            sample_regex = os.path.join(test_dir, "%05d.png")
            command = f"ffmpeg -r {video_args.fps} -start_number {0} -i {sample_regex} -c:v libx264 -r 30 -pix_fmt yuv420p {os.path.join(test_dir, video_name)}.mp4"   #TODO do that stuff                         
            os.system(command)

    return





#Placeholder "main" code
cfg_path = 'configs/stable-diffusion/v1-inference.yaml'
optimized_cfg_path = 'optimizedSD/v1-inference.yaml'
ckpt_path = 'models/ldm/stable-diffusion-v1/model.ckpt'
model_state = ModelState()
load_model(model_state, optimized_cfg_path, ckpt_path, optimized=True)

image_args = ImageArgs()
video_args = VideoArgs()
video_args.prompts = ["A 19th century dreamy colored drawing for a child book, surrealist, clouds, yellow glowing stars, moon with a face, sun with a face, paper perspective, art station"]
video_args.fps = 20
video_args.zoom = 1.015
video_args.x = -3
video_args.frames = 120
image_args.steps = 40
image_args.W = 768

generate_video(image_args, video_args, model_state)
