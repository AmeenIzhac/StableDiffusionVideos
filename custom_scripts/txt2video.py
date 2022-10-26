# general outline :

#1. generate a starting image, with parameters handed out by client

#2. then n-1 times, generate an new image with img2img, using the last generated image

#3. use a library to aggregate the images in a video

#4. send that back to the client




#lib imports
import argparse, os, sys, glob
import cv2
import torch
import torch.nn as nn
import numpy as np
import math
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from skimage.exposure import match_histograms
import random

#repo imports
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from k_diffusion import sampling
from k_diffusion.external import CompVisDenoiser


# load safety model
#safety_model_id = "CompVis/stable-diffusion-safety-checker"
#safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
#safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


#taken from deforum's video repo
def maintain_colors(prev_img, color_match_sample, hsv=False):
    if hsv:
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else:
        return match_histograms(prev_img, color_match_sample, multichannel=True)

def sample_to_cv2(sample: torch.Tensor) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(
        np.float32
    )
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255).astype(np.uint8)
    return sample_int8


def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample


def add_noise(sample: torch.Tensor, noise_amt: float):
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def make_callback(sampler, dynamic_threshold=None, static_threshold=None):
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image after each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1, img.ndim)))
        s = np.max(np.append(s, 1.0))
        torch.clamp_(img, -1 * s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback(args_dict):
        if static_threshold is not None:
            torch.clamp_(args_dict["x"], -1 * static_threshold, static_threshold)
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict["x"], dynamic_threshold)

    # Function that is called on the image (img) and step (i) at each step
    def img_callback(img, i):
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1 * static_threshold, static_threshold)

    if sampler in ["plms", "ddim"]:
        # Callback function formated for compvis latent diffusion samplers
        callback = img_callback
    else:
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback

    return callback


def make_xform_2d(width, height, translation_x, translation_y, angle, scale):
    center = (height // 2, width // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return np.matmul(rot_mat, trans_mat)


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--prompt2",
        type=str,
        nargs="?",
        default=None,
        help="the prompt to render"
    )
    parser.add_argument(
        "--prompt3",
        type=str,
        nargs="?",
        default="a painting of a virus monster haha playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2vid-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--no_color_match",
        action="store_true",
        help="disables the color coherency feature, which keeps the color palette similar across frames"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        help="sampler used to generate or denoise images"
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="the framerate of the rendered video"
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="name of the rendered video"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=40,
        help="the number of frames in the video"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.4,
        help="the strength at which the previous image is denoised to obtain the following one. Increasing this parameter makes the video change faster, but decreasing it too much make it loose complexity"
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="the zoom into the image"
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=0.0,
        help="the angle of the rotation of the scene, in degrees per frame"
    )
    parser.add_argument(
        "--x",
        type=float,
        default=0.0,
        help="the x translation of the image"
    )
    parser.add_argument(
        "--y",
        type=float,
        default=0.0,
        help="the y translation of the image"
    )

    opt = parser.parse_args()

    print("\n\n=========================\n first bench : \n")
    os.system("nvidia-smi")
    print("=========================\n\n")

    if opt.seed < 0 :
        opt.seed = random.randint(0, 10 ** 6)
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    print("\n\n=========================\n second bench : \n")
    os.system("nvidia-smi")
    print("=========================\n\n")


    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    

    sample_path = os.path.join(outpath, "samples")
    video_path = os.path.join(outpath, "videos")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    start_number = base_count

    #listen
    print("\n\n=====LISTENING=====\n\n")

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                #listen to requests, parse the prompt, replace it with the opt.prompt here, that is currently given when running the script (not how it should happen in production ideallys) 
                print("\n\n=====OR=MAYBE=HERE=====\n\n")


                assert opt.prompt is not None
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning([""])
                c1 = model.get_learned_conditioning([opt.prompt])
                if opt.prompt2 != None:
                    c2 = model.get_learned_conditioning([opt.prompt2])

                
                #=====================FIRST_IMAGE_GENERATION=========================#
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c1,
                                                 batch_size=1,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 x_T=None)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                #x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                #x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                #x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1

                #=====================SUBSEQUENT_IMAGES_GENERATION=========================#
                #there's the thing I think I got wrong : when we do just once img2img, we take the image as noised.
                # this is an approach with works for just one image I guess. 
                # but when we do it over and over, we seem to converge to an ugly state. therefore, we must
                # noise it each time, to avoid it. I need to do that. 
                
                
                #sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
                strength = opt.strength
                noise = 0.03
                frame_count = opt.frames
                xform = make_xform_2d(opt.W, opt.H, opt.x, opt.y, opt.angle, opt.zoom)

                t_enc = int(strength * opt.ddim_steps)
                previous_sample = model.decode_first_stage(samples_ddim)
                color_sample = sample_to_cv2(previous_sample).copy()
                c = c1
                

                model_wrap = CompVisDenoiser(model)
                callback = make_callback(
                    sampler=opt.sampler,
                    dynamic_threshold=None,
                    static_threshold=None,
                )
                
                for i in trange(frame_count, desc="Generating subsequent frames"):
                    
                    if opt.prompt2 != None:
                        #semantic spherical interpolation
                        if i < 0.8 * frame_count:
                            t = i / (0.8 * frame_count)
                            c = slerp(t, c1, c2)
                        #c_slerp = c * math.sin((1-t) * omega) / sin_omega + c2 * math.sin(t * omega) / sin_omega
                        else:
                            c = c2

                    #seeding
                    opt.seed += 1
                    seed_everything(opt.seed)

                    #previous image processing (color coherency, noise, encoding)
                    previous_img = sample_to_cv2(previous_sample)
                    #if opt.zoom != 1.0:
                    previous_img = cv2.warpPerspective(
                        previous_img,
                        xform,
                        (previous_img.shape[1], previous_img.shape[0]),
                        borderMode=cv2.BORDER_WRAP
                    )
                    if not opt.no_color_match:
                        previous_img = maintain_colors(previous_img, color_sample, (i % 2) == 0)
                    previous_sample = sample_from_cv2(previous_img)
                    previous_noised = add_noise(previous_sample, noise).half().to(device)
                    previous_latent = model.get_first_stage_encoding(model.encode_first_stage(previous_noised))
                    

                    ## encode (scaled latent) -> here we noise the image, according to t_enc
                    #z_enc = sampler.stochastic_encode(previous_latent, 
                    #    torch.tensor([t_enc]).to(device))
                    ## decode it -> here we denoise it, with t_enc too, and with prompt conditioning 
                    #new_latent = sampler.decode(z_enc, c, t_enc, 
                    #    unconditional_guidance_scale= opt.scale, unconditional_conditioning=uc,)
                   
                    sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                    sigmas = sigmas[len(sigmas) - t_enc - 1 :]
                    x = (previous_latent + torch.randn([1, *shape], device=device) * sigmas[0])
                    model_wrap_cfg = CFGDenoiser(model_wrap)
                    extra_args = {
                                "cond": c,
                                "uncond": uc,
                                "cond_scale": opt.scale,
                            }
                    samples = sampling.sample_euler_ancestral(
                                    model_wrap_cfg,
                                    x,
                                    sigmas,
                                    extra_args=extra_args,
                                    disable=False,
                                    callback=callback,
                    )
                    os.system("nvidia-smi")
                    new_latent = samples


                    x_new = model.decode_first_stage(new_latent)
                    x_new_clamp = torch.clamp((x_new + 1.0) / 2.0, min=0.0, max=1.0)

                    #save the latent
                    for x_sample in x_new_clamp:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1   
                    
                    #prepare for the next iteration             
                    previous_sample = x_new

                
                #==========COMPILING=THE=VIDEO==========
                if(opt.video_name == None):
                    video_count = len(os.listdir(video_path))
                    opt.video_name = f"video{video_count}"
                sample_regex = os.path.join(sample_path, "%05d.png")
                command = f"ffmpeg -r {opt.fps} -start_number {start_number} -i {sample_regex} -c:v libx264 -r 30 -pix_fmt yuv420p {os.path.join(video_path, opt.video_name)}.mp4"                            
                os.system(command)

                #=======SEND=IT=BACK=TO=CLIENT========

if __name__ == "__main__":
    main()

