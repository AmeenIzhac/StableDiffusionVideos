import k_diffusion as K
import torch
import torch.nn as nn

class KDiffusionSampler:
    def __init__(self, m, sampler, callback=None):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler
        self.generation_callback = callback
    def get_sampler_name(self):
        return self.schedule
    def sample(self, S, conditioning, unconditional_guidance_scale, unconditional_conditioning, x_T, img2img=False, t_enc=None):
        if img2img:
            assert t_enc is not None
            sigmas = self.model_wrap.get_sigmas(S)
            sigmas = sigmas[len(sigmas) - t_enc - 1 :]
            noise = torch.randn(*x_T.shape, device=torch.device("cuda"))
            x = (x_T + noise * sigmas[0])
            model_wrap_cfg = CFGDenoiser(self.model_wrap)
            extra_args = {
                        "cond": conditioning,
                        "uncond": unconditional_conditioning,
                        "cond_scale": unconditional_guidance_scale,
                    }
            samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](
                model_wrap_cfg, x, sigmas,
                extra_args=extra_args,
                disable=False, callback=self.generation_callback)
            return samples_ddim, None


        else:
            sigmas = self.model_wrap.get_sigmas(S)
            x = x_T * sigmas[0]
            model_wrap_cfg = CFGDenoiser(self.model_wrap)
            samples_ddim = None
            samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](
                model_wrap_cfg, x, sigmas,
                extra_args={'cond': conditioning, 'uncond': unconditional_conditioning,'cond_scale': unconditional_guidance_scale},
                disable=False, callback=self.generation_callback)
            return samples_ddim, None


class CFGMaskedDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
        x_in = x
        x_in = torch.cat([x_in] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        denoised = uncond + (cond - uncond) * cond_scale

        if mask is not None:
            assert x0 is not None
            img_orig = x0
            mask_inv = 1. - mask
            denoised = (img_orig * mask_inv) + (mask * denoised)

        return denoised

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