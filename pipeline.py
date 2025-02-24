import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def get_latent_from_image(input_image, models={}, seed=None, device=None, idle_device=None):

    latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
    if input_image: 
        # img2img case

        with torch.no_grad():
            
            if idle_device:
                to_idle = lambda x: x.to(idle_device)
            else:
                to_idle = lambda x: x
            
            generator = torch.Generator(device=device)
            if seed is None:
                generate.seed()
            else:
                generator.manual_seed(seed)

            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (h, w, channels) -> (batch_size, h, w, channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch_size, h, w, channels) -> (batch_size, channels, h, w)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            to_idle(encoder)

            return latents
    
def get_image_from_latent(latents, models={}, device=None, idle_device=None):
    with torch.no_grad():
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        # (batch_size, channels, h, w) -> (batche_size, h, w, channels)
        images = images.permute(0, 2, 3, 1)

        images = images.to("cpu", torch.uint8).numpy()

        return images[0]


def generate(
    prompt: str,
    uncond_prompt: str,
    input_image=None,
    strength=0.75,
    do_cfg=True,
    cfg_scale=7,
    sampler_name="ddpm",
    steps=30,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
    weight_path=None,
    word=None,
    coeff_attn=1,
    th=0.3,
    save=False,
    ):
    
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError('strength must be in ]0;1]')
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # (batch_size, seq_length) -> (batch_size, seq_length, d_embed)
            cond_context = clip(cond_tokens)

            # Convert the negative prompt into tokens using the tokenizer
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids

            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            # (batch_size, seq_length) -> (batch_size, seq_length, d_embed)
            uncond_context = clip(uncond_tokens)

            # (2 * batch_size, seq_length, d_embed) = (2 * batch_size, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert the prompt into tokens using the tokenizer
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            # (batch_size, seq_length) -> (batch_size, seq_length, d_embed) = (batch_size, 77, 768)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image: 
            # img2img case

            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (h, w, channels) -> (batch_size, h, w, channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch_size, h, w, channels) -> (batch_size, channels, h, w)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # txt2img case

            # start with random noise N(0,I)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (batch_size, 4, h/8, w/8)
            model_input = latents

            if do_cfg:
                # (batch_size, 4, h/8, w/8) -> (2 * batch_size, 4, h/8, w/8)
                model_input = model_input.repeat(2, 1, 1, 1)
                
            weight=None
            if weight_path:
                w = torch.load(weight_path + f"/{i}.pt")
                weight = {"down": w[0:6], "mid":[w[6]], "up": w[7:]}

            # model_output is the predicted noise by unet
            model_output = diffusion(model_input, context, time_embedding, i, weight=weight, word=word, coeff_attn=coeff_attn, th=th, save=save)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)

                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # remove noise predicted by unet
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)

        # (batch_size, channels, h, w) -> (batche_size, h, w, channels)
        images = images.permute(0, 2, 3, 1)

        images = images.to("cpu", torch.uint8).numpy()

        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x = new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # transformer-like

    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)

    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    # (1, 320)
    time_embedding = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    return time_embedding
