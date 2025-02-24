import torch
import numpy as np

class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=30):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        #TODO: handle exception when empty
        self.timesteps = torch.from_numpy(timesteps)

    def add_noise(self, x: torch.FloatTensor, t: torch.IntTensor) -> torch.FloatTensor:
        # from DDPM paper equations

        alpha_cumprod = self.alpha_cumprod.to(device=x.device, dtype=x.dtype)
        t = t.to(device=x.device)

        mean_coeff = alpha_cumprod[t] ** 0.5
        mean_coeff = mean_coeff.flatten()

        while len(mean_coeff.shape) < len(x.shape):
            mean_coeff = mean_coeff.unsqueeze(-1)

        stdev = (1 - alpha_cumprod[t]) ** 0.5
        stdev = stdev.flatten()

        while len(stdev.shape) < len(x.shape):
            stdev = stdev.unsqueeze(-1)

        noise = torch.randn(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)

        noisy_samples = mean_coeff * x + stdev * noise

        return noisy_samples
    
    def _get_previous_timestep(self, t: int) -> int:
        prev_t = t - self.num_training_steps // self.num_inference_steps
        return prev_t
    
    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    
    def step(self, t: int, x: torch.Tensor, model_output: torch.Tensor):
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        alpha_t = alpha_prod_t / alpha_prod_t_prev
        beta_t = 1 - alpha_t

        # using formula (15) of DDPM paper
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # using formula (7)
        coeff_x0 = alpha_prod_t_prev ** 0.5 * beta_t / beta_prod_t
        coeff_xt = alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t
        
        pred_mean = coeff_x0 * pred_x0 + coeff_xt * x

        stdev = 0

        if t > 0:
            variance = beta_prod_t_prev / beta_prod_t * beta_t
            variance = torch.clamp(variance, min=1e-20)
            stdev = variance ** 0.5

            noise = torch.randn(model_output.shape, generator=self.generator, device=model_output.device, dtype=model_output.dtype)

            stdev = stdev * noise

        pred_x_prev = pred_mean + stdev

        return pred_x_prev
