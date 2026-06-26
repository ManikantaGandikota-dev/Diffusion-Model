import torch
import torch.nn as nn

class Noise(nn.Module):
    def __init__(self, dim,device):
        super().__init__()
       
        beta = torch.linspace(0.0001, 0.02, dim)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)  
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_bar_t = self.alpha_bar.to(x.device)[t].view(-1, *([1] * (x.dim() - 1)))
        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_x, noise

    def denoise(self, noisy_x, t, pred_noise):

        shape = (-1,) + (1,) * (noisy_x.dim() - 1)
    
        alpha_t = self.alpha[t].view(shape).to(noisy_x.device)
        beta_t = self.beta[t].view(shape).to(noisy_x.device)
        alpha_bar_t = self.alpha_bar[t].view(shape).to(noisy_x.device)
    
        alpha_bar_prev = self.alpha_bar[
            torch.clamp(t - 1, min=0)
        ].view(shape).to(noisy_x.device)
    
        x0_pred = (
            noisy_x -
            torch.sqrt(1 - alpha_bar_t) * pred_noise
        ) / torch.sqrt(alpha_bar_t)
    
        mean = (
            noisy_x -
            (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        ) / torch.sqrt(alpha_t)
    
        posterior_variance = (
            beta_t *
            (1 - alpha_bar_prev) /
            (1 - alpha_bar_t)
        )
    
        if t[0] > 0:
            noise = torch.randn_like(noisy_x)
            x_prev = mean + torch.sqrt(posterior_variance) * noise
        else:
            x_prev = mean
    
        return x_prev, x0_pred

class DiffusionTransformer(nn.Module):
    def __init__(self,dim,device):
        super().__init__()
        self.noise=Noise(dim,device)

    def forward(self,x,t):
        noisy_x,noise=self.noise.add_noise(x,t)
        return noisy_x,noise

    def reverse(self,noisy_x,t,pred_noise):
        x_hat=self.noise.denoise(noisy_x,t,pred_noise)
        return x_hat