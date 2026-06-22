import torch

class noise:
    def __init__(self, dim):
        self.dim = dim
        self.beta = torch.linspace(0.0001, 0.02, self.dim)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_bar_t = self.alpha_bar[t].view(-1, *([1] * (x.dim() - 1))).to(x.device)
        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_x, noise

    def denoise(self, noisy_x, t):
        alpha_bar_t = self.alpha_bar[t].view(-1, *([1] * (noisy_x.dim() - 1))).to(noisy_x.device)
        x_hat = noisy_x / torch.sqrt(alpha_bar_t)
        return x_hat

class DiffusionTransformer:
    def __init__(self,dim):
        self.noise=noise(dim)

    def forward(self,x,t):
        noisy_x,noise=self.noise.add_noise(x,t)
        return noisy_x,noise

    def reverse(self,noisy_x,t):
        x_hat=self.noise.denoise(noisy_x,t)
        return x_hat