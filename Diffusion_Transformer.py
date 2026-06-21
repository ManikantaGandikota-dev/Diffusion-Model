import numpy as np

class noise:
    def __init__(self,dim):
        self.dim=dim
        self.beta=np.linspace(0.0001,0.02,self.dim)
        self.alpha=1-self.beta
        self.alpha_bar=np.cumprod(self.alpha)

    def add_noise(self,x,t):
        noise=np.random.randn(*x.shape)
        noisy_x=np.sqrt(self.alpha_bar[t])*x+np.sqrt(1-self.alpha_bar[t])*noise
        return noisy_x,noise

    def denoise(self,noisy_x,t):
        x_hat=noisy_x/np.sqrt(self.alpha_bar[t])
        return x_hat

