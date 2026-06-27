# Recreate the model architecture
import torch

from Diffusion_Transformer import DiffusionTransformer
from Transformer import VisionTransformer
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from Train import img_size, patch_size, in_channels, dim, heads, layers, num_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionTransformer(img_size, patch_size, in_channels, dim, heads, layers, num_classes=num_classes).to(device)


state_dict = torch.load("/kaggle/input/datasets/manigandikota/chech-mnist2/dit_mnist.pth", map_location=device)

model.load_state_dict(state_dict)
model.to(device)
model.eval()


def sample_one(model, diffusion, label, timesteps=1000, image_size=64):
    model.eval()
    with torch.no_grad():
        x = torch.randn((1, 3, image_size, image_size), device=device)        
        for t in reversed(range(timesteps)):
            t_tensor = torch.tensor([t], device=device)
            pred_noise = model(x, t_tensor,class_labels=torch.tensor([label], device=device))
            x, pred_x0 = diffusion.reverse(x, t_tensor, pred_noise)
            # Show every step (change to t % 10 == 0 for faster display)
            if t%100==0:
                torchvision.utils.save_image(
                    x,
                    "one_sample.png",
                    normalize=True,
                    value_range=(-1, 1)
                )
                img = x[0].detach().cpu()
                img = (img.clamp(-1, 1) + 1) / 2  # Convert from [-1,1] to [0,1]
                img = img.permute(1, 2, 0)
            
                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.axis("off")
                plt.title("Final Generated Image")
                plt.show()

diffusion = DiffusionTransformer(dim=1000, device=device).to(device)

label=9
sample_one(model, diffusion,label, timesteps=1000,image_size=32)
