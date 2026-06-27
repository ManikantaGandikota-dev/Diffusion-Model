import torch
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from Transformer import VisionTransformer
from Diffusion_Transformer import DiffusionTransformer
from huggingface_hub import hf_hub_download

epoch = 1000
img_size = 32
display_size = 256
patch_size = 4
in_channels = 3
dim = 512
heads = 8
layers = 6
timesteps = 1000
learning_rate = 1e-4
num_classes = 10
cfg_prob = 0.1 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = hf_hub_download(
    repo_id="Manikantagandikota143/Diffusion_model",
    filename="dit_mnist.pth"
)

model = VisionTransformer(
    img_size,
    patch_size,
    in_channels,
    dim,
    heads,
    layers,
    num_classes=num_classes,
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()


diffusion = DiffusionTransformer(timesteps,device=device).to(device)

@torch.no_grad()
def generate_image(label):

    x = torch.randn(
        (1, 3, img_size, img_size),
        device=device,
    )

    label = torch.tensor([label], device=device)

    for t in reversed(range(timesteps)):

        tt = torch.tensor([t], device=device)

        pred_noise = model(x, tt, class_labels=label)

        x, _ = diffusion.reverse(x, tt, pred_noise)

        if t % 10 == 0:

            img = x[0].cpu()
            img = (img.clamp(-1, 1) + 1) / 2
            img = img.permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img).resize((display_size, display_size), resample=Image.BICUBIC)

            yield img

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Dropdown(
        choices=[0,1,2,3,4,5,6,7,8,9],
        value=0,
        label="Digit"
    ),
    outputs=gr.Image(type="pil"),
    title="MNIST Diffusion Transformer",
    description="Generate handwritten digits using a Diffusion Transformer (DiT).",
    examples=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]],
)

if __name__ == "__main__":
    demo.launch()