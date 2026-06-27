import torch
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from Transformer import VisionTransformer
from Diffusion_Transformer import DiffusionTransformer
from Train import img_size, patch_size, in_channels, dim, heads, layers, num_classes, timesteps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "dit_mnist.pth"

model = VisionTransformer(
    img_size,
    patch_size,
    in_channels,
    dim,
    heads,
    layers,
    num_classes=num_classes,
).to(device)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()


diffusion = DiffusionTransformer(
    timesteps=timesteps,
    device=device,
)

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

            yield Image.fromarray(img)

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