import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Diffusion_Transformer import DiffusionTransformer
from Transformer import VisionTransformer
from Transformer import block_list
from dataset import train_loader, test_loader

epoch = 1000
img_size = 32
patch_size = 4
in_channels = 3
dim = 512
heads = 8
layers = 6
timesteps = 1000
learning_rate = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")

model = VisionTransformer(img_size, patch_size, in_channels, dim, heads, layers).to(device)
diffusion = DiffusionTransformer(timesteps)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for i in range(epoch+1):
    model.train()
    total_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        t = torch.randint(0, timesteps, (images.size(0),)).to(device)
        noisy_images, noise = diffusion.forward(images, t)
        outputs = model(noisy_images)
        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    if i%100 == 0:
        print(f"Epoch [{i}/{epoch}], Loss: {avg_loss:.4f}")
