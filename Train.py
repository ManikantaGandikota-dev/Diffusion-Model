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
num_classes = 10
cfg_prob = 0.1  # Classifier-free guidance: probability of training without labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")

model = VisionTransformer(img_size, patch_size, in_channels, dim, heads, layers, num_classes=num_classes).to(device)
diffusion = DiffusionTransformer(timesteps)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')

for i in range(epoch+1):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        t = torch.randint(0, timesteps, (images.size(0),)).to(device)
        noisy_images, noise = diffusion.forward(images, t)
        
        # Classifier-free guidance: sometimes train without labels
        if np.random.rand() < cfg_prob:
            # Train without conditioning (unconditional)
            outputs = model(noisy_images, class_labels=None)
            loss = criterion(outputs, noise)
        else:
            # Train with conditioning (conditional)
            outputs = model(noisy_images, class_labels=labels)
            loss = criterion(outputs, noise)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    if i % 100 == 0:
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                t = torch.randint(0, timesteps, (images.size(0),)).to(device)
                noisy_images, noise = diffusion.forward(images, t)
                
                # Evaluate with conditioning
                outputs = model(noisy_images, class_labels=labels)
                loss = criterion(outputs, images)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch [{i}/{epoch}], Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        model.train()
