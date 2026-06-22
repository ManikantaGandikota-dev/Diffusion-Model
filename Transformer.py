import numpy as np
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H/patch, W/patch]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x + self.pos_embed

class Transformer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(B, T, self.heads, self.dim // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, self.dim // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, self.dim // self.heads).transpose(1, 2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / (self.dim**0.5)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ff = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.ReLU(),
            nn.Linear(self.dim * 4, self.dim)
        )

    def forward(self, x):
        return self.ff(x)

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.transformer = Transformer(dim, heads)
        self.ff = FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.norm1(self.transformer(x))
        x = x + self.norm2(self.ff(x))
        return x

class block_list(nn.Module):
    def __init__(self, dim, heads, layers):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, dim, heads, layers):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_size = img_size // patch_size
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, dim)
        self.encoder = block_list(dim, heads, layers)
        self.to_pixels = nn.Linear(dim, patch_size * patch_size * in_channels)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.to_pixels(x)
        x = x.view(B, self.grid_size, self.grid_size, self.in_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, self.in_channels, self.img_size, self.img_size)
        return x
            
