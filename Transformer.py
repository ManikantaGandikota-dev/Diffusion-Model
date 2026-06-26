import numpy as np
import torch
import torch.nn as nn

import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
        assert dim % heads == 0
        self.head_dim = self.dim // self.heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(B, T, self.heads, self.dim // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, self.dim // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, self.dim // self.heads).transpose(1, 2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / (self.head_dim**0.5)
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.proj_drop(self.proj(out))
        return out



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.dim = dim

    def forward(self, t):
        device = t.device

        half_dim = self.dim // 2

        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half_dim, device=device) /
            max(half_dim - 1,1)
        )

        args = t[:, None].float() * freqs[None]

        emb = torch.cat(
            [torch.sin(args), torch.cos(args)],
            dim=-1
        )

        return self.mlp(emb)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ff = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.dim * 4, self.dim),
            nn.Dropout(0.1)
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

    def forward(self, x,class_embed):
        if class_embed is not None:
            cond = class_embed.unsqueeze(1)
            x = x + self.transformer(self.norm1(x+cond))
            x = x + self.ff(self.norm2(x+cond))
        else:
            x = x + self.transformer(self.norm1(x))
            x = x + self.ff(self.norm2(x))
        
        return x

class block_list(nn.Module):
    def __init__(self, dim, heads, layers):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])

    def forward(self, x,class_embed):
        for block in self.blocks:
            x = block(x,class_embed)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, dim, heads, layers, num_classes=10):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_size = img_size // patch_size
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, dim)
        self.time_embedding = TimeEmbedding(dim)
        
        # Class conditioning
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, dim)
        
        self.encoder = block_list(dim, heads, layers)
        self.to_pixels = nn.Linear(dim, patch_size * patch_size * in_channels)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x,t, class_labels=None):
        B = x.shape[0]
        x = self.patch_embedding(x)
        time_embed = self.time_embedding(t)
        class_embed = None
        
        # Add class conditioning if provided
        if class_labels is not None:
            class_embed = self.class_embedding(class_labels)  # [B, dim]
            cond=class_embed + time_embed
        else:
            cond=time_embed
        
        x = self.encoder(x,cond)
        x = self.final_norm(x)
        
        x = self.to_pixels(x)
        x = x.view(B, self.grid_size, self.grid_size, self.in_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.contiguous().view(B, self.in_channels, self.img_size, self.img_size)
        return x
            
