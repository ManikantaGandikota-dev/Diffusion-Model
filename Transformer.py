import numpy as np
import torch
import torch.nn as nn

class Transformer:
    def __init__(self,dim,heads):
        self.dim=dim
        self.heads=heads
        self.q=nn.Linear(dim,dim)
        self.k=nn.Linear(dim,dim)
        self.v=nn.Linear(dim,dim)

    def forward(self,x):
        B,T,D=x.shape
        q=self.q(x)
        k=self.k(x)
        v=self.v(x)

        q=q.view(B,T,self.heads,self.dim//self.heads).transpose(1,2)
        k=k.view(B,T,self.heads,self.dim//self.heads).transpose(1,2)
        v=v.view(B,T,self.heads,self.dim//self.heads).transpose(1,2)

        attn=q@k.transpose(-2,-1)
        attn=attn/(self.dim**0.5)
        attn=attn.softmax(dim=-1)

        out=attn@v
        out=out.transpose(1,2).contiguous().view(B,T,D)
        return out
    
class FeedForward:
    def __init__(self,dim):
        self.dim=dim
        self.ff=nn.Sequential(
                    nn.Linear(self.dim,self.dim*4),
                    nn.ReLU(),
                    nn.Linear(self.dim*4,self.dim)
                )
    def forward(self,x):
        return self.ff(x)
        
class Block:
    def __init__(self,dim,heads):
        self.transformer=Transformer(dim,heads)
        self.ff=FeedForward(dim)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
    def forward(self,x):
        x=x+self.norm1(self.transformer.forward(x))
        x=x+self.norm2(self.ff.forward(x))
        return x

class block_list:
    def __init__(self,dim,heads,layers):
        self.layers=layers
        self.blocks=[Block(dim,heads) for _ in range(layers)]
    def forward(self,x):
        for block in self.blocks:
            x=block.forward(x)
        return x
            
