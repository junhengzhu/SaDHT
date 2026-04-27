import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module): #ctf
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.last_attn = None
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        x,x1 = torch.split(x,[225,225], dim=1)
        _, _, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        # head_num*head_dim = dim
        qkv = self.to_qkv(x).chunk(3, dim = -1) # split to_qkv(x) from [b, n, inner_dim*3] to [b, n, inner_dim]*3 tuple
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        qkv1 = self.to_qkv(x1).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv1)
        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        dots1 = torch.einsum('bhid,bhjd->bhij', q1, k1) * self.scale
        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1) # [b,head_num,n,n]
        attn1 = dots1.softmax(dim=-1)  # [b,head_num,n,n]
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v1) # [b,head_num,n,head_dim]
        out1 = torch.einsum('bhij,bhjd->bhid', attn1, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out1 = self.to_out(out1)
        out = torch.cat((out,out1),dim=1)
        return out

class Attention2(nn.Module): #sf
    def __init__(self, dim, heads, dim_head, dropout,return_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.last_attn = None
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.return_attention = return_attention
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        _, _, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        # head_num*head_dim = dim
        qkv = self.to_qkv(x).chunk(3, dim = -1) # split to_qkv(x) from [b, n, inner_dim*3] to [b, n, inner_dim]*3 tuple
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        self.last_attn = attn
        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1) # [b,head_num,n,n]
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v) # [b,head_num,n,head_dim]
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if self.return_attention:
            return out, attn  # 返回输出和注意力权重
        else:
            return out

class Attention1(nn.Module): #mgmc
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, proj_drop=0., expand_ratio=2):
        super().__init__()
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads
        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."
        self.last_attn = None
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.split_groups = self.dim // ca_num_heads
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)
        for i in range(self.ca_num_heads):
            local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(1 + i * 2),
                                   padding=(0 + i), stride=1, groups=dim // self.ca_num_heads)
            setattr(self, f"local_conv_{i + 1}", local_conv)
        self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups)
        self.bn = nn.BatchNorm2d(dim * expand_ratio)
        self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)


    def forward(self, x):
        B, N, C = x.shape # 64,169,64
        v = self.v(x)
        H = int(math.sqrt(N))
        W = H
        s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]
            s_i = local_conv(s_i)
            s_i = s_i.reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 2)
        s_out = s_out.reshape(B, C, H, W)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
        self.modulator = s_out
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
        x = s_out * v
        x = self.proj(x)
        x = self.proj_drop(x)

        return x




class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, mode):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention1(dim=dim))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))),
                Residual(PreNorm(dim, Attention2(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))),
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
            ]))
        self.mode = mode

    def forward(self, x, mask = None):

        if self.mode == 'MViT':
            for attn1, ff1, attn2, ff2, attn3, ff3 in self.layers:
                x = attn1(x)
                x = ff1(x)
                x = attn2(x, mask=mask)
                x = ff2(x)
            return x
        elif self.mode == 'MSA':
            for attn1, ff1, attn2, ff2 ,attn3, ff3  in self.layers:
                x = attn2(x,mask=mask)
                x = ff2(x)
                # x = attn4(x, mask=mask)
                # x = ff4(x)
            return x
        else:
            for attn1, ff1, attn2, ff2 ,attn3, ff3  in self.layers:
                # x = attn(x)
                # x = ff(x)
                x = attn3(x, mask=mask)
                x = ff3(x)
            return x






class MViT(nn.Module):
    def __init__(self, patch_size, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., mode='MViT'):
        super().__init__()

        nout = 16
        self.intermediate_features = {}
        samesize = 1
        self.separable1 = nn.Sequential(
            nn.Conv2d(num_patches[0], num_patches[0], kernel_size=3, padding=samesize, groups=num_patches[0]),
            nn.Conv2d(num_patches[0], nout, kernel_size=1),
            nn.BatchNorm2d(nout),
            nn.GELU(),
            nn.Conv2d(nout, nout, kernel_size=3, padding=samesize, groups=nout),
            nn.Conv2d(nout, nout*2, kernel_size=1),
            nn.BatchNorm2d(nout*2),
            nn.GELU(),
            nn.Conv2d(nout*2, nout*2, kernel_size=3, padding=samesize, groups=nout*2),
            nn.Conv2d(nout*2, nout*4, kernel_size=1),
            nn.BatchNorm2d(nout*4),
            nn.GELU()
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(num_patches[0], num_patches[0], kernel_size=3, padding=samesize),
            nn.Conv2d(num_patches[0], nout, kernel_size=1),
            nn.BatchNorm2d(nout),
            nn.GELU(),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(nout, nout, kernel_size=3, padding=samesize),
            nn.Conv2d(nout, nout * 2, kernel_size=1),
            nn.BatchNorm2d(nout * 2),
            nn.GELU(),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(nout * 2 , nout * 2, kernel_size=3, padding=samesize),
            nn.Conv2d(nout * 2, nout * 4, kernel_size=1),
            nn.BatchNorm2d(nout * 4),
            nn.GELU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_patches[0], 64, kernel_size=3, padding=samesize),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_patches[1], 64, kernel_size=3, padding=samesize),
            nn.GELU(),
        )
        self.separable2 = nn.Sequential(
            nn.Conv2d(num_patches[1], num_patches[1], kernel_size=3, padding=samesize, groups=num_patches[1]),
            nn.Conv2d(num_patches[1], nout, kernel_size=1),
            nn.BatchNorm2d(nout),
            nn.GELU(),
            nn.Conv2d(nout, nout, kernel_size=3, padding=samesize, groups=nout),
            nn.Conv2d(nout, nout*2, kernel_size=1),
            nn.BatchNorm2d(nout*2),
            nn.GELU(),
            nn.Conv2d(nout*2, nout*2, kernel_size=3, padding=samesize, groups=nout*2),
            nn.Conv2d(nout*2, nout*4, kernel_size=1),
            nn.BatchNorm2d(nout*4),
            nn.GELU()
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(num_patches[1], num_patches[1], kernel_size=3, padding=samesize),
            nn.Conv2d(num_patches[1], nout, kernel_size=1),
            nn.BatchNorm2d(nout),
            nn.GELU(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(nout, nout, kernel_size=3, padding=samesize),
            nn.Conv2d(nout, nout * 2, kernel_size=1),
            nn.BatchNorm2d(nout * 2),
            nn.GELU(),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(nout*2, nout * 2, kernel_size=3, padding=samesize),
            nn.Conv2d(nout * 2, nout * 4, kernel_size=1),
            nn.BatchNorm2d(nout * 4),
            nn.GELU()
        )

        grid_size = 1
        vit_patches = (patch_size // grid_size) ** 2
        self.to_patch_embedding2 = nn.Linear(nout*4, dim)
        self.to_patch_embedding2c = nn.Linear(nout*4, dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, vit_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth-4, heads, dim_head, mlp_dim, dropout, mode='MSA')
        self.transformer1 = Transformer(dim, depth-4, heads, dim_head, mlp_dim, dropout, mode='MViT')
        self.transformer2 = Transformer(dim, depth-5, heads, dim_head, mlp_dim, dropout, mode='MViT')
        
        self.mlp_head0 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Softmax(dim=1)
            )

        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
            )
        self.attention_maps = []
        self.intermediate_features = {}
        self.hooks = []

    def _register_hooks(self):
        self.intermediate_features = {}
        self.hooks = []
        att1_counter = 0
        att2_counter = 0

        def hook_fn(layer_type, idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                key = f'{layer_type}_Layer{idx}'
                self.intermediate_features[key] = out.detach()
                print(f"Hook captured: {key}")  # 调试用

            return hook

        for name, module in self.named_modules():
            class_name = module.__class__.__name__
            if class_name == 'Attention1':
                self.hooks.append(module.register_forward_hook(hook_fn('Attention1', att1_counter)))
                att1_counter += 1
            elif class_name == 'Attention2':
                self.hooks.append(module.register_forward_hook(hook_fn('Attention2', att2_counter)))
                att2_counter += 1

        print(f"Registered {len(self.hooks)} hooks (Attention1: {att1_counter}, Attention2: {att2_counter})")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    def forward(self, x1, x2, mask = None, return_intermediate=False):
        self.intermediate_features = {}

        self.feature_maps = []
        self.attention_maps = []

        x1 = self.separable1(x1)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x1 = self.to_patch_embedding2(x1) #[b, n, c to dim], n=hw
        b, n, _ = x1.shape
        x1 += self.pos_embedding[:, :n]
        x1 = self.dropout(x1)




        # x2 = self.csa(x2)
        # x2 = self.conv2_3(x2)
        # x2 = self.csa1(x2)
        x2 = self.separable2(x2)
        # x2 = self.conv2(x2)
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x2 = self.to_patch_embedding2c(x2) #common subpsace projection: better to be different with self.to_patch_embedding1
        x2 += self.pos_embedding[:, :n]
        x2 = self.dropout(x2)



        # Attention Fusion
        # x  = torch.cat((x1,x2),dim=1)
        x1 = self.transformer1(x1)
        x2 = self.transformer2(x2)
        
        x = torch.cat((x1,x2), dim=1)
        x = self.transformer(x)
        
        # MLP Pre-Head & Head
        xs = self.mlp_head0(x).squeeze(-1)  # 仅压缩最后一维，保留 [b, n]
        if xs.dim() == 1:  # 当 n=1 时，补充序列维度
            xs = xs.unsqueeze(1)
        x = torch.einsum('bn,bnd->bd', xs, x)
        
        x = self.mlp_head1(x)
        if return_intermediate:
            return x, self.intermediate_features
        return x


