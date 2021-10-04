import torch
import torch.nn as nn
import math

class Attention(nn.Module):

    def __init__(self, in_channels, embed_dim):
        super(Attention, self).__init__()

        self.qkv = nn.Conv2d(in_channels,3 * embed_dim, 1)
        self.embed_dim = embed_dim
        self.out_transform = nn.Conv2d(embed_dim, in_channels, 1)

    def forward(self, x):
        N, _, H, W = x.shape
        q, k, v = self.qkv(x).reshape(N, -1, 3 * self.embed_dim).chunk(3, dim=-1)
        out = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(H*W)
        soft_out = nn.functional.softmax(out, dim=-2)
        out = torch.matmul(soft_out, v).reshape(N, self.embed_dim, H, W)
        return self.out_transform(out)

class MultiheadAttention(nn.Module):

    def __init__(self, in_channels, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        heads = [Attention(in_channels, embed_dim) for _ in range(num_heads)]
        self.heads = nn.Sequential(*heads)
        self.out_blend = nn.Conv2d(in_channels * num_heads, in_channels, 1)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=1)
        return self.out_blend(x)


class NystromAttention(nn.Module):

    def __init__(self, in_channels, embed_dim, num_landmarks):
        super(NystromAttention, self).__init__()

        self.qkv = nn.Conv2d(in_channels,3 * embed_dim, 1)
        self.embed_dim = embed_dim
        self.num_landmarks = num_landmarks
        self.out_transform = nn.Conv2d(embed_dim, in_channels, 1)


    def forward(self, x):
        N, _, H, W = x.shape
        q, k, v = self.qkv(x).reshape(N, H*W, 3 * self.embed_dim).chunk(3, dim=-1)

        # if self.num_landmarks == self.embed_dim:
        #     attn = torch.nn.functional.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
        #     out = torch.matmul(attn, v)
        # else:
        q_landmarks = q.reshape(N, self.num_landmarks, (H*W) // self.num_landmarks, self.embed_dim).mean(dim=-2).div_((H*W) ** .5)
        k_landmarks = k.reshape(N, self.num_landmarks, (H*W) // self.num_landmarks, self.embed_dim).mean(dim=-2).div_((H*W) ** .5)

        bmmp = lambda a, b: torch.einsum('...ij,...kj->...ik', a, b)
        k1 = nn.functional.softmax(bmmp(q, k_landmarks), dim=-2)
        k2 = nn.functional.softmax(bmmp(q_landmarks, k_landmarks), dim=-2)
        k3 = nn.functional.softmax(bmmp(q_landmarks, k), dim=-2)
        k2_inv = self.iterative_inv(k2)

        bmmp = lambda a, b: torch.einsum('...ij,...jk->...ik', a, b)
        out = bmmp(bmmp(k1, k2_inv), bmmp(k3, v)).add(v)

        return self.out_transform(out.reshape(N, -1, H, W))

    def iterative_inv(self, mat, n_iter=6):
        def matmul(a, b):
            return torch.einsum('...ij,...jk->...ik', a, b)

        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat
        KA = K.abs()
        V0 = KA.sum(dim=-2, keepdim=True).amax(dim=-1, keepdim=True)
        VI = KA.sum(dim=-1, keepdim=True).amax(dim=-2, keepdim=True)
        V = K.transpose(-1, -2).div(V0 * VI)
        for _ in range(n_iter):
            KV = matmul(K, V)
            V = matmul(0.25 * V, 13 * I - matmul(KV, 15 * I - matmul(KV, 7 * I - KV)))
        return V

class NystromMultiheadAttention(nn.Module):

    def __init__(self, in_channels, embed_dim, num_heads):
        super(NystromMultiheadAttention, self).__init__()
        self.num_heads = num_heads
        heads = [NystromAttention(in_channels, embed_dim, num_landmarks=4) for _ in range(num_heads)]
        self.heads = nn.Sequential(*heads)
        self.out_blend = nn.Conv2d(in_channels * num_heads, in_channels, 1)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=1)
        return self.out_blend(x)