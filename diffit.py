import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDependentSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.temporal_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, t):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        temporal = self.temporal_proj(t).reshape(B, 1, 3, self.num_heads, self.head_dim)
        q, k, v = (qkv + temporal).chunk(3, dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out


class DiffiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = TimeDependentSelfAttention(embed_dim, num_heads, bias=bias)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x, t):
        x = x + self.attn(self.norm1(x), t)
        x = x + self.mlp(self.norm2(x))
        return x


class DiffiT(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_heads, depth, mlp_ratio=4.0, bias=True):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blocks = nn.ModuleList([
            DiffiTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, bias=bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x, t):
        t = self.time_embed(t)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x, t)
        x = self.norm(x)
        x = self.head(x)
        return x


# モデルの使用例
img_size = 32  # 画像サイズ
patch_size = 4  # パッチサイズ
embed_dim = 128  # 埋め込み次元
num_heads = 4  # 注意ヘッド数
depth = 6  # トランスフォーマーブロックの深さ
batch_size = 16  # バッチサイズ

model = DiffiT(img_size, patch_size, embed_dim, num_heads, depth)

x = torch.randn(batch_size, 3, img_size, img_size)  # ダミー入力画像
t = torch.randn(batch_size, embed_dim)  # ダミー時間埋め込み

out = model(x, t)  # モデルの出力
print(out.shape)  # 出力形状の確認 (batch_size, num_patches, 3)