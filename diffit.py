import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# --- Helper Modules ---
class TimeEmbeddings(nn.Module):
    """
    Converts time steps into embeddings using sinusoidal positional embeddings
    followed by an MLP.
    Based on Figure 3 & 4 and standard practice (e.g., DDPM, DiT).
    """
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        assert embedding_dim % 2 == 0
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor):
        # t shape: (batch_size,)
        # half_dim = self.embedding_dim // 2 # Original code used hidden_dim here, paper implies input dim
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        # emb shape: (half_dim,)
        emb = t[:, None] * emb[None, :]
        # emb shape: (batch_size, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # emb shape: (batch_size, embedding_dim)
        return self.mlp(emb) # shape: (batch_size, hidden_dim)

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class MLP(nn.Module):
    """ Standard FeedForward MLP block used in Transformers """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(), # Or GeLU, paper often uses Swish/SiLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2) # B, embed_dim, H/p, W/p -> B, embed_dim, N -> B, N, embed_dim
        return x

# --- Core DiffiT Components ---

class TMSA(nn.Module):
    """
    Time-dependant Multihead Self Attention (TMSA)
    As described in Section 3.2, Equations 3-6.
    Includes relative position bias.
    """
    def __init__(self, dim: int, time_embed_dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, num_patches: int = 196):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Linear projections for spatial (s) and temporal (t) components
        # Note: The paper implies separate weights Wqs, Wqt, etc.
        # Let's implement it this way. dim_in_s = dim, dim_in_t = time_embed_dim
        self.Wqs = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wqt = nn.Linear(time_embed_dim, dim, bias=qkv_bias)
        self.Wks = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wkt = nn.Linear(time_embed_dim, dim, bias=qkv_bias)
        self.Wvs = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wvt = nn.Linear(time_embed_dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # --- Relative Position Bias ---
        # Simplified implementation: Learnable embedding table based on relative coordinates
        self.num_patches = num_patches
        self.grid_size = int(num_patches ** 0.5) # Assumes square patch grid
        assert self.grid_size * self.grid_size == num_patches, "num_patches must be a perfect square for RPB"

        # Define the maximum relative distance
        self.rel_pos_max_dist = 2 * self.grid_size - 1
        # Embedding table size: (2*max_dist - 1) x (2*max_dist - 1) x num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.rel_pos_max_dist, self.rel_pos_max_dist, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # Get relative coordinates or indices for lookup
        coords_h = torch.arange(self.grid_size)
        coords_w = torch.arange(self.grid_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, N
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        # Shift to start from 0
        relative_coords[:, :, 0] += self.grid_size - 1
        relative_coords[:, :, 1] += self.grid_size - 1
        # Ensure indices are within bounds (should be due to calculation)
        # relative_coords[:, :, 0] *= (self.relative_position_bias_table.shape[1] // self.rel_pos_max_dist) # Scaling if table size != max_dist
        # relative_coords[:, :, 1] *= (self.relative_position_bias_table.shape[0] // self.rel_pos_max_dist) # No scaling needed here
        self.register_buffer("relative_position_index", relative_coords.sum(-1)) # N, N
        # --- End Relative Position Bias ---


    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        # x_s shape: (B, N, D) - Spatial tokens (image patches)
        # x_t shape: (B, D_time) - Time embedding token
        B, N, D = x_s.shape
        assert N == self.num_patches, f"Input spatial token number ({N}) doesn't match RPB patches ({self.num_patches})."

        # Add dimension for broadcasting time token
        x_t = x_t.unsqueeze(1) # (B, 1, D_time)

        # Calculate Q, K, V using Eq. 3, 4, 5
        q = self.Wqs(x_s) + self.Wqt(x_t) # (B, N, D) + (B, 1, D) -> (B, N, D) via broadcasting
        k = self.Wks(x_s) + self.Wkt(x_t) # (B, N, D)
        v = self.Wvs(x_s) + self.Wvt(x_t) # (B, N, D)

        # Reshape for multi-head attention
        # (B, N, D) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        # Calculate attention scores: (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # --- Add Relative Position Bias ---
        # Look up bias values: self.relative_position_index is (N, N)
        # We need to select values based on this index from the table (max_dist, max_dist, H)
        # This index directly maps 2D relative position to a single index for a flattened table,
        # but our table is 2D. We need indices for height and width relative distances.
        # Let's re-calculate indices for the 2D table lookup.
        relative_position_indices_h = self.relative_position_index // self.rel_pos_max_dist # Use integer division
        relative_position_indices_w = self.relative_position_index % self.rel_pos_max_dist # Use modulo
        
        # Lookup: table is (max_dist_h, max_dist_w, H), indices are (N, N)
        rel_pos_bias = self.relative_position_bias_table[
            relative_position_indices_h.view(-1), # Flatten indices for lookup
            relative_position_indices_w.view(-1)
        ].view(N, N, -1) # Reshape back to (N, N, H)
        
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()  # H, N, N
        attn = attn + rel_pos_bias.unsqueeze(0) # Add bias (broadcast batch dim B)
        # --- End Relative Position Bias ---


        # Apply softmax and dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values: (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
        x = (attn @ v)

        # Reshape back: (B, H, N, d) -> (B, N, H, d) -> (B, N, D)
        x = x.transpose(1, 2).reshape(B, N, D)

        # Apply output projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DiffiTTransformerBlock(nn.Module):
    """
    The core Transformer block for DiffiT (ViT-style).
    Uses TMSA. Based on Section 3.2, Equations 7, 8.
    """
    def __init__(self, dim: int, time_embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, dropout: float = 0.0, attn_drop: float = 0.0,
                 num_patches: int = 196): # Need num_patches for TMSA RPB
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = TMSA(
            dim=dim, time_embed_dim=time_embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=dropout, num_patches=num_patches
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, hidden_dim=mlp_hidden_dim, dropout=dropout)

        # NOTE: The paper applies dropout after the MLP output and after the attention output projection.
        # We include proj_drop in TMSA and dropout in MLP.

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        # x_s shape: (B, N, D) - Spatial tokens
        # x_t shape: (B, D_time) - Time embedding token

        # Eq 7: x_hat_s = TMSA(LN(x_s), x_t) + x_s
        attn_output = self.attn(self.norm1(x_s), x_t)
        x_s = x_s + attn_output # Residual connection 1

        # Eq 8: x_s = MLP(LN(x_hat_s)) + x_hat_s (using updated x_s as x_hat_s)
        mlp_output = self.mlp(self.norm2(x_s))
        x_s = x_s + mlp_output # Residual connection 2

        return x_s


class DiffiTResBlock(nn.Module):
    """
    Residual block for Image Space DiffiT (U-Net style).
    Combines a convolutional block and a DiffiT Transformer block.
    Based on Section 4.1, Equations 9, 10 and Figure 4 diagram.
    """
    def __init__(self, dim: int, time_embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, dropout: float = 0.0, attn_drop: float = 0.0,
                 num_patches: int = 196, # Patches at this resolution level
                 use_conv: bool = True # Flag to include the conv path
                 ):
        super().__init__()
        self.use_conv = use_conv
        if self.use_conv:
            # Convolutional path (Eq 9 describes output, Fig 4 shows parallel path)
            # Assumes input x is (B, C, H, W) format for conv
            self.norm_conv = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True) # Typical GN params
            self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
            self.act_conv = nn.SiLU()

        # Transformer path (Eq 10 involves this block)
        self.norm_tfm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = TMSA(
            dim=dim, time_embed_dim=time_embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=dropout, num_patches=num_patches
        )
        self.norm_tfm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, hidden_dim=mlp_hidden_dim, dropout=dropout)


    def forward(self, x: torch.Tensor, x_t: torch.Tensor):
        # x shape: (B, C, H, W) assumed for conv path compatibility
        # x_t shape: (B, D_time)
        B, C, H, W = x.shape
        # Store residual
        residual = x

        # --- Convolutional Path (Eq 9) ---
        if self.use_conv:
            # x_hat_s = Conv(Act(Norm(xs)))
            x_hat_s = self.norm_conv(x)
            x_hat_s = self.act_conv(x_hat_s)
            x_hat_s = self.conv(x_hat_s) # Output: (B, C, H, W)
        else:
            # If no conv path, the input to Tfm block is the original input
            x_hat_s = x

        # --- Transformer Path (Part of Eq 10) ---
        # Input to DiffiT-Block is x_hat_s (output of conv path)
        # Reshape input for Transformer: (B, C, H, W) -> (B, N, C) where N = H*W
        x_tfm_in = x_hat_s.flatten(2).transpose(1, 2) # (B, N, C)

        # Apply Transformer blocks (LN -> TMSA -> Add -> LN -> MLP -> Add)
        # This corresponds to DiffiT-Block(x_hat_s, x_t) in Eq 10
        attn_output = self.attn(self.norm_tfm1(x_tfm_in), x_t)
        x_tfm_mid = x_tfm_in + attn_output # Residual 1 within Tfm path
        mlp_output = self.mlp(self.norm_tfm2(x_tfm_mid))
        x_tfm_out = x_tfm_mid + mlp_output # Residual 2 within Tfm path

        # Reshape output back: (B, N, C) -> (B, C, N) -> (B, C, H, W)
        diffit_block_output = x_tfm_out.transpose(1, 2).view(B, C, H, W)

        # --- Combine according to Eq 10 ---
        # xs = DiffiT-Block(x_hat_s, xt) + xs (where xs is the original residual)
        output = diffit_block_output + residual

        return output

# --- Main Models ---

class LatentDiffiT(nn.Module):
    """
    Latent Diffusion Vision Transformer (DiffiT).
    ViT architecture without up/downsampling in the main body.
    Based on Section 3.2 and Figure 3.
    """
    def __init__(self, img_size=32, patch_size=2, in_channels=4, embed_dim=768, time_embed_dim=256,
                 depth=12, num_heads=12, mlp_ratio=4.0, num_classes=1000,
                 qkv_bias=True, dropout=0.0, attn_drop=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.time_embed = TimeEmbeddings(embedding_dim=time_embed_dim, hidden_dim=embed_dim)

        # if self.num_classes > 0:
        #      # Classifier-free guidance requires ability to condition on class
        #      self.label_embed = LabelEmbedder(num_classes=num_classes, hidden_size=embed_dim, dropout_prob=0.0)
        #      # Need to combine time and label embeddings. Simple addition is common.
        #      # Input tokens will be: patch_tokens + pos_embed + combined_time_label_embed
        #      # For simplicity here, let's assume time+label embed is passed as x_t to blocks.
        #      # The paper doesn't specify how label embedding is integrated in TMSA.
        #      # Common practice in DiT is to add time, label, and positional embeds before first layer.
        #      # Let's make time_embed output embed_dim and add label embed to it.
        #      self.time_embed = TimeEmbeddings(embedding_dim=time_embed_dim, hidden_dim=embed_dim) # Output embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.blocks = nn.ModuleList([
            DiffiTTransformerBlock(
                dim=embed_dim, time_embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, dropout=dropout, attn_drop=attn_drop, num_patches=num_patches
            )
            for _ in range(depth)])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # Final layer predicts the noise (same shape as input patches)
        self.decoder_pred = nn.Linear(embed_dim, patch_size * patch_size * in_channels, bias=True)

        # Initialize weights
        self.apply(self._init_weights)


    def _init_weights(self, m):
         if isinstance(m, nn.Linear):
             nn.init.trunc_normal_(m.weight, std=.02)
             if isinstance(m, nn.Linear) and m.bias is not None:
                 nn.init.constant_(m.bias, 0)
         elif isinstance(m, nn.LayerNorm):
             nn.init.constant_(m.bias, 0)
             nn.init.constant_(m.weight, 1.0)
         elif isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Or other init
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0)


    def unpatchify(self, x):
        """ Converts token sequence back into image. """
        # x: (B, N, P*P*C)
        B, N, C_full = x.shape
        P = self.patch_size
        C = C_full // (P * P)
        H_patch = W_patch = int(N ** 0.5)
        assert H_patch * W_patch == N, "Output token count must be a perfect square."

        # Reshape: (B, N, P*P*C) -> (B, N, C, P, P) -> (B, Hp, Wp, C, P, P)
        x = x.view(B, N, C, P, P)
        x = x.view(B, H_patch, W_patch, C, P, P)
        # Permute: (B, C, H*, W*) -> (B, C, H_patch, P, W_patch, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        # Reshape: -> (B, C, H_patch*P, W_patch*P)
        img = x.view(B, C, H_patch * P, W_patch * P)
        return img


    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None):
        # x: (B, C, H, W) - Noisy latent image
        # t: (B,) - Timesteps
        # y: (B,) - Class labels (optional)

        B, C, H, W = x.shape
        x_s = self.patch_embed(x) # (B, N, D)
        x_s = x_s + self.pos_embed # Add positional embedding

        # Prepare time and optional label embedding
        x_t = self.time_embed(t) # (B, D)
        # if y is not None and self.num_classes > 0:
        #     label_emb = self.label_embed(y, self.training) # (B, D)
        #     x_t = x_t + label_emb # Combine by adding

        # Apply Transformer blocks
        for block in self.blocks:
            x_s = block(x_s, x_t)

        x_s = self.norm(x_s)
        x_s = self.decoder_pred(x_s) # Predict noise per patch: (B, N, P*P*C_in)

        # Reshape back to image format
        noise_pred = self.unpatchify(x_s) # (B, C_in, H, W)

        return noise_pred


class ImageSpaceDiffiT(nn.Module):
    """
    Image Space Diffusion Vision Transformer (DiffiT).
    U-Net architecture with DiffiTResBlocks.
    Based on Section 3.2, Section 4.1 and Figure 4.
    Uses windowed attention implicitly if DiffiTResBlock is adapted.
    """
    def __init__(self, img_size=64, in_channels=3, out_channels=3, embed_dim=128, time_embed_dim=256,
                 depths=[2, 2, 4, 4], num_heads=[4, 4, 8, 8], mlp_ratio=4.0, num_classes=1000,
                 qkv_bias=True, dropout=0.0, attn_drop=0.0,
                 patch_size=4 # Initial patch size for tokenization (or use Conv tokenizer)
                 ):
        super().__init__()
        assert len(depths) == len(num_heads)
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # --- Time Embedding ---
        self.time_embed = TimeEmbeddings(embedding_dim=time_embed_dim, hidden_dim=int(embed_dim * mlp_ratio))
        # Project time embedding to the dimension needed by TMSA in ResBlocks
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim) # Match ResBlock's time_embed_dim input
        )
        # --- Class Label Embedding ---
        # if num_classes > 0:
        #     self.label_embed = LabelEmbedder(num_classes=num_classes, hidden_size=embed_dim, dropout_prob=0.0)
        #     # Combine time and label embeddings
        #     self.time_mlp = nn.Sequential(
        #         nn.SiLU(),
        #         nn.Linear(int(embed_dim * mlp_ratio), embed_dim) # Match ResBlock's time_embed_dim input
        #     )


        # --- Input Tokenizer ---
        # Fig 4 uses Conv 3x3. Let's follow that.
        self.tokenizer = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)
        current_dim = embed_dim
        current_size = img_size

        # --- Encoder ---
        self.encoder_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i in range(self.num_levels):
            resblocks = nn.ModuleList()
            num_patches_level = (current_size * current_size) # Assuming square
            for _ in range(depths[i]):
                resblocks.append(DiffiTResBlock(
                    dim=current_dim,
                    time_embed_dim=embed_dim, # Use the projected time embed dim
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    num_patches=num_patches_level,
                    use_conv=True # Include conv path in standard blocks
                ))
            self.encoder_stages.append(resblocks)

            if i < self.num_levels - 1:
                # Downsample: Conv stride 2 usually doubles channels
                new_dim = current_dim * 2
                self.downsamplers.append(nn.Conv2d(current_dim, new_dim, kernel_size=3, stride=2, padding=1))
                current_dim = new_dim
                current_size //= 2
            else:
                self.downsamplers.append(nn.Identity()) # No downsampling at last level


        # --- Bottleneck ---
        num_patches_bottleneck = (current_size * current_size)
        self.bottleneck = nn.ModuleList([
             DiffiTResBlock(
                dim=current_dim, time_embed_dim=embed_dim, num_heads=num_heads[-1],
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, dropout=dropout,
                attn_drop=attn_drop, num_patches=num_patches_bottleneck, use_conv=True
             ) for _ in range(depths[-1]) # Use last depth/heads for bottleneck too
        ])


        # --- Decoder ---
        self.decoder_stages = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        # Calculate encoder dimensions for skip connections
        encoder_dims = [embed_dim]
        temp_dim = embed_dim
        for i in range(self.num_levels - 1):
            temp_dim *= 2
            encoder_dims.append(temp_dim)
        # Example: embed_dim=128, num_levels=3 -> encoder_dims = [128, 256, 512]
        # current_dim is the dimension after the bottleneck (e.g., 512)

        for i in range(self.num_levels - 1): # Loop num_levels-1 times for upsampling
            level_idx = self.num_levels - 2 - i # Index for depths/heads/skips (e.g., 1, 0 for num_levels=3)

            # Upsample: Halves channels of the current feature map
            upsample_out_dim = current_dim // 2
            self.upsamplers.append(nn.ConvTranspose2d(current_dim, upsample_out_dim, kernel_size=2, stride=2))
            current_dim_after_upsample = upsample_out_dim # Dimension after upsampling
            current_size *= 2
            num_patches_level = (current_size * current_size)

            # Dimension after concatenation with skip connection
            skip_dim = encoder_dims[level_idx] # Get skip connection dim from corresponding encoder level
            stage_dim = current_dim_after_upsample + skip_dim # Dimension after concat

            resblocks = nn.ModuleList()
            for _ in range(depths[level_idx]): # Use depth from corresponding encoder level
                resblocks.append(DiffiTResBlock(
                    dim=stage_dim, # Use the concatenated dimension
                    time_embed_dim=embed_dim, # Projected time embed dim
                    num_heads=num_heads[level_idx], # Use heads from corresponding encoder level
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    num_patches=num_patches_level,
                    use_conv=True
                ))
            self.decoder_stages.append(resblocks)
            current_dim = stage_dim # Update current_dim for the input of the next upsampler


        # --- Head ---
        # current_dim is now the dimension after the last decoder stage
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=current_dim)
        self.act_out = nn.SiLU()
        self.head = nn.Conv2d(current_dim, out_channels, kernel_size=3, stride=1, padding=1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
         # Same init as LatentDiffiT or specific U-Net init
         if isinstance(m, nn.Linear):
             nn.init.trunc_normal_(m.weight, std=.02)
             if m.bias is not None: nn.init.constant_(m.bias, 0)
         elif isinstance(m, nn.LayerNorm):
             nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
         elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
              if m.bias is not None: nn.init.constant_(m.bias, 0)
         elif isinstance(m, nn.GroupNorm):
             if m.weight is not None: nn.init.constant_(m.weight, 1.0)
             if m.bias is not None: nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None):
        # x: (B, C_in, H, W) - Noisy input image
        # t: (B,) - Timesteps

        # Time embedding
        time_emb = self.time_embed(t)
        time_emb = self.time_mlp(time_emb) # Project to embed_dim: (B, D)

        # if y is not None and self.num_classes > 0:
        #     # Classifier-free guidance requires ability to condition on class
        #     label_emb = self.label_embed(y, self.training)
        #     # Combine time and label embeddings
        #     time_emb = time_emb + label_emb
        #     # time_emb is now (B, embed_dim)

        # Tokenize
        h = self.tokenizer(x) # (B, embed_dim, H, W)

        # --- Encoder ---
        skip_connections = []
        for i in range(self.num_levels):
            for block in self.encoder_stages[i]:
                h = block(h, time_emb)
            if i < self.num_levels -1: # Store before downsampling (except last)
                skip_connections.append(h)
            h = self.downsamplers[i](h)


        # --- Bottleneck ---
        for block in self.bottleneck:
            h = block(h, time_emb)

        # --- Decoder ---
        for i in range(self.num_levels - 1): # num_levels-1 upsampling steps
            # Upsample (index i corresponds to levels -2, -3, ..., 0)
            h = self.upsamplers[i](h)
            # Concatenate with skip connection (from last to first)
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1) # Concat along channel dimension
            # Apply ResBlocks for this level
            for block in self.decoder_stages[i]: # index i is 0..num_levels-2
                 h = block(h, time_emb)


        # --- Head ---
        h = self.norm_out(h)
        h = self.act_out(h)
        output = self.head(h) # (B, C_out, H, W)

        return output


# --- Example Usage ---
if __name__ == '__main__':

    # === Latent DiffiT Example ===
    print("--- Testing LatentDiffiT ---")
    latent_model = LatentDiffiT(
        img_size=64,       # Example latent size
        patch_size=2,
        in_channels=4,        # Example latent channels
        embed_dim=256,     # Smaller dimension for faster test
        time_embed_dim=128,
        depth=6,           # Shallower depth
        num_heads=4,
        num_classes=10,    # Example with class conditioning
        dropout=0.1
    )

    dummy_latent = torch.randn(4, 4, 64, 64) # B, C, H, W
    dummy_time = torch.randint(0, 1000, (4,)) # B
    dummy_labels = torch.randint(0, 10, (4,)) # B

    try:
        print("LatentDiffiT Input Shape:", dummy_latent.shape)
        output_noise = latent_model(dummy_latent, dummy_time, dummy_labels)
        print("LatentDiffiT Output Shape:", output_noise.shape) # Should match input H, W
        assert output_noise.shape == dummy_latent.shape
        print("LatentDiffiT OK")
    except Exception as e:
        print(f"LatentDiffiT Error: {e}")
        import traceback
        traceback.print_exc()


    # === Image Space DiffiT Example ===
    print("\n--- Testing ImageSpaceDiffiT ---")
    image_model = ImageSpaceDiffiT(
        img_size=32,
        in_channels=3,
        out_channels=3,
        embed_dim=128,      # Base dimension
        time_embed_dim=256,
        depths=[2, 2, 2],  # Shallower U-Net for test (3 levels)
        num_heads=[4, 4, 4],
        mlp_ratio=4.0,
        num_classes=10,    # Example with class conditioning
        dropout=0.1
    )

    dummy_image = torch.randn(2, 3, 32, 32) # B, C, H, W
    dummy_time_img = torch.randint(0, 1000, (2,)) # B
    dummy_label = torch.randint(0, 10, (2,)) # B

    try:
        print("ImageSpaceDiffiT Input Shape:", dummy_image.shape)
        output_image = image_model(dummy_image, dummy_time_img)
        print("ImageSpaceDiffiT Output Shape:", output_image.shape) # Should match input
        assert output_image.shape == dummy_image.shape
        print("ImageSpaceDiffiT OK")
    except Exception as e:
        print(f"ImageSpaceDiffiT Error: {e}")
        import traceback
        traceback.print_exc()