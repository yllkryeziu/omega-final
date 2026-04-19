"""
U-TAE: U-Net with Temporal Attention Encoder for satellite image time series segmentation.

Based on: Garnot & Landrieu, "Panoptic Segmentation of Satellite Image Time Series
with Convolutional Temporal Attention Networks" (ICCV 2021)

Architecture:
  - Spatial encoder: convolutional blocks with downsampling
  - Temporal encoder: L-TAE (Lightweight Temporal Attention Encoder) at each scale
  - Spatial decoder: transpose conv upsampling with skip connections
  - Output: per-pixel binary deforestation probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    """Two conv layers with BatchNorm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsample then ConvBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsample, concat skip, then ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from non-power-of-2 inputs
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from day-of-year or month index."""

    def __init__(self, d_model, max_len=366):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, positions):
        """positions: (B, T) integer indices -> (B, T, d_model)"""
        return self.pe[positions]


class LTAE(nn.Module):
    """Lightweight Temporal Attention Encoder.

    Takes a sequence of spatial feature maps and produces a single
    temporally-aggregated feature map using multi-head attention.
    """

    def __init__(self, in_ch, n_heads=4, d_k=32, d_model=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        d_model = d_model or in_ch

        self.fc_q = nn.Linear(d_model, n_heads * d_k)
        self.fc_k = nn.Linear(in_ch, n_heads * d_k)
        self.fc_v = nn.Linear(in_ch, n_heads * d_k)

        self.pos_enc = TemporalPositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.out_proj = nn.Linear(n_heads * d_k, in_ch)
        self.norm = nn.LayerNorm(in_ch)
        self.scale = d_k ** -0.5

    def forward(self, x, positions=None):
        """
        x: (B, T, C, H, W) - temporal sequence of feature maps
        positions: (B, T) - temporal position indices (e.g. month 0-71)
        Returns: (B, C, H, W) - temporally aggregated feature map
        """
        B, T, C, H, W = x.shape

        # Reshape to process all spatial positions together
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)  # (BHW, T, C)

        # Query from learnable class token + positional encoding
        if positions is not None:
            pos_emb = self.pos_enc(positions)  # (B, T, d_model)
            # Expand cls token and add positional info
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            # Use mean positional encoding as query context
            q_input = cls + pos_emb.mean(dim=1, keepdim=True)  # (B, 1, d_model)
        else:
            q_input = self.cls_token.expand(B, -1, -1)

        q_input = q_input.unsqueeze(1).expand(-1, H * W, -1, -1)  # (B, HW, 1, d_model)
        q_input = q_input.reshape(B * H * W, 1, -1)  # (BHW, 1, d_model)

        Q = self.fc_q(q_input)  # (BHW, 1, n_heads*d_k)
        K = self.fc_k(x_flat)   # (BHW, T, n_heads*d_k)
        V = self.fc_v(x_flat)   # (BHW, T, n_heads*d_k)

        # Multi-head reshape
        Q = Q.view(B * H * W, 1, self.n_heads, self.d_k).transpose(1, 2)  # (BHW, heads, 1, d_k)
        K = K.view(B * H * W, T, self.n_heads, self.d_k).transpose(1, 2)  # (BHW, heads, T, d_k)
        V = V.view(B * H * W, T, self.n_heads, self.d_k).transpose(1, 2)  # (BHW, heads, T, d_k)

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (BHW, heads, 1, T)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).reshape(B * H * W, 1, self.n_heads * self.d_k)
        out = self.out_proj(out.squeeze(1))  # (BHW, C)
        out = self.norm(out + x_flat.mean(dim=1))  # residual with temporal mean

        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)


class UTAE(nn.Module):
    """U-TAE: U-Net with Temporal Attention Encoder.

    Args:
        in_channels: number of input channels per timestep (e.g. 12 for S2 bands)
        num_classes: number of output classes (1 for binary deforestation)
        encoder_widths: channel widths for encoder stages
        decoder_widths: channel widths for decoder stages
        n_heads: number of attention heads in L-TAE
        d_k: key dimension per head
        aef_channels: number of AEF embedding channels to fuse (0 to disable)
    """

    def __init__(
        self,
        in_channels=12,
        num_classes=1,
        encoder_widths=(64, 64, 128, 128),
        decoder_widths=(128, 64, 64, 32),
        n_heads=4,
        d_k=32,
        aef_channels=0,
    ):
        super().__init__()
        self.encoder_widths = encoder_widths
        self.aef_channels = aef_channels

        # Encoder
        self.enc0 = ConvBlock(in_channels, encoder_widths[0])
        self.enc_blocks = nn.ModuleList()
        for i in range(1, len(encoder_widths)):
            self.enc_blocks.append(DownBlock(encoder_widths[i - 1], encoder_widths[i]))

        # Temporal attention at each encoder scale
        self.ltae_blocks = nn.ModuleList()
        for w in encoder_widths:
            self.ltae_blocks.append(LTAE(w, n_heads=n_heads, d_k=d_k))

        # Optional AEF fusion at bottleneck
        bottleneck_ch = encoder_widths[-1]
        if aef_channels > 0:
            self.aef_proj = nn.Sequential(
                nn.Conv2d(aef_channels, bottleneck_ch, 1),
                nn.BatchNorm2d(bottleneck_ch),
                nn.ReLU(inplace=True),
            )
            bottleneck_ch = bottleneck_ch * 2

        # Decoder
        self.dec_blocks = nn.ModuleList()
        dec_in = bottleneck_ch
        for i, w in enumerate(decoder_widths[:-1]):
            skip_ch = encoder_widths[-(i + 2)]
            self.dec_blocks.append(UpBlock(dec_in, skip_ch, w))
            dec_in = w

        # Final output
        self.head = nn.Sequential(
            nn.Conv2d(decoder_widths[-2] if len(decoder_widths) > 1 else dec_in, decoder_widths[-1], 3, padding=1),
            nn.BatchNorm2d(decoder_widths[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_widths[-1], num_classes, 1),
        )

    def forward(self, x, positions=None, aef_diff=None):
        """
        x: (B, T, C, H, W) — S2 time series
        positions: (B, T) — temporal positions (month indices 0-71)
        aef_diff: (B, C_aef, H, W) — AEF embedding difference (optional)
        Returns: (B, 1, H, W) — deforestation logits
        """
        B, T, C, H, W = x.shape

        # Encode each timestep through shared spatial encoder, collect at each scale
        scale_features = [[] for _ in self.encoder_widths]

        for t in range(T):
            xt = x[:, t]  # (B, C, H, W)
            feat = self.enc0(xt)
            scale_features[0].append(feat)
            for i, enc in enumerate(self.enc_blocks):
                feat = enc(feat)
                scale_features[i + 1].append(feat)

        # Apply L-TAE at each scale
        aggregated = []
        for i, ltae in enumerate(self.ltae_blocks):
            stacked = torch.stack(scale_features[i], dim=1)  # (B, T, C_i, H_i, W_i)
            agg = ltae(stacked, positions)  # (B, C_i, H_i, W_i)
            aggregated.append(agg)

        # Bottleneck = deepest aggregated features
        out = aggregated[-1]

        # Fuse AEF at bottleneck
        if self.aef_channels > 0 and aef_diff is not None:
            aef_resized = F.interpolate(aef_diff, size=out.shape[2:], mode="bilinear", align_corners=False)
            aef_proj = self.aef_proj(aef_resized)
            out = torch.cat([out, aef_proj], dim=1)

        # Decode with skip connections from temporally-aggregated encoder features
        for i, dec in enumerate(self.dec_blocks):
            skip_idx = len(self.encoder_widths) - 2 - i
            out = dec(out, aggregated[skip_idx])

        return self.head(out)


def build_utae(in_channels=12, aef_channels=66, small=False):
    """Factory function for U-TAE model.

    Args:
        in_channels: S2 band count (12)
        aef_channels: AEF diff channels (66 = 64 dims + cosine + L2), 0 to disable
        small: if True, use smaller architecture for faster iteration
    """
    if small:
        return UTAE(
            in_channels=in_channels,
            num_classes=1,
            encoder_widths=(32, 32, 64),
            decoder_widths=(64, 32, 16),
            n_heads=4, d_k=16,
            aef_channels=aef_channels,
        )
    return UTAE(
        in_channels=in_channels,
        num_classes=1,
        encoder_widths=(64, 64, 128, 128),
        decoder_widths=(128, 64, 64, 32),
        n_heads=4, d_k=32,
        aef_channels=aef_channels,
    )


if __name__ == "__main__":
    # Quick test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_utae(in_channels=12, aef_channels=66, small=True).to(device)

    B, T, C, H, W = 2, 12, 12, 128, 128
    x = torch.randn(B, T, C, H, W, device=device)
    pos = torch.arange(T).unsqueeze(0).expand(B, -1).to(device)
    aef = torch.randn(B, 66, H, W, device=device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    out = model(x, pos, aef)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
