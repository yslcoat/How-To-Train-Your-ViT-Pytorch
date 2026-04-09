import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange


class LearnableFourierPE(nn.Module):
    def __init__(self, d_pos: int, sigma: float = 10.0):
        super().__init__()
        assert d_pos % 2 == 0, "d_pos must be even"
        n_freqs = d_pos // 2
        # B ∈ R^(2 * n_freqs): rows are [f_x, f_y] frequencies
        self.B = nn.Parameter(torch.randn(2, n_freqs) * sigma)

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        if not hasattr(self, "_coords") or self._coords.shape[:2] != (H, W) or self._coords.device != device:
            ys = torch.arange(H, device=device, dtype=torch.float32) / max(H - 1, 1)
            xs = torch.arange(W, device=device, dtype=torch.float32) / max(W - 1, 1)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            self._coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        phase = self._coords @ self.B.to(device)  # (H, W, n_freqs)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (H, W, d_pos)


class SinusoidalPE(nn.Module):
    def __init__(self, d_pos: int, denominator: float = 10000.0):
        super().__init__()
        assert d_pos % 2 == 0, "d_pos must be even"
        enc_dim = d_pos // 2
        div_term = torch.exp(
            torch.arange(0.0, enc_dim, 2) * -(torch.log(torch.tensor(denominator)) / enc_dim)
        )
        freqs = torch.zeros(2, enc_dim)
        freqs[0, :enc_dim // 2] = div_term
        freqs[0, enc_dim // 2:] = div_term
        freqs[1, :enc_dim // 2] = div_term
        freqs[1, enc_dim // 2:] = div_term
        self.freqs: torch.Tensor
        self.register_buffer("freqs", freqs)

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        if not hasattr(self, "_coords") or self._coords.shape[:2] != (H, W) or self._coords.device != device:
            ys = torch.arange(H, device=device, dtype=torch.float32) / max(H - 1, 1)
            xs = torch.arange(W, device=device, dtype=torch.float32) / max(W - 1, 1)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            self._coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        phase = self._coords @ self.freqs  # (H, W, enc_dim)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (H, W, d_pos)


class PixelLevelEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_dim: int,
        out_dim: int,
        downsample: int = 2,
        pe_type: str = "fourier",
        pe_injection: str = "concat",
        use_proj: bool = True,
    ):
        super().__init__()
        self.pe_injection = pe_injection
        self.use_proj = use_proj

        self.local_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                base_dim, 
                kernel_size=7, 
                stride=downsample, 
                padding=3, 
                bias=False
            ),
            nn.BatchNorm2d(base_dim),
            nn.GELU(),
        )

        if pe_type == "fourier":
            self.pos_encoding = LearnableFourierPE(base_dim)
        elif pe_type == "sinusoidal":
            self.pos_encoding = SinusoidalPE(base_dim)
        else:
            raise ValueError(f"pe_type must be 'fourier' or 'sinusoidal', got '{pe_type}'")

        if use_proj:
            proj_in = base_dim * 2 if pe_injection == "concat" else base_dim
            self.projection = nn.Conv2d(proj_in, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        local_feats = self.local_extractor(x)                       # (B, base_dim, H', W')
        H, W = local_feats.shape[2], local_feats.shape[3]

        pos_enc = self.pos_encoding(H, W, x.device).permute(2, 0, 1).unsqueeze(0)

        if self.pe_injection == "concat":
            out = torch.cat([local_feats, pos_enc.expand(B, -1, -1, -1)], dim=1)
        else:  # add
            out = local_feats + pos_enc

        if self.use_proj:
            out = self.projection(out)  # (B, out_dim, H', W')

        return out.permute(0, 2, 3, 1)  # (B, H', W', out_dim)


class SuperpixelAggregation(nn.Module):
    SUPPORTED = ("avg", "max", "min", "std")
    def __init__(self, n_superpixels: int, aggregate: list[str] = ("avg", "max")):
        super().__init__()
        for m in aggregate:
            if m not in self.SUPPORTED:
                raise ValueError(f"Unsupported aggregation '{m}'. Choose from {self.SUPPORTED}.")
        self.n_superpixels = n_superpixels
        self.aggregate = list(aggregate)

    def forward(
        self,
        pixel_embeddings: torch.Tensor,  # (B, Hf, Wf, C)
        superpixel_map: torch.Tensor,    # (B, Hf, Wf) long, indices in [0, K-1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Hf, Wf, C = pixel_embeddings.shape
        K = self.n_superpixels

        emb_flat = pixel_embeddings.reshape(B, Hf * Wf, C)          # (B, N, C)
        sp_flat  = superpixel_map.reshape(B, Hf * Wf).clamp(0, K-1) # (B, N)
        sp_idx   = sp_flat.unsqueeze(-1).expand(-1, -1, C)           # (B, N, C)

        count = emb_flat.new_zeros(B, K, 1)
        ones  = torch.ones(B, Hf * Wf, 1, device=emb_flat.device, dtype=emb_flat.dtype)
        count.scatter_add_(1, sp_flat.unsqueeze(-1), ones)           # (B, K, 1)
        valid = count.squeeze(-1) > 0                                # (B, K) bool

        results = []

        if "avg" in self.aggregate:
            z_avg = emb_flat.new_zeros(B, K, C)
            z_avg.scatter_add_(1, sp_idx, emb_flat)
            z_avg = z_avg / count.clamp(min=1.0)
            results.append(z_avg)

        if "max" in self.aggregate:
            z_max = emb_flat.new_full((B, K, C), float("-inf"))
            z_max.scatter_reduce_(1, sp_idx, emb_flat, reduce="amax", include_self=True)
            z_max = z_max.masked_fill(~valid.unsqueeze(-1), 0.0)
            results.append(z_max)

        if "min" in self.aggregate:
            z_min = emb_flat.new_full((B, K, C), float("inf"))
            z_min.scatter_reduce_(1, sp_idx, emb_flat, reduce="amin", include_self=True)
            z_min = z_min.masked_fill(~valid.unsqueeze(-1), 0.0)
            results.append(z_min)

        if "std" in self.aggregate:
            z_mean = emb_flat.new_zeros(B, K, C)
            z_mean.scatter_add_(1, sp_idx, emb_flat)
            z_mean = z_mean / count.clamp(min=1.0)

            z_sq_mean = emb_flat.new_zeros(B, K, C)
            z_sq_mean.scatter_add_(1, sp_idx, emb_flat ** 2)
            z_sq_mean = z_sq_mean / count.clamp(min=1.0)

            z_std = (z_sq_mean - z_mean ** 2).clamp(min=0.0).sqrt()
            z_std = z_std.masked_fill(~valid.unsqueeze(-1), 0.0)
            results.append(z_std)

        tokens = torch.cat(results, dim=-1)  # (B, K, C * n_methods)

        cls_valid = valid.new_ones(B, 1)
        mask = rearrange(torch.cat([cls_valid, valid], dim=1), "b n -> b 1 1 n")

        return tokens, mask


class MaskedMSA(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if mask is not None:
            attn_bias = torch.zeros(q.shape[0], 1, 1, q.shape[2], device=q.device, dtype=q.dtype)
            attn_bias.masked_fill_(~mask, float("-inf"))
        else:
            attn_bias = None

        out = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class MLP(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class MaskedTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msa = MaskedMSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = MLP(dim, hidden_dim=mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.msa(x, mask) + x
        x = self.mlp(x) + x
        return x


class MaskedTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            MaskedTransformerBlock(dim=dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.layers:
            x = block(x, mask)
        return self.norm(x)


class SuiTTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        emb_dim: int,
        base_dim: int | None = None,
        pe_type: str = "fourier",
        pe_injection: str = "concat",
        use_proj: bool = True,
        aggregate: list[str] = ("avg", "max"),
        n_superpixels: int = 196,
        compactness: float = 10.0,
        n_slic_iter: int = 10,
        downsample: int = 2,
    ):
        super().__init__()
        n_methods = len(aggregate)
        if emb_dim % n_methods != 0:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be divisible by len(aggregate) ({n_methods})"
            )
        out_dim = emb_dim // n_methods
        if base_dim is None:
            base_dim = emb_dim // 4

        self.n_superpixels = n_superpixels
        self.compactness   = compactness
        self.n_slic_iter   = n_slic_iter

        self.pixel_embedding = PixelLevelEmbedding(
            in_channels=in_channels,
            base_dim=base_dim,
            out_dim=out_dim,
            downsample=downsample,
            pe_type=pe_type,
            pe_injection=pe_injection,
            use_proj=use_proj,
        )
        self.aggregation = SuperpixelAggregation(n_superpixels, aggregate=aggregate)

    def compute_superpixels(
        self,
        images: torch.Tensor,
        n_superpixels: int,
        compactness: float,
        n_iter: int,
    ) -> torch.Tensor:
        B, C, _, _ = images.shape
        imgs_np = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()  # (B, H, W, C)

        sp_maps = []
        try:
            from fast_slic import Slic  # type: ignore
            for i in range(B):
                img_u8 = (imgs_np[i].clip(0.0, 1.0) * 255).astype(np.uint8)
                if img_u8.shape[-1] == 1:
                    img_u8 = np.concatenate([img_u8] * 3, axis=-1)
                slic = Slic(num_components=n_superpixels, compactness=compactness, max_iter=n_iter)
                sp_maps.append(slic.iterate(img_u8))
        except ImportError:
            from skimage.segmentation import slic as sk_slic  # type: ignore
            ch_axis = -1 if C > 1 else None
            for i in range(B):
                sp_maps.append(
                    sk_slic(
                        imgs_np[i].clip(0.0, 1.0),
                        n_segments=n_superpixels,
                        compactness=compactness,
                        max_num_iter=n_iter,
                        channel_axis=ch_axis,
                        start_label=0,
                    )
                )

        return torch.from_numpy(np.stack(sp_maps, axis=0)).long()  # (B, H, W)

    def forward(
        self,
        x: torch.Tensor,
        superpixel_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if superpixel_map is None:
            superpixel_map = self.compute_superpixels(
                x, self.n_superpixels, self.compactness, self.n_slic_iter
            )
        superpixel_map = superpixel_map.to(x.device)

        pixel_embs = self.pixel_embedding(x)           # (B, H', W', out_dim)
        Hf, Wf = pixel_embs.shape[1], pixel_embs.shape[2]

        sp_map_down = F.interpolate(
            superpixel_map.unsqueeze(1).float(), size=(Hf, Wf), mode="nearest"
        ).squeeze(1).long()

        return self.aggregation(pixel_embs, sp_map_down)  # tokens, mask


class SuiT(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int,
        emb_dim: int,
        n_heads: int,
        attn_head_dim: int,
        mlp_dim: int,
        n_blocks: int,
        base_dim: int | None = None,
        pe_type: str = "fourier",
        pe_injection: str = "concat",
        use_proj: bool = True,
        aggregate: list[str] = ("avg", "max"),
        n_superpixels: int = 196,
        compactness: float = 10.0,
        n_slic_iter: int = 10,
        downsample: int = 2,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()

        self.tokenizer = SuiTTokenizer(
            in_channels=in_channels,
            emb_dim=emb_dim,
            base_dim=base_dim,
            pe_type=pe_type,
            pe_injection=pe_injection,
            use_proj=use_proj,
            aggregate=list(aggregate),
            n_superpixels=n_superpixels,
            compactness=compactness,
            n_slic_iter=n_slic_iter,
            downsample=downsample,
        )

        self.cls_token   = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = MaskedTransformer(
            dim=emb_dim,
            depth=n_blocks,
            heads=n_heads,
            dim_head=attn_head_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        superpixel_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        tokens, mask = self.tokenizer(x, superpixel_map)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # zero-copy view, no allocation
        tokens = self.emb_dropout(torch.cat([cls_tokens, tokens], dim=1))
        tokens = self.transformer(tokens, mask)
        return self.mlp_head(tokens[:, 0])
    

if __name__ == "__main__":
    model = SuiT(
        n_superpixels=196,
        in_channels=3,
        num_classes=1000,
        emb_dim=1024,
        mlp_dim=2048,
        n_heads=12,
        n_blocks=12,
        attn_head_dim=64,
        dropout=0.1
    ).to("cuda")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    input_tensor = torch.randn(2, 3, 224, 224).to("cuda")
    output = model(input_tensor)
    print(output.shape)
