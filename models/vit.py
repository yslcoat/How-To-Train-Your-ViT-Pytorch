import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class MLP(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MSA(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, attn_bias: bool = False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=attn_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            qkv,
        )

        out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.msa = MSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = MLP(dim, hidden_dim=mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.msa(x) + x
        x = self.mlp(x) + x
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x)

        return self.norm(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        image_size: tuple[int, int] | int,
        n_channels: int,
        patch_size: int,
        num_classes: int,
        emb_dim: int,
        mlp_dim: int,
        attn_head_dim: int,
        n_heads: int,
        n_blocks: int,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()

        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size}")

        self.patch_size = patch_size
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        if image_size[0] % patch_size != 0:
            raise ValueError(f"Image height must be divisible by patch size. Got {image_size[0]} and {patch_size}.")
        if image_size[1] % patch_size != 0:
            raise ValueError(f"Image width must be divisible by patch size. Got {image_size[1]} and {patch_size}.")

        self.n_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        patch_dim = n_channels * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, emb_dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
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

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_parameters(self) -> None:
        self.apply(self._init_module)
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(x)
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :x.shape[1], :]
        x = self.dropout(x)
        x = self.transformer(x)
        logits = self.mlp_head(x[:, 0])
        return logits


if __name__ == "__main__":
    model = VisionTransformer(
        image_size=224,
        n_channels=3,
        patch_size=16,
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
