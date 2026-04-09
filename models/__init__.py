import torchvision.models as torch_models
from vit_pytorch import ViT
from models.suit import SuiT

MODEL_REGISTRY = {
    "resnet18": torch_models.resnet18,
    "resnet50": torch_models.resnet50,
    "vgg16": torch_models.vgg16,
    "lucidrain_vit": ViT,
    "suit": SuiT,
}


def create_model(args):
    model_constructor = MODEL_REGISTRY.get(args.arch)

    if not model_constructor:
        raise ValueError(f"Model {args.arch} not supported.")

    if args.arch == "lucidrain_vit":
        model = model_constructor(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout,
        )
    elif args.arch == "suit":
        model = model_constructor(
            num_classes=args.num_classes,
            emb_dim=args.dim,
            n_heads=args.heads,
            attn_head_dim=args.dim // args.heads,
            mlp_dim=args.mlp_dim,
            n_blocks=args.depth,
            base_dim=args.suit_base_dim,
            n_superpixels=args.suit_n_superpixels,
            compactness=args.suit_compactness,
            n_slic_iter=args.suit_n_slic_iter,
            downsample=args.suit_downsample,
            pe_type=args.suit_pe_type,
            pe_injection=args.suit_pe_injection,
            aggregate=args.suit_aggregate,
            use_proj=not args.suit_no_proj,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout,
        )
    else:
        model = model_constructor(pretrained=args.pretrained)

    return model
