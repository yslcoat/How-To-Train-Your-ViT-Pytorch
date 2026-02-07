"""Learnable Edge-Detection Superpixel Tokenizer (LedSpit)"""

import math
import torch
import torch.nn as nn

# from torch_scatter import scatter_mean, scatter_max
from vit_pytorch import ViT


def scatter_mean(src, index, dim, dim_size):
    out = torch.zeros(
        (src.shape[0], dim_size, src.shape[2]), device=src.device, dtype=src.dtype
    )
    count = torch.zeros(
        (src.shape[0], dim_size, src.shape[2]), device=src.device, dtype=src.dtype
    )

    # Expand index to [B, H*W, C]
    index_expanded = index.unsqueeze(-1).expand_as(src)

    out.scatter_add_(dim, index_expanded, src)
    count.scatter_add_(dim, index_expanded, torch.ones_like(src))

    return out / count.clamp(min=1e-6)


def scatter_max(src, index, dim, dim_size):
    # Initialize with a very small number
    out = torch.full(
        (src.shape[0], dim_size, src.shape[2]),
        -1e38,
        device=src.device,
        dtype=src.dtype,
    )
    index_expanded = index.unsqueeze(-1).expand_as(src)

    # scatter_reduce_ with 'amax' is available in PyTorch 1.12+
    out.scatter_reduce_(dim, index_expanded, src, reduce="amax", include_self=False)

    # Mask empty bins to 0.0 (optional, but standard for features)
    out[out == -1e38] = 0.0
    return out, None


class VoronoiPropagation(nn.Module):
    def __init__(self, num_clusters=196, height=224, width=224, device="cpu"):
        super(VoronoiPropagation, self).__init__()
        self.C = num_clusters
        self.H = height
        self.W = width
        self.device = torch.device(device)

    def place_centroids_on_grid(self, batch_size):
        # Calculate grid dimensions to approximate square superpixels
        num_cols = int(math.sqrt(self.C * self.W / self.H))
        num_rows = int(math.ceil(self.C / num_cols))

        grid_spacing_y = self.H / num_rows
        grid_spacing_x = self.W / num_cols

        centroids = []
        for i in range(num_rows):
            for j in range(num_cols):
                if len(centroids) >= self.C:
                    break
                y = int((i + 0.5) * grid_spacing_y)
                x = int((j + 0.5) * grid_spacing_x)
                centroids.append([y, x])

        # Ensure we exactly match self.C (pad if necessary, though grid usually covers it)
        while len(centroids) < self.C:
            centroids.append([self.H // 2, self.W // 2])

        centroids = torch.tensor(centroids, device=self.device).float()
        return centroids.unsqueeze(0).repeat(batch_size, 1, 1)

    def find_nearest_minima(self, centroids, grad_map, neighborhood_size=10):
        # Move centroids to the lowest gradient point in their local neighborhood
        updated_centroids = []
        B, _, H, W = grad_map.shape

        for batch_idx in range(B):
            updated_centroids_batch = []
            occupied_positions = set()

            for centroid in centroids[batch_idx]:
                y, x = centroid
                y_min = max(0, int(y) - neighborhood_size)
                y_max = min(H, int(y) + neighborhood_size)
                x_min = max(0, int(x) - neighborhood_size)
                x_max = min(W, int(x) + neighborhood_size)

                # Check neighborhood in gradient map
                neighborhood = grad_map[batch_idx, 0, y_min:y_max, x_min:x_max]
                min_val = torch.min(neighborhood)

                # Get local coordinates of minima
                min_coords = torch.nonzero(neighborhood == min_val, as_tuple=False)

                # Find first unoccupied minimum
                found = False
                for coord in min_coords:
                    new_y = y_min + coord[0].item()
                    new_x = x_min + coord[1].item()
                    position = (new_y, new_x)
                    if position not in occupied_positions:
                        occupied_positions.add(position)
                        updated_centroids_batch.append([new_y, new_x])
                        found = True
                        break

                if not found:
                    updated_centroids_batch.append([y.item(), x.item()])

            updated_centroids.append(
                torch.tensor(updated_centroids_batch, device=self.device)
            )

        return torch.stack(updated_centroids, dim=0)

    def distance_weighted_propagation(
        self, centroids, grad_map, num_iters=50, gradient_weight=10.0, edge_exponent=4.0
    ):
        B, _, H, W = grad_map.shape
        # Initialize mask with -1
        mask = torch.full(
            (B, H, W), fill_value=-1, dtype=torch.long, device=grad_map.device
        )
        dist_map = torch.full(
            (B, H, W), fill_value=float("inf"), device=grad_map.device
        )

        # Initialize centroids
        # Clamp coordinates to ensure they are within bounds
        cy = centroids[:, :, 0].long().clamp(0, H - 1)
        cx = centroids[:, :, 1].long().clamp(0, W - 1)

        # Vectorized initialization
        batch_indices = (
            torch.arange(B, device=grad_map.device)
            .unsqueeze(1)
            .expand(-1, centroids.shape[1])
        )
        cluster_indices = (
            torch.arange(centroids.shape[1], device=grad_map.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        mask[batch_indices, cy, cx] = cluster_indices
        dist_map[batch_indices, cy, cx] = 0.0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Precompute weighted gradient penalty
        weighted_grad_map = (grad_map**edge_exponent) * gradient_weight

        for _ in range(num_iters):
            changed = False
            for dy, dx in directions:
                shifted_dist = torch.roll(dist_map, shifts=(dy, dx), dims=(1, 2))
                shifted_mask = torch.roll(mask, shifts=(dy, dx), dims=(1, 2))

                # Penalty logic
                weighted_dist = shifted_dist + weighted_grad_map[:, 0, :, :]

                # Valid updates: new distance is smaller AND the neighbor was actually visited (mask != -1)
                # Note: If shifted_dist is inf, weighted_dist is inf, so < check handles unvisited safely
                update_mask = (weighted_dist < dist_map) & (shifted_mask != -1)

                if update_mask.any():
                    dist_map[update_mask] = weighted_dist[update_mask]
                    mask[update_mask] = shifted_mask[update_mask]
                    changed = True

            if not changed:
                break

        # Fill any remaining -1 holes (rare but possible) with nearest valid neighbor
        if (mask == -1).any():
            mask[mask == -1] = 0  # Fallback safety

        return mask

    def forward(self, x, grad_map):
        batch_size = x.shape[0]
        centroids = self.place_centroids_on_grid(batch_size)
        centroids = self.find_nearest_minima(centroids, grad_map)
        mask = self.distance_weighted_propagation(centroids, grad_map)

        return centroids, mask


class PDCBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(PDCBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_planes,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class NanoPiDiNet(nn.Module):
    def __init__(self, in_channels=1, width=32):
        super(NanoPiDiNet, self).__init__()
        self.init_block = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.block1 = PDCBlock(width, width)
        self.block2 = PDCBlock(width, width)
        self.pred = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        x = self.init_block(x)
        x = self.block1(x)
        x = self.block2(x)
        return torch.sigmoid(self.pred(x))


class LearnableEdgeDetectionSuperpixelTokenizer(nn.Module):
    def __init__(
        self,
        max_segments,
        n_channels=3,
        embed_dim=768,
        use_positional_embeddings=True,
        reconstruction=False,
        device="cuda",
    ):
        super().__init__()
        self.voronoi_prop = VoronoiPropagation(
            max_segments, height=224, width=224, device=device
        )
        self.max_segments = max_segments
        self.embed_dim = embed_dim
        self.use_positional_embeddings = use_positional_embeddings
        self.reconstruction = reconstruction

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

        self.pidinet = NanoPiDiNet(in_channels=1, width=32)

        if self.use_positional_embeddings:
            self.positional_embedding = nn.Linear(2, embed_dim)

        self.fusion = nn.Linear(2 * embed_dim, embed_dim)

        if self.reconstruction:
            hidden_dim = embed_dim * 4
            self.reconstruction_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_channels),
            )

    def forward(self, img):
        batch_size, n_channels, height, width = img.shape
        features = self.cnn(img)
        B, C, Hf, Wf = features.shape

        gray_img = torch.mean(img, dim=1, keepdim=True)
        gradient_map = self.pidinet(gray_img)

        centroid_coords, segments = self.voronoi_prop(img, gradient_map)

        # Similarity Weight Calculation
        similarity = 1.0 - gradient_map
        similarity = torch.clamp(similarity, 0.0, 1.0)

        segments_flat = segments.view(B, -1)
        similarity_flat = similarity.view(B, -1)

        n_K = (
            torch.zeros((B, self.max_segments), device=img.device)
            .scatter_add(
                dim=1, index=segments_flat, src=torch.ones_like(similarity_flat)
            )
            .clamp(min=1)
        )

        similarity_sum = torch.zeros(
            (B, self.max_segments), device=img.device
        ).scatter_add(dim=1, index=segments_flat, src=similarity_flat)
        W_k = similarity_sum / n_K

        # Feature Aggregation
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        segments_flat_global = segments.view(-1)
        batch_indices = (
            torch.arange(B, device=img.device)
            .unsqueeze(1)
            .expand(B, Hf * Wf)
            .reshape(-1)
        )
        unique_segment_ids = batch_indices * self.max_segments + segments_flat_global
        dim_size = B * self.max_segments

        embeddings_mean = scatter_mean(
            features_flat, unique_segment_ids, dim=0, dim_size=dim_size
        )
        embeddings_mean = embeddings_mean.view(B, self.max_segments, C)

        embeddings_max, _ = scatter_max(
            features_flat, unique_segment_ids, dim=0, dim_size=dim_size
        )
        embeddings_max = embeddings_max.view(B, self.max_segments, C)

        embeddings_concat = torch.cat([embeddings_mean, embeddings_max], dim=-1)
        embeddings_fused = self.fusion(embeddings_concat)
        weighted_embeddings = embeddings_fused * W_k.unsqueeze(-1)

        if self.use_positional_embeddings:
            centroids_normalized = centroid_coords.clone().float()
            centroids_normalized[:, :, 0] /= float(width)
            centroids_normalized[:, :, 1] /= float(height)
            pos_embeddings = self.positional_embedding(
                centroids_normalized.to(img.device)
            )
            n_centroids = pos_embeddings.shape[1]
            pos_embeddings_padded = torch.zeros(
                B, self.max_segments, self.embed_dim, device=img.device
            )
            limit = min(n_centroids, self.max_segments)
            pos_embeddings_padded[:, :limit, :] = pos_embeddings[:, :limit, :]

            final_embeddings = weighted_embeddings + pos_embeddings_padded
        else:
            final_embeddings = weighted_embeddings

        if self.reconstruction:
            superpixel_recon = self.reconstruction_head(final_embeddings)  # [B, S, 3]
            batch_offsets = (
                torch.arange(B, device=img.device) * self.max_segments
            ).view(B, 1, 1)
            segments_global_idx = segments + batch_offsets  # [B, H, W]

            recon_flat_source = superpixel_recon.view(-1, n_channels)
            reconstructed_img = recon_flat_source[segments_global_idx]

            # Permute to [B, 3, H, W] for standard image format
            reconstructed_img = reconstructed_img.permute(0, 3, 1, 2)

            return final_embeddings, reconstructed_img, segments, gradient_map

        return final_embeddings, gradient_map, segments


class LedSpit(ViT):
    def __init__(self, max_segments=196, reconstruction=False, **kwargs):
        if "patch_size" not in kwargs:
            kwargs["patch_size"] = 16

        super().__init__(**kwargs)

        self.max_segments = max_segments
        self.reconstruction = reconstruction
        dim = kwargs["dim"]
        channels = kwargs.get("channels", 3)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        del self.to_patch_embedding

        self.tokenizer = LearnableEdgeDetectionSuperpixelTokenizer(
            max_segments=max_segments,
            n_channels=channels,
            embed_dim=dim,
            reconstruction=reconstruction,
            device=device,
        )

        del self.pos_embedding
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, img):
        if self.reconstruction:
            x, recon_img, segments, grad_map = self.tokenizer(img)
        else:
            x, grad_map, segments = self.tokenizer(img)

        # x.shape = [B, max_segments, dim]
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        cls_tokens = cls_tokens + self.cls_pos_embedding
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        logits = self.mlp_head(x)

        if self.reconstruction:
            return logits, recon_img, grad_map
        return logits
