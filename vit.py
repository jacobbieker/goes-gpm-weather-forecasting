"""This is a modified version of the 3D ViT implementation from the following repository:
https://github.com/lucidrains/vit-pytorch to have it output a sequence of images."""

import torch
from torch import nn
import lightning as L
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        output_image_size,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        output_channels: int = 3,
        output_frames: int = 7
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        assert frames % frame_patch_size == 0, "Frames must be divisible by frame patch size"

        num_patches = (
            (image_height // patch_height)
            * (image_width // patch_width)
            * (frames // frame_patch_size)
        )
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)",
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_image_size**2 * output_channels * output_frames),
        )
        # Reshape to an image
        self.image_output = Rearrange(
            "b (t h w c) -> b c t h w",
            h=output_image_size,
            w=output_image_size,
            c=output_channels,
            t=output_frames,
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        return self.image_output(x)


class LitViT(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _common_step(self, batch, batch_idx, tag: str = "train"):
        x, y = batch
        y_hat = self.model(x)
        # Calculate loss per timestep
        for i in range(y_hat.shape[1]):
            loss = torch.nn.functional.mse_loss(y_hat[:, i], y[:, i])
            self.log(f"{tag}_loss_variable_{i}", loss)
            self.log(f"{tag}_rmse_variable_{i}", torch.sqrt(loss), prog_bar=True)
        # Calculate per timestep
        for i in range(y_hat.shape[2]):
            loss = torch.nn.functional.mse_loss(y_hat[:, :, i], y[:, :, i])
            self.log(f"{tag}_loss_timestep_{i}", loss)
            self.log(f"{tag}_rmse_timestep_{i}", torch.sqrt(loss))
        loss = torch.nn.functional.mse_loss(y_hat, y.float())
        # Logging to TensorBoard (if installed) by default
        self.log(f"{tag}_loss", loss)
        self.log(f"{tag}_rmse", torch.sqrt(loss), prog_bar=True)
        if tag == "test" and batch_idx % 10 == 0:
            self._plot_outputs(batch, y_hat, tag)
        return loss

    def _plot_outputs(self, batch, y_hat, tag: str = "train"):
        # Only plot the first example in the batch
        x, y = batch
        y = y[0]
        y_hat = y_hat[0]
        # Plot the GT 3 variables and 7 timesteps
        fig, ax = plt.subplots(3, 7)
        for i in range(3):
            for j in range(7):
                ax[i, j].imshow(y[i, j].cpu().numpy())
        self.logger.experiment.add_figure(f"{tag}_gt", fig, self.global_step)
        # Plot the predicted 3 variables and 7 timesteps
        fig, ax = plt.subplots(3, 7)
        for i in range(3):
            for j in range(7):
                ax[i, j].imshow(y_hat[i, j].cpu().numpy())
        self.logger.experiment.add_figure(f"{tag}_pred", fig, self.global_step)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, tag="train", batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, tag="val", batch_idx=batch_idx)

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, tag="test", batch_idx=batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
