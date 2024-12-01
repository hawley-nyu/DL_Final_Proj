from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob
from models import  BYOL
from training import train_byol


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


class ViTBackbone(nn.Module):
    def __init__(self, device="cuda", bs=64, n_steps=17, img_size=64, patch_size=8,
                 in_channels=3, embed_dim=256, num_heads=8, num_layers=6, mlp_ratio=4):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer layers
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, states, actions):
        """
        Args:
            states: [B, T, C, H, W]
            actions: [B, T-1, 2]
        """
        B, T, C, H, W = states.shape

        # Process each timestep
        features = []
        for t in range(T):
            x = states[:, t]  # [B, C, H, W]

            # Patch embedding
            x = self.patch_embed(x)  # [B, embed_dim, h, w]
            x = x.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]

            # Add cls token and position embedding
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.pos_embed

            # Apply transformer blocks
            for block in self.blocks:
                x = block(x)

            x = self.norm(x)
            features.append(x[:, 0])  # Take CLS token embedding

        return torch.stack(features, dim=1)  # [B, T, embed_dim]


def load_model(device):
    backbone = ViTBackbone(device=device)
    model = BYOL(backbone=backbone)
    return model



def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    train_loader, val_loader, probe_train_ds, probe_val_ds = load_data(device)
    model = load_model(device)

    # Training
    train_byol(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        device=device,
        save_path="checkpoints/byol"
    )

    # Evaluation
    evaluate_model(device, model, probe_train_ds, probe_val_ds)