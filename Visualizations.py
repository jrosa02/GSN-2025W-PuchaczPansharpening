import matplotlib.pyplot as plt
import numpy as np
from SentinelData import SentinelCroppedDataset
import random
import torch
import torch.nn.functional as  F

from models.PansharpeningUnetppLightning import denormalize

# ---------------------------------------------------------
# Utility: Display multispectral tensor as a grid of bands
# ---------------------------------------------------------
def show_ms_grid(ms_tensor, title=""):
    """
    ms_tensor: shape (C, H, W)
    Shows each band separately using a viridis colormap.
    """
    arr = ms_tensor.cpu().numpy()
    C = arr.shape[0]

    cols = 4
    rows = int(np.ceil(C / cols))

    plt.figure(figsize=(4 * cols, 3.5 * rows))
    for i in range(C):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(arr[i], cmap="viridis")
        plt.title(f"{title} | Band {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Main visualization
# ---------------------------------------------------------
def visualize_one_sample(
    data_path,
    loader_type="train",       # "train", "val", or "test"
    batch_size=16,
):
    """
    Load dataset, pick a random sample from the selected dataloader,
    and display RGB, low-res MS, and high-res MS tiles.
    """

    dataset = SentinelCroppedDataset(data_path)

    # Create dataloaders
    train_loader, val_loader, test_loader = dataset.produce_dataloaders(
        batch_size=batch_size
    )

    # Pick loader
    if loader_type == "train":
        loader = train_loader
    elif loader_type == "val":
        loader = val_loader
    elif loader_type == "test":
        loader = test_loader
    else:
        raise ValueError("loader_type must be 'train', 'val', or 'test'.")

    # Convert loader to list so we can sample a batch randomly
    batches = list(loader)
    batch = random.choice(batches)

    (rgb_batch, ms_low_batch), ms_gt_batch = batch

    # Pick a random sample from the batch
    idx = random.randint(0, rgb_batch.size(0) - 1)

    rgb = rgb_batch[idx]          # (3, H, W)
    ms_low = ms_low_batch[idx]    # (C_mul, H2, W2)
    ms_gt = ms_gt_batch[idx]      # (C_out, H, W)

    # ---------------------------------------------------------
    # Plot RGB tile
    # ---------------------------------------------------------
    rgb_np = rgb.permute(1, 2, 0).numpy()

    # Normalize for display
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-8)

    plt.figure(figsize=(6, 5))
    plt.imshow(rgb_np)
    plt.title("RGB Tile (High Resolution)")
    plt.axis("off")
    plt.show()

    # ---------------------------------------------------------
    # Plot Low-Res MS tile (128x64)
    # ---------------------------------------------------------
    show_ms_grid(ms_low, "Low-Res MS Input")

    # ---------------------------------------------------------
    # Plot Target High-Res MS tile (256x128)
    # ---------------------------------------------------------
    show_ms_grid(ms_gt, "High-Res MS Ground Truth")

    print("Displayed sample tiles from:", loader_type, "loader.")


def normalize_img(x):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x

def plot_pansharpen_examples(
    model,
    test_loader,
    num_samples=5,
):
    device = next(model.parameters()).device
    model.eval()

    # Get one batch large enough
    batch = next(iter(test_loader))
    (rgb, ms_lr), ms_hr = batch
    rgb = rgb.to(device)
    ms_lr = ms_lr.to(device)
    ms_hr = ms_hr.to(device)
    # (rgb, ms_lr), ms_hr = [x.to(device) for x in batch]

    B = rgb.size(0)
    idxs = random.sample(range(B), min(num_samples, B))

    with torch.no_grad():
        # Forward
        pred_norm = model(rgb, ms_lr)
        pred = (
            denormalize(pred_norm)
        )

        # Upsample MS input for visualization
        ms_lr_up = F.interpolate(
            ms_lr,
            size=pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    for i, idx in enumerate(idxs):
        fig = plt.figure(figsize=(15, 25))
        fig.suptitle(f"Sample {idx}", fontsize=16)

        # ---- RGB ----
        ax = plt.subplot2grid((5, 4), (0, 0), colspan=4)
        rgb_img = rgb[idx].permute(1, 2, 0).cpu()
        ax.imshow(normalize_img(rgb_img))
        ax.set_title("RGB input")
        ax.axis("off")

        # ---- MS bands ----
        for b in range(4):
            # MS LR
            ax = plt.subplot2grid((5, 4), (b + 1, 0))
            ax.imshow(normalize_img(ms_lr_up[idx, b].cpu()), cmap="gray")
            ax.set_title(f"MS LR band {b}")
            ax.axis("off")

            # Prediction
            ax = plt.subplot2grid((5, 4), (b + 1, 1))
            ax.imshow(normalize_img(pred[idx, b].cpu()), cmap="gray")
            ax.set_title(f"Pred band {b}")
            ax.axis("off")

            # GT
            ax = plt.subplot2grid((5, 4), (b + 1, 2))
            ax.imshow(normalize_img(ms_hr[idx, b].cpu()), cmap="gray")
            ax.set_title(f"GT band {b}")
            ax.axis("off")

            # Error
            ax = plt.subplot2grid((5, 4), (b + 1, 3))
            err = (pred[idx, b] - ms_hr[idx, b]).abs()
            ax.imshow(normalize_img(err.cpu()), cmap="gray")
            ax.set_title(f"Error band {b}")
            ax.axis("off")

        # plt.tight_layout()
        plt.show()
