import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn.functional as F

from SentinelData import SentinelCroppedDataset, SentinelDataset
from models.PansharpeningUnetppLightning import denormalize

class SentinelVisualization:
    """
    Utility class for visualizing Sentinel pansharpening data and model outputs.
    """

    def __init__(self, data_path="./dataset_sentinel", batch_size=16, dataset: SentinelDataset=SentinelCroppedDataset("./dataset_sentinel")):
        self.data_path = data_path
        self.batch_size = batch_size

        self.dataset = dataset
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataset.produce_dataloaders(batch_size=batch_size)
        )

    # ---------------------------------------------------------
    # Tensor validity checks
    # ---------------------------------------------------------
    @staticmethod
    def is_degenerate(tensor: torch.Tensor) -> bool:
        """
        Returns True if tensor has no dynamic range (max == min).
        """
        return tensor.max().item() == tensor.min().item()

    def sample_is_valid(self, *tensors) -> bool:
        """
        Returns False if ANY provided tensor is degenerate.
        """
        for t in tensors:
            if self.is_degenerate(t):
                return False
        return True

    # ---------------------------------------------------------
    # Static utilities
    # ---------------------------------------------------------
    @staticmethod
    def normalize_img(x):
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        return x

    @staticmethod
    def show_ms_grid(ms_tensor, title=""):
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
    # Dataset visualization
    # ---------------------------------------------------------
    def visualize_one_sample(self, loader_type="train", max_tries=50):
        """
        Randomly samples until a non-degenerate example is found.
        """

        if loader_type == "train":
            loader = self.train_loader
        elif loader_type == "val":
            loader = self.val_loader
        elif loader_type == "test":
            loader = self.test_loader
        else:
            raise ValueError("loader_type must be 'train', 'val', or 'test'.")

        batches = list(loader)

        for _ in range(max_tries):
            batch = random.choice(batches)
            (rgb_batch, ms_low_batch), ms_gt_batch = batch

            idx = random.randint(0, rgb_batch.size(0) - 1)

            rgb = rgb_batch[idx]
            ms_low = ms_low_batch[idx]
            ms_gt = ms_gt_batch[idx]

            if self.sample_is_valid(rgb, ms_low, ms_gt):
                break
        else:
            raise RuntimeError("No valid sample found (all samples degenerate).")

        # ---- RGB ----
        rgb_np = rgb.permute(1, 2, 0).numpy()
        rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-8)

        plt.figure(figsize=(6, 5))
        plt.imshow(rgb_np)
        plt.title("RGB Tile (High Resolution)")
        plt.axis("off")
        plt.show()

        # ---- MS ----
        self.show_ms_grid(ms_low, "Low-Res MS Input")
        self.show_ms_grid(ms_gt, "High-Res MS Ground Truth")

        print(f"Displayed valid sample from {loader_type} loader.")

    # ---------------------------------------------------------
    # Model output visualization
    # ---------------------------------------------------------
    def plot_pansharpen_bands(self, model, num_samples=5, max_tries=50):
        """
        Visualize pansharpening results while discarding degenerate samples.
        """

        device = next(model.parameters()).device
        model.eval()

        batch = next(iter(self.test_loader))
        (rgb, ms_lr), ms_hr = batch

        rgb = rgb.to(device)
        ms_lr = ms_lr.to(device)
        ms_hr = ms_hr.to(device)

        with torch.no_grad():
            pred_norm = model(rgb, ms_lr)
            pred = denormalize(pred_norm)

            ms_lr_up = F.interpolate(
                ms_lr,
                size=pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        valid_idxs = []
        for idx in range(rgb.size(0)):
            if self.sample_is_valid(
                rgb[idx],
                ms_lr[idx],
                ms_hr[idx],
                pred[idx],
            ):
                valid_idxs.append(idx)

            if len(valid_idxs) >= num_samples:
                break

        if not valid_idxs:
            raise RuntimeError("No valid prediction samples found.")

        for idx in valid_idxs:
            fig = plt.figure(figsize=(18, 22))
            fig.suptitle(f"Sample {idx}", fontsize=16)

            # ---- RGB ----
            ax = plt.subplot2grid((5, 5), (0, 0), colspan=5)
            rgb_img = rgb[idx].permute(1, 2, 0).cpu()
            ax.imshow(self.normalize_img(rgb_img))
            ax.set_title("RGB input")
            ax.axis("off")

            for b in range(4):
                row = b + 1

                ax = plt.subplot2grid((5, 5), (row, 0))
                ax.imshow(
                    self.normalize_img(ms_lr_up[idx, b].cpu()), cmap="gray"
                )
                ax.set_title(f"MS LR band {b}")
                ax.axis("off")

                ax = plt.subplot2grid((5, 5), (row, 1))
                ax.imshow(
                    self.normalize_img(pred[idx, b].cpu()), cmap="gray"
                )
                ax.set_title(f"Pred band {b}")
                ax.axis("off")

                ax = plt.subplot2grid((5, 5), (row, 2))
                ax.imshow(
                    self.normalize_img(ms_hr[idx, b].cpu()), cmap="gray"
                )
                ax.set_title(f"GT band {b}")
                ax.axis("off")

                ax = plt.subplot2grid((5, 5), (row, 3))
                err = (pred[idx, b] - ms_hr[idx, b]).abs()
                ax.imshow(self.normalize_img(err.cpu()), cmap="gray")
                ax.set_title(f"Pred error, max: {err.max():0.0f}")
                ax.axis("off")

                ax = plt.subplot2grid((5, 5), (row, 4))
                inter_err = (ms_lr_up[idx, b] - ms_hr[idx, b]).abs()
                ax.imshow(self.normalize_img(inter_err.cpu()), cmap="gray")
                ax.set_title(f"Interp error, max: {inter_err.max():0.0f}")
                ax.axis("off")

            plt.tight_layout()
            plt.show()
