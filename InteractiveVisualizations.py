import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
import math
from SentinelData import SentinelDataset
from models.PansharpeningUnetppLightning import denormalize, normalize, PanSharpenUnetppLightning


class InteractiveBandViewer:
    """
    Interactive Jupyter-based viewer for pansharpening results.

    Displays:
    - RGB composite
    - Ground Truth HR band
    - LR band
    - Predicted HR band
    - Absolute error
    - Log absolute error

    Features:
    - Band switching via buttons
    - Rectangle zoom with synchronized axes
    - Compact gradient scales
    """

    def __init__(self, ms_gt, ms_pred, pan=None, lr=None, figsize=(18, 10)):
        """
        ms_gt   : Tensor (C, H, W) ground truth HR
        ms_pred : Tensor (C, H, W) predicted HR
        pan     : Tensor (1, H, W) panchromatic image (optional, for RGB)
        lr      : Tensor (C, h, w) low resolution input (optional)
        figsize : Tuple (width, height) for figure size
        """

        assert ms_gt.shape == ms_pred.shape, "GT and prediction must match"

        self.ms_gt = ms_gt.cpu()
        self.ms_pred = ms_pred.cpu()
        self.pan = pan.cpu() if pan is not None else None
        self.lr = lr.cpu() if lr is not None else None
        self.num_bands = ms_gt.shape[0]

        self.current_band = 0
        self.zoom_limits = None
        
        # Flag to prevent recursion in zoom synchronization
        self._updating_axes = False
        self._imgs = {}  # Store image objects

        self._setup_figure(figsize)
        self._draw_images()
        self._setup_buttons()
        self._setup_zoom()

        plt.show()

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------
    @staticmethod
    def normalize(x, per_channel=False):
        """
        Normalize a tensor to [0,1].

        x : torch.Tensor, shape (C,H,W) or (H,W)
        per_channel : if True, normalize each channel independently
        """
        if per_channel and x.ndim == 3:  # (C,H,W)
            x_norm = torch.zeros_like(x)
            for c in range(x.shape[0]):
                x_c = x[c]
                x_norm[c] = (x_c - x_c.min()) / (x_c.max() - x_c.min() + 1e-8)
            return x_norm
        else:
            x = x - x.min()
            return x / (x.max() + 1e-8)


    @staticmethod
    def log_abs_error(pred, gt):
        err = (pred - gt).abs()
        return torch.log1p(err + 1e-8)  # Add small epsilon for log stability

    # ---------------------------------------------------------
    # Figure setup
    # ---------------------------------------------------------
    def _setup_figure(self, figsize):
        # Create figure with adjusted layout
        self.fig = plt.figure(figsize=figsize, constrained_layout=False)
        
        # Create main grid: 3 rows
        # Rows 1-2: 3x2 images grid
        # Row 3: Gradient scales
        gs_main = self.fig.add_gridspec(
            nrows=3, ncols=1, 
            height_ratios=[2, 2, 0.8],  # Images take most space
            hspace=0.15,
            left=0.05, right=0.95, bottom=0.12, top=0.92
        )
        
        # Grid for images (2 rows, 3 columns)
        gs_images = gs_main[0:2].subgridspec(2, 3, wspace=0.03, hspace=0.05)
        
        # Create axes for 6 images
        self.axes = {}
        titles = ['RGB Composite', 'GT HR Band', 'LR Input',
                 'Predicted HR', 'Absolute Error', 'Log Abs Error']
        
        positions = [(0, 0), (0, 1), (0, 2),
                    (1, 0), (1, 1), (1, 2)]
        
        for (row, col), title in zip(positions, titles):
            ax = self.fig.add_subplot(gs_images[row, col])
            self.axes[title] = ax
            ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
            ax.axis("off")
            ax.set_aspect('equal')
        
        # Gradient scales row (bottom)
        gs_gradients = gs_main[2].subgridspec(1, 2, wspace=0.1)
        
        # Create axes for gradient scales
        self.ax_log_scale = self.fig.add_subplot(gs_gradients[0])
        self.ax_abs_scale = self.fig.add_subplot(gs_gradients[1])
        
        # Create a small axis for minimal stats
        stats_pos = [0.82, 0.02, 0.15, 0.08]  # Bottom right corner
        self.ax_stats = self.fig.add_axes(stats_pos)

    # ---------------------------------------------------------
    # Drawing logic
    # ---------------------------------------------------------
    def _draw_images(self):
        band = self.current_band

        # Prepare all images
        images = {}
        # RGB Composite
        rgb = self.normalize(self.pan, per_channel=True)
        
        images['RGB Composite'] = rgb
        
        # Ground Truth HR Band
        gt = self.normalize(self.ms_gt[band])
        images['GT HR Band'] = gt
        
        # LR Input (upsampled to match HR size)
        if self.lr is not None:
            lr_band = self.lr[band]
            # Upsample if needed
            if lr_band.shape != gt.shape:
                lr_band = torch.nn.functional.interpolate(
                    lr_band.unsqueeze(0).unsqueeze(0),
                    size=gt.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            images['LR Input'] = self.normalize(lr_band)
        else:
            images['LR Input'] = torch.zeros_like(gt)
        
        # Predicted HR Band
        pred = self.normalize(self.ms_pred[band])
        images['Predicted HR'] = pred
        
        # Absolute Error
        abs_err = (self.ms_pred[band] - self.ms_gt[band]).abs()
        images['Absolute Error'] = self.normalize(abs_err)
        
        # Log Absolute Error
        log_err = self.log_abs_error(self.ms_pred[band], self.ms_gt[band])
        images['Log Abs Error'] = self.normalize(log_err)
        
        # Calculate error statistics for gradient scales
        abs_min, abs_max = abs_err.min().item(), abs_err.max().item()
        log_min, log_max = log_err.min().item(), log_err.max().item()
        
        # Display all images
        for title, img in images.items():
            ax = self.axes[title]
            
            if title in self._imgs:
                
                display_img = img

                if 'RGB' in title:
                    # img shape is (3, H, W), need (H, W, 3) for matplotlib
                    display_img = img.permute(1, 2, 0)

                self._imgs[title].set_data(display_img)
                
                if title == 'Absolute Error':
                    self._imgs[title].set_clim(0, 1)
                elif title == 'Log Abs Error':
                    self._imgs[title].set_clim(0, 1)
            else:
                # First time initialization
                if 'RGB' in title:
                    self._imgs[title] = ax.imshow(img.permute(1, 2, 0))
                elif 'Error' in title:
                    cmap = 'inferno'
                    self._imgs[title] = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
                else:
                    cmap = 'gray'
                    self._imgs[title] = ax.imshow(img, cmap=cmap)

        # Update gradient scales
        self._update_gradient_scales(abs_min, abs_max, log_min, log_max)
        
        # Update minimal statistics
        self._update_minimal_stats(abs_min, abs_max, log_min, log_max)

        # Apply stored zoom limits if they exist
        if self.zoom_limits is not None:
            xmin, xmax, ymin, ymax = self.zoom_limits
            self._apply_zoom(xmin, xmax, ymin, ymax, store=False)

        # Update band info in title
        self.fig.suptitle(f'Band {self.current_band + 1}/{self.num_bands}', 
                         fontsize=14, y=0.98, fontweight='bold')
        
        self.fig.canvas.draw_idle()

    def _update_gradient_scales(self, abs_min, abs_max, log_min, log_max):
        """Update the compact gradient scales."""
        
        # Clear previous scales
        self.ax_log_scale.clear()
        self.ax_abs_scale.clear()
        
        # Create gradient bars
        log_gradient = np.linspace(log_min, log_max, 100).reshape(1, -1)
        abs_gradient = np.linspace(abs_min, abs_max, 100).reshape(1, -1)
        
        # Display gradient bars
        self.ax_log_scale.imshow(log_gradient, aspect='auto', cmap='inferno',
                                extent=[log_min, log_max, 0, 1])
        self.ax_abs_scale.imshow(abs_gradient, aspect='auto', cmap='inferno',
                                extent=[abs_min, abs_max, 0, 1])
        
        # Configure log scale axis
        self.ax_log_scale.set_xlim(log_min, log_max)
        self.ax_log_scale.set_ylim(0, 1)
        self.ax_log_scale.set_yticks([])
        self.ax_log_scale.set_xlabel('Log Error', fontsize=9, fontweight='bold')
        self.ax_log_scale.set_title('Log Error Scale', fontsize=10, pad=5)
        self.ax_log_scale.tick_params(axis='x', labelsize=8)
        
        # Configure abs scale axis
        self.ax_abs_scale.set_xlim(abs_min, abs_max)
        self.ax_abs_scale.set_ylim(0, 1)
        self.ax_abs_scale.set_yticks([])
        self.ax_abs_scale.set_xlabel('Abs Error', fontsize=9, fontweight='bold')
        self.ax_abs_scale.set_title('Abs Error Scale', fontsize=10, pad=5)
        self.ax_abs_scale.tick_params(axis='x', labelsize=8)
        
        # Add range labels
        self.ax_log_scale.text(0.5, 1.1, f'[{log_min:.3f}, {log_max:.3f}]',
                              ha='center', fontsize=8, fontweight='bold',
                              transform=self.ax_log_scale.transAxes)
        self.ax_abs_scale.text(0.5, 1.1, f'[{abs_min:.3f}, {abs_max:.3f}]',
                              ha='center', fontsize=8, fontweight='bold',
                              transform=self.ax_abs_scale.transAxes)

    def _update_minimal_stats(self, abs_min, abs_max, log_min, log_max):
        """Update minimal statistics display."""
        
        # Clear previous stats
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Display only the most critical stats
        stats_text = [
            "ERROR RANGES",
            "─" * 15,
            f"Abs: [{abs_min:.4f}, {abs_max:.4f}]",
            f"Log: [{log_min:.4f}, {log_max:.4f}]",
            f"Band: {self.current_band + 1}/{self.num_bands}"
        ]
        
        for i, line in enumerate(stats_text):
            y_pos = 0.9 - i * 0.2
            fontsize = 9 if i == 0 else 8
            fontweight = 'bold' if i == 0 else 'normal'
            
            self.ax_stats.text(0.05, y_pos, line,
                              fontsize=fontsize,
                              fontweight=fontweight,
                              transform=self.ax_stats.transAxes,
                              verticalalignment='top',
                              family='monospace')
        
        # Add subtle background
        self.ax_stats.set_facecolor('#f5f5f5')
        for spine in self.ax_stats.spines.values():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(0.5)

    # ---------------------------------------------------------
    # Band selection buttons
    # ---------------------------------------------------------
    def _setup_buttons(self):
        # Position buttons at the bottom left
        button_height = 0.05
        button_width = 0.1
        button_y = 0.02
        
        # Reset zoom button
        ax_reset = plt.axes([0.15, button_y, button_width, button_height])
        self.btn_reset = Button(ax_reset, "⟲ Reset")
        self.btn_reset.on_clicked(self.reset_zoom)
        
        # Prev/next buttons
        ax_prev = plt.axes([0.30, button_y, button_width, button_height])
        ax_next = plt.axes([0.45, button_y, button_width, button_height])
        
        # Simple band info
        ax_info = plt.axes([0.60, button_y, button_width * 0.8, button_height])
        ax_info.axis('off')
        self.info_text = ax_info.text(0.5, 0.5, '', 
                                     ha='center', va='center', 
                                     fontsize=9, fontweight='bold')

        self.btn_prev = Button(ax_prev, "← Prev")
        self.btn_next = Button(ax_next, "Next →")

        self.btn_prev.on_clicked(self._prev_band)
        self.btn_next.on_clicked(self._next_band)
        
        # Update info text
        self._update_info_text()

    def _prev_band(self, event):
        self.current_band = (self.current_band - 1) % self.num_bands
        self._draw_images()
        self._update_info_text()

    def _next_band(self, event):
        self.current_band = (self.current_band + 1) % self.num_bands
        self._draw_images()
        self._update_info_text()
        
    def _update_info_text(self):
        """Update the info text display."""
        self.info_text.set_text(f'Band {self.current_band + 1}/{self.num_bands}')

    # ---------------------------------------------------------
    # Zoom logic (rectangle selector)
    # ---------------------------------------------------------
    def _setup_zoom(self):
        # Create a rectangle selector for each image axis
        self.selectors = []
        for title, ax in self.axes.items():
            selector = RectangleSelector(
                ax,
                self._on_zoom_select,
                useblit=True,
                button=[1],
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True,
                drag_from_anywhere=True
            )
            self.selectors.append(selector)

        # Connect zoom callbacks for each axis
        for ax in self.axes.values():
            ax.callbacks.connect("xlim_changed", self._on_axis_zoom)
            ax.callbacks.connect("ylim_changed", self._on_axis_zoom)

    def _on_zoom_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return
            
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        # Apply zoom to all axes
        self._apply_zoom(xmin, xmax, ymin, ymax, store=True)

    def _on_axis_zoom(self, ax):
        """
        When any axis is zoomed, synchronize limits to all axes.
        """
        if self._updating_axes:
            return
            
        self._updating_axes = True
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Update all other image axes
        for other_ax in self.axes.values():
            if other_ax is not ax:
                other_ax.set_xlim(xlim)
                other_ax.set_ylim(ylim)

        # Update stored zoom limits
        self.zoom_limits = (xlim[0], xlim[1], ylim[0], ylim[1])
        
        self._updating_axes = False
        self.fig.canvas.draw_idle()

    def _apply_zoom(self, xmin, xmax, ymin, ymax, store=True):
        """
        Apply zoom to all image axes.
        """
        self._updating_axes = True
        
        for ax in self.axes.values():
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)  # inverted y-axis for images
            
        if store:
            self.zoom_limits = (xmin, xmax, ymin, ymax)
            
        self._updating_axes = False
        self.fig.canvas.draw_idle()

    def reset_zoom(self, event=None):
        """Reset zoom to show full image."""
        self.zoom_limits = None
        
        if hasattr(self, 'ms_gt'):
            H, W = self.ms_gt.shape[1], self.ms_gt.shape[2]
            xmin, xmax = -0.5, W - 0.5
            ymin, ymax = H - 0.5, -0.5
            
            self._apply_zoom(xmin, xmax, ymax, ymin, store=False)
            
            for selector in self.selectors:
                selector.set_active(True)
                if hasattr(selector, 'extents'):
                    selector.extents = (xmin, xmax, ymax, ymin)
                    
        self._update_info_text()