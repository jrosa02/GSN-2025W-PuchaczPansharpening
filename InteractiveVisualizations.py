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
    - Ground Truth HR band
    - Predicted HR band
    - Logarithmic absolute error

    Features:
    - Band switching via buttons
    - Rectangle zoom with synchronized axes
    - Enhanced gradient bar for error visualization
    """

    def __init__(self, ms_gt, ms_pred, figsize=(16, 6)):
        """
        ms_gt   : Tensor (C, H, W) ground truth
        ms_pred : Tensor (C, H, W) prediction
        figsize : Tuple (width, height) for figure size
        """

        assert ms_gt.shape == ms_pred.shape, "GT and prediction must match"

        self.ms_gt = ms_gt.cpu()
        self.ms_pred = ms_pred.cpu()
        self.num_bands = ms_gt.shape[0]

        self.current_band = 0
        self.zoom_limits = None  # Store zoom limits (xmin, xmax, ymin, ymax)
        
        # Flag to prevent recursion in zoom synchronization
        self._updating_axes = False
        self._colorbar = None  # Reference to error colorbar
        self._gradient_bar = None  # Reference to gradient visualization
        self._error_stats_text = None  # Reference to stats text

        self._setup_figure(figsize)
        self._draw_images()
        self._setup_buttons()
        self._setup_zoom()

        plt.show()

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------
    @staticmethod
    def normalize(x):
        x = x - x.min()
        return x / (x.max() + 1e-8)

    @staticmethod
    def log_abs_error(pred, gt):
        err = (pred - gt).abs()
        return torch.log1p(err)

    # ---------------------------------------------------------
    # Figure setup
    # ---------------------------------------------------------
    def _setup_figure(self, figsize):
        # Create figure with adjusted layout
        self.fig = plt.figure(figsize=figsize, constrained_layout=False)
        
        # Create grid with space for gradient bar and stats
        # 4 columns: GT, Pred, Error, Gradient Bar
        gs = self.fig.add_gridspec(
            nrows=1, ncols=5, 
            width_ratios=[1, 1, 1, 0.12, 0.18],  # Last two for gradient and stats
            wspace=0.03, hspace=0.05,
            left=0.03, right=0.97, bottom=0.12, top=0.92
        )
        
        self.ax_gt = self.fig.add_subplot(gs[0])
        self.ax_pred = self.fig.add_subplot(gs[1])
        self.ax_err = self.fig.add_subplot(gs[2])
        self.axes = [self.ax_gt, self.ax_pred, self.ax_err]
        
        # Gradient bar axis (vertical)
        self.ax_gradient = self.fig.add_subplot(gs[3])
        
        # Stats axis (for text display)
        self.ax_stats = self.fig.add_subplot(gs[4])
        self.ax_stats.axis('off')

        self.ax_gt.set_title("Ground Truth", fontsize=12, fontweight='bold', pad=10)
        self.ax_pred.set_title("Prediction", fontsize=12, fontweight='bold', pad=10)
        self.ax_err.set_title("Log |Error|", fontsize=12, fontweight='bold', pad=10)

        for ax in self.axes:
            ax.axis("off")
            ax.set_aspect('equal')

    # ---------------------------------------------------------
    # Drawing logic
    # ---------------------------------------------------------
    def _draw_images(self):
        band = self.current_band

        gt = self.normalize(self.ms_gt[band])
        pred = self.normalize(self.ms_pred[band])
        err = self.log_abs_error(self.ms_pred[band], self.ms_gt[band])
        
        # Calculate error stats
        err_min = err.min().item()
        err_max = err.max().item()
        err_mean = err.mean().item()
        err_std = err.std().item()
        err_median = err.median().item()
        
        # Calculate percentiles for better gradient visualization
        percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
        err_percentiles = torch.quantile(err, torch.tensor([p/100 for p in percentiles])).tolist()
        
        # Add a small margin to error range for better visualization
        err_range = err_max - err_min
        if err_range > 0:
            vmin = err_min - 0.02 * err_range
            vmax = err_max + 0.02 * err_range
        else:
            vmin = err_min - 0.1
            vmax = err_max + 0.1

        if hasattr(self, "im_gt"):
            self.im_gt.set_data(gt)
            self.im_pred.set_data(pred)
            self.im_err.set_data(err)
            self.im_err.set_clim(vmin, vmax)
        else:
            # First time initialization
            self.im_gt = self.ax_gt.imshow(gt, cmap="gray", aspect='auto')
            self.im_pred = self.ax_pred.imshow(pred, cmap="gray", aspect='auto')
            self.im_err = self.ax_err.imshow(err, cmap="inferno", aspect='auto', 
                                            vmin=vmin, vmax=vmax)

        # Update or create gradient visualization
        self._update_gradient_bar(err, vmin, vmax, err_percentiles, percentiles)
        
        # Update error statistics display
        self._update_error_stats(err_min, err_max, err_mean, err_std, err_median)

        # Apply stored zoom limits if they exist
        if self.zoom_limits is not None:
            xmin, xmax, ymin, ymax = self.zoom_limits
            self._apply_zoom(xmin, xmax, ymin, ymax, store=False)

        # Update band info in title
        self.fig.suptitle(f'Band {self.current_band + 1}/{self.num_bands} - Log Absolute Error Scale', 
                         fontsize=14, y=0.98, fontweight='bold')
        
        self.fig.canvas.draw_idle()

    def _update_gradient_bar(self, err_tensor, vmin, vmax, percentiles, percentile_labels):
        """Create or update the gradient bar visualization."""
        
        # Clear previous gradient bar
        self.ax_gradient.clear()
        self.ax_gradient.set_title("Error Scale", fontsize=10, pad=10, fontweight='bold')
        
        # Create gradient bar
        gradient = np.linspace(vmin, vmax, 256).reshape(-1, 1)
        
        # Display gradient
        self.ax_gradient.imshow(gradient, aspect='auto', cmap='inferno', 
                               extent=[0, 1, vmin, vmax])
        
        # Set up axis for gradient bar
        self.ax_gradient.set_xlim(0, 1)
        self.ax_gradient.set_ylim(vmin, vmax)
        self.ax_gradient.set_xticks([])
        
        # Add y-axis label
        self.ax_gradient.set_ylabel('Log Error Magnitude', fontsize=9)
        
        # Format y-axis ticks
        self.ax_gradient.yaxis.set_major_locator(plt.MaxNLocator(8))
        self.ax_gradient.tick_params(axis='y', labelsize=8)
        
        # Add grid lines for better readability
        self.ax_gradient.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add percentile markers on the right side
        for val, label in zip(percentiles, percentile_labels):
            # Position marker
            marker_x = 1.05
            self.ax_gradient.plot([0.95, 1.05], [val, val], 'k-', linewidth=0.8, alpha=0.7)
            
            # Add percentile label
            self.ax_gradient.text(marker_x + 0.05, val, f'{label}%', 
                                 fontsize=7, va='center', ha='left',
                                 bbox=dict(boxstyle="round,pad=0.2", 
                                          facecolor='white', 
                                          edgecolor='gray', 
                                          alpha=0.8))
        
        # Add value range at top
        range_text = f"Range: {vmin:.3f} - {vmax:.3f}"
        self.ax_gradient.text(0.5, vmax + (vmax-vmin)*0.05, range_text,
                             ha='center', fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", 
                                      facecolor='lightyellow', 
                                      edgecolor='gold', 
                                      alpha=0.9))

    def _update_error_stats(self, err_min, err_max, err_mean, err_std, err_median):
        """Update the error statistics display."""
        
        # Clear previous stats
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Create formatted statistics text
        stats_text = [
            "ERROR STATISTICS",
            "=" * 20,
            f"Min:      {err_min:.4f}",
            f"Max:      {err_max:.4f}",
            f"Mean:     {err_mean:.4f}",
            f"Median:   {err_median:.4f}",
            f"Std Dev:  {err_std:.4f}",
            f"Range:    {err_max-err_min:.4f}",
            "",
            "Percentiles:",
            f"25%:      {np.percentile([err_min, err_max], 25):.4f}",
            f"50%:      {err_median:.4f}",
            f"75%:      {np.percentile([err_min, err_max], 75):.4f}",
            f"95%:      {np.percentile([err_min, err_max], 95):.4f}"
        ]
        
        # Display statistics
        for i, line in enumerate(stats_text):
            if "=" in line or "ERROR STATISTICS" in line or "Percentiles:" in line:
                fontweight = 'bold'
                fontsize = 9 if "ERROR STATISTICS" in line else 8
                color = 'darkred' if "ERROR STATISTICS" in line else 'black'
            else:
                fontweight = 'normal'
                fontsize = 8
                color = 'black'
                
            y_pos = 0.95 - i * 0.065
            self.ax_stats.text(0.1, y_pos, line, 
                              fontsize=fontsize, 
                              fontweight=fontweight,
                              color=color,
                              transform=self.ax_stats.transAxes,
                              verticalalignment='top',
                              family='monospace')
        
        # Add a colored background based on error magnitude
        avg_error_norm = (err_mean - err_min) / (err_max - err_min + 1e-8)
        if avg_error_norm < 0.33:
            bg_color = 'lightgreen'
        elif avg_error_norm < 0.66:
            bg_color = 'lightyellow'
        else:
            bg_color = 'lightcoral'
            
        self.ax_stats.set_facecolor(bg_color)
        self.ax_stats.set_alpha(0.2)

    # ---------------------------------------------------------
    # Band selection buttons
    # ---------------------------------------------------------
    def _setup_buttons(self):
        # Position buttons at the bottom
        button_height = 0.06
        button_width = 0.1
        button_y = 0.03
        
        # Reset zoom button
        ax_reset = plt.axes([0.15, button_y, button_width, button_height])
        self.btn_reset = Button(ax_reset, "Reset Zoom")
        self.btn_reset.on_clicked(self.reset_zoom)
        
        # Prev/next buttons
        ax_prev = plt.axes([0.35, button_y, button_width, button_height])
        ax_next = plt.axes([0.55, button_y, button_width, button_height])
        
        # Add info display area
        ax_info = plt.axes([0.75, button_y, button_width * 1.5, button_height])
        ax_info.axis('off')
        self.info_text = ax_info.text(0.5, 0.5, '', 
                                     ha='center', va='center', 
                                     fontsize=9, fontweight='bold')

        self.btn_prev = Button(ax_prev, "← Prev Band")
        self.btn_next = Button(ax_next, "Next Band →")

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
        band_name = f"Band {self.current_band + 1}"
        total_bands = f"/{self.num_bands}"
        self.info_text.set_text(f'{band_name}{total_bands}')

    # ---------------------------------------------------------
    # Zoom logic (rectangle selector)
    # ---------------------------------------------------------
    def _setup_zoom(self):
        # Create a rectangle selector for each axis
        self.selectors = []
        for ax in self.axes:
            selector = RectangleSelector(
                ax,
                self._on_zoom_select,
                useblit=True,
                button=[1],  # Left mouse button
                minspanx=5, minspany=5,  # Minimum drag size in pixels
                spancoords='pixels',
                interactive=True,
                drag_from_anywhere=True,
            )
            self.selectors.append(selector)

        # Connect zoom callbacks for each axis
        for ax in self.axes:
            ax.callbacks.connect("xlim_changed", self._on_axis_zoom)
            ax.callbacks.connect("ylim_changed", self._on_axis_zoom)

    def _on_zoom_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Handle None values (click outside image)
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return
            
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        # Apply zoom to all axes
        self._apply_zoom(xmin, xmax, ymin, ymax, store=True)

    def _on_axis_zoom(self, ax):
        """
        When any axis is zoomed (manually or programmatically),
        synchronize limits to all axes.
        """
        # Prevent recursion
        if self._updating_axes:
            return
            
        self._updating_axes = True
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Update all other axes
        for other_ax in self.axes:
            if other_ax is not ax:
                other_ax.set_xlim(xlim)
                other_ax.set_ylim(ylim)

        # Update stored zoom limits
        self.zoom_limits = (xlim[0], xlim[1], ylim[0], ylim[1])
        
        self._updating_axes = False
        self.fig.canvas.draw_idle()

    def _apply_zoom(self, xmin, xmax, ymin, ymax, store=True):
        """
        Apply zoom to all axes.
        
        Parameters:
        -----------
        store : bool
            Whether to store these zoom limits for band switching
        """
        # Prevent recursion in callbacks
        self._updating_axes = True
        
        for ax in self.axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)  # inverted y-axis for images
            
        if store:
            self.zoom_limits = (xmin, xmax, ymin, ymax)
            
        self._updating_axes = False
        self.fig.canvas.draw_idle()

    def reset_zoom(self, event=None):
        """Reset zoom to show full image."""
        self.zoom_limits = None
        
        # Get full extent from images
        if hasattr(self, 'im_gt'):
            # Get the shape of the image
            H, W = self.ms_gt.shape[1], self.ms_gt.shape[2]
            
            # Reset to full image with proper margins
            xmin, xmax = -0.5, W - 0.5
            ymin, ymax = H - 0.5, -0.5  # Inverted for images
            
            self._apply_zoom(xmin, xmax, ymax, ymin, store=False)
            
            # Clear rectangle selectors
            for selector in self.selectors:
                selector.set_active(True)
                if hasattr(selector, 'extents'):
                    selector.extents = (xmin, xmax, ymax, ymin)
                    
        self._update_info_text()