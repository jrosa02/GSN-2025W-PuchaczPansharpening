import logging
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from pystac_client import Client
import planetary_computer as pc
import rasterio
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
import pytest
from typing import Tuple
from ConfigParser import ConfigParser
import sys
from functools import lru_cache


logger = logging.getLogger(__name__)

def augment_tensor(data_tensor, crop_size, rotation_deg=45, noise_std=0.01):
    """
    Augment multispectral tensors using torchvision transforms.
    
    Input: (C,H,W) or (H,W,C) or batched variants.
    Output: (C,H_crop,W_crop)
    """

    # -------- Normalize input shape to (C,H,W) --------
    if data_tensor.ndim == 5:           # [B, C, 1, H, W]
        data_tensor = data_tensor.squeeze(2)

    if data_tensor.ndim == 4 and data_tensor.shape[0] == 1:
        data_tensor = data_tensor.squeeze(0)  # [C,H,W]

    if data_tensor.ndim == 3 and data_tensor.shape[0] <= 10:
        # Already (C,H,W)
        pass
    elif data_tensor.ndim == 3:
        # Assume (H,W,C)
        data_tensor = data_tensor.permute(2, 0, 1)  # -> (C,H,W)

    C, H, W = data_tensor.shape

    # # -------- torchvision augmentation pipeline --------
    # transform = T.Compose([
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.RandomVerticalFlip(p=0.5),
    #     T.RandomRotation(degrees=rotation_deg, expand=False),
    #     T.RandomCrop(size=crop_size)
    # ])

    # # Apply transform
    # augmented = transform(data_tensor)

    return data_tensor

GeoBBOXs = {
    "Poland": [14.124562, 49.002063, 24.145562, 54.835563],
    "Poland_with_bounds": [9.086269710379241, 47.900026791895904, 27.30727373945674, 54.21477439048045],
    "Paris": [2.2241, 48.8156, 2.4699, 48.9022],
}

# ========================================================================================================================

class Sentinel2DownLoader:
    """
    Iterator for Sentinel-2 L2A images from Planetary Computer.
    
    Parameters
    ----------
    bbox : list
        Bounding box [min_lon, min_lat, max_lon, max_lat]
    time : str
        Time range in 'YYYY-MM-DD/YYYY-MM-DD'
    bands : list, optional
        List of bands to download (default: ["B02","B03","B04","B06","B08"])
    save_tiff : bool, optional
        If True, save downloaded TIFFs locally
    download_folder : str, optional
        Dataset location
    normalize : bool, optional
        If True, normalize bands to 0-1
    """

    def __init__(self, bbox, time, max_items: int | None = None, download_folder="dataset_sentinel"):
        logger.info("Initializing Sentinel2Loader")
        self.configparser = ConfigParser()
        self.bbox = bbox
        self.time = time
        self.max_items = max_items
        self.download_folder = Path(download_folder).resolve()

            # Log parameters
        logger.info(
            "Sentinel2Loader parameters:\n"
            "  bbox: %s\n"
            "  time range: %s\n"
            "  max_items: %s\n"
            "  download_folder: %s",
            self.bbox,
            self.time,
            self.max_items if self.max_items is not None else "unlimited",
            self.download_folder,
        )

        logger.info(f"Making dir structure at {self.download_folder}")
        os.makedirs(self.download_folder/"input/", exist_ok=True)
        os.makedirs(self.download_folder/"input/rgb/", exist_ok=True)
        os.makedirs(self.download_folder/"input/mul/", exist_ok=True)
        os.makedirs(self.download_folder/"output/", exist_ok=True)

        self.stac_connect()
    
    def stac_connect(self):
        # Connect to STAC
        stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        logger.info(f"Connecting to STAC catalog: {stac_url}")
        self.catalog = Client.open(stac_url)

        # Search items
        logger.info("Searching for Sentinel-2 items")
        search = self.catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=self.bbox,
            datetime=self.time,
            query={"eo:cloud_cover": {"lt": 10}}
        )
        self.items = list(search.items())
        self._download_index = 0
        logger.info(f"Found {len(self.items)} items.")

    def stac_download(self, item):

        logger.info(f"Downloading item: {item.id}")
        signed_item = pc.sign(item)
        band_arrays = []
        band_profiles = []
        shapes = []

        for band in self.configparser.get_sentinel_2_bands():
            band_name = band["band"]
            logger.debug(f"Processing band: {band_name}")
            asset = signed_item.assets[band_name]

            # Read band data at native resolution
            logger.debug(f"Reading data for band: {band_name}")
            with rasterio.open(asset.href) as src:
                data = src.read(1)  # (H, W) numpy array
                profile = src.profile

            band_arrays.append(data)
            band_profiles.append(profile)
            shapes.append(data.shape)  # (H, W)

        return (band_arrays, band_profiles, shapes)

    def download_next(self):
        if self._download_index >= len(self.items) or \
        (self.max_items is not None and self._download_index >= int(self.max_items)):
            logger.info("No more items to iterate over")
            raise StopIteration

        item = self.items[self._download_index]
        self._download_index += 1

        # Read all bands at native resolution and keep profiles for optional save
        band_arrays, band_profiles, shapes = self.stac_download(item)

        # Determine common (max) shape among bands so we only upsample smaller bands once
        heights = [s[0] for s in shapes]
        widths = [s[1] for s in shapes]
        common_h = max(heights)
        common_w = max(widths)

        # Convert numpy arrays to torch and upsample those that are smaller than common shape
        processed_tensors = []
        for idx, arr in enumerate(band_arrays):
            h, w = arr.shape
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)

            if (h, w) != (common_h, common_w):
                # Upsample to common size once (use bilinear for upsampling, area for downsampling)
                # Choose mode based on whether we're up or downsampling for better quality
                mode = "bilinear" if (common_h >= h and common_w >= w) else "area"
                t = F.interpolate(t, size=(common_h, common_w), mode=mode, align_corners=False if mode == "bilinear" else None)
                # resulting shape: (1,1,common_h,common_w)

            processed_tensors.append(t.squeeze(0))  # each is (1,Hc,Wc)

        # Stack to (C, Hc, Wc)
        stacked = torch.cat(processed_tensors, dim=0)  # (C, Hc, Wc) because processed_tensors are (1,Hc,Wc)

        # Final resize to TARGET_SHAPE in ONE call (only if needed)
        target_h, target_w = self.configparser.get_output_cv2_shape()[:2]
        if (common_h, common_w) != (target_h, target_w):
            # interpolate expects (N, C, H, W)
            stacked = F.interpolate(stacked.unsqueeze(0), size=(target_h, target_w), mode="area").squeeze(0)  # (C, target_h, target_w)

        # Match original output format:
        # original code produced a tensor shaped like (C,1,H,W) via building a list of (1,H,W) and np.array(...)
        final_np = stacked.cpu().numpy()  # (C, 1, H, W)
        final_torch = torch.from_numpy(final_np)        # preserve dtype/behavior similar to original

        logger.info(f"Completed processing item: {item.id}")
        logger.debug(f"Item Shapes: {[t.shape for t in processed_tensors]}")
        return final_torch, item.id

    def generate_target_tensor(self, sentinel_tensor):
        # sentinel_tensor is CHW (5,4096,2048)
        # select 4 MS bands (example: 1,2,3,4)
        ms_tensor = sentinel_tensor[1:, :, :]   # -> (4,4096,2048)
        augmented = augment_tensor(ms_tensor, self.configparser.get_mul_input_cv2_shape()   [:2], noise_std=0)

        return augmented  # (4,4096,2048)

    def generate_target_tensors(self, sentinel_tensor, n):
        return [self.generate_target_tensor(sentinel_tensor) for _ in range(n)]

    def save_target_tensor(self, target_tensor, filename):
        torch.save(target_tensor, f"{self.download_folder}/output/{filename}.pt")

    def save_input_2tensor(self, input_2tensor, filename):
        torch.save(input_2tensor[0], f"{self.download_folder}/input/rgb/{filename}.pt")
        torch.save(input_2tensor[1], f"{self.download_folder}/input/mul/{filename}.pt")

    def generate_input_2tensor(self, rgbms_tensor, target_tensor):
        rgb = rgbms_tensor[:3, :, :]

        # Downsample MS
        ms = target_tensor.unsqueeze(0)  # (1,x, x, x)
        ms = F.interpolate(ms, size=self.configparser.get_mul_input_tensor_shape()[1:], mode="area")  # (1,4,H,W)
        ms = ms.squeeze(0)

        return rgb, ms
    
    def __iter__(self):
        self._count = 0
        return self
    
    def __next__(self):
        # Enforce download quantity limit
        if self.max_items is not None and self._count >= self.max_items:
            logger.info("Reached maximum download count: %d", self.max_items)
            raise StopIteration

        try:
            sentinel_tensor, id = self.download_next()
        except StopIteration:
            # Propagate natural exhaustion
            raise

        logger.info(f"Generating target_tensors for: {id}")
        target_tensors = self.generate_target_tensors(sentinel_tensor, 1)

        for i, tensor in enumerate(target_tensors):
            new_id = str(id)+str(i)
            logger.debug(f"Generating input_tensors for: {new_id}")
            input_2tensor= self.generate_input_2tensor(sentinel_tensor, tensor)
            logger.debug(f"saving tensors: {new_id}")
            self.save_target_tensor(tensor, new_id)
            self.save_input_2tensor(input_2tensor, new_id)

        logger.info(f"Saved target_tensors: {id}")
        self._count += 1


# ================================================================================================================================

class SentinelDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        super().__init__()
        path = Path(path).resolve()

        self.config = ConfigParser()

        self.input_path_rgb = path / "input/rgb"
        self.input_path_mul = path / "input/mul"
        self.output_path = path / "output"

        # Gather files
        self.files_rgb = sorted(p for p in self.input_path_rgb.rglob("*") if p.is_file())
        self.files_mul = sorted(p for p in self.input_path_mul.rglob("*") if p.is_file())
        self.files_out = sorted(p for p in self.output_path.rglob("*") if p.is_file())

        if not (len(self.files_rgb) and len(self.files_mul) and len(self.files_out)):
            raise ValueError("Empty input directories or missing files.")

        if not (len(self.files_rgb) == len(self.files_mul) == len(self.files_out)):
            raise ValueError(
                "Number of RGB, MUL and OUTPUT files must be equal and corresponding."
            )

        self.num_samples = len(self.files_rgb)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index):
        rgb = torch.load(str(self.files_rgb[index]), weights_only=False)
        mul = torch.load(str(self.files_mul[index]), weights_only=False)
        out = torch.load(str(self.files_out[index]), weights_only=False)

        if rgb.ndim != 3 or mul.ndim != 3 or out.ndim != 3:
            raise ValueError("Loaded tensors must be CHW format.")

        return (rgb, mul), out

    def produce_dataloaders(
        self,
        train_frac=0.7,
        val_frac=0.2,
        batch_size=32,
        pin_memory=torch.cuda.is_available()
    ):
        num_workers = self.config.get_training_num_workers()

        total_len = len(self)
        train_len = int(train_frac * total_len)
        val_len = int(val_frac * total_len)
        test_len = total_len - train_len - val_len

        train_ds, val_ds, test_ds = random_split(
            self, [train_len, val_len, test_len]
        )

        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory),
        )


# ================================================================================================================================

class SentinelCroppedDataset(SentinelDataset):
    def __init__(
        self,
        path: str | Path,
        rgb_crop: Tuple[int, int] | None = None,
        mul_crop: Tuple[int, int] | None = None,
        out_crop: Tuple[int, int] | None = None,
        cache_size: int = 4,  # Number of images to cache per worker
    ):
        super().__init__(path)

        if rgb_crop is None:
            rgb_crop = tuple(self.config.get_chunk_size())

        if mul_crop is None:
            mul_crop = (self.config.get_chunk_size()[0]//2, self.config.get_chunk_size()[1]//2)

        if out_crop is None:
            out_crop = tuple(self.config.get_chunk_size())

        self.rgb_crop_h, self.rgb_crop_w = rgb_crop
        self.mul_crop_h, self.mul_crop_w = mul_crop
        self.out_crop_h, self.out_crop_w = out_crop

        # Load just the first image to get dimensions
        rgb = torch.load(str(self.files_rgb[0]), weights_only=False, map_location='cpu')
        _, H_rgb, W_rgb = rgb.shape
        
        # Calculate total tiles per image
        self.tiles_h = H_rgb // self.rgb_crop_h
        self.tiles_w = W_rgb // self.rgb_crop_w
        self.tiles_per_image = self.tiles_h * self.tiles_w
        
        # Build index map: simple sequential mapping
        self.num_samples = len(self.files_rgb) * self.tiles_per_image
        
        # Cache size per worker
        self.cache_size = cache_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index):
        # Calculate which image and tile this index refers to
        img_idx = index // self.tiles_per_image
        tile_idx = index % self.tiles_per_image
        tile_row = tile_idx // self.tiles_w
        tile_col = tile_idx % self.tiles_w
        
        # Use the worker's own cache (simple LRU cache)
        # Each DataLoader worker will have its own instance of this cache
        rgb, mul, out = self._get_cached_tensors(img_idx)
        
        # Calculate crop positions
        top_rgb = tile_row * self.rgb_crop_h
        left_rgb = tile_col * self.rgb_crop_w
        top_mul = tile_row * self.mul_crop_h
        left_mul = tile_col * self.mul_crop_w
        
        # Extract crops
        rgb_crop = rgb[:, top_rgb:top_rgb + self.rgb_crop_h,
                          left_rgb:left_rgb + self.rgb_crop_w]
        
        mul_crop = mul[:, top_mul:top_mul + self.mul_crop_h,
                          left_mul:left_mul + self.mul_crop_w]
        
        out_crop = out[:, top_rgb:top_rgb + self.out_crop_h,
                          left_rgb:left_rgb + self.out_crop_w]
        
        return (rgb_crop, mul_crop), out_crop
    
    # Simple cache using Python's lru_cache - each worker gets its own cache
    @lru_cache(maxsize=5)  # Default cache size of 4 images per worker
    def _load_tensors(self, img_idx: int):
        """Load and cache tensors for an image"""
        rgb = torch.load(str(self.files_rgb[img_idx]), weights_only=False, map_location='cpu')
        mul = torch.load(str(self.files_mul[img_idx]), weights_only=False, map_location='cpu')
        out = torch.load(str(self.files_out[img_idx]), weights_only=False, map_location='cpu')
        
        if rgb.ndim != 3 or mul.ndim != 3 or out.ndim != 3:
            raise ValueError("Loaded tensors must be CHW format.")
            
        return rgb, mul, out
    
    def _get_cached_tensors(self, img_idx: int):
        """Wrapper to use the cache"""
        return self._load_tensors(img_idx)
    
    def produce_dataloaders(
        self,
        train_frac=0.7,
        val_frac=0.2,
        batch_size=32,
        pin_memory=torch.cuda.is_available()
    ):
        num_workers = self.config.get_training_num_workers()
        
        # IMPORTANT: With caching, each worker needs its own dataset instance
        # We'll create separate instances for each split
        
        total_len = len(self)
        train_len = int(train_frac * total_len)
        val_len = int(val_frac * total_len)
        test_len = total_len - train_len - val_len
        
        # Get indices for splits
        indices = torch.randperm(total_len).tolist()
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]
        
        # Create subset datasets with their own caches
        train_ds = torch.utils.data.Subset(self, train_indices)
        val_ds = torch.utils.data.Subset(self, val_indices)
        test_ds = torch.utils.data.Subset(self, test_indices)
        
        # Use persistent_workers=True to keep workers alive between epochs
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory,
                       persistent_workers=True if num_workers > 0 else False),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory,
                       persistent_workers=True if num_workers > 0 else False),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory,
                       persistent_workers=True if num_workers > 0 else False),
        )


# ====================================================================================================
# import cProfile
# def profileDataloader():
#     dataset = SentinelCroppedDataset("./dataset_sentinel")
    
#     for (x_rgb, x_mul), y in dataset:
#         pass
    
# if __name__ == "__main__":
#     cProfile.run('profileDataloader()', sort='cumtime', filename="sentinel_loader")
    
if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("./logs/sentinel2_loader.log"),
            logging.StreamHandler()
        ]
    )

    if len(sys.argv) < 2:
        raise RuntimeError(
            "Missing argument: max_items\n"
            "Usage: python ./SentinelData.py <max_items>"
    )

    try:
        max_items = int(sys.argv[1])
    except ValueError:
        raise RuntimeError("max_items must be an integer")

    logger.info("Starting Sentinel2Loader main execution")
    loader = Sentinel2DownLoader(
        bbox=GeoBBOXs["Poland_with_bounds"],
        time="2024-01-01/2024-12-31",
        max_items=max_items
    )

    i = 0

    for image in loader:
        i = i + 1

    data_path = Path("./dataset_sentinel/")
    dataset = SentinelCroppedDataset(data_path)
    train_loader, val_loader, test_loader = dataset.produce_dataloaders()


# ----- Sentinel2DownLoader -------------------------------------------------------------------

def test_sentinel2downloader_init():
    loader = Sentinel2DownLoader(
        bbox=GeoBBOXs["Paris"],
        time="2024-01-01/2024-12-31",
    )

# ----- SentinelDataset -------------------------------------------------------------------
@pytest.fixture
def full_dataset():
    data_path = "./dataset_sentinel/"
    return SentinelDataset(data_path)

def test_full_shapes(full_dataset):
    config = ConfigParser()

    (x_rgb, x_mul), y = full_dataset[0]

    assert x_rgb.shape == config.get_rgb_input_tensor_shape(), f"rgb expected:{config.get_rgb_input_tensor_shape()} != {x_rgb.shape}"
    assert x_mul.shape == config.get_mul_input_tensor_shape(), f"mul expected:{config.get_mul_input_tensor_shape()} != {x_mul.shape}"
    assert y.shape[1:] == x_rgb.shape[1:], "out"

# ----- SentinelCroppedDataset -------------------------------------------------------------------
@pytest.fixture
def cropped_dataset():
    data_path = "./dataset_sentinel/"
    return SentinelCroppedDataset(data_path)

def test_cropped_shapes(cropped_dataset):
    (x_rgb, x_mul), y = cropped_dataset[0]

    assert x_rgb.shape == (3, 256, 256)
    assert x_mul.shape[1:] == (128, 128)
    assert y.shape[1:] == x_rgb.shape[1:]


def test_cropped_alignment(cropped_dataset):
    """
    Ensures that rgb[i], mul[i], and out[i] correspond to the same crop index.
    """
    for i in range(len(cropped_dataset)):
        (rgb_i, mul_i), out_i = cropped_dataset[i]

        assert rgb_i is not None
        assert mul_i is not None
        assert out_i is not None

        # Ensure same number of elements
        assert rgb_i.shape[1:] == out_i.shape[1:]


def test_iteration(cropped_dataset):
    for (x_rgb, x_mul), y in cropped_dataset:
        assert x_rgb.shape == (3, 256, 256)
        assert x_mul.shape[1:] == (128, 128)
        assert y.shape[1:] == x_rgb.shape[1:]


def test_full_pass(cropped_dataset):
    """
    Ensures full iteration does not crash and returns correct count.
    """
    count = 0
    for _ in cropped_dataset:
        count += 1

    assert count == len(cropped_dataset)

def test_dataloaders(cropped_dataset):
    train_loader, val_loader, test_loader = cropped_dataset.produce_dataloaders(
        train_frac=0.5, val_frac=0.25, batch_size=4
    )

    # Just verify that loaders produce batches without errors
    batch = next(iter(train_loader))
    (x_rgb, x_mul), y = batch

    assert x_rgb.shape[0] <= 4  # batch size
    assert x_rgb.shape[1:] == (3, 256, 256)
