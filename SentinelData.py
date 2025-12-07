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


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentinel2_loader.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

MAX_DATA_SIZE = 100 * 2**30  # 100 GB

RPI_SHAPE = (2592, 4608)
MULTI_SHAPE = (1152 ,2048)

RPI_SPATIAL_RES = 27
MULTI_SPATIAL_RES = int(27*4608/2048)

TARGET_SHAPE = (4096, 2048)
INPUT_SHAPE = (2048, 1024)

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

    # -------- torchvision augmentation pipeline --------
    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=rotation_deg, expand=False),
        T.RandomCrop(size=crop_size)
    ])

    # Apply transform
    augmented = transform(data_tensor)

    return augmented


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
        Folder to save TIFFs if save_tiff=True
    normalize : bool, optional
        If True, normalize bands to 0–1
    """
    
    def __init__(self, bbox, time, bands=None, save_tiff=False, download_folder="sentinel2_data"):
        logger.info("Initializing Sentinel2Loader")
        self.bbox = bbox
        self.time = time
        self.bands = ["B02","B03","B04","B06","B08"]
        self.input_spatial_res_m = [10, 10, 10, 20, 10]
        self.imput_shapes = [(10980, 10980), (10980, 10980), (10980, 10980), (5490, 5490), (10980, 10980), ]
        self.output_spatial_res_m = 27
        # self.output_shape = (int(10980/2.7), int(10980/2.7))
        self.output_shape = (4096, 2048)
        self.save_tiff = save_tiff
        self.download_folder = download_folder

        if self.save_tiff:
            logger.info(f"Creating download folder: {self.download_folder}")
            os.makedirs(self.download_folder, exist_ok=True)

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
            query={"eo:cloud_cover": {"lt": 20}}
        )
        self.items = list(search.items())
        self._download_index = 0
        logger.info(f"Found {len(self.items)} items.")

    def __iter__(self):
        return self


    def download_next(self):
        if self._download_index >= len(self.items):
            logger.info("No more items to iterate over")
            raise StopIteration

        item = self.items[self._download_index]
        self._download_index += 1
        logger.info(f"Downloading item: {item.id}")
        signed_item = pc.sign(item)

        # Read all bands at native resolution and keep profiles for optional save
        band_arrays = []
        band_profiles = []
        shapes = []

        for band in self.bands:
            logger.debug(f"Processing band: {band}")
            asset = signed_item.assets[band]

            # Read band data at native resolution
            logger.debug(f"Reading data for band: {band}")
            with rasterio.open(asset.href) as src:
                data = src.read(1)  # (H, W) numpy array
                profile = src.profile

            band_arrays.append(data)
            band_profiles.append(profile)
            shapes.append(data.shape)  # (H, W)

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

            # squeeze batch dim -> (1, common_h, common_w)
            processed_tensors.append(t.squeeze(0))  # each is (1,Hc,Wc)

        # Stack to (C, Hc, Wc)
        stacked = torch.cat(processed_tensors, dim=0)  # (C, Hc, Wc) because processed_tensors are (1,Hc,Wc)

        # Final resize to TARGET_SHAPE in ONE call (only if needed)
        target_h, target_w = TARGET_SHAPE
        if (common_h, common_w) != (target_h, target_w):
            # interpolate expects (N, C, H, W)
            stacked = F.interpolate(stacked.unsqueeze(0), size=(target_h, target_w), mode="area").squeeze(0)  # (C, target_h, target_w)

        # Match original output format:
        # original code produced a tensor shaped like (C,1,H,W) via building a list of (1,H,W) and np.array(...)
        final_np = stacked.cpu().numpy()  # (C, 1, H, W)
        final_torch = torch.from_numpy(final_np)        # preserve dtype/behavior similar to original

        logger.debug(f"Completed processing item: {item.id}")
        logger.debug(f"Item Shapes: {[t.shape for t in processed_tensors]}")
        return final_torch, item.id

    
    def generate_target_tensor(self, sentinel_tensor):
        # sentinel_tensor is CHW (5,4096,2048)
        # select 4 MS bands (example: 1,2,3,4)
        ms_tensor = sentinel_tensor[1:, :, :]   # -> (4,4096,2048)
        augmented = augment_tensor(ms_tensor, TARGET_SHAPE, noise_std=0)

        return augmented  # (4,4096,2048)

    def generate_target_tensors(self, sentinel_tensor, n):
        return [self.generate_target_tensor(sentinel_tensor) for _ in range(n)]
    
    def save_target_tensor(self, target_tensor, filename):
        os.makedirs("./dataset_sentinel/output/", exist_ok=True)
        torch.save(target_tensor, f"./dataset_sentinel/output/{filename}.pt", pickle_protocol=5, _use_new_zipfile_serialization=True)

    def save_input_2tensor(self, input_2tensor, filename):
        os.makedirs("./dataset_sentinel/input/", exist_ok=True)
        os.makedirs("./dataset_sentinel/input/rgb/", exist_ok=True)
        os.makedirs("./dataset_sentinel/input/mul/", exist_ok=True)
        torch.save(input_2tensor[0], f"./dataset_sentinel/input/rgb/{filename}.pt", pickle_protocol=5, _use_new_zipfile_serialization=True)
        torch.save(input_2tensor[1], f"./dataset_sentinel/input/mul/{filename}.pt", pickle_protocol=5, _use_new_zipfile_serialization=True)

    def generate_input_2tensor(self, rgbms_tensor, target_tensor):
         # rgbms_tensor = (5,4096,2048) original
        # target_tensor = (4,4096,2048) output

        rgb = rgbms_tensor[:3, :, :]  # (3,4096,2048)

        # Downsample MS to (4,2048,1024)
        ms = target_tensor.unsqueeze(0)  # (1,4,4096,2048)
        ms = F.interpolate(ms, size=(2048,1024), mode="area")  # (1,4,H,W)
        ms = ms.squeeze(0)

        return rgb, ms

    def __next__(self):
        sentinel_tensor, id = self.download_next()
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


# ================================================================================================================================

class SentinelDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        path = Path(path)
        
        self.input_path_rgb = path / "input/rgb"
        self.input_path_mul = path / "input/mul"
        self.output_path = path / "output"

        files_rgb = sorted([f for f in self.input_path_rgb.rglob("*") if f.is_file()])
        files_mul = sorted([f for f in self.input_path_mul.rglob("*") if f.is_file()])
        assert len(files_rgb) == len(files_mul), \
            "RGB and Multispectral input file lists have different lenghts!"
        
        self.input_file_list = list(zip(files_rgb, files_mul)) #sorted list of tuples like

        self.output_file_list = sorted([f for f in self.output_path.rglob("*") if f.is_file()])
    
    def __len__(self):
        return len(self.output_file_list) #input file list has twice the number of output elements

    def __getitem__(self, index):
        input_files = self.input_file_list[index]
        input_file_rgb = input_files[0]
        input_file_mul = input_files[1]
        
        output_file = self.output_file_list[index]

        x_rgb = torch.load(input_file_rgb, weights_only=False)
        x_mul = torch.load(input_file_mul, weights_only=False)
        y = torch.load(output_file, weights_only=False)

        return (x_rgb, x_mul), y
    
    def produce_dataloaders(self, train_frac = 0.7, val_frac = 0.2, batch_size = 16, num_workers = 16):
        total_len = len(self)
        train_len = int(train_frac * total_len)
        val_len = int(val_frac * total_len)
        test_len = total_len - train_len - val_len

        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            self, [train_len, val_len, test_len]
        )

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return train_loader, val_loader, test_loader
    


# ==================================================================================================================================

class SentinelCroppedDataset(Dataset):
    """
    Tile-based dataset created from saved CHW tensors on disk.

    Expected stored tensor shapes:
      - RGB:  (C_rgb, H_rgb, W_rgb)    (e.g. (3, 4096, 2048))
      - MUL:  (C_mul, H_mul, W_mul)    (e.g. (4, 2048, 1024))
      - OUT:  (C_out, H_out, W_out)    (e.g. (4, 4096, 2048))

    Default cropping behavior (non-overlapping):
      - RGB tiles -> (C_rgb, 256, 128)
      - MUL tiles -> (C_mul, 128, 64)
      - OUT tiles -> (C_out, 256, 128)

    Tiles are aligned so that tile (r, c) in RGB ↔ tile (r, c) in MUL ↔ tile (r, c) in OUT.
    """

    def __init__(
        self,
        path: str | Path,
        rgb_crop: Tuple[int, int] = (256, 128),
        mul_crop: Tuple[int, int] = (128, 64),
        out_crop: Tuple[int, int] = (256, 128),
    ):
        super().__init__()
        path = Path(path)

        self.input_path_rgb = path / "input/rgb"
        self.input_path_mul = path / "input/mul"
        self.output_path = path / "output"

        self.rgb_crop_h, self.rgb_crop_w = rgb_crop
        self.mul_crop_h, self.mul_crop_w = mul_crop
        self.out_crop_h, self.out_crop_w = out_crop

        self.num_samples = 0

        # Gather files (sorted to preserve deterministic order)
        self.files_rgb = sorted([p for p in self.input_path_rgb.rglob("*") if p.is_file()])
        self.files_mul = sorted([p for p in self.input_path_mul.rglob("*") if p.is_file()])
        self.files_out = sorted([p for p in self.output_path.rglob("*") if p.is_file()])

        if not (len(self.files_rgb) and len(self.files_mul) and len(self.files_out)):
            raise ValueError("Empty input directories or missing files.")

        if not (len(self.files_rgb) == len(self.files_mul) == len(self.files_out)):
            raise ValueError("Number of RGB, MUL and OUTPUT files must be equal and corresponding by order.")

        # Precompute index mapping (dataset index -> (image_index, tile_row, tile_col))
        self.index_map = []

        for img_idx, (rgb_file, mul_file, out_file) in enumerate(zip(self.files_rgb, self.files_mul, self.files_out)):
            # Load minimal metadata only (here we load the tensor but do not keep it in memory)
            rgb = torch.load(str(rgb_file), weights_only=False)
            mul = torch.load(str(mul_file), weights_only=False)
            out = torch.load(str(out_file), weights_only=False)

            # Validate shapes are CHW
            if rgb.ndim != 3 or mul.ndim != 3 or out.ndim != 3:
                raise ValueError(f"Files {rgb_file}, {mul_file} or {out_file} do not contain 3D CHW tensors.")

            _, H_rgb, W_rgb = rgb.shape
            _, H_mul, W_mul = mul.shape
            _, H_out, W_out = out.shape

            # Determine how many tiles in each axis (floor division -> remainder truncated)
            tiles_h_rgb = H_rgb // self.rgb_crop_h
            tiles_w_rgb = W_rgb // self.rgb_crop_w

            tiles_h_mul = H_mul // self.mul_crop_h
            tiles_w_mul = W_mul // self.mul_crop_w

            tiles_h_out = H_out // self.out_crop_h
            tiles_w_out = W_out // self.out_crop_w

            # Ensure tile grid counts match between RGB / MUL / OUT
            if not (tiles_h_rgb == tiles_h_mul == tiles_h_out and tiles_w_rgb == tiles_w_mul == tiles_w_out):
                raise ValueError(
                    f"Tile grid mismatch for image index {img_idx}:\n"
                    f"RGB tiles (h,w)=({tiles_h_rgb},{tiles_w_rgb}), "
                    f"MUL tiles=({tiles_h_mul},{tiles_w_mul}), OUT tiles=({tiles_h_out},{tiles_w_out}).\n"
                    "Ensure the stored images sizes are consistent and divisible by respective crop sizes "
                    "or choose crop sizes that align across inputs."
                )

            # Append mapping entries (row-major order)
            for th in range(tiles_h_rgb):
                self.num_samples = self.num_samples + 1
                for tw in range(tiles_w_rgb):
                    self.index_map.append((img_idx, th, tw))

        if len(self.index_map) == 0:
            raise ValueError("No tiles were produced — check image and crop sizes.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index):
        img_idx, tile_row, tile_col = self.index_map[index]

        rgb_file = self.files_rgb[img_idx]
        mul_file = self.files_mul[img_idx]
        out_file = self.files_out[img_idx]

        # Load full tensors (CHW)
        rgb = torch.load(str(rgb_file), weights_only=False)   # (C_rgb, H_rgb, W_rgb)
        mul = torch.load(str(mul_file), weights_only=False)   # (C_mul, H_mul, W_mul)
        out = torch.load(str(out_file), weights_only=False)   # (C_out, H_out, W_out)

        # Compute crop coordinates for RGB / OUT (they share crop size)
        top_rgb = tile_row * self.rgb_crop_h
        left_rgb = tile_col * self.rgb_crop_w
        rgb_crop = rgb[:, top_rgb : top_rgb + self.rgb_crop_h, left_rgb : left_rgb + self.rgb_crop_w]

        out_crop = out[:, top_rgb : top_rgb + self.out_crop_h, left_rgb : left_rgb + self.out_crop_w]

        # Compute crop coordinates for MUL
        top_mul = tile_row * self.mul_crop_h
        left_mul = tile_col * self.mul_crop_w
        mul_crop = mul[:, top_mul : top_mul + self.mul_crop_h, left_mul : left_mul + self.mul_crop_w]

        # Final sanity checks (shapes)
        if rgb_crop.shape[1:] != (self.rgb_crop_h, self.rgb_crop_w):
            raise RuntimeError(f"RGB crop has unexpected size: {rgb_crop.shape}")
        if mul_crop.shape[1:] != (self.mul_crop_h, self.mul_crop_w):
            raise RuntimeError(f"MUL crop has unexpected size: {mul_crop.shape}")
        if out_crop.shape[1:] != (self.out_crop_h, self.out_crop_w):
            raise RuntimeError(f"OUT crop has unexpected size: {out_crop.shape}")

        return (rgb_crop, mul_crop), out_crop

    def produce_dataloaders(self, train_frac=0.7, val_frac=0.2, batch_size=32, num_workers=8, pin_memory=True):
        total_len = len(self)
        train_len = int(train_frac * total_len)
        val_len = int(val_frac * total_len)
        test_len = total_len - train_len - val_len

        train_ds, val_ds, test_ds = random_split(self, [train_len, val_len, test_len])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader, test_loader


# ====================================================================================================

# if __name__ == "__main__":
#     logger.info("Starting Sentinel2Loader main execution")
#     loader = Sentinel2DownLoader(
#         bbox=[2.2241, 48.8156, 2.4699, 48.9022],
#         time="2024-01-01/2024-12-31",
#         save_tiff=False
#     )

#     i = 0

#     for image in loader:
#         if i > 9: break
#         i = i + 1

#     data_path = Path("/home/karolina/studia/GSN-2025W-PuchaczPansharpening/dataset_sentinel/")
#     dataset = SentinelDataset(data_path)
#     train_loader, val_loader, test_loader = dataset.produce_dataloaders()


# =====================================================================================================

@pytest.fixture
def dataset():
    data_path = Path("/home/karolina/studia/GSN-2025W-PuchaczPansharpening/dataset_sentinel/")
    return SentinelDataset(data_path)


def test_file_list_lengths(dataset):
    """RGB and MUL lists must contain the same number of elements."""
    assert len(dataset.input_file_list) == len(dataset.output_file_list)


def test_len_method(dataset):
    """Check that __len__ returns correct value."""
    assert len(dataset) == len(dataset.output_file_list)


def test_first_sample_shapes(dataset):
    """Check that first sample loads and has correct pansharpening-friendly shapes."""
    (x_rgb, x_mul), y = dataset[0]

    # Tensor type
    assert isinstance(x_rgb, torch.Tensor)
    assert isinstance(x_mul, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    # Must be (C, H, W)
    assert x_rgb.ndim == 3
    assert x_mul.ndim == 3
    assert y.ndim == 3

    # RGB high-res and Y high-res must match
    assert x_rgb.shape[1:] == y.shape[1:], \
        "RGB and output MS must have same spatial resolution."

    # MUL must be lower resolution
    mul_h, mul_w = x_mul.shape[1], x_mul.shape[2]
    hr_h, hr_w = x_rgb.shape[1], x_rgb.shape[2]

    assert mul_h < hr_h and mul_w < hr_w, \
        "Multispectral input should be lower resolution than RGB."

    # Channels non-empty
    assert x_rgb.shape[0] > 0
    assert x_mul.shape[0] > 0
    assert y.shape[0] > 0


def test_middle_sample(dataset):
    """Check that a middle sample loads properly."""
    idx = len(dataset) // 2
    (x_rgb, x_mul), y = dataset[idx]

    assert x_rgb.ndim == 3
    assert x_mul.ndim == 3
    assert y.ndim == 3


def test_random_access(dataset):
    """Test random indexing and pansharpening shape relations."""
    import random
    idx = random.randint(0, len(dataset) - 1)
    (x_rgb, x_mul), y = dataset[idx]

    # RGB and Y must match
    assert x_rgb.shape[1:] == y.shape[1:]

    # MUL must be smaller
    assert x_mul.shape[1] < x_rgb.shape[1]
    assert x_mul.shape[2] < x_rgb.shape[2]

def test_data_loaders(dataset):
    """Test the dataloaders from SentinelDataset."""
    train_loader, val_loader, test_loader = dataset.produce_dataloaders()
    total = len(dataset)
    train_len = int(0.7 * total)
    val_len   = int(0.2 * total)
    test_len  = total - train_len - val_len

    assert len(train_loader.dataset) == train_len
    assert len(val_loader.dataset) == val_len
    assert len(test_loader.dataset) == test_len

    # Check batches
    xb, yb = next(iter(train_loader))
    (rgb_batch, mul_batch), out_batch = xb, yb

    assert rgb_batch.shape[1:] == (3, 4096, 2048)
    assert mul_batch.shape[1:] == (4, 2048, 1024)
    assert out_batch.shape[1:] == (4, 4096, 2048)


# ----- SentinelCroppedDataset -------------------------------------------------------------------
@pytest.fixture
def cropped_dataset():
    data_path = Path("/home/karolina/studia/GSN-2025W-PuchaczPansharpening/dataset_sentinel/")
    return SentinelCroppedDataset(data_path)

def test_cropped_dataset_len(cropped_dataset):
    # Expect 16 crops: (1024/256) × (512/128) = 4×4 = 16
    # For each image (currently: 11 images)
    # So 16x11 = 176
    assert len(cropped_dataset) == 176


def test_cropped_shapes(cropped_dataset):
    (x_rgb, x_mul), y = cropped_dataset[0]

    assert x_rgb.shape == (3, 256, 128)
    assert x_mul.shape[1:] == (128, 64)
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
        assert x_rgb.shape == (3, 256, 128)
        assert x_mul.shape[1:] == (128, 64)
        assert y.shape[1:] == x_rgb.shape[1:]


def test_full_pass(cropped_dataset):
    """
    Ensures full iteration does not crash and returns correct count.
    """
    count = 0
    for _ in cropped_dataset:
        count += 1

    assert count == len(cropped_dataset)

# def test_dataloaders(cropped_dataset):
#     train_loader, val_loader, test_loader = cropped_dataset.produce_dataloaders(
#         train_frac=0.5, val_frac=0.25, batch_size=4, num_workers=0
#     )

#     # Just verify that loaders produce batches without errors
#     batch = next(iter(train_loader))
#     (x_rgb, x_mul), y = batch

#     assert x_rgb.shape[0] <= 4  # batch size
#     assert x_rgb.shape[1:] == (3, 256, 128)