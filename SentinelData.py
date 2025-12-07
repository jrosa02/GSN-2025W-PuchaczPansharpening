import logging
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from pystac_client import Client
import planetary_computer as pc
import rasterio

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


import torch
import torch.nn.functional as F
import random

def augment_tensor(data_tensor, crop_size=RPI_SHAPE, rotation_deg=45, noise_std=0.01):
    """
    Apply random augmentations to a multispectral tensor.
    
    Parameters
    ----------
    data_tensor : torch.Tensor
        Tensor of shape (C, H, W) or (H, W, C)
    crop_size : tuple
        Size of random crop (height, width)
    rotation_deg : float
        Maximum rotation angle in degrees (+/-)
    noise_std : float
        Standard deviation of added Gaussian noise
    """
    # After loading and resizing all bands
    # sentinel_tensor shape: [C,H,W]
    if data_tensor.ndim == 5:
        # probably [B, C, 1, H, W]
        data_tensor = data_tensor.squeeze(2)  # remove extra dim -> [B,C,H,W]

    # For single sample, remove batch dim
    if data_tensor.shape[0] == 1:
        data_tensor = data_tensor.squeeze(0)  # -> [C,H,W]
    data_tensor = data_tensor.squeeze(1)

    # Ensure shape is (H, W, C) for grid_sample
    if data_tensor.shape[0] <= 10:  # assume C < 10 → (C,H,W)
        data_tensor = data_tensor.permute(1, 2, 0)  # (H,W,C)
    
    H, W, C = data_tensor.shape

    # --- Random horizontal flip ---
    if random.random() < 0.5:
        data_tensor = torch.flip(data_tensor, dims=[1])  # flip width

    # --- Random vertical flip ---
    if random.random() < 0.5:
        data_tensor = torch.flip(data_tensor, dims=[0])  # flip height

    # --- Random rotation ---
    if random.random() < 0.9:
        angle = random.uniform(-rotation_deg, rotation_deg)
        angle_rad = torch.tensor(angle * torch.pi / 180)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        R = torch.tensor([[cos_a, -sin_a],
                          [sin_a,  cos_a]], dtype=torch.float32)

        # Create coordinate grid
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        x_c = x - (W - 1) / 2
        y_c = y - (H - 1) / 2
        coords = torch.stack([x_c, y_c], dim=0).reshape(2, -1)

        coords_rot = R @ coords
        coords_rot = coords_rot.reshape(2, H, W)
        x_rot = coords_rot[0] + (W - 1) / 2
        y_rot = coords_rot[1] + (H - 1) / 2

        # Normalize to [-1,1] for grid_sample
        x_norm = 2 * (x_rot / (W - 1)) - 1
        y_norm = 2 * (y_rot / (H - 1)) - 1
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # (1,H,W,2)

        # Apply rotation
        data_tensor = F.grid_sample(
            data_tensor.permute(2, 0, 1).unsqueeze(0),  # (1,C,H,W)
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)  # back to (H,W,C)

    # --- Random crop ---
    crop_h, crop_w = crop_size
    if H > crop_h and W > crop_w:
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        data_tensor = data_tensor[top:top+crop_h, left:left+crop_w]

    return data_tensor  # (H_crop, W_crop, C)

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
        band_tensors = []

        for band in self.bands:
            logger.debug(f"Processing band: {band}")
            asset = signed_item.assets[band]

            # Read band data
            logger.debug(f"Reading data for band: {band}")
            with rasterio.open(asset.href) as src:
                data = src.read(1)  # (H,W)
                profile = src.profile

            # Optional save
            if self.save_tiff:
                logger.debug(f"Saving TIFF for band: {band}")
                out_path = os.path.join(self.download_folder, f"{item.id}_{band}.tif")
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data, 1)

            # Convert to torch tensor
            logger.debug(f"Converting band {band} to tensor")
            tensor = torch.from_numpy(data).float()

            target_h, target_w = TARGET_SHAPE

            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # -> [1, 1, H, W]
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)  # -> [1, C, H, W]

            # target_h, target_w = desired size
            tensor = F.interpolate(
                tensor,
                size=(target_h, target_w),
                mode="area"
            ).squeeze(0)

            band_tensors.append(tensor.unsqueeze(0))  # shape: (1,H,W)

        logger.debug(f"Completed processing item: {item.id}")
        logger.debug(f"Item Shapes: {[b_t.shape for b_t in band_tensors]}")
        return torch.from_numpy(np.array(band_tensors)), item.id
    
    def generate_target_tensor(self, sentinel_tensor):
        return augment_tensor(sentinel_tensor, RPI_SHAPE, noise_std=0)

    def generate_target_tensors(self, sentinel_tensor, n):
        return [self.generate_target_tensor(sentinel_tensor) for _ in range(n)]
    
    def save_target_tensor(self, target_tensor, filename):
        os.makedirs("./dataset_sentinel/output/", exist_ok=True)
        torch.save(target_tensor, f"./dataset_sentinel/output/{filename}.pt", pickle_protocol=5, _use_new_zipfile_serialization=True)

    def save_input_2tensor(self, input_2tensor, filename):
        os.makedirs("./dataset_sentinel/input/", exist_ok=True)
        torch.save(input_2tensor[0], f"./dataset_sentinel/input/rgb_{filename}.pt", pickle_protocol=5, _use_new_zipfile_serialization=True)
        torch.save(input_2tensor[1], f"./dataset_sentinel/input/mul_{filename}.pt", pickle_protocol=5, _use_new_zipfile_serialization=True)

    def generate_input_2tensor(self, target_tensor):
        # target_tensor: [H,W,C] (e.g., 4096,2048,5)

        # Split RGB vs multispectral
        rgb_tensor = target_tensor[:, :, :3]   # [H,W,3]
        multi_tensor = target_tensor[:, :, 3:] # [H,W, remaining bands]

        # Convert multispectral to [C,H,W] for interpolation
        multi_tensor = multi_tensor.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
        multi_tensor = F.interpolate(multi_tensor, size=INPUT_SHAPE, mode="area")
        multi_tensor = multi_tensor.squeeze(0)  # -> [C,H,W]

        # RGB can stay HWC or convert to CHW if needed
        rgb_tensor = rgb_tensor.permute(2, 0, 1)  # -> [3,H,W]

        return (rgb_tensor, multi_tensor)

    def __next__(self):
        sentinel_tensor, id = self.download_next()
        logger.info(f"Generating target_tensors for: {id}")
        target_tensors = self.generate_target_tensors(sentinel_tensor, 4)
        for i, tensor in enumerate(target_tensors):
            new_id = str(id)+str(i)
            logger.debug(f"Generating input_tensors for: {new_id}")
            input_2tensor= self.generate_input_2tensor(tensor)
            logger.debug(f"saving tensors: {new_id}")
            self.save_target_tensor(tensor, new_id)
            self.save_input_2tensor(input_2tensor, new_id)
        
        logger.info(f"Saved target_tensors: {id}")


        




# class SmallDataset(Dataset):
#     def __init__(self) -> None:
#         super().__init__()

#     def __getitem__(self, index) -> Any:
#         return None
    
#     def __len__(self) -> int:
#         return None


if __name__ == "__main__":
    logger.info("Starting Sentinel2Loader main execution")
    loader = Sentinel2DownLoader(
        bbox=[2.2241, 48.8156, 2.4699, 48.9022],
        time="2024-01-01/2024-12-31",
        save_tiff=False
    )


    for image in loader:
        break


