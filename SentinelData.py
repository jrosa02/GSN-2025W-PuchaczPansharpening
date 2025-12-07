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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
        return augment_tensor(sentinel_tensor, TARGET_SHAPE, noise_std=0)

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
        pass

