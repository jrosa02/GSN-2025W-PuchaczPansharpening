import torch
from torch.utils.data import DataLoader, Dataset, random_split
import os
from pathlib import Path


class SmallDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        path = Path(path)

        self.input_path = path / "input"
        self.output_path = path / "output"

        # List files only inside each directory (not mixed)
        self.input_file_list = sorted([f for f in self.input_path.rglob("*") if f.is_file()])
        self.output_file_list = sorted([f for f in self.output_path.rglob("*") if f.is_file()])

        # They must match in count
        assert len(self.input_file_list) == len(self.output_file_list), \
            "input and output file counts do not match!"

    def __len__(self):
        return len(self.input_file_list)

    def __getitem__(self, index):
        input_file = self.input_file_list[index]
        output_file = self.output_file_list[index]

        # Load tensors
        x = torch.load(input_file)   # (96, 96, 7)
        y = torch.load(output_file)  # (96, 96, 7)

        # Swap axes to (C, H, W)
        x = x.permute(2, 0, 1).float()
        y = y.permute(2, 0, 1).float()

        return x, y
    

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

if __name__ == "__main__":
    dataset = SmallDataset("dataset/multispectral_field_images")
    dataset.__getitem__(10)
    # Define split sizes
    dataset.produce_dataloaders()
