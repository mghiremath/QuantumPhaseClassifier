import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from PIL import ImageDraw
import os
import re

# ---- Crop helper for your specific image setup ----
def crop_lattice(img):
    # Crops to the 370x370 lattice region, top=58, left=143
    return TF.crop(img, top=58, left=143, height=370, width=370)

def mask_temp_box(img):
    draw = ImageDraw.Draw(img)
    # Rectangle: left, top, right, bottom
    draw.rectangle([16, 44, 16+148, 44+16], fill="black")  # or fill=img.getpixel((0,0)) for background
    return img

class IsingFolderDataset(Dataset):
    """
    Loads Ising images from folders labeled by temperature.
    Crops to central lattice, maps grayscale to [-1, 1].
    """
    def __init__(self, data_dir, flatten=True, transform=None):
        self.samples = []
        self.flatten = flatten
        self.transform = transform or transforms.Compose([
            transforms.Lambda(crop_lattice),
            transforms.Lambda(mask_temp_box),  # Mask the temperature box
            transforms.Grayscale(),
            transforms.ToTensor(),  # [C, H, W]
            transforms.Lambda(lambda x: x * 2 - 1),  # [0,1] -> [-1,1]
        ])
        temp_pat = re.compile(r'Temp=(\d+)-(\d+)')
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            m = temp_pat.match(folder)
            if m:
                temp = float(m.group(1) + "." + m.group(2))
                # Map temperature to label
                if temp < 2.0:
                    label = 0
                elif temp < 2.5:
                    label = 1
                else:
                    label = 2
                for fname in os.listdir(folder_path):
                    if fname.endswith(".png"):
                        self.samples.append((os.path.join(folder_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        x = self.transform(img)  # [1, 370, 370]
        if self.flatten:
            x = x.view(-1)  # flatten for MLP: [1*370*370]
        return x, torch.tensor(label).long()

def get_image_dataloaders(
    data_dir,
    batch_size=32,
    train_ratio=0.8,
    flatten=True,
    shuffle=True,
    seed=42
):
    """
    Splits dataset into train/test DataLoaders.
    """
    ds = IsingFolderDataset(data_dir, flatten=flatten)
    train_len = int(train_ratio * len(ds))
    test_len = len(ds) - train_len
    train_ds, test_ds = random_split(ds, [train_len, test_len], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader