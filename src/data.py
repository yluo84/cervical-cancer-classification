"""Dataset, splits and transforms for cervix VIA image classification.

The train/val/test assignment is read from the label CSV rather than being
re-generated at run time. The split is grouped at the case level: every image
belonging to one case is assigned to exactly one split, so no case contributes
images to both training and evaluation. `scripts/check_split_leakage.py`
verifies this property on the shipped CSV.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["Normal", "Precancerous", "Cancerous"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_labels(csv_path, images_root):
    """Read the label CSV and resolve each relative path to an absolute one.

    Rows whose image file is not present on disk are dropped and reported, so a
    partially downloaded dataset fails loudly rather than silently changing the
    evaluation set.

    Returns
    -------
    df : pandas.DataFrame
        Rows with an existing image file, with an added ``abs_path`` column.
    missing : pandas.DataFrame
        The rows that were dropped.
    """
    images_root = Path(images_root)
    df = pd.read_csv(csv_path)

    required = {"image_path", "class", "split", "case_id"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Label CSV is missing columns: {sorted(missing_cols)}")

    df["abs_path"] = df["image_path"].apply(
        lambda rel: images_root / str(rel).strip()
    )
    exists = df["abs_path"].apply(lambda p: p.exists())

    missing = df.loc[~exists].copy()
    df = df.loc[exists].reset_index(drop=True)
    return df, missing


def build_transforms(image_size):
    """Training and evaluation transforms.

    VIA images are field-acquired photographs with non-standardised framing,
    illumination and white balance, so the training augmentations target those
    specific nuisance factors. The random crop keeps at least 90% of the frame
    to avoid cropping the lesion out of the image.
    """
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.90, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03
        ),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tfms, eval_tfms


class CervixDataset(Dataset):
    """Images and integer class labels for one split."""

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(Path(row["abs_path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["class"])
