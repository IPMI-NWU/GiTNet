import os, glob
import pandas as pd
import torch
from PIL import Image
import numpy as np
from .base_dataset import BaseImageDataset

from torchvision import transforms
from monai.transforms import (
    MapLabelValue, EnsureChannelFirstd, NormalizeIntensityd, Resized, RandFlipd,
    ToTensord, Compose, RandAffined, RandRotated, RandSpatialCrop, RandSpatialCropd
)


class KvasirSegDataset(BaseImageDataset):
    def __init__(self, *argv, **kargs):
        super().__init__(*argv, **kargs)

        split_file = pd.read_csv(os.path.join(self.root, f"{self.split}.txt"), sep=" ", header=None)
        self.sample_list = split_file.iloc[:, 0].tolist()

        self.images = np.array([os.path.join(self.root, "images", f"{file}.jpg") for file in self.sample_list])[
            : int(len(self.sample_list) * self.size_rate)
        ]
        self.labels = np.array([os.path.join(self.root, "masks", f"{file}.jpg") for file in self.sample_list])[
            : int(len(self.sample_list) * self.size_rate)
        ]

    def _fetch_data(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"), dtype=np.float32)
        label = np.array(Image.open(self.labels[idx]).convert("L"), dtype=np.int16)

        return {"image": image, "label": label}

    def _transform_custom(self, data):
        label = data["label"]
        label = label.float() / 255.0

        label[label > 0.5] = 1
        label[label < 0.5] = 0

        data["label"] = label.long()

        return data


class KvasirGazeDataset(KvasirSegDataset):
    def __init__(self, pseudo_mask_root, fixation_path, *argv, **kargs):
        super().__init__(*argv, **kargs)
        self.pseudo_mask_root = pseudo_mask_root
        self.pseudo_labels = [np.array([os.path.join(pseudo_mask_root, f"heatmap", f"{file}.jpg",) for file in self.sample_list])]
        self.fixation_data = pd.read_csv(fixation_path)

    def _transform_custom(self, data):
        data = super()._transform_custom(data)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = data["image"]
        img = normalize(img / 255)
        data["image"] = img
        return data

    def get_transform(self):
        resize_keys = ["image", "label", "pseudo_label", "fixation"] if self.resize_label else ["image"]
        resize_mode = ["bilinear", "nearest", "bilinear", "nearest"] if self.resize_label else ["bilinear"]

        if self.split == "train" and self.do_augmentation:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image"],
                        channel_dim=2,
                    ),
                    EnsureChannelFirstd(
                        keys=["label", "pseudo_label", "fixation"],
                        channel_dim="no_channel",
                    ),
                    # RandSpatialCropd(
                    #     keys=["image", "label", "pseudo_label"],
                    #     roi_size=(512, 512),
                    # ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    RandFlipd(
                        keys=["image", "label", "pseudo_label", "fixation"],
                        prob=0.5,
                        spatial_axis=0,
                    ),
                    RandFlipd(
                        keys=["image", "label", "pseudo_label", "fixation"],
                        prob=0.5,
                        spatial_axis=1,
                    ),
                    RandRotated(
                        keys=["image", "label", "pseudo_label", "fixation"],
                        mode=['bilinear', 'nearest', 'bilinear', 'nearest'],
                        range_x=np.pi / 18,
                        range_y=np.pi / 18,
                        prob=0.5,
                        padding_mode=['reflection', 'reflection', 'reflection', 'reflection'],
                    ),
                    RandAffined(
                        keys=["image", "label", "pseudo_label", "fixation"],
                        mode=('bilinear', 'nearest', 'bilinear', 'nearest'),
                        prob=0.3,
                        rotate_range=(np.pi / 2, np.pi / 2),
                        scale_range=(0.05, 0.05)
                    ),
                ]
            )
        else:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image"],
                        channel_dim=2,
                    ),
                    EnsureChannelFirstd(
                        keys=["label", "pseudo_label", "fixation"],
                        channel_dim="no_channel",
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                ]
            )

    def _fetch_data(self, idx):
        data = super()._fetch_data(idx)
        pseudo_label = np.array(Image.open(self.pseudo_labels[0][idx])).astype(np.float32)
        data[f"pseudo_label"] = pseudo_label / 255

        img_name = self.images[idx].split("\\")[-1]
        image_data = self.fixation_data[self.fixation_data['IMAGE'] == img_name]

        # image_height = int(image_data['IMAGE_HEIGHT'].iloc[0])
        # image_width = int(image_data['IMAGE_WIDTH'].iloc[0])
        image_height, image_width = 224, 224
        fixation_array = np.full((image_height, image_width), -1, dtype=int)
        for idx, row in enumerate(image_data.itertuples(), start=1):
            x = int(row.CURRENT_FIX_X * image_width)
            y = int(row.CURRENT_FIX_Y * image_height)
            x = min(x, image_width - 1)
            y = min(y, image_height - 1)
            fixation_array[y, x] = idx
        data[f"fixation"] = fixation_array
        return data


if __name__ == "__main__":
    dataset = KvasirGazeDataset(
                root="Kvasir-SEG/",
                pseudo_mask_root=os.path.join("Kvasir-SEG/", "gaze"),
                split='train',
                spatial_size=224,
                do_augmentation=True,
                resize_label=True,
                size_rate=1,
            )

    data = dataset[0]
    print(data["image"].shape, data["label"].shape, data["pseudo_label"].shape)
    print(torch.min(data["pseudo_label"]), torch.max(data["pseudo_label"]))
    #
    # dataset = KvasirSegDataset(
    #     root="Kvasir-SEG/",
    #     split='test',
    #     spatial_size=224,
    #     do_augmentation=True,
    #     resize_label=True,
    #     size_rate=1,
    # )
    # data = dataset[0]
    # print(data["image"].shape, data["label"].shape)