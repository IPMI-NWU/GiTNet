import os

from torch.utils.data import Dataset, DataLoader

from .kvasir_seg import (
    KvasirSegDataset,
    KvasirGazeDataset,
)
from .nci_isbi import (
    NCIISBIProstateDataset,
    NCIISBIProstateGazeDataset,
)


def get_dataloader(args, split, resize_label):
    if args.dataset == "Kvasir" and split == "train":

        dataset = KvasirGazeDataset(
            root="../Data/Kvasir-SEG/",
            pseudo_mask_root=os.path.join("../Data/Kvasir-SEG/", "gaze"),
            fixation_path=os.path.join("../Data/Kvasir-SEG/", "kvasir_fixation.csv"),
            split='train',
            spatial_size=224,
            do_augmentation=True,
            resize_label=resize_label,
            size_rate=1,
        )
    if args.dataset == "Kvasir" and split == "test":
        dataset = KvasirGazeDataset(
            root="../Data/Kvasir-SEG/",
            pseudo_mask_root=os.path.join("../Data/Kvasir-SEG/", "gaze"),
            fixation_path=os.path.join("../Data/Kvasir-SEG/", "kvasir_fixation.csv"),
            split='test',
            spatial_size=224,
            do_augmentation=False,
            resize_label=resize_label,
            size_rate=1,
        )

    if args.dataset == "NCI" and split == "train":
        dataset = NCIISBIProstateGazeDataset(
            root="../Data/NCI-ISBI-2013/",
            pseudo_mask_root=os.path.join("../Data/NCI-ISBI-2013/", "gaze"),
            fixation_path=os.path.join("../Data/NCI-ISBI-2013/", "nci-isbi_fixation.csv"),
            split='train',
            spatial_size=224,
            do_augmentation=True,
            resize_label=resize_label,
            size_rate=1,
        )
    if args.dataset == "NCI" and split == "test":
        dataset = NCIISBIProstateGazeDataset(
            root="../Data/NCI-ISBI-2013/",
            pseudo_mask_root=os.path.join("../Data/NCI-ISBI-2013/", "gaze"),
            fixation_path=os.path.join("../Data/NCI-ISBI-2013/", "nci-isbi_fixation.csv"),
            split='test',
            spatial_size=224,
            do_augmentation=False,
            resize_label=resize_label,
            size_rate=1,
        )

    if split == "train":
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    else:
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
