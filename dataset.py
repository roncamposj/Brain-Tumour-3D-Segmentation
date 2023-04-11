import argparse 
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import tempfile
import time
import json
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
from unet import Unet_3D
from glob import glob
from monai.data import partition_dataset
from torch.utils.data import RandomSampler
import random

print_config()

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
    

class BrainTumourData():
    def __init__(self):
        super().__init__()

        self.train_transform = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        self.val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

        self.data = self.loadjson()

    def loadjson(self):
        with open('dataset.json','r') as f:
            metadata=json.load(f)

        data = metadata['training']
        return data


    def __call__(self):


        train_data = random.sample(self.data, int(0.8 * len(self.data)))
        val_data = [d for d in self.data if d not in train_data]

        train_ds = CacheDataset(
            data=train_data,
            cache_rate=0.0,
            num_workers=4,
            transform=self.train_transform
        )

        val_ds = CacheDataset(
            data=val_data,
            cache_rate=0.0,
            num_workers=4,
            transform=self.val_transform
        )

        # sampler = RandomSampler(dataset, replacement=True, num_samples=int(0.8*len(dataset)))

        train_loader = DataLoader(train_ds, batch_size=1, num_workers=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, shuffle=True)

        return train_ds, val_ds, train_loader, val_loader

