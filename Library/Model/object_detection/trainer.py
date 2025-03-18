import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.build import InfiniteDataLoader
from dataset import PhysicalCellsObjectDetectionDataset
from torch.utils.data import DataLoader
from callbacks.plot_custom_val_batch import plot_callbackval
from ultralytics.utils.plotting import  plot_labels
import sys
from validator import CustomObjectDetectionValidator
from copy import deepcopy
import numpy as np

def get_custom_transforms():
    return A.Compose([
        A.OneOf([
            A.Blur(p=0.5),
            A.MedianBlur(p=0.5),
        ], p=0.7),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.5), contrast_limit=0.5, p=0.5
            ),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=60, p=0.5),
            A.ToGray(p=0.5),
        ], p=0.5),
        A.GaussNoise(var_limit=(50, 200)),
        A.ImageCompression(
            quality_lower=15,
            quality_upper=20,
            p=0.5,
        ),
        A.GaussianBlur(
            blur_limit=(3, 15),
            p=0.5,
        ),
        A.CoarseDropout(
            max_holes=10,
            max_height=200,
            max_width=200,
            min_holes=1,
            min_height=10,
            min_width=10,
            fill_value=(0, 0, 0),
            always_apply=False , # Ensures CoarseDropout is applied to every image
            p=0.4,
        ),
        ToTensorV2(),  # Convert to tensor at the end
    ])

def collate_fn(batch):
    if not batch:
        return {}
    
    new_batch = {}
    keys = batch[0][0].keys()
    values = list(zip(*[list(b[0].values()) for b in batch]))
    
    for i, k in enumerate(keys):
        value = values[i]
        if k == "img":
            value = torch.stack(value, 0)
        if k in {"bboxes", "cls"}:
            value = torch.cat(value, 0)
        new_batch[k] = value
    
    new_batch["batch_idx"] = torch.cat([torch.full([len(b[0]["cls"])], i) for i, b in enumerate(batch)]).to(batch[0][0]['img'].device)
    new_batch["cls"] = new_batch["cls"][:, None]
    new_batch["sample_infos"] = [b[1] for b in batch]
    return new_batch


from torch.utils.data import DistributedSampler, RandomSampler

class CustomObjectDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, plot_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_callback = plot_callback

    def get_dataloader(self, images_dir, batch_size, rank, mode):
        parent_path, last_folder_name = os.path.split(images_dir)
        dataset = PhysicalCellsObjectDetectionDataset(
            data_dir=parent_path,
            transform=get_custom_transforms(),
            use_keypoint_clipping=True,
        )
        
        if rank == -1:  # Single GPU or CPU
            sampler = RandomSampler(dataset) #if mode == "train" else None
        else:  # Distributed training
            sampler = DistributedSampler(dataset) #if mode == "train" else None

        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),  # Shuffle if no sampler is used
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4  # Adjust according to your system
        )

    def train_step(self, batch, batch_idx):
        batch_data, sample_infos = batch
        print(f"Training with batch {batch_idx}")
        print(f"Image files in batch: {batch_data['im_file']}")
        print(f"Bounding Boxes in batch: {batch_data['bboxes']}")
        print(f"Labels in batch: {batch_data['cls']}")
        
        return super().train_step(batch_data, batch_idx)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CustomObjectDetectionValidator(
            dataloader=self.test_loader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
            plot_callbackval=plot_callbackval  # Pass the custom plotting callback
        )

    def plot_training_samples(self, batch, ni):
        """Override plot_training_batch to use custom callback if provided."""
        if self.plot_callback:
            self.plot_callback(batch, ni, self.save_dir)
        else:
            # Call the parent class's plot_training_batch method
            super().plot_training_samples(batch, ni)

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        # Initialize empty lists to store valid boxes and classes
        valid_boxes = []
        valid_cls = []

        # Loop over the labels in the dataset
        for lb in self.train_loader.dataset.labels:

            # Handle bounding boxes: Check if bboxes exist, otherwise create an empty (0, 4) array
            if lb["bboxes"].shape[0] == 0:
                valid_boxes.append(np.zeros((0, 4)))  # Append an empty (0, 4) array
            else:
                valid_boxes.append(lb["bboxes"])  # Append the valid bboxes

            # Handle classes: Add classes if they exist
            if lb["cls"].shape[0] == 0:
                valid_cls.append(np.zeros((0, 1)))  # Append an empty array for classes
            else:
                if lb["cls"].ndim == 1:
                    lb["cls"] = lb["cls"][:, None]  # Convert 1D array to 2D array with shape (N, 1)
                valid_cls.append(lb["cls"])  # Append the valid classes
        # Concatenate the valid bounding boxes and classes        if valid_boxes:
            boxes = np.concatenate(valid_boxes, axis=0)
        if valid_cls:
            cls = np.concatenate(valid_cls, axis=0)

        # Ensure classes are squeezed (remove any extra dimensions)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)
