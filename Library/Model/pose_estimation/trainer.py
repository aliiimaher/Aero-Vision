import torch
from torch.utils.data import Dataset
from ultralytics.models.yolo.pose import PoseTrainer
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from dataset import PhysicalCellsSampledDataset
from ultralytics.data.build import InfiniteDataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
import numpy as np
from copy import deepcopy
from validator import CustomPoseValidator
from call_backs.plot_custom_train_batch import plot_training_samples 

# Function to get training transformations
def get_train_transforms():
    return A.Compose([
        A.Blur(p=0.1),
        A.Resize(1024, 1024),
        ToTensorV2(),
    ])

# Collate function for batching
def collate_fn(batch):
    new_batch = {}
    keys = batch[0][0].keys()
    values = list(zip(*[list(b[0].values()) for b in batch]))
    for i, k in enumerate(keys):
        value = values[i]
        if k == "img":
            value = torch.stack(value, 0)
        if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
            value = torch.cat(value, 0)
        new_batch[k] = value
    new_batch["batch_idx"] = torch.cat([torch.full([len(b[0]["cls"])], i) for i, b in enumerate(batch)]).to(batch[0][0]['img'].device)
    new_batch["cls"] = new_batch["cls"][:, None]
    new_batch["sample_infos"] = [b[1] for b in batch]  # Add sample info
    return new_batch

# Custom PoseTrainer class to use new dataset and InfiniteDataLoader
class CustomPoseTrainer(PoseTrainer):
    def __init__(self, *args, plot_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_callback = plot_callback  # Store the callback

    def get_dataloader(self, images_dir, batch_size, rank, mode):
        parent_path, last_folder_name = os.path.split(images_dir)
        dataset = PhysicalCellsSampledDataset(
            data_dir=parent_path,
            transform=get_train_transforms()
        )
        
        shuffle = mode == "train"
        sampler = DistributedSampler(dataset, shuffle=shuffle)

        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )

    def train_step(self, batch, batch_idx, epoch=None):
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        batch_data, sample_infos = batch
        print(f"Training with batch {batch_idx}")
        print(f"Image files in batch: {batch_data['im_file']}")
        print(f"Labels in batch: {batch_data['cls']}")
        
        return super().train_step(batch_data, batch_idx)
    
    def get_validator(self):
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return CustomPoseValidator(
            dataloader=self.test_loader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with keypoints only."""
        if self.plot_callback:
            self.plot_callback(
                batch["img"],
                batch["batch_idx"],
                batch["keypoints"],
                batch["sample_infos"],
                batch["im_file"],
                self.save_dir,
                ni
            )
        else:
            super().plot_training_samples(batch, ni)  # Fallback to the original method
