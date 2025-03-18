import torch
from ultralytics.models.yolo.pose import PoseTrainer
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from torch.utils.data import DistributedSampler, RandomSampler
from dataset import PhysicalCellsObjectDetectionDataset
from ultralytics.data.build import InfiniteDataLoader
from torch.utils.data.distributed import DistributedSampler
#from callbacks.plot_custom_val_batch import plot_callbackval
from copy import deepcopy
from validator import CustomPoseValidator
from callbacks.plot_custom_val_batch import plot_callbackval
import torch.distributed as dist

def get_train_transforms():
    return A.Compose([
        ToTensorV2(),
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
        elif k in {"bboxes", "cls", "keypoints"}:  # Added "keypoints" here
            value = torch.cat(value, 0)
        new_batch[k] = value

    new_batch["batch_idx"] = torch.cat(
        [torch.full([len(b[0]["cls"])], i) for i, b in enumerate(batch)]
    ).to(batch[0][0]['img'].device)
    new_batch["cls"] = new_batch["cls"][:, None]
    new_batch["sample_infos"] = [b[1] for b in batch]
    
    return new_batch  # Ensure it’s a dictionary, not a tuple

# Custom PoseTrainer class to use new dataset and InfiniteDataLoader
class CustomPoseTrainer(PoseTrainer):
    def __init__(self, *args, plot_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_callback = plot_callback  # Store the callback
        self.workers = getattr(self.args, 'workers')
    def get_dataloader(self, images_dir, batch_size, rank, mode):
        parent_path, last_folder_name = os.path.split(images_dir)
        dataset = PhysicalCellsObjectDetectionDataset(
            data_dir=parent_path,
            transform=get_train_transforms()
        )
        
        if rank == -1:  # Single GPU or CPU
            sampler = RandomSampler(dataset) if mode == "train" else None
        else:  # Distributed training
            sampler = DistributedSampler(dataset) if mode == "train" else None
  
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=8  ,
            shuffle=(sampler is None),  # Shuffle if no sampler is used
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=24,
        )
 
    def plot_training_samples(self, batch, ni):
        """Override plot_training_batch to ensure it runs on one rank."""
        if self.plot_callback:
            # Only execute on the main process (rank 0)
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.plot_callback(self, batch, ni)
        else:
            # Call the parent class's plot_training_batch method
            super().plot_training_samples(batch, ni)
  
    def train_step(self, batch, batch_idx, epoch=None):
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
            _callbacks=self.callbacks,
           plot_callbackval=plot_callbackval 
        )
    
