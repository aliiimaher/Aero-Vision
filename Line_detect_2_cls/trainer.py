import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.build import InfiniteDataLoader
from dataset import Line_2_objectdetectiondataset
from callbacks.plot_custom_val_batch import plot_callbackval
from validator import CustomObjectDetectionValidator
from copy import deepcopy
from library.Model.object_detection.trainer import collate_fn, get_custom_transforms
from torch.utils.data import DistributedSampler, RandomSampler
import torch.distributed as dist
from transforms import get_dsm_transform

class CustomObjectDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, plot_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_callback = plot_callback
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def get_dataloader(self, images_dir, batch_size, rank, mode):
        parent_path, last_folder_name = os.path.split(images_dir)
        dataset = Line_2_objectdetectiondataset(
            data_dir=parent_path,
            transform=get_dsm_transform(),
            use_keypoint_clipping=False,
            use_aligner=False,
        )

        # sampler = RandomSampler(dataset)
        if rank == -1:  # Single GPU or CPU
            sampler = RandomSampler(dataset) if mode == "train" else None
        else:  # Distributed training
            sampler = DistributedSampler(dataset) if mode == "train" else None
        batch_size = batch_size // (2 if mode == "val" else 1)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size * 2,
            shuffle=(sampler is None),  # Shuffle if no sampler is used
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=8  # Adjust according to your system
        )

    def train_step(self, batch, batch_idx):
        batch_data, sample_infos = batch
        print(f"Training with batch {batch_idx}")
        print(f"Image files in batch: {batch_data['im_file']}")
        print(f"Bounding Boxes in batch: {batch_data['bboxes']}")
        print(f"Labels in batch: {batch_data['cls']}")
        
        return super().train_step(batch_data, batch_idx)

    def plot_training_samples(self, batch, ni):
        try:
            print("plot_training_samples")
            """Override plot_training_batch to ensure it runs on one rank."""
            if self.plot_callback:
                # Only execute on the main process (rank 0)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    self.plot_callback(self, batch, ni)
            else:
                # Call the parent class's plot_training_batch method
                super().plot_training_samples(batch, ni)
        except:
            pass

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CustomObjectDetectionValidator(
            dataloader=self.test_loader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
            plot_callbackval=plot_callbackval  # Pass the custom plotting callback
        )
