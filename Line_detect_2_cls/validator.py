import torch
from ultralytics.models.yolo.detect import DetectionValidator
from dataset import Line_2_objectdetectiondataset
from torch.utils.data import DistributedSampler, RandomSampler
from ultralytics.data.build import InfiniteDataLoader
import matplotlib as mpl
import albumentations as A
import numpy as np
import os
import cv2
import sys
sys.path.insert(0, os.path.abspath('/home/training/Aerialytic_AI/New-Graph-Extraction'))
from ultralytics.utils.plotting import output_to_target
from library.Model.object_detection.validator import  plot_custom_images, get_custom_transforms, collate_fn
from transforms import get_dsm_transform

class CustomObjectDetectionValidator(DetectionValidator):
    def __init__(self, *args, plot_callbackval=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_callbackval = plot_callbackval

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
            sampler = RandomSampler(dataset) #if mode == "train" else None
        else:  # Distributed training
            sampler = DistributedSampler(dataset) #if mode == "train" else None
        batch_size = batch_size // (2 if mode == "val" else 1)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size * 2,
            shuffle=(sampler is None),  # Shuffle if no sampler is used
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=8  # Adjust according to your system
        )
    
    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        try:
            val_dir = self.save_dir / f"val"
            val_dir.mkdir(exist_ok=True)
            e = (len(os.listdir(val_dir))//2) * self.args.val_interval
            batch_dir = val_dir / f"e{e}_b{ni}_pred"
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(len(batch["img"])):
                individual_batch = {
                    "img": batch["img"][i:i+1],
                    "im_file": batch["im_file"][i:i+1],
                }
                
                individual_preds = output_to_target(preds[i:i+1], max_det=self.args.max_det)
                
                # Plot the custom images with bounding boxes
                img_with_predictions = plot_custom_images(
                    individual_batch["img"],
                    *individual_preds,
                    paths=individual_batch["im_file"],
                    fname=batch_dir / f"val_batch{ni}_img{i}_pred.jpg",
                    names=self.names,
                    on_plot=self.on_plot,
                    save=False  # Don't save immediately, we'll handle it below
                )
                
                if img_with_predictions is None:
                    continue  # Handle the case where the image was not generated
                
                # Create a white canvas of 1800x1800 pixels
                canvas = np.full((1800, 1800, 3), 255, dtype=np.uint8)

                # Get the dimensions of the image
                h, w, _ = img_with_predictions.shape
                
                # Calculate starting positions to center the image on the canvas
                start_y = (canvas.shape[0] - h) // 2
                start_x = (canvas.shape[1] - w) // 2
                
                # Place the image on the canvas
                canvas[start_y:start_y+h, start_x:start_x+w] = img_with_predictions

                canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(batch_dir / f"val_batch{ni}_img{i}.jpg"), canvas_bgr)
        except:
            pass


    def plot_val_samples(self, batch, ni):
        """Plot validation image samples using the custom plot_callback."""
        try:
            if self.plot_callbackval:
                self.plot_callbackval(self, batch, ni)
            else:
                super().plot_val_samples(batch, ni)
        except:
            pass