#validator
import torch
from ultralytics.models.yolo.pose import PoseValidator
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
import matplotlib.cm as cm
from call_backs.plot_custom_val_batch import plot_val_samples

class CustomPoseValidator(PoseValidator):
    """Custom Pose Validator for YOLO Pose Estimation."""

    def plot_val_samples(self, batch, ni):
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
            super().plot_val_samples(batch, ni)  # Fallback to the original method
   