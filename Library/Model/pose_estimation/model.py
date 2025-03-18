import torch
from pathlib import Path
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
import sys
import sys
sys.path.append("/home/training/Yousefi/pose_estimation")  # Replace with the directory containing posetrainer.py

from trainer import CustomPoseTrainer
from validator import CustomPoseValidator
# Example usage in the YOLOPose class
class YOLOPose(Model):
    """YOLO Pose Estimation Model."""

    def __init__(self, model="yolov8n-pose.pt", task="pose", verbose=False):
        """
        Initialize YOLOPose model for pose estimation task.

        Args:
            model (str | Path): Path to the pre-trained model file.
            task (str): Task type, default is "pose".
            verbose (bool): If True, prints additional information during initialization.
        """
        path = Path(model)
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes for pose estimation."""
        return {
            "pose": {
                "model": PoseModel,
                "trainer": CustomPoseTrainer,  # Use your custom trainer here
                "validator": CustomPoseValidator,  # Use your custom validator here
                "predictor": yolo.pose.PosePredictor,  # Default predictor
            },
        }
