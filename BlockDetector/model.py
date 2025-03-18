import torch
from pathlib import Path
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
import sys
import sys
from callbacks.plot_custom_train_batch import plot_callback
from callbacks.plot_custom_val_batch import plot_callbackval
from trainer import CustomPoseTrainer
sys.path.append("/home/training/BlockDetector") 
from validator import CustomPoseValidator
# Example usage in the YOLOPose class
class YOLOPose(Model):
    """YOLO Pose Estimation Model."""

    def __init__(self, model="yolov8n-pose.pt", task="pose"):
        """
        Initialize YOLOPose model for pose estimation task.

        Args:
            model (str | Path): Path to the pre-trained model file.
            task (str): Task type, default is "pose".
            verbose (bool): If True, prints additional information during initialization.
        """
        path = Path(model)
        super().__init__(model=model, task=task)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes for pose estimation."""
        return {
            "pose": {
                "model": PoseModel,
                "trainer": lambda *args, **kwargs: CustomPoseTrainer(*args, plot_callback=plot_callback, **kwargs),
                "validator": lambda *args, **kwargs: CustomPoseValidator(*args, plot_callbackval=plot_callbackval,**kwargs),  # Use your custom validator here
                "predictor": yolo.pose.PosePredictor,  # Default predictor
            },
        }
