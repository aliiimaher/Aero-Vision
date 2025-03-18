import torch
from pathlib import Path
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
import sys

# Add paths to custom modules
sys.path.append("/home/4TDrive/Yousefi/OBB/linedetector_4_cls")  # Replace with the directory containing obbtrainer.py

from trainer import CustomOBBTrainer  # Import custom OBB trainer
from validator import CustomOBBValidator
from callbacks.plot_custom_train_batch import plot_callback
from callbacks.plot_custom_train_batch import plot_callback
# YOLOPose class adapted for Oriented Bounding Box (OBB) task
class YOLOOBB(Model):
    """YOLO Oriented Bounding Box (OBB) Model."""

    def __init__(self, model="yolov8n-obb.pt", task="obb", verbose=False):
        """
        Initialize YOLOOBB model for oriented bounding box estimation task.

        Args:
            model (str | Path): Path to the pre-trained model file.
            task (str): Task type, default is "obb".
            verbose (bool): If True, prints additional information during initialization.
        """
        path = Path(model)
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes for OBB estimation."""
        return {
            "obb": {
                "model": OBBModel,  # Use the detection model for OBB tasks
                "trainer": lambda *args, **kwargs: CustomOBBTrainer(*args, plot_callback=plot_callback, **kwargs),
                "validator": lambda *args, **kwargs: CustomOBBValidator(*args, plot_callbackval=plot_callbackval, **kwargs),
                "predictor": yolo.obb.OBBPredictor,  # Default predictor for detection tasks
            },
        }
