
import torch
from pathlib import Path
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from trainer import CustomObjectDetectionTrainer
from validator import CustomObjectDetectionValidator
from ultralytics import NAS
import sys 
import os

# Add the parent directory of `callbacks` to `sys.path` for module imports
sys.path.insert(0, os.path.abspath('/home/training/Aerialytic_AI/New-Graph-Extraction/library'))

# Import custom plotting callbacks for training and validation
from callbacks.plot_custom_val_batch import plot_callbackval
from callbacks.plot_custom_train_batch import plot_callback

class YOLOObjectDetection(Model):
    def __init__(self, model, task="detect", verbose=False):
        """
        Initialize the YOLOObjectDetection model.

        Args:
            model (str): Path to the model file (must be provided).
            task (str): The task type (default is 'detect').
            verbose (bool): Flag to control verbosity (default is False).
        """
        super().__init__(model=model, task=task)  # Initialize the parent Model class
        self.verbose = verbose  # Save the verbose argument


    @property
    def task_map(self):
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": lambda *args, **kwargs: CustomObjectDetectionTrainer(*args, plot_callback=plot_callback, **kwargs),
                "validator": lambda *args, **kwargs: CustomObjectDetectionValidator(*args, plot_callbackval=plot_callbackval, **kwargs),
                "predictor": yolo.detect.DetectionPredictor,
            },
        }
