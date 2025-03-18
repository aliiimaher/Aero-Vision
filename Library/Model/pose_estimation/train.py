import torch
import yaml
from model import YOLOPose  # Import your YOLOPose class

def load_yaml_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    import sys
    sys.path.append("/home/training/Yousefi/pose_estimation")  # Append to sys.path within the block

    # Load all arguments from YAML file, including 'model' and 'data'
    config_path = "/home/training/Yousefi/pose_estimation/train_config.yaml"
    args = load_yaml_config(config_path)

    # Initialize and train the pose estimation model using all arguments from YAML
    yolo_pose = YOLOPose(model=args['model'], task=args['task'], verbose=args['verbose'])

    # Pass all arguments to the training function
    yolo_pose.train(**args)
