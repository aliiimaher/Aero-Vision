import torch
import yaml
from model import YOLOOBB  # Import your YOLOPose class

def load_yaml_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    import sys
    sys.path.append("/home/4TDrive/Yousefi/OBB/linedetector_4_cls")  # Append to sys.path within the block
    
    checkpoint_file = "/home/4TDrive/Yousefi/OBB/linedetector_4_cls/yolo12n.pt"
    # Load all arguments from YAML file, including 'model' and 'data'
    config_path = "/home/4TDrive/Yousefi/OBB/linedetector_4_cls/train_config.yaml"
    
    args = load_yaml_config(config_path)
    
    yolo_OBB = YOLOOBB(model=args['model'], task=args['task'], verbose=args['verbose'])
    
    state_dict = torch.load(checkpoint_file)
    
    modified_state_dict = {f"model.model.{k}": v for k, v in state_dict['model'].model.state_dict().items()}

    
    yolo_OBB.load_state_dict(modified_state_dict, strict=False)
    
    yolo_OBB.train(**args)
