import yaml
from model import YOLOObjectDetection  # Import your YOLOObjectDetection class

def load_yaml_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    # Load all arguments from YAML file, including 'model' and 'data'
    config_path = "/home/training/Aerialytic_AI/New-Graph-Extraction/library/Model/object_detection/train_config.yaml"
    args = load_yaml_config(config_path)

    # Ensure the model path is retrieved from the config
    model_path = args['model']  # Get the model path from the config

    # Initialize the object detection model using the model path from the config
    yolo_detection = YOLOObjectDetection(model=model_path, task=args['task'], verbose=args.get('verbose', False))

    # Start training with all arguments passed as keyword arguments
    yolo_detection.train(**args)
