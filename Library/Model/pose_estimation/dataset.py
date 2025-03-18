import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import random
from shapely.geometry import Point
from shapely.affinity import rotate, scale
from shapely.geometry import box, Polygon, LineString
from call_backs.clip_keypoint import KeypointClipper
from call_backs.reorder_keypoint import KeypointReorderer, apply_keypoint_reordering  # Import the reorderer

# Function to get a rectangular bounding polygon centered at a given point with specified width and height
def get_bounding_polygon(center: Point, width: int, height: int) -> box:
    # Return a rectangular bounding box centered at 'center' with the specified width and height
    return box(
        center.x - width / 2,  # Left x-coordinate
        center.y - height / 2, # Bottom y-coordinate
        center.x + width / 2,  # Right x-coordinate
        center.y + height / 2, # Top y-coordinate
    )

# Function to scale and pad a rectangle
def rect_scale_pad(rect, scale=1.0, pad=0):
    center, size, angle = rect  # Unpack rectangle parameters
    # Compute new size after scaling and padding
    new_size = (size[0] * scale + pad, size[1] * scale + pad)
    # Return the updated rectangle with scaled size
    return (center, new_size, angle)

# Function to crop an image based on a rectangle and apply optional horizontal flipping
def img_rectangle_cut(img, rect, target_size=(1024, 1024), flipy=False):
    # Get the four corner points of the rectangle
    box = cv2.boxPoints(rect)
    # Convert points to integer type
    box = np.int0(box)

    width, height = target_size

    # Define source points for perspective transform
    src_pts = box.astype("float32")
    # Define destination points for perspective transform
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # Apply the perspective transformation to the image
    warped = cv2.warpPerspective(img, M, (width, height))
    
    if flipy:
        # Flip the image horizontally if required
        warped = cv2.flip(warped, 1)
    
    # Return the cropped image, the original rectangle, and the transform matrix
    return warped, rect, M

# Function to find keypoints that intersect with a bounding box
def find_intersecting_keypoints(bbox, lines):
    # Create a polygon from the bounding box coordinates
    bbox_polygon = Polygon(bbox)
    keypoints = []

    for line in lines:
        # Create a LineString from the line's points
        line_string = LineString(line['points'])
        if bbox_polygon.intersects(line_string):
            # Check if the line intersects with the bounding box
            for point in line['points']:
                # Append intersecting points
                keypoints.append(point)

    # Return the list of keypoints as a numpy array
    return np.array(keypoints)

# Function to transform keypoints using a transformation matrix and handle optional horizontal flipping
def transform_keypoints(keypoints, M, flipy, width):
    if keypoints.size == 0:
        return keypoints
    
    # Convert keypoints to homogeneous coordinates
    keypoints_homogeneous = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
    # Apply the transformation matrix
    transformed_keypoints_homogeneous = keypoints_homogeneous @ M.T
    # Convert back to Cartesian coordinates
    transformed_keypoints = transformed_keypoints_homogeneous[:, :2] / transformed_keypoints_homogeneous[:, 2][:, np.newaxis]
    
    if flipy:
        # Adjust x-coordinates if horizontal flip is applied
        transformed_keypoints[:, 0] = width - transformed_keypoints[:, 0]
        
    return transformed_keypoints

class PhysicalCellsSampledDataset(Dataset):
    # Initialize the dataset
    def __init__(self, data_dir, transform=None, num_samples_per_cell=10, sample_size=1024, keypoint_clipper_callback=None, keypoint_reorder_callback=None):
        # Directory containing images
        self.images_dir = os.path.join(data_dir, 'images')
        # Directory containing labels
        self.labels_dir = os.path.join(data_dir, 'labels')
        # List of image files
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
        # List of label files
        self.label_files = [f for f in os.listdir(self.labels_dir) if f.endswith('.pkl')]
        self.transform = transform  # Optional transformation function
        self.num_samples_per_cell = num_samples_per_cell  # Number of samples per cell
        self.sample_size = sample_size  # Size of the sampled image

        # Pair image files with their corresponding label files
        self.data_pairs = self._pair_images_and_labels()
        # Initialize labels from label files
        self.labels = self._initialize_labels()
        # Optional keypoint clipping and reordering callbacks
        self.keypoint_clipper = keypoint_clipper_callback
        self.keypoint_reorderer = keypoint_reorder_callback

    # Pair image files with label files
    def _pair_images_and_labels(self):
        data_pairs = []
        for image_file in self.image_files:
            # Get base name without extension
            base_name = os.path.splitext(image_file)[0]
            # Create corresponding label file name
            label_file = f"{base_name}.pkl"
            if label_file in self.label_files:
                # Add the pair to the list
                data_pairs.append((image_file, label_file))
        return data_pairs

    # Initialize labels from label files
    def _initialize_labels(self):
        labels = []
        for img_file, label_file in self.data_pairs:
            # Load annotations from the label file
            label_path = os.path.join(self.labels_dir, label_file)
            with open(label_path, 'rb') as f:
                annotations = pickle.load(f)
            
            image_labels = []
            for annotation in annotations:
                points = annotation['points']
                if len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    # Compute bounding box coordinates
                    x_min, y_min = min(x1, x2), min(y1, y2)
                    x_max, y_max = max(x1, x2), max(y1, y2)
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2

                    # Append bounding box and center coordinates
                    image_labels.append([x_center, y_center, bbox_width, bbox_height])
            
            # Append labels for the image
            labels.append({"bboxes": np.array(image_labels), "cls": np.zeros((len(image_labels),), dtype=np.int64)})
        
        return labels

    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.data_pairs) * self.num_samples_per_cell

    # Get a sample from the dataset
    def __getitem__(self, idx):
        # Compute cell index and sample index within the cell
        cell_idx = idx // self.num_samples_per_cell
        sample_idx = idx % self.num_samples_per_cell
        
        # Get the image and label file for the current cell
        img_file, label_file = self.data_pairs[cell_idx]
        
        # Read the image
        img_path = os.path.join(self.images_dir, img_file)
        image = cv2.imread(img_path)
        img_height, img_width = image.shape[:2]
        # Define the center of the cell
        cell_center = Point(img_width // 2, img_height // 2)
        
        # Load annotations from the label file
        label_path = os.path.join(self.labels_dir, label_file)
        with open(label_path, 'rb') as f:
            annotations = pickle.load(f)

        # Define random parameters for sampling
        r_x = random.random()
        r_y = random.random()
        sample_center = Point(cell_center.x + r_x * 512, cell_center.y + r_y * 512)
        rotation_angle = random.uniform(0, 365)
        scale_val = random.uniform(0.5, 2)
        flipy = random.choice([True, False])
        
        # Create a bounding box and apply transformations
        bounding_box = get_bounding_polygon(sample_center, self.sample_size, self.sample_size)
        bounding_box = scale(bounding_box, xfact=scale_val, yfact=scale_val, origin=sample_center)
        bounding_box = rotate(bounding_box, rotation_angle, origin=sample_center)
        
        # Get minimum area rectangle for the bounding box
        box_coords = np.array(bounding_box.exterior.coords[:-1], dtype="float32")
        rect = cv2.minAreaRect(box_coords)
        # Scale and pad the rectangle
        rect_scaled = rect_scale_pad(rect, scale=1.0, pad=40)
        # Crop the image using the scaled rectangle
        cropped_image, rect_target, M = img_rectangle_cut(image, rect_scaled, target_size=(self.sample_size, self.sample_size), flipy=flipy)
        
        # Find keypoints that intersect with the bounding box
        keypoints = find_intersecting_keypoints(box_coords, annotations)
        # Transform keypoints based on the perspective transform
        transformed_keypoints = transform_keypoints(keypoints, M, flipy=flipy, width=self.sample_size)
        
        # Clip keypoints if a clipping callback is provided
        if self.keypoint_clipper:
            transformed_keypoints = self.keypoint_clipper.clip_keypoints_to_image(transformed_keypoints)

        # Reorder keypoints if a reordering callback is provided
        if self.keypoint_reorderer:
            transformed_keypoints = apply_keypoint_reordering(transformed_keypoints, M, flipy=flipy, width=self.sample_size)

        boxes = []
        labels = []
        final_keypoints = []
        
        # Process keypoints into bounding boxes and labels
        if len(transformed_keypoints) > 0:
            for i in range(0, len(transformed_keypoints) - 1, 2):
                pt1 = transformed_keypoints[i]
                pt2 = transformed_keypoints[i + 1]
                
                # Compute bounding box based on keypoints
                x_min, y_min = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
                x_max, y_max = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
                bbox_width = (x_max - x_min) / self.sample_size
                bbox_height = (y_max - y_min) / self.sample_size
                x_center = (x_min + x_max) / 2 / self.sample_size
                y_center = (y_min + y_max) / 2 / self.sample_size

                # Append bounding box and keypoints
                boxes.append([x_center, y_center, bbox_width, bbox_height])
                
                final_keypoints.append([pt1[0] / self.sample_size, pt1[1] / self.sample_size, 2])
                final_keypoints.append([pt2[0] / self.sample_size, pt2[1] / self.sample_size, 2])
                
                labels.append(0)  # Append label (0 is used here)

        # Handle cases with no keypoints
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            final_keypoints = torch.zeros((0, 2, 3), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # Convert lists to tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            final_keypoints = torch.tensor(final_keypoints, dtype=torch.float32).reshape(len(boxes), -1, 3)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Apply transformation if provided
        if self.transform:
            cropped_image = self.transform(image=cropped_image)['image']
        else:
            # Convert the image to a tensor
            cropped_image = torch.from_numpy(cropped_image).permute(2, 0, 1)

        # Additional sample information
        sample_info = {
            "rotation_angle": rotation_angle,
            "scale_val": scale_val,
            "flipy": flipy,
            "sample_center": sample_center,
            "img_file": img_file
        }

        # Prepare the result dictionary
        result = {
            "img": cropped_image.to(torch.float32),
            "bboxes": boxes,
            "cls": labels,
            "keypoints": final_keypoints,
            "image_id": torch.tensor([idx]),
            "ori_shape": (img_width, img_height),
            "ratio_pad": ((1, 1), (0, 0)),
            "im_file": img_file,
        }

        # Return the sample with additional information
        return result, sample_info
