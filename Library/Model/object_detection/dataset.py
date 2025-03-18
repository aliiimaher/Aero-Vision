import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import random
from shapely.geometry import Point
from shapely.affinity import rotate, scale
from shapely.geometry import box, Polygon, LineString
import sys

# Add the parent directory of `callbacks` to `sys.path`
sys.path.insert(0, os.path.abspath('/home/training/Aerialytic_AI/New-Graph-Extraction/library'))

# Now import the KeypointClipper class
from callbacks.clip_keypoint import KeypointClipper

def get_bounding_polygon(center: Point, width: int, height: int) -> box:
    """
    Create a bounding box polygon centered at a given point.
    """
    return box(
        center.x - width / 2,
        center.y - height / 2,
        center.x + width / 2,
        center.y + height / 2,
    )

def rect_scale_pad(rect, scale=1.0, pad=0):
    """
    Scale and pad a rectangle.
    """
    center, size, angle = rect
    new_size = (size[0] * scale + pad, size[1] * scale + pad)
    return (center, new_size, angle)

def img_rectangle_cut(img, rect, target_size=(1024, 1024), flipy=False):
    """
    Cut a region from an image defined by a rectangle and apply flipping if necessary.
    """
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width, height = target_size

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)  # Get perspective transform matrix
    warped = cv2.warpPerspective(img, M, (width, height))  # Apply the transformation
    
    if flipy:
        warped = cv2.flip(warped, 1)  # Flip the image vertically if specified
    
    return warped, rect, M

def find_intersecting_keypoints(bbox, lines, tolerance=0):
    bbox_polygon = Polygon(bbox).buffer(tolerance)  # Expand by a small tolerance
    keypoints = []

    for line in lines:
        if 'points' not in line:
            continue
        line_string = LineString(line['points'])
        if bbox_polygon.intersects(line_string):
            for point in line['points']:
                keypoints.append(point)

    return np.array(keypoints)

def transform_keypoints(keypoints, M, flipy, width):
    """
    Transform keypoints using the provided transformation matrix and apply flipping if necessary.
    """
    if keypoints.size == 0:
        return keypoints
    
    keypoints_homogeneous = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
    transformed_keypoints_homogeneous = keypoints_homogeneous @ M.T  # Apply transformation
    transformed_keypoints = transformed_keypoints_homogeneous[:, :2] / transformed_keypoints_homogeneous[:, 2][:, np.newaxis]
    
    if flipy:
        transformed_keypoints[:, 0] = width - transformed_keypoints[:, 0]  # Apply flipping
    
    return transformed_keypoints

def classify_line_slope(point1, point2):
    """
    Classify the slope of a line segment defined by two points.
    """
    x1, y1 = point1
    x2, y2 = point2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx <= 1 or x1 == x2:
        return 1  # Class for vertical lines
    
    if dy <= 1 or y1 == y2:
        return 0  # Class for horizontal lines
    
    slope = (y2 - y1) / (x2 - x1)
    
    if slope > 0:
        return 1  # Class for positive slope
    else:
        return 0  # Class for negative slope

class PhysicalCellsObjectDetectionDataset(Dataset):
    annotations_dict = dict()

    def __init__(self, data_dir, transform=None, num_samples_per_cell=1, sample_size=1024, use_keypoint_clipping=False, use_aligner=False):
        """
        Initialize the dataset for physical cells object detection.
        """
        self.images_dir = os.path.join(data_dir, 'images')  # Directory for images
        self.labels_dir = os.path.join(data_dir, 'labels')  # Directory for labels
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]  # List of image files
        self.label_files = [f for f in os.listdir(self.labels_dir) if f.endswith('.pkl')]  # List of label files
        self.transform = transform  # Transform function (if any)
        self.num_samples_per_cell = num_samples_per_cell  # Number of samples per cell
        self.sample_size = sample_size  # Size of the samples
        self.use_keypoint_clipping = use_keypoint_clipping  # Flag for using keypoint clipping
        self.use_aligner = use_aligner
        # Create a KeypointClipper instance
        self.keypoint_clipper = KeypointClipper(img_width=sample_size, img_height=sample_size)
        # Pairing images and labels
        self.data_pairs = self._pair_images_and_labels()  # Pair images with their corresponding labels
        # Initializing labels
        self.labels = self._initialize_labels()  # Load and initialize labels

    def _pair_images_and_labels(self):
        """
        Pair images with their corresponding label files.
        """
        data_pairs = []
        for image_file in self.image_files:
            base_name = os.path.splitext(image_file)[0]  # Get the base name of the image file
            label_file = f"{base_name}.pkl"  # Construct the corresponding label file name
            if label_file in self.label_files:
                label_path = os.path.join(self.labels_dir, label_file)
                try:
                    if not label_file in PhysicalCellsObjectDetectionDataset.annotations_dict:
                        with open(label_path, 'rb') as f:
                            annotations = pickle.load(f)
                            PhysicalCellsObjectDetectionDataset.annotations_dict[label_file] = annotations
                    data_pairs.append((image_file, label_file))  # Add the pair to the list
                    print(f"success - {label_file}")
                except:
                    print(f"error - {label_file}")
        return data_pairs

    def _initialize_labels(self):
        labels = []
        for img_file, label_file in self.data_pairs:
            annotations = PhysicalCellsObjectDetectionDataset.annotations_dict[label_file]
            label_path = os.path.join(self.labels_dir, label_file)
            # with open(label_path, 'rb') as f:
            #     annotations = pickle.load(f)
            
            image_labels = []
            for annotation in annotations:
                if 'points' not in annotation:
                    continue
                
                points = annotation['points']
                if len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x_min, y_min = min(x1, x2), min(y1, y2)
                    x_max, y_max = max(x1, x2), max(y1, y2)
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2

                    image_labels.append([x_center, y_center, bbox_width, bbox_height])
            
            if image_labels:
                labels.append({"bboxes": np.array(image_labels), "cls": np.zeros((len(image_labels),), dtype=np.int64)})
        
        return labels

    def _sort_keypoints(self, keypoints):
        """
        Sort keypoints such that for each line, the keypoint on the right or the top is first.
        """
        sorted_keypoints = []
        if keypoints.shape[0] % 2 != 0:
            raise ValueError("Keypoints should be in pairs of two")  # Ensure keypoints are in pairs

        for i in range(0, len(keypoints), 2):
            pt1, pt2 = keypoints[i], keypoints[i+1]
            if pt1[0] > pt2[0]:  # pt1 is on the right of pt2
                sorted_keypoints.extend([pt1, pt2])  # Add points in order
            elif pt1[0] < pt2[0]:  # pt1 is on the left of pt2
                sorted_keypoints.extend([pt2, pt1])  # Add points in order
            else:  # Vertical line
                if pt1[1] > pt2[1]:  # pt1 is higher than pt2
                    sorted_keypoints.extend([pt1, pt2])  # Add points in order
                else:
                    sorted_keypoints.extend([pt2, pt1])  # Add points in order
        
        return np.array(sorted_keypoints)

    def _load_aligner_data(self, label_path):
        with open(label_path, 'rb') as f:
            data = pickle.load(f)
        aligner_data = None
        for item in data:
            if isinstance(item, dict) and 'aligner' in item:
                aligner_data = item['aligner']
                break
        return aligner_data

    def determine_grid_and_orientation(self, point, cell_width, cell_height, aligner_data):
        if aligner_data is None or 'grid' not in aligner_data or 'orientation' not in aligner_data:
            return None, random.uniform(0, 360)

        grid = aligner_data['grid']
        orientations = aligner_data['orientation']

        grid_width = cell_width // grid[0]
        grid_height = cell_height // grid[1]
        
        grid_x = int(point.x // grid_width)
        grid_y = int(point.y // grid_height)
        
        grid_index = grid_y * grid[0] + grid_x
        
        if grid_index < len(orientations) and orientations[grid_index] is not None:
            base_orientation = orientations[grid_index]
        else:
            base_orientation = random.uniform(0, 360)
        
        rotation_choice = random.choice([0, 90, 180, 270])
        final_orientation = (base_orientation + rotation_choice) % 360
        
        return grid_index, final_orientation

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data_pairs) * self.num_samples_per_cell

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        """
        cell_idx = idx // self.num_samples_per_cell  # Determine the cell index
        sample_idx = idx % self.num_samples_per_cell  # Determine the sample index
        
        img_file, label_file = self.data_pairs[cell_idx]  # Get the corresponding image and label file
        
        img_path = os.path.join(self.images_dir, img_file)  # Get the full image path
        image = cv2.imread(img_path)  # Read the image
        img_height, img_width = image.shape[:2]  # Get the dimensions of the image
        cell_center = Point(img_width // 2, img_height // 2)  # Define the cell center
        
        label_path = os.path.join(self.labels_dir, label_file)  # Get the full label path
        with open(label_path, 'rb') as f:
            annotations = pickle.load(f)  # Load annotations from the label file
        aligner_data = self._load_aligner_data(label_path) if self.use_aligner else None
        r_x = random.random()  # Random x offset
        r_y = random.random()  # Random y offset
        sample_center = Point(cell_center.x + r_x * 512, cell_center.y + r_y * 512)  # Define the sample center
        if self.use_aligner and aligner_data:
            grid_index, rotation_angle = self.determine_grid_and_orientation(sample_center, img_width, img_height, aligner_data)
        else:
            grid_index, rotation_angle = None, random.uniform(0, 360) # Random rotation angle
        scale_val = random.uniform(0.5, 2)  # Random scale value
        flipy = random.choice([True, False])  # Randomly choose whether to flip the image vertically
        
        bounding_box = get_bounding_polygon(sample_center, self.sample_size, self.sample_size)  # Get the bounding polygon
        bounding_box = scale(bounding_box, xfact=scale_val, yfact=scale_val, origin=sample_center)  # Scale the bounding box
        bounding_box = rotate(bounding_box, rotation_angle, origin=sample_center)  # Rotate the bounding box
        
        box_coords = np.array(bounding_box.exterior.coords[:-1], dtype="float32")  # Get the coordinates of the bounding box
        rect = cv2.minAreaRect(box_coords)  # Get the minimum area rectangle
        rect_scaled = rect_scale_pad(rect, scale=1.0, pad=40)  # Scale and pad the rectangle
        cropped_image, rect_target, M = img_rectangle_cut(image, rect_scaled, target_size=(self.sample_size, self.sample_size), flipy=flipy)  # Cut the image
        tolerance = 30
        keypoints = find_intersecting_keypoints(box_coords, annotations, tolerance=tolerance)  # Find intersecting keypoints
        sorted_keypoints = self._sort_keypoints(keypoints)  # Sort keypoints
        transformed_keypoints = transform_keypoints(sorted_keypoints, M, flipy=flipy, width=self.sample_size)  # Transform keypoints
        
        # Apply keypoint clipping if enabled
        if self.use_keypoint_clipping:
            clipped_keypoints = self.keypoint_clipper.clip_keypoints_to_image(transformed_keypoints)  # Clip keypoints to image
        else:
            clipped_keypoints = transformed_keypoints  # Use transformed keypoints directly
        
        boxes = []  # Initialize boxes list
        labels = []  # Initialize labels list
        final_keypoints = []  # Initialize final keypoints list

        if len(clipped_keypoints) > 0:
            for i in range(0, len(clipped_keypoints) - 1, 2):
                pt1 = clipped_keypoints[i]  # Get the first keypoint
                pt2 = clipped_keypoints[i + 1]  # Get the second keypoint
                
                # Calculate bounding box
                x_min, y_min = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])  # Calculate min coordinates
                x_max, y_max = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])  # Calculate max coordinates
                bbox_width = (x_max - x_min) / self.sample_size  # Calculate bounding box width
                bbox_height = (y_max - y_min) / self.sample_size  # Calculate bounding box height
                x_center = (x_min + x_max) / 2 / self.sample_size  # Calculate x center
                y_center = (y_min + y_max) / 2 / self.sample_size  # Calculate y center

                # Assign class based on line slope
                line_class = classify_line_slope(pt1, pt2)  # Classify the line based on slope
                boxes.append([x_center, y_center, bbox_width, bbox_height])  # Append bounding box
                # Store keypoints
                final_keypoints.append([pt1[0] / self.sample_size, pt1[1] / self.sample_size, 2])  # Append first keypoint
                final_keypoints.append([pt2[0] / self.sample_size, pt2[1] / self.sample_size, 2])  # Append second keypoint
                
                labels.append(line_class)  # Append line class

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # Create empty tensor for boxes
            labels = torch.zeros((0,), dtype=torch.int64)  # Create empty tensor for labels
            final_keypoints = torch.zeros((0, 2, 3), dtype=torch.float32)  # Create empty tensor for keypoints
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)  # Convert boxes to tensor
            labels = torch.tensor(labels, dtype=torch.int64)  # Convert labels to tensor
            final_keypoints = torch.tensor(final_keypoints, dtype=torch.float32).reshape(len(boxes), -1, 3)  # Convert keypoints to tensor and reshape

        if self.transform:
            cropped_image = self.transform(image=cropped_image)['image']  # Apply transformation if specified
        else:
            cropped_image = torch.from_numpy(cropped_image).permute(2, 0, 1)  # Convert cropped image to tensor

        # Update sample_info with keypoint details
        sample_info = {
            "rotation_angle": rotation_angle,
            "scale_val": scale_val,
            "flipy": flipy,
            "sample_center": sample_center,
            "img_file": img_file,
            "keypoints": final_keypoints,
            "grid_index": grid_index
        }

        result = {
            "img": cropped_image.to(torch.float32),  # Return cropped image as float tensor
            "bboxes": boxes,  # Return bounding boxes
            "cls": labels,  # Return class labels
            "image_id": torch.tensor([idx]),  # Return image ID
            "ori_shape": (img_width, img_height),  # Return original image shape
            "ratio_pad": ((1, 1), (0, 0)),  # Return padding ratio
            "im_file": img_file,  # Return image file name
        }

        return result, sample_info  # Return the result and sample information