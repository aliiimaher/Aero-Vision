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
sys.path.insert(0, os.path.abspath('/home/training/Aerialytic_AI/New-Graph-Extraction/line_detector_4_cls'))
# Now import the KeypointClipper class
from callbacks.clip_keypoint import KeypointClipper
sys.path.insert(0, os.path.abspath('/home/training/Aerialytic_AI/New-Graph-Extraction'))
from library.Model.object_detection.dataset import PhysicalCellsObjectDetectionDataset
from library.Model.object_detection.dataset import  transform_keypoints ,find_intersecting_keypoints ,img_rectangle_cut , rect_scale_pad ,get_bounding_polygon

def classify_line_slope(point1, point2):
    """
    Classify the slope of a line segment defined by two points.
    """
    x1, y1 = point1
    x2, y2 = point2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    slope = (y2 - y1) / (x2 - x1) if dx != 0 else 0

    vertical = dx <= 2 or x1 == x2
    horizontal = dy <= 2 or y1 == y2
    
    pos_slope = slope > 0 or (dx <= 1 or x1 == x2)

    results = []
    if vertical: results.append(3) # Class for vertical lines
    elif horizontal: results.append(2)  # Class for horizontal lines
    if pos_slope: results.append(1) # Class for positive slope
    else: results.append(0) # Class for negative slope

    return results

class Line_2_objectdetectiondataset(PhysicalCellsObjectDetectionDataset):
    def __init__(self, data_dir, transform=None, num_samples_per_cell=1, sample_size=1024, use_keypoint_clipping=False, use_aligner=False):
        super().__init__(data_dir, transform, num_samples_per_cell, sample_size, use_keypoint_clipping)

        self.images_dir = os.path.join(data_dir, 'images')
        self.labels_dir = os.path.join(data_dir, 'labels')
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
        self.label_files = [f for f in os.listdir(self.labels_dir) if f.endswith('.pkl')]
        self.transform = transform
        self.num_samples_per_cell = num_samples_per_cell
        self.sample_size = sample_size
        self.use_keypoint_clipping = use_keypoint_clipping
        self.use_aligner = use_aligner

        self.keypoint_clipper = KeypointClipper(img_width=sample_size, img_height=sample_size)

        self.data_pairs = self._pair_images_and_labels()
        self.labels = self._initialize_labels()

    def _pair_images_and_labels(self):
        data_pairs = []
        for image_file in self.image_files:
            base_name = os.path.splitext(image_file)[0]
            label_file = f"{base_name}.pkl"
            if label_file in self.label_files:
                data_pairs.append((image_file, label_file))
        return data_pairs

    def _initialize_labels(self):
        labels = []
        for img_file, label_file in self.data_pairs:
            label_path = os.path.join(self.labels_dir, label_file)
            with open(label_path, 'rb') as f:
                annotations = pickle.load(f)
            
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
        sorted_keypoints = []
        if keypoints.shape[0] % 2 != 0:
            raise ValueError("Keypoints should be in pairs of two")

        for i in range(0, len(keypoints), 2):
            pt1, pt2 = keypoints[i], keypoints[i+1]
            if pt1[0] > pt2[0]:
                sorted_keypoints.extend([pt1, pt2])
            elif pt1[0] < pt2[0]:
                sorted_keypoints.extend([pt2, pt1])
            else:
                if pt1[1] > pt2[1]:
                    sorted_keypoints.extend([pt1, pt2])
                else:
                    sorted_keypoints.extend([pt2, pt1])
        
        return np.array(sorted_keypoints)

    def _load_aligner_data(self, data):
        # with open(label_path, 'rb') as f:
        #     data = pickle.load(f)
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
        return len(self.data_pairs) * self.num_samples_per_cell

    def __getitem__(self, idx):
        cell_idx = idx // self.num_samples_per_cell
        sample_idx = idx % self.num_samples_per_cell
        
        img_file, label_file = self.data_pairs[cell_idx]
        
        img_path = os.path.join(self.images_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
                return self.__getitem__((idx + 1) % len(self.data_pairs))
        img_height, img_width = image.shape[:2]
        cell_center = Point(img_width // 2, img_height // 2)
        
        label_path = os.path.join(self.labels_dir, label_file)
        with open(label_path, 'rb') as f:
            annotations = pickle.load(f)

        r_x = random.random()
        r_y = random.random()
        sample_center = Point(cell_center.x + r_x * 512, cell_center.y + r_y * 512)
        
        grid_index, rotation_angle = None, random.uniform(0, 360)
        if self.use_aligner:
            aligner_data = self._load_aligner_data(annotations)
            if aligner_data:
                grid_index, rotation_angle = self.determine_grid_and_orientation(sample_center, img_width, img_height, aligner_data)

        scale_val = random.uniform(0.5, 2)
        flipy = random.choice([True, False])
        
        bounding_box = get_bounding_polygon(sample_center, self.sample_size, self.sample_size)
        bounding_box = scale(bounding_box, xfact=scale_val, yfact=scale_val, origin=sample_center)
        bounding_box = rotate(bounding_box, rotation_angle, origin=sample_center)
        
        box_coords = np.array(bounding_box.exterior.coords[:-1], dtype="float32")
        rect = cv2.minAreaRect(box_coords)
        rect_scaled = rect_scale_pad(rect, scale=1.0, pad=40)
        cropped_image, rect_target, M = img_rectangle_cut(image, rect_scaled, target_size=(self.sample_size, self.sample_size), flipy=flipy)
        tolerance=30
        keypoints = find_intersecting_keypoints(box_coords, annotations,tolerance)
        sorted_keypoints = self._sort_keypoints(keypoints)
        transformed_keypoints = transform_keypoints(sorted_keypoints, M, flipy=flipy, width=self.sample_size)
        
        if self.use_keypoint_clipping:
            clipped_keypoints = self.keypoint_clipper.clip_keypoints_to_image(transformed_keypoints)
        else:
            clipped_keypoints = transformed_keypoints
        
        boxes = []
        labels = []
        final_keypoints = []

        if len(clipped_keypoints) > 0:
            for i in range(0, len(clipped_keypoints) - 1, 2):
                pt1 = clipped_keypoints[i]
                pt2 = clipped_keypoints[i + 1]
                
                x_min, y_min = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
                x_max, y_max = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
                bbox_width = max(0.002, abs(x_max - x_min) / self.sample_size)
                bbox_height = max(0.002, abs(y_max - y_min) / self.sample_size)
                x_center = (x_min + x_max) / 2 / self.sample_size
                y_center = (y_min + y_max) / 2 / self.sample_size

                line_classes = classify_line_slope(pt1, pt2)
                for line_class in line_classes:
                    if line_class == 2: boxes.append([x_center, y_center, bbox_width, max(0.01, bbox_height)])
                    elif line_class == 3: boxes.append([x_center, y_center, max(0.01, bbox_width), bbox_height])
                    else: boxes.append([x_center, y_center, bbox_width, bbox_height])
                    final_keypoints.append([pt1[0] / self.sample_size, pt1[1] / self.sample_size, 2])
                    final_keypoints.append([pt2[0] / self.sample_size, pt2[1] / self.sample_size, 2])
                    labels.append(line_class)
                    break

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            final_keypoints = torch.zeros((0, 2, 3), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            final_keypoints = torch.tensor(final_keypoints, dtype=torch.float32).reshape(len(boxes), -1, 3)

        if self.transform:
            cropped_image = self.transform(image=cropped_image)['image']
        else:
            cropped_image = torch.from_numpy(cropped_image).permute(2, 0, 1)

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
            "img": cropped_image.to(torch.float32),
            "bboxes": boxes,
            "cls": labels,
            "image_id": torch.tensor([idx]),
            "ori_shape": (img_width, img_height),
            "ratio_pad": ((1, 1), (0, 0)),
            "im_file": img_file,
        }

        return result, sample_info