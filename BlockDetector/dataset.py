import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from shapely.geometry import Point, Polygon
from shapely.affinity import rotate, scale
from shapely.geometry import box

class PhysicalCellsObjectDetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_samples_per_cell=1, sample_size=1024):
        self.images_dir = os.path.join(data_dir, 'images')
        self.labels_dir = os.path.join(data_dir, 'labels')
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
        self.label_files = [f for f in os.listdir(self.labels_dir) if f.endswith('.pkl')]
        self.transform = transform
        self.num_samples_per_cell = num_samples_per_cell
        self.sample_size = sample_size
        
        self.data_pairs = self._pair_images_and_labels()
        self.labels = self._cache_all_labels()
        
    def _cache_all_labels(self):
        """Cache all labels for YOLO visualization compatibility"""
        all_labels = []
        for _ in range(len(self)):
            idx = _
            cell_idx = idx // self.num_samples_per_cell
            
            _, label_file = self.data_pairs[cell_idx]
            label_path = os.path.join(self.labels_dir, label_file)
            
            with open(label_path, 'rb') as f:
                annotations = pickle.load(f)
            
            label_dict = {
                "bboxes": [],
                "cls": [],
                "segments": [],
                "keypoints": [],
            }
            
            for obj in annotations:
                points = np.array(obj['points'])
                keypoints = points[points[:, 2] == 2]
                
                if len(keypoints) >= 2:
                    x_coords = keypoints[:, 0]
                    y_coords = keypoints[:, 1]
                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    y_min, y_max = np.min(y_coords), np.max(y_coords)
                    
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    label_dict["bboxes"].append([x_center, y_center, width, height])
                    label_dict["cls"].append(int(obj['label']))
                    label_dict["keypoints"].append(points)
            
            label_dict["bboxes"] = np.array(label_dict["bboxes"], dtype=np.float32)
            label_dict["cls"] = np.array(label_dict["cls"], dtype=np.int64)
            label_dict["keypoints"] = np.array(label_dict["keypoints"], dtype=np.float32)
            
            all_labels.append(label_dict)
        
        return all_labels
    
    def _pair_images_and_labels(self):
        data_pairs = []
        for image_file in self.image_files:
            base_name = os.path.splitext(image_file)[0]
            label_file = f"{base_name}.pkl"
            if label_file in self.label_files:
                data_pairs.append((image_file, label_file))
        return data_pairs

    def calculate_bbox_from_keypoints(self, keypoints):
        
        visible_keypoints = keypoints[keypoints[:, 2] == 2]
        if len(visible_keypoints) < 2:
            return None
        
        x_coords = visible_keypoints[:, 0]
        y_coords = visible_keypoints[:, 1]
        
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = (x_max - x_min)
        height = (y_max - y_min)
        
        return [x_center, y_center, width, height]

    def find_objects_with_points(self, bbox, objects, tolerance=30):
      
        bbox_polygon = Polygon(bbox).buffer(tolerance)
        objects_and_points = []

        for obj in objects:
            points = np.array(obj['points'])
            visible_points = points[points[:, 2] == 2]
            
      
            valid_points = []
            has_visible_point_in_bbox = False
            
            for i, point in enumerate(points):
                if point[2] == 2:  
                    if bbox_polygon.contains(Point(point[0], point[1])):
                        has_visible_point_in_bbox = True
                        valid_points.append(i)
            
        
            if has_visible_point_in_bbox:
      
                for i, point in enumerate(points):
                    if point[2] == 2 and i not in valid_points:
                        valid_points.append(i)
                
                objects_and_points.append((obj, valid_points))
        
        return objects_and_points

    def _transform_keypoints(self, keypoints, M, flipy, width):
     
        keypoints_xy = keypoints[:, :2]
        keypoints_homogeneous = np.hstack([keypoints_xy, np.ones((keypoints_xy.shape[0], 1))])
        transformed_homogeneous = keypoints_homogeneous @ M.T
        transformed_xy = transformed_homogeneous[:, :2] / transformed_homogeneous[:, 2][:, np.newaxis]
        
        if flipy:
            transformed_xy[:, 0] = width - transformed_xy[:, 0]
        
        transformed_xy /= width
        transformed_points = np.hstack([transformed_xy, keypoints[:, 2].reshape(-1, 1)])
        return transformed_points

    def _get_bounding_polygon(self, center: Point, width: int, height: int) -> box:

        return box(
            center.x - width / 2,
            center.y - height / 2,
            center.x + width / 2,
            center.y + height / 2,
        )

    def _rect_scale_pad(self, rect, scale=1.0, pad=40):

        center, size, angle = rect
        new_size = (size[0] * scale + pad, size[1] * scale + pad)
        return (center, new_size, angle)

    def _img_rectangle_cut(self, img, rect, target_size=(1024, 1024), flipy=False):
        """برش و تغییر اندازه تصویر"""
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width, height = target_size

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                           [0, 0],
                           [width-1, 0],
                           [width-1, height-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (width, height))
        
        if flipy:
            warped = cv2.flip(warped, 1)
        
        return warped, rect, M
    def _check_and_fix_keypoint_order(self, keypoints):
        """
        بررسی و تصحیح ترتیب کی‌پوینت‌ها مطابق قوانین داده‌شده:
        - محور افقی (x):
            - کی‌پوینت 2 باید سمت راست کی‌پوینت 1 باشد.
            - کی‌پوینت 4 باید سمت راست کی‌پوینت 3 باشد.
            - کی‌پوینت 6 باید سمت راست کی‌پوینت 5 باشد.
            - کی‌پوینت 12 باید سمت راست کی‌پوینت 11 باشد.
            - کی‌پوینت 14 باید سمت راست کی‌پوینت 13 باشد.
            - کی‌پوینت 16 باید سمت راست کی‌پوینت 15 باشد.
            - کی‌پوینت 18 باید سمت راست کی‌پوینت 17 باشد.
        - محور عمودی (y):
            - کی‌پوینت 8 باید بالای کی‌پوینت 7 باشد.
            - کی‌پوینت 10 باید بالای کی‌پوینت 9 باشد.
            - کی‌پوینت 1 باید بالای کی‌پوینت 3 باشد.
            - کی‌پوینت 3 باید بالای کی‌پوینت 5 باشد.
            - کی‌پوینت 2 باید بالای کی‌پوینت 4 باشد.
            - کی‌پوینت 4 باید بالای کی‌پوینت 6 باشد.
        """
        # تبدیل به لیست برای تغییرات
        keypoints = keypoints.tolist()

        def swap_keypoints_if_needed(kp1_idx, kp2_idx, axis='x'):
            """
            جابجایی کی‌پوینت‌ها در صورت نیاز.
            - axis='x': محور افقی (بررسی سمت راست/چپ).
            - axis='y': محور عمودی (بررسی بالا/پایین).
            """
            if axis == 'x':  # محور افقی
                if keypoints[kp1_idx][0] > keypoints[kp2_idx][0]:  # اگر کی‌پوینت دوم سمت چپ کی‌پوینت اول بود
                    keypoints[kp1_idx], keypoints[kp2_idx] = keypoints[kp2_idx], keypoints[kp1_idx]
            elif axis == 'y':  # محور عمودی
                if keypoints[kp1_idx][1] > keypoints[kp2_idx][1]:  # اگر کی‌پوینت دوم پایین کی‌پوینت اول بود
                    keypoints[kp1_idx], keypoints[kp2_idx] = keypoints[kp2_idx], keypoints[kp1_idx]

          # بررسی محور افقی
        swap_keypoints_if_needed(0, 1, axis='x')  # بررسی کی‌پوینت 1 و 2
        swap_keypoints_if_needed(2, 3, axis='x')  # بررسی کی‌پوینت 3 و 4
        swap_keypoints_if_needed(4, 5, axis='x')  # بررسی کی‌پوینت 5 و 6
        swap_keypoints_if_needed(10, 11, axis='x')  # بررسی کی‌پوینت 11 و 12
        swap_keypoints_if_needed(12, 13, axis='x')  # بررسی کی‌پوینت 13 و 14
        swap_keypoints_if_needed(14, 15, axis='x')  # بررسی کی‌پوینت 15 و 16
        swap_keypoints_if_needed(16, 17, axis='x')  # بررسی کی‌پوینت 17 و 18

        # بررسی محور عمودی
        swap_keypoints_if_needed(6, 7, axis='y')  # بررسی کی‌پوینت 7 و 8
        swap_keypoints_if_needed(8, 9, axis='y')  # بررسی کی‌پوینت 9 و 10
        swap_keypoints_if_needed(18, 19, axis='y')  # 19 , 20
        swap_keypoints_if_needed(20, 21, axis='y')  # بررسی کی‌پوینت 22 و 21
 

        return np.array(keypoints, dtype=np.float32)


    def __len__(self):
        return len(self.data_pairs) * self.num_samples_per_cell

    def __getitem__(self, idx):
        cell_idx = idx // self.num_samples_per_cell
        
        img_file, label_file = self.data_pairs[cell_idx]
        
        img_path = os.path.join(self.images_dir, img_file)
        image = cv2.imread(img_path)
        assert image is not None, f"Failed to load image: {img_path}"
        
        img_height, img_width = image.shape[:2]
        cell_center = Point(img_width // 2, img_height // 2)
        
        label_path = os.path.join(self.labels_dir, label_file)
        with open(label_path, 'rb') as f:
            annotations = pickle.load(f)

        r_x = np.random.random()
        r_y = np.random.random()
        sample_center = Point(cell_center.x + r_x * 512, cell_center.y + r_y * 512)
        rotation_angle = np.random.uniform(0, 360)
        scale_val = np.random.uniform(0.3, 0.5)
        flipy = np.random.choice([True, False])
        
        bounding_box = self._get_bounding_polygon(sample_center, self.sample_size, self.sample_size)
        bounding_box = scale(bounding_box, xfact=scale_val, yfact=scale_val, origin=sample_center)
        bounding_box = rotate(bounding_box, rotation_angle, origin=sample_center)
        
        box_coords = np.array(bounding_box.exterior.coords[:-1], dtype="float32")
        rect = cv2.minAreaRect(box_coords)
        rect_scaled = self._rect_scale_pad(rect, scale=1.0, pad=40)
        cropped_image, rect_target, M = self._img_rectangle_cut(image, rect_scaled, 
                                                        target_size=(self.sample_size, self.sample_size), 
                                                        flipy=flipy)
        
        # Finding objects with valid keypoints within the bounding box
        objects_and_points = self.find_objects_with_points(box_coords, annotations)
        
        boxes = []
        final_labels = []
        all_keypoints = []

        for obj, valid_points in objects_and_points:
            points = np.array(obj['points'])
            transformed_points = self._transform_keypoints(points, M, flipy, self.sample_size)
            
            # Retain original visibility states for keypoints
            filtered_points = transformed_points.copy()
            
            # Disable keypoints outside bounds (that are not visibility 2)
            for i, point in enumerate(filtered_points):
                if (point[2] != 2) and (point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1):
                    filtered_points[i, 2] = 0
            
            # Check and fix the order of keypoints
            filtered_points = self._check_and_fix_keypoint_order(filtered_points)
            
            # Calculate bounding box based on keypoints with visibility 2
            bbox = self.calculate_bbox_from_keypoints(filtered_points)
            if bbox is not None:
                boxes.append(bbox)
                final_labels.append(int(obj['label']))
                all_keypoints.append(filtered_points)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            final_labels = torch.zeros((0,), dtype=torch.int64)
            final_keypoints = torch.zeros((0, 22, 3), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            final_labels = torch.tensor(final_labels, dtype=torch.int64)
            
            filtered_keypoints = []
            for kps in all_keypoints:
                kps_vis2 = kps[kps[:, 2] == 2]  # Keep keypoints with visibility 2
                if len(kps_vis2) < 22:
                    pad_length = 22 - len(kps_vis2)
                    padding = np.zeros((pad_length, 3))  # Pad missing keypoints
                    kps_vis2 = np.vstack([kps_vis2, padding])
                elif len(kps_vis2) > 22:
                    kps_vis2 = kps_vis2[:22]  # Trim to 22 keypoints
                filtered_keypoints.append(kps_vis2)
            final_keypoints = torch.tensor(np.array(filtered_keypoints), dtype=torch.float32)


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
            "cls": final_labels,
            "bboxes": boxes,
        }
        
        result = {
            "img": cropped_image.to(torch.float32),
            "bboxes": boxes,
            "cls": final_labels,
            "image_id": torch.tensor([idx]),
            "keypoints": final_keypoints,
            "ori_shape": (img_width, img_height),
            "ratio_pad": ((1, 1), (0, 0)),
            "im_file": img_file,
        }
        
        return result, sample_info
