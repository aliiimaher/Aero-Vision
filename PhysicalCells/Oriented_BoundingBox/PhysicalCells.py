import os
import numpy as np
import cv2
from PIL import Image, ImageFile
import shapely.geometry as geom
import pickle
import argparse


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PhysicalCellCreator:
    def __init__(self, image_path, label_path, output_dir, grid_height, grid_width):
        self.image_path = image_path
        self.label_path = label_path
        self.output_dir = output_dir
        self.grid_height = grid_height
        self.grid_width = grid_width

        # Load the image and labels
        self.image = self.load_image(self.image_path)
        self.labels = self.load_labels(self.label_path)

        # Create output directories if they do not exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_image(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as img:
            image = np.array(img)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def save_image(self, image: np.ndarray, output_path: str):
        cv2.imwrite(output_path, image)

    def load_labels(self, label_path: str) -> list:
        with open(label_path, 'r') as f:
            labels = f.readlines()
        return labels

    def save_pickle(self, data, output_path: str):
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

    def create_physical_cells_with_labels(self):
        img_height, img_width = self.image.shape[:2]
        S = self.grid_width
        physical_size = int(np.sqrt(2) * 2 * S + S)

        for y in range(0, img_height, self.grid_height):
            for x in range(0, img_width, self.grid_width):
                center_x = x + self.grid_width // 2
                center_y = y + self.grid_height // 2

                top_left_x = center_x - physical_size // 2
                top_left_y = center_y - physical_size // 2
                bottom_right_x = center_x + physical_size // 2
                bottom_right_y = center_y + physical_size // 2

                physical_cell = np.zeros((physical_size, physical_size, 3), dtype=np.uint8)
                start_x = max(0, top_left_x)
                start_y = max(0, top_left_y)
                end_x = min(img_width, bottom_right_x)
                end_y = min(img_height, bottom_right_y)

                offset_x = max(0, -top_left_x)
                offset_y = max(0, -top_left_y)

                physical_cell[offset_y:offset_y+(end_y-start_y), offset_x:offset_x+(end_x-start_x)] = self.image[start_y:end_y, start_x:end_x]

                # Save physical cell image
                img_output_path = f"{self.output_dir}/physical_cell_{y}_{x}.png"
                self.save_image(physical_cell, img_output_path)
                print(f"Saved {img_output_path}")

                # Check and save bounding boxes in this physical cell
                physical_cell_polygon = geom.box(start_x, start_y, end_x, end_y)
                bboxes_in_cell = []

                for line in self.labels:
                    label = line.strip().split()
                    class_id = int(label[0])
                    x_center = float(label[1]) * img_width
                    y_center = float(label[2]) * img_height
                    width = float(label[3]) * img_width
                    height = float(label[4]) * img_height

                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2

                    bbox = geom.box(x_min, y_min, x_max, y_max)
                    if physical_cell_polygon.intersects(bbox):
                        # Adjust bounding box coordinates to the new coordinates
                        adjusted_x_min = x_min - top_left_x
                        adjusted_y_min = y_min - top_left_y
                        adjusted_x_max = x_max - top_left_x
                        adjusted_y_max = y_max - top_left_y
                        if 0 <= adjusted_x_min < physical_size and 0 <= adjusted_y_min < physical_size:
                            bboxes_in_cell.append({
                                'class_id': class_id,
                                'bbox': [adjusted_x_min, adjusted_y_min, adjusted_x_max, adjusted_y_max]
                            })

                pickle_output_path = f"{self.output_dir}/physical_cell_{y}_{x}.pkl"
                self.save_pickle(bboxes_in_cell, pickle_output_path)
                print(f"Saved {pickle_output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Create physical cells with labels from an image and its associated text labels.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--label_path", type=str, required=True, help="Path to the text label file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output physical cells and labels.")
    parser.add_argument("--grid_height", type=int, default=1024, help="Height of the grid cells.")
    parser.add_argument("--grid_width", type=int, default=1024, help="Width of the grid cells.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Create an instance of PhysicalCellCreator with the provided arguments
    creator = PhysicalCellCreator(
        image_path=args.image_path,
        label_path=args.label_path,
        output_dir=args.output_dir,
        grid_height=args.grid_height,
        grid_width=args.grid_width
    )

    # Create physical cells with labels
    creator.create_physical_cells_with_labels()

if __name__ == "__main__":
    main()
