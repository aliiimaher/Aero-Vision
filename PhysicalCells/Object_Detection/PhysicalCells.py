import os
import json
import numpy as np
import cv2
from PIL import Image, ImageFile
import shapely.geometry as geom
import pickle
import argparse
import shutil
from pathlib import Path
import re
import concurrent.futures
from PIL import Image

# Increase maximum image size
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        image = np.array(img)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def save_image(image: np.ndarray, output_path: str):
    #cv2.imwrite(output_path, image)
    image = image[:, :, ::-1]
    pil_image = Image.fromarray(image)
    # image_8bit = pil_image.quantize(colors=256, method=Image.MEDIANCUT, kmeans=2, palette=None, dither=Image.FLOYDSTEINBERG)
    # image_8bit.save(output_path)
    pil_image.convert("L").save(output_path)


def load_labels(label_path: str) -> dict:
    with open(label_path, 'r') as f:
        labels = json.load(f)
    return labels

def save_pickle(data, output_path: str):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

def create_physical_cells_with_labels(image_idx:int, image: np.ndarray, grid_height: int, grid_width: int, labels: dict, output_dir: str):
    img_height, img_width = image.shape[:2]
    S = grid_width
    physical_size = int(np.sqrt(2) * 2 * S + S)

    physical_cells = []

    for y in range(0, img_height, grid_height):
        for x in range(0, img_width, grid_width):
            center_x = x + grid_width // 2
            center_y = y + grid_height // 2

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

            physical_cell[offset_y:offset_y+(end_y-start_y), offset_x:offset_x+(end_x-start_x)] = image[start_y:end_y, start_x:end_x]

            # Save physical cell image
            img_output_path = f"{output_dir}/physical_cell_{image_idx}_{y}_{x}.png"
            if not os.path.exists(img_output_path):
                save_image(physical_cell, img_output_path)
                print(f"Saved physical cell image: {img_output_path}")
            else:
                print(f"Physical cell image already exists: {img_output_path}")

            # Check and save lines in this physical cell
            physical_cell_polygon = geom.box(start_x, start_y, end_x, end_y)
            lines_in_cell = []

            for shape in labels['shapes']:
                if shape['shape_type'] == 'line':
                    line_points = shape['points']
                    line = geom.LineString(line_points)
                    if physical_cell_polygon.intersects(line):
                        adjusted_points = []
                        for point in line_points:
                            adjusted_x = point[0] - top_left_x
                            adjusted_y = point[1] - top_left_y
                            if 0 <= adjusted_x < physical_size and 0 <= adjusted_y < physical_size:
                                adjusted_points.append([adjusted_x, adjusted_y])
                        if len(adjusted_points) == len(line_points):
                            lines_in_cell.append({
                                'label': shape['label'],
                                'points': adjusted_points,
                                'shape_type': shape['shape_type']
                            })
            
            pickle_output_path = f"{output_dir}/physical_cell_{image_idx}_{y}_{x}.pkl"
            if not os.path.exists(pickle_output_path):
                save_pickle(lines_in_cell, pickle_output_path)
                print(f"Saved pickle file: {pickle_output_path}")
            else:
                print(f"Pickle file already exists: {pickle_output_path}")

            physical_cells.append((img_output_path, pickle_output_path))
            print(f"Added physical cell: Image - {img_output_path}, Labels - {pickle_output_path}")

    return physical_cells

def sort_key(filename):
    match = re.match(r'User(\d+)_(\d+)\.png', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (float('inf'), float('inf'))  # For any non-matching filenames

def create_train_val_dirs(output_dir: str, physical_cells: list):
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

    # Shuffle and split data
    np.random.shuffle(physical_cells)
    split_index = int(len(physical_cells) * 0.7)

    train_cells = physical_cells[:split_index]
    val_cells = physical_cells[split_index:]

    for img_path, label_path in train_cells:
        shutil.move(img_path, os.path.join(train_dir, 'images', os.path.basename(img_path)))
        shutil.move(label_path, os.path.join(train_dir, 'labels', os.path.basename(label_path)))
        print(f"Moved to train: {os.path.basename(img_path)}, {os.path.basename(label_path)}")

    for img_path, label_path in val_cells:
        shutil.move(img_path, os.path.join(val_dir, 'images', os.path.basename(img_path)))
        shutil.move(label_path, os.path.join(val_dir, 'labels', os.path.basename(label_path)))
        print(f"Moved to val: {os.path.basename(img_path)}, {os.path.basename(label_path)}")

def main():
    parser = argparse.ArgumentParser(description='Process images and labels.')
    parser.add_argument('--image_path', required=False, help='Path to the input image directory', default="/home/training/Aerialytic_AI/RGB_to_DSM/DepthModel/Depth-Anything-V2/result")
    parser.add_argument('--label_path', required=False, help='Path to the input label directory', default="/home/training/Aerialytic_AI/AY-dep/datasets/processed/megatile/keypoints")
    parser.add_argument('--output_dir', required=False, help='Output directory for physical cells', default="/home/training/Aerialytic_AI/Graph-Extraction/dsm_dataset")

    args = parser.parse_args()

    # Read all image files
    image_files = [f for f in os.listdir(args.image_path) if f.lower().endswith('.png')]

    # Sort image files based on the new sorting criteria
    image_files.sort(key=sort_key)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    physical_cells = []
    physical_cells_results = []

    def proccess_image(image_idx, image_fn, label_fn, height, width, output_dir):
        # Load image and labels
        image = load_image(image_fn)
        labels = load_labels(label_fn)
        create_physical_cells_with_labels(image_idx, image, height, width, labels, output_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        for image_idx, image_file in enumerate(image_files, start=1):
            if image_idx < 18: continue

            filename = Path(image_file).stem
            image_fn = os.path.join(args.image_path, image_file)
            label_fn = os.path.join(args.label_path, f'{filename}.json')
            
            print(f"Processing {image_fn}")

            # Process image
            physical_cells_results += [executor.submit(lambda: proccess_image(image_idx, image_fn, label_fn, 1024, 1024, args.output_dir))]

    for future in concurrent.futures.as_completed(physical_cells_results):
        physical_cells.append(future.result())

    # After processing all images, create train and val directories
    create_train_val_dirs(args.output_dir, physical_cells)

if __name__ == "__main__":
    main()
