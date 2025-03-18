import os
import numpy as np
import cv2
from PIL import Image, ImageFile
import json
import pickle
import glob


mega_tile_dir = "/home/training/BlockDetector/Megatile dataset" 
physical_output_dir = "/home/training/BlockDetector/physicall_Cell"  
json_dir = "/home/training/BlockDetector/Megatile dataset"  
grid_height = 1024  
grid_width = 1024  


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(image_path: str) -> np.ndarray:
    """Load an image from a file path using Pillow and convert to NumPy array."""
    print(f"Loading image: {image_path}")
    with Image.open(image_path) as img:
        image = np.array(img)
        # Convert RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def save_image(image: np.ndarray, output_path: str):
    """Save an image to a file."""
    print(f"Saving image to: {output_path}")
    cv2.imwrite(output_path, image)

def create_physical_cells(image: np.ndarray, grid_height: int, grid_width: int, output_dir: str, json_data: dict, tile_number: int):
    """Create physical cells with the specified size centered on the grid cells and save them."""
    img_height, img_width = image.shape[:2]
    S = grid_width
    physical_size = int(np.sqrt(2) * 2 * S + S)
    canvas_size = physical_size + 200  

    for y in range(0, img_height, grid_height):
        for x in range(0, img_width, grid_width):
            center_x = x + grid_width // 2
            center_y = y + grid_height // 2

            top_left_x = center_x - canvas_size // 2
            top_left_y = center_y - canvas_size // 2
            bottom_right_x = center_x + canvas_size // 2
            bottom_right_y = center_y + canvas_size // 2

            # ایجاد بوم سفید با اندازه canvas_size
            canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
            start_x = max(0, top_left_x)
            start_y = max(0, top_left_y)
            end_x = min(img_width, bottom_right_x)
            end_y = min(img_height, bottom_right_y)

            offset_x = max(0, -top_left_x)
            offset_y = max(0, -top_left_y)

         
            canvas[offset_y:offset_y + (end_y - start_y), offset_x:offset_x + (end_x - start_x)] = image[start_y:end_y, start_x:end_x]

            pickle_data = []

            for shape in json_data['shapes']:
                points = shape['points']
                class_id = shape['label']
                bbox = shape['bbox']

              
                points_in_canvas = [
                    [point[0] - top_left_x, point[1] - top_left_y, point[2]]
                    for point in points
                    if (top_left_x <= point[0] < bottom_right_x) and (top_left_y <= point[1] < bottom_right_y)
                ]

           
                if points_in_canvas:
                    adjusted_points = [[point[0] - top_left_x, point[1] - top_left_y, point[2]] for point in points]
                    adjusted_bbox = [
                        bbox[0] - top_left_x,
                        bbox[1] - top_left_y,
                        bbox[2] - top_left_x,
                        bbox[3] - top_left_y
                    ]

                 
                    obj_data = {
                        'label': str(class_id),
                        'points': adjusted_points,
                        'shape_type': shape['shape_type'],
                        'bbox': adjusted_bbox
                    }
                    pickle_data.append(obj_data)

   
            output_image_path = f"{output_dir}/physical_cell_{tile_number}_{y}_{x}.png"
            save_image(canvas, output_image_path)

            pickle_file_path = f"{output_dir}/physical_cell_{tile_number}_{y}_{x}.pkl"
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(pickle_data, pickle_file)
            print(f"Saved {output_image_path} and {pickle_file_path}")


def main():
    os.makedirs(physical_output_dir, exist_ok=True)

    mega_tile_paths = sorted(glob.glob(f"{mega_tile_dir}/*.png"))
    json_files = sorted([f for f in glob.glob(f"{json_dir}/*.json") if not f.endswith('_Complete.json')])

    for idx, mega_tile_path in enumerate(mega_tile_paths):
        print(f"Processing mega-tile image {idx + 1}/{len(mega_tile_paths)}: {mega_tile_path}")
        image = load_image(mega_tile_path)

        json_path = json_files[idx]
        print(f"Loading JSON file: {json_path}")
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)

        create_physical_cells(image, grid_height, grid_width, physical_output_dir, json_data, idx + 1)


main()
