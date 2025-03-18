import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json

# Function to read the annotation file and extract information
def read_annotation_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        annotations = []
        for line in lines:
            data = list(map(float, line.strip().split()))
            annotations.append(data)
        
        print(f"Read annotations from {file_path}")
        return annotations
    except Exception as e:
        print(f"Error reading annotation file {file_path}: {e}")
        return []

def create_mega_tiles(image_folder, annotation_folder, output_folder, num_tiles, tile_rows, tile_cols):
    images = []
    annotations_complete = []
    annotations_empty = []

    print("Starting to read images and annotations...")

    # Read all image and annotation files
    for filename in sorted(os.listdir(image_folder)):
        try:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_path = os.path.join(image_folder, filename)
                complete_annotation_path = os.path.join(annotation_folder, filename.replace('.png', '_Complete.txt').replace('.jpg', '_Complete.txt'))
                empty_annotation_path = os.path.join(annotation_folder, filename.replace('.png', '.txt').replace('.jpg', '.txt'))

                # Check if image and both annotations exist
                if not os.path.exists(image_path):
                    print(f"Skipping {filename}: Image not found.")
                    continue
                if not os.path.exists(complete_annotation_path) or not os.path.exists(empty_annotation_path):
                    print(f"Skipping {filename}: One or both annotation files are missing.")
                    continue

                # Read image and annotations
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    complete_annotation_data = read_annotation_file(complete_annotation_path)
                    empty_annotation_data = read_annotation_file(empty_annotation_path)
                    
                    annotations_complete.append(complete_annotation_data)
                    annotations_empty.append(empty_annotation_data)
                    print(f"Loaded image and annotations for {filename}")
                else:
                    print(f"Error: Could not load image from path {image_path}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # Calculate dimensions for each mega-tile
    tile_image_height, tile_image_width, _ = images[0].shape
    mega_tile_height = tile_image_height * tile_rows
    mega_tile_width = tile_image_width * tile_cols
    num_images_per_tile = tile_rows * tile_cols

    # Create specified number of mega-tiles
    for tile_index in range(num_tiles):
        try:
            # Create a blank mega-tile
            mega_tile = np.zeros((mega_tile_height, mega_tile_width, 3), dtype=np.uint8)
            json_data_complete = {
                "flags": {},
                "imageData": "",
                "imageHeight": mega_tile_height,
                "imagePath": "",
                "imageWidth": mega_tile_width,
                "shapes": []
            }
            json_data_empty = {
                "flags": {},
                "imageData": "",
                "imageHeight": mega_tile_height,
                "imagePath": "",
                "imageWidth": mega_tile_width,
                "shapes": []
            }

            print(f"\nCreating mega-tile {tile_index + 1}...")

            # Place images into the mega-tile and process annotations
            for idx in range(num_images_per_tile):
                image_idx = tile_index * num_images_per_tile + idx
                if image_idx >= len(images):
                    print(f"No more images to process for mega-tile {tile_index + 1}.")
                    break
                
                row = idx // tile_cols
                col = idx % tile_cols
                y_offset = row * tile_image_height
                x_offset = col * tile_image_width

                image = images[image_idx]
                mega_tile[y_offset:y_offset + tile_image_height, x_offset:x_offset + tile_image_width] = image
                print(f"Placed image {image_idx + 1} at position ({row}, {col})")

                for annotation, json_data in [(annotations_complete[image_idx], json_data_complete), (annotations_empty[image_idx], json_data_empty)]:
                    for obj in annotation:
                        class_id = int(obj[0])
                        points = []
                        
                        # Prepare keypoints data
                        for i in range(5, len(obj), 3):
                            keypoint_x = obj[i] * tile_image_width + x_offset
                            keypoint_y = obj[i + 1] * tile_image_height + y_offset
                            visibility = obj[i + 2]
                            points.append([keypoint_x, keypoint_y, visibility])

                        # Bounding box information
                        x_center = obj[1] * tile_image_width + x_offset
                        y_center = obj[2] * tile_image_height + y_offset
                        w = obj[3] * tile_image_width
                        h = obj[4] * tile_image_height

                        bbox_x1 = int(x_center - w / 2)
                        bbox_y1 = int(y_center - h / 2)
                        bbox_x2 = int(x_center + w / 2)
                        bbox_y2 = int(y_center + h / 2)

                        shape = {
                            "flags": {},
                            "group_id": None,
                            "label": str(class_id),
                            "points": points,
                            "shape_type": "points",
                            "bbox": [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
                        }
                        json_data["shapes"].append(shape)
                    print(f"Processed annotations for image {image_idx + 1}")

            # Save the mega-tile image
            output_image_path = os.path.join(output_folder, f"mega_tile_{tile_index+1}.png")
            cv2.imwrite(output_image_path, mega_tile)
            print(f"Mega-tile {tile_index + 1} saved at {output_image_path}")

            # Save JSON label files
            complete_json_path = os.path.join(output_folder, f"mega_tile_{tile_index+1}_Complete.json")
            empty_json_path = os.path.join(output_folder, f"mega_tile_{tile_index+1}.json")
            with open(complete_json_path, 'w') as json_file:
                json.dump(json_data_complete, json_file, indent=4)
            with open(empty_json_path, 'w') as json_file:
                json.dump(json_data_empty, json_file, indent=4)
            print(f"Annotation files for mega-tile {tile_index + 1} saved.")

            # Display each mega-tile (optional)
            plt.imshow(cv2.cvtColor(mega_tile, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
        except Exception as e:
            print(f"Error creating mega-tile {tile_index + 1}: {e}")

# Parameters for function
image_folder = '/home/training/BlockDetector/Megatile_Test'       # Image folder path
annotation_folder = '/home/training/BlockDetector/Megatile_Test'  # Annotation folder path
output_folder = '/home/training/BlockDetector/Megatile dataset'     # Output folder path
num_tiles = 15                              # Number of mega-tiles
tile_rows = 41                              # Number of rows in each mega-tile
tile_cols = 41                              # Number of columns in each mega-tile

# Call the function
create_mega_tiles(image_folder, annotation_folder, output_folder, num_tiles, tile_rows, tile_cols)
