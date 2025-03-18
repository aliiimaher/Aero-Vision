import os
import numpy as np
import cv2
from PIL import Image, ImageFile
import argparse

# Increase the maximum allowed image size
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PhysicalCellCreator:
    def __init__(self, image_path, mask_path, output_image_dir, output_mask_dir, grid_height, grid_width):
        self.image_path = image_path
        self.mask_path = mask_path
        self.output_image_dir = output_image_dir
        self.output_mask_dir = output_mask_dir
        self.grid_height = grid_height
        self.grid_width = grid_width

        # Load the image and mask
        self.image = self.load_image(self.image_path)
        self.mask = self.load_image(self.mask_path)

        # Create output directories if they do not exist
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_mask_dir, exist_ok=True)

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from a file path using Pillow and convert it to a NumPy array.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Loaded image in BGR format.
        """
        with Image.open(image_path) as img:
            image = np.array(img)
            # Convert RGB to BGR for OpenCV compatibility
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def save_image(self, image: np.ndarray, output_path: str):
        """
        Save an image to a file.

        Args:
            image (np.ndarray): Image to be saved.
            output_path (str): Path to save the image file.
        """
        cv2.imwrite(output_path, image)

    def pad_image(self, image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """
        Pad the image with black pixels to reach the target size.

        Args:
            image (np.ndarray): Input image.
            target_height (int): Target height.
            target_width (int): Target width.

        Returns:
            np.ndarray: Padded image.
        """
        height, width = image.shape[:2]
        padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded_image[:height, :width] = image
        return padded_image

    def create_physical_cells(self):
        """
        Create physical cells from the image and mask, dividing them into grid cells, and saving them.
        """
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
                physical_mask = np.zeros((physical_size, physical_size, 3), dtype=np.uint8)

                start_x = max(0, top_left_x)
                start_y = max(0, top_left_y)
                end_x = min(img_width, bottom_right_x)
                end_y = min(img_height, bottom_right_y)

                offset_x = max(0, -top_left_x)
                offset_y = max(0, -top_left_y)

                physical_cell[offset_y:offset_y+(end_y-start_y), offset_x:offset_x+(end_x-start_x)] = self.image[start_y:end_y, start_x:end_x]
                physical_mask[offset_y:offset_y+(end_y-start_y), offset_x:offset_x+(end_x-start_x)] = self.mask[start_y:end_y, start_x:end_x]

                image_output_path = f"{self.output_image_dir}/physical_cell_{y}_{x}.png"
                mask_output_path = f"{self.output_mask_dir}/physical_cell_{y}_{x}_mask.png"

                self.save_image(physical_cell, image_output_path)
                self.save_image(physical_mask, mask_output_path)

                print(f"Saved {image_output_path}")
                print(f"Saved {mask_output_path}")

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Create physical cells from an image and its associated mask.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask file.")
    parser.add_argument("--output_image_dir", type=str, required=True, help="Directory to save the output physical cell images.")
    parser.add_argument("--output_mask_dir", type=str, required=True, help="Directory to save the output physical cell masks.")
    parser.add_argument("--grid_height", type=int, default=1024, help="Height of the grid cells.")
    parser.add_argument("--grid_width", type=int, default=1024, help="Width of the grid cells.")
    
    return parser.parse_args()

def main():
    """
    Main function to execute the PhysicalCellCreator class.
    """
    args = parse_args()

    # Create an instance of PhysicalCellCreator with the provided arguments
    creator = PhysicalCellCreator(
        image_path=args.image_path,
        mask_path=args.mask_path,
        output_image_dir=args.output_image_dir,
        output_mask_dir=args.output_mask_dir,
        grid_height=args.grid_height,
        grid_width=args.grid_width
    )

    # Create physical cells from the image and mask
    creator.create_physical_cells()

if __name__ == "__main__":
    main()
