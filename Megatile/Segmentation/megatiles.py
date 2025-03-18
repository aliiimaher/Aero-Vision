import os
import cv2
import json
import numpy as np
import argparse

class MegatileCreator:
    def __init__(self, image_folder, mask_folder, annotation_folder, output_image, output_mask, output_annotation, tile_size):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.annotation_folder = annotation_folder
        self.output_image = output_image
        self.output_mask = output_mask
        self.output_annotation = output_annotation
        self.tile_size = tile_size
        self.image_files = sorted([f for f in os.listdir(self.image_folder) if f.endswith('.png')])

        # Determine the number of images along each dimension (assuming square layout)
        self.num_images = int(np.ceil(np.sqrt(len(self.image_files))))
        self.megatile_size = self.tile_size * self.num_images

        # Initialize megatile image and mask
        self.megatile_image = np.zeros((self.megatile_size, self.megatile_size, 3), dtype=np.uint8)
        self.megatile_mask = np.zeros((self.megatile_size, self.megatile_size, 3), dtype=np.uint8)
        self.megatile_annotations = {
            'description': '',
            'tags': [],
            'size': {
                'height': self.megatile_size,
                'width': self.megatile_size
            },
            'objects': []
        }

    def read_annotations(self, ann_path):
        with open(ann_path, 'r') as f:
            annotations = json.load(f)
        return annotations

    def adjust_coordinates(self, points, offset_x, offset_y):
        return [[x + offset_x, y + offset_y] for x, y in points]

    def pad_image_to_tile_size(self, image):
        h, w, _ = image.shape
        padded_image = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
        padded_image[:h, :w, :] = image
        return padded_image

    def create_megatile(self):
        for idx, img_file in enumerate(self.image_files):
            img_path = os.path.join(self.image_folder, img_file)
            mask_path = os.path.join(self.mask_folder, img_file.replace('.png', '_mask.png'))
            ann_path = os.path.join(self.annotation_folder, os.path.splitext(img_file)[0] + '.json')

            # Read image and mask
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            # Pad image and mask if necessary
            if image.shape[0] < self.tile_size or image.shape[1] < self.tile_size:
                image = self.pad_image_to_tile_size(image)
                mask = self.pad_image_to_tile_size(mask)

            # Calculate position in the megatile
            row = idx // self.num_images
            col = idx % self.num_images
            offset_x = col * self.tile_size
            offset_y = row * self.tile_size

            # Place the image and mask in the megatile
            self.megatile_image[offset_y:offset_y + self.tile_size, offset_x:offset_x + self.tile_size] = image
            self.megatile_mask[offset_y:offset_y + self.tile_size, offset_x:offset_x + self.tile_size] = mask

            # Read and adjust annotations
            annotations = self.read_annotations(ann_path)
            for obj in annotations['objects']:
                new_obj = obj.copy()
                new_obj['points']['exterior'] = self.adjust_coordinates(obj['points']['exterior'], offset_x, offset_y)
                self.megatile_annotations['objects'].append(new_obj)

        # Save the megatile image, mask, and annotations
        cv2.imwrite(self.output_image, self.megatile_image)
        cv2.imwrite(self.output_mask, self.megatile_mask)
        with open(self.output_annotation, 'w') as ann_file:
            json.dump(self.megatile_annotations, ann_file, indent=4)

        print(f'Megatile image saved at {self.output_image}')
        print(f'Megatile mask saved at {self.output_mask}')
        print(f'Megatile annotations saved at {self.output_annotation}')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create a megatile image, mask, and annotations from smaller tiles.")
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing the image tiles.')
    parser.add_argument('--mask_folder', type=str, required=True, help='Path to the folder containing the mask tiles.')
    parser.add_argument('--annotation_folder', type=str, required=True, help='Path to the folder containing the annotation files.')
    parser.add_argument('--output_image', type=str, required=True, help='Path to save the output megatile image.')
    parser.add_argument('--output_mask', type=str, required=True, help='Path to save the output megatile mask.')
    parser.add_argument('--output_annotation', type=str, required=True, help='Path to save the output megatile annotations.')
    parser.add_argument('--tile_size', type=int, required=True, help='Size of each tile (assumed square, e.g., 640 for 640x640 images).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    megatile_creator = MegatileCreator(
        image_folder=args.image_folder,
        mask_folder=args.mask_folder,
        annotation_folder=args.annotation_folder,
        output_image=args.output_image,
        output_mask=args.output_mask,
        output_annotation=args.output_annotation,
        tile_size=args.tile_size
    )
    
    megatile_creator.create_megatile()
