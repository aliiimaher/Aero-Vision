import os
import random
import argparse
from PIL import Image

class MegatileCreator:
    def __init__(self, image_dir, label_dir, output_dir, num_columns, num_rows):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.num_columns = num_columns
        self.num_rows = num_rows

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Get all image and label files
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.label_files = [f for f in os.listdir(self.label_dir) if f.endswith('.txt')]

        # Check if enough images are available
        if len(self.image_files) < self.num_columns * self.num_rows:
            raise ValueError("Not enough images to create the megatile.")

        # Select the required number of unique images and corresponding labels
        self.selected_images = random.sample(self.image_files, self.num_columns * self.num_rows)
        self.selected_labels = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in self.selected_images]

    def create_megatile(self):
        # Load images
        images = [Image.open(os.path.join(self.image_dir, img)) for img in self.selected_images]

        # Get image dimensions (assuming all images have the same dimensions)
        img_width, img_height = images[0].size

        # Create a blank canvas for the megatile
        megatile_width = self.num_columns * img_width
        megatile_height = self.num_rows * img_height
        megatile = Image.new('RGB', (megatile_width, megatile_height))

        # Paste images into the megatile
        for idx, image in enumerate(images):
            row = idx // self.num_columns
            col = idx % self.num_columns
            megatile.paste(image, (col * img_width, row * img_height))

        # Save the megatile image
        megatile_image_path = os.path.join(self.output_dir, 'megatile_image.jpg')
        megatile.save(megatile_image_path)

        # Create the megatile label file
        megatile_labels = []

        # Adjust bounding box coordinates for each image and write to the megatile label file
        for idx, label_file in enumerate(self.selected_labels):
            label_path = os.path.join(self.label_dir, label_file)
            with open(label_path, 'r') as file:
                label_data = file.readlines()

            row = idx // self.num_columns
            col = idx % self.num_columns

            for line in label_data:
                label = line.strip().split()

                # Extract the class and bounding box coordinates
                class_id = int(label[0])
                x_center = float(label[1])
                y_center = float(label[2])
                width = float(label[3])
                height = float(label[4])

                # Adjust coordinates
                x_center_new = (x_center + col) / self.num_columns
                y_center_new = (y_center + row) / self.num_rows
                width_new = width / self.num_columns
                height_new = height / self.num_rows

                # Append the new label
                megatile_labels.append(f"{class_id} {x_center_new:.6f} {y_center_new:.6f} {width_new:.6f} {height_new:.6f}\n")

        # Write the megatile label file
        megatile_label_path = os.path.join(self.output_dir, 'megatile_labels.txt')
        with open(megatile_label_path, 'w') as file:
            file.writelines(megatile_labels)

        print(f"Megatile image saved to: {megatile_image_path}")
        print(f"Megatile labels saved to: {megatile_label_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create a megatile image from smaller images and their labels.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the folder containing the images.')
    parser.add_argument('--label_dir', type=str, required=True, help='Path to the folder containing the labels.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the output megatile image and label file.')
    parser.add_argument('--num_columns', type=int, required=True, help='Number of columns for the megatile grid.')
    parser.add_argument('--num_rows', type=int, required=True, help='Number of rows for the megatile grid.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    megatile_creator = MegatileCreator(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        num_columns=args.num_columns,
        num_rows=args.num_rows
    )

    megatile_creator.create_megatile()
