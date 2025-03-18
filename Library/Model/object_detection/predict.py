import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib as mpl

# Load a pretrained YOLOv8n model
model = YOLO("/home/training/Aerialytic_AI/Graph-Extraction/runs/detect21/weights/best.pt")

# Run inference on the image
results = model.predict("/home/training/Yousefi/small-train/sampled_physical_cell_0_64512.png_no_lines_2.png", save=True, imgsz=1024, show_labels=False, save_txt=True, save_conf=True, conf=0.01)

# Load the imag
image_path = "/home/training/Yousefi/small-train/sampled_physical_cell_0_64512.png_no_lines_2.png"
image = cv2.imread(image_path)

# Extract bounding box information and draw them on the image
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer
        conf = box.conf[0].cpu().numpy()  # Move to CPU and convert to numpy
        cls = int(box.cls[0])  # Class index (convert to integer)

        # Normalize confidence to a value between 0 and 1 for color mapping
        conf_line = np.clip(conf, 0, 1)  # Ensure it stays within [0, 1]
        
        # Get color based on confidence (winter colormap)
        color = mpl.cm.winter(conf_line)  
        color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))  # Convert to BGR format

        # Draw positive or negative diagonal based on the class
        if cls == 2:  # Class 2: Draw positive diagonal
            cv2.line(image, (x1, y1), (x2, y2), color, 2)  # Line with mapped color
        elif cls == 3:  # Class 3: Draw negative diagonal
            cv2.line(image, (x1, y2), (x2, y1), color, 2)  # Line with mapped color
       
# Save the resulting image
output_path = "/home/training/Yousefi/result_image.png"
cv2.imwrite(output_path, image)
