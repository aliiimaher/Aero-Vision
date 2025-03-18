import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

# Load a pretrained YOLOv8n model
model = YOLO("C:/Users/F/OneDrive/Desktop/aligner/best.pt")

# Run inference on the image
results = model("C:/Users/F/OneDrive/Desktop/dataloader/Physical_Cells2/train/images/physical_cell_0_35840.png", save=True, show_labels=False, save_txt=True, save_conf=True, conf=0.0001)

# Load the image
image_path = "C:/Users/F/OneDrive/Desktop/dataloader/Physical_Cells2/train/images/physical_cell_0_35840.png"
image = cv2.imread(image_path)

# Prepare lines from YOLO predictions
lines = []

# Iterate over the results and extract predicted lines
for result in results:
    boxes = result.boxes  # Extract the boxes from the result
    for box in boxes:
        # Extract bounding box coordinates and class
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert to integer list
        conf = box.conf[0].item()  # Extract confidence score
        cls = int(box.cls[0].item())  # Extract class index

        # Depending on the class, determine whether to draw positive or negative diagonal
        if cls == 1:  # Class 1: Draw positive diagonal
            lines.append({'points': [(x1, y1), (x2, y2)], 'confidence': conf})
        elif cls == 0:  # Class 0: Draw negative diagonal
            lines.append({'points': [(x1, y2), (x2, y1)], 'confidence': conf})

# Function to calculate line properties (angle and length)
def calculate_line_properties(points):
    x1, y1 = points[0]
    x2, y2 = points[-1]
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 180
    length = math.sqrt(dx**2 + dy**2)
    return angle, length

# Clustering lines and visualizing the dominant orientation per grid cell
def display_lines_and_clusters(image, lines, grid_size=(3, 3), thickness=2):
    height, width = image.shape[:2]
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]

    # Initialize grid cells
    grid_cells = [[[] for _ in range(grid_size[1])] for _ in range(grid_size[0])]

    # Assign lines to grid cells
    for line in lines:
        points = line['points']
        angle, length = calculate_line_properties(points)

        # Determine which grid cell(s) the line belongs to
        start_row, start_col = int(points[0][1]) // cell_height, int(points[0][0]) // cell_width
        end_row, end_col = int(points[-1][1]) // cell_height, int(points[-1][0]) // cell_width

        min_row, max_row = min(start_row, end_row), max(start_row, end_row)
        min_col, max_col = min(start_col, end_col), max(start_col, end_col)

        # Add line to relevant grid cells
        for row in range(min_row, min(grid_size[0], max_row + 1)):
            for col in range(min_col, min(grid_size[1], max_col + 1)):
                grid_cells[row][col].append((angle, length, points))

        # Draw the original line
        cv2.polylines(image, [np.array(points, dtype=np.int32)], False, (0, 0, 255), thickness)

    # Process each grid cell
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            cell_lines = grid_cells[row][col]
            if cell_lines:
                angles, lengths, points = zip(*cell_lines)

                # Calculate cell center
                cell_center_x = (col + 0.5) * cell_width
                cell_center_y = (row + 0.5) * cell_height
                cell_center = (cell_center_x, cell_center_y)

                # Determine number of clusters
                n_clusters = min(6, len(angles))

                # Apply k-means clustering
                kmeans = KMeans(n_clusters=n_clusters)
                labels = kmeans.fit_predict(np.array(angles).reshape(-1, 1))

                # Find the dominant cluster
                cluster_weights = np.bincount(labels, weights=lengths)
                dominant_cluster = np.argmax(cluster_weights)
                dominant_angle = kmeans.cluster_centers_[dominant_cluster][0]

                # Draw the dominant orientation line for this cell
                line_length = min(cell_width, cell_height) * 0.8
                angle_rad = math.radians(dominant_angle)
                dx = math.cos(angle_rad) * line_length / 2
                dy = math.sin(angle_rad) * line_length / 2
                start_x = int(cell_center_x - dx)
                start_y = int(cell_center_y - dy)
                end_x = int(cell_center_x + dx)
                end_y = int(cell_center_y + dy)

                # Draw the dominant orientation line
                cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)

                # Add text to show the dominant angle
                text_position = (int(cell_center_x), int(cell_center_y) - 20)
                cv2.putText(image, f"{dominant_angle:.1f}°", text_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw grid lines
    for i in range(1, grid_size[0]):
        cv2.line(image, (0, i * cell_height), (width, i * cell_height), (255, 0, 0), 3)
    for j in range(1, grid_size[1]):
        cv2.line(image, (j * cell_width, 0), (j * cell_width, height), (255, 0, 0), 3)

    return image

# Apply the clustering and display the results
image_with_lines_and_clusters = display_lines_and_clusters(image.copy(), lines, grid_size=(3, 3), thickness=2)

# Display the final image with clusters and dominant orientations
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(image_with_lines_and_clusters, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the resulting image
output_path = "C:/Users/F/OneDrive/Desktop/aligner/result_image_with_clusters.png"
cv2.imwrite(output_path, image_with_lines_and_clusters)
