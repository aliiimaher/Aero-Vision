###  Clustering Lines in an Image Using YOLO and K-Means

This Python script combines **YOLOv8** for object detection and **K-Means clustering** to identify dominant line orientations in an image. The process involves detecting lines, classifying their orientations, assigning them to grid cells, and applying K-Means clustering to find the dominant orientation per grid cell. Finally, the results are visualized by drawing these orientations on the image.
![out put of aligner ](smbm65/aerialytic_yousefi/Alligner/output.png)

#### 1. **Loading the YOLOv8 Model**
The code begins by loading a custom-trained YOLOv8 model (`best.pt`). YOLO (You Only Look Once) is an object detection model capable of identifying and localizing objects within an image. Here, the model is used to detect lines in an image. The model predicts bounding boxes, classes, and confidence scores for detected objects.

#### 2. **Performing Inference on the Image**
After loading the model, the script performs inference on the target image, detecting objects such as lines. The results include bounding box coordinates, the class of each object (representing the type of line), and the confidence score. This step is critical for extracting information that will later be used for clustering the line orientations.

#### 3. **Extracting Line Properties**
For each detected bounding box, the code extracts the coordinates of the top-left and bottom-right corners. Depending on the class (positive or negative diagonal), the code determines the orientation of the lines and stores them in a list. Additionally, the code computes each line's properties, such as its **angle** (orientation) and **length**, based on the coordinates of the bounding box corners.

#### 4. **Assigning Lines to Grid Cells**
Next, the image is divided into a grid (default size of 3x3). Each line is assigned to the grid cell based on its starting and ending points. The code calculates the grid cell that each line intersects and stores the line's properties (angle and length) in the corresponding grid cells.

#### 5. **Applying K-Means Clustering**
For each grid cell, the angles of all the lines within that cell are clustered using the **K-Means** algorithm. K-Means groups similar angles together, helping to identify dominant orientations in each grid cell. The number of clusters is dynamically set to a maximum of 6, ensuring that even a small number of lines are clustered effectively.

The code then calculates the **dominant orientation** for each cell by determining the cluster with the highest weighted length (longer lines are given more weight). This dominant angle is used to draw a line representing the prevailing orientation in each cell.

#### 6. **Visualizing the Results**
The code proceeds to visualize the results by drawing the following:

- **Grid lines** to divide the image into smaller cells.
- **Detected lines** drawn in red, showing the orientation and position of each detected line.
- **Dominant orientation lines** in green, representing the prevailing line orientation in each grid cell.
- **Angle annotations** are added near the center of each grid cell, showing the angle of the dominant orientation.

This helps in visualizing how the lines are distributed in the image and how the orientations vary across different regions.

#### 7. **Displaying and Saving the Image**
Finally, the processed image with the detected lines, grid, and dominant orientation lines is displayed using **Matplotlib**. The image is saved to a specified output file, allowing the user to view and further analyze the results.

### Key Components of the Code:
- **YOLOv8**: Used for detecting objects (lines) and extracting bounding boxes, classes, and confidence scores.
- **Line properties**: Calculated for each detected line, including angle and length.
- **Grid division**: The image is divided into cells to analyze lines in local regions.
- **K-Means clustering**: Applied to cluster line orientations within each grid cell and determine the dominant angle.
- **Visualization**: The final image displays detected lines, dominant orientations, and grid annotations.

### Summary:
This script effectively combines advanced object detection (YOLOv8) with clustering (K-Means) to analyze line orientations in an image. By dividing the image into grid cells and clustering line angles within each cell, it provides a comprehensive analysis of line distribution and orientation, visualized through a series of drawn lines and annotations. The process allows for a deeper understanding of the spatial organization of lines in the image, making it useful for applications such as image analysis, pattern recognition, and object alignment.
