import cv2
import numpy as np
import torch
from pathlib import Path
import os
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils import ops
def plot_val_samples(images, batch_idx, sample_infos, paths, save_dir, ni):
    """Plot each training sample image with keypoints, bounding boxes and other annotations on an individual 1800x1800 canvas."""
    
    # Extract keypoints, bboxes and classes from sample_infos
    keypoints = [info['keypoints'].numpy() for info in sample_infos]  # Assuming keypoints are tensors
    bboxes = [info['bboxes'].numpy() for info in sample_infos]  # Assuming bboxes are tensors
    classes = [info['cls'].numpy() for info in sample_infos]  # Assuming classes are tensors
    
    # Call the plotting function with individual images and save path
    plot_images_with_keypoints(
        images,
        batch_idx,
        keypoints,
        bboxes,
        classes,
        sample_infos,
        paths,
        save_dir,
        ni
    )

def plot_images_with_keypoints(
    images: torch.Tensor,
    batch_idx: torch.Tensor,
    keypoints: list,
    bboxes: list,
    classes: list,
    sample_infos: list,
    paths: list,
    save_dir: Path,
    ni: int
):
    """Plot each image individually with keypoints and bounding boxes on a white canvas and save them."""

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    bs, _, h, w = images.shape  # Batch size, height, width
    bs = min(bs, 16)  # Limit the number of images to 16

    if np.max(images[0]) <= 1:
        images *= 255  # Scale to 0-255 if needed

    canvas_size = 1800  # Define the canvas size (1800x1800)

    # Define a color for each of the 22 keypoints
    keypoint_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
        (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0),
        (0, 0, 192), (192, 192, 0)
    ]

    for i in range(bs):
        img_with_keypoints = images[i].transpose(1, 2, 0).astype(np.uint8)

        # Create a white canvas
        canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

        # Calculate the position to center the image on the canvas
        start_y = (canvas_size - h) // 2
        start_x = (canvas_size - w) // 2
        canvas[start_y:start_y + h, start_x:start_x + w] = img_with_keypoints

        # Create annotator object for the canvas
        fs = int((canvas_size) * 0.01)  # font size
        annotator = Annotator(canvas, line_width=round(fs / 10), font_size=fs)

        # Plot keypoints on the canvas
        if len(keypoints) > 0 and len(keypoints[i]) > 0:
            img_keypoints = keypoints[i]
            img_keypoints[..., 0] *= w  # Convert normalized X to pixel scale
            img_keypoints[..., 1] *= h  # Convert normalized Y to pixel scale
            
            for kpts in img_keypoints:
                for kp_idx in range(len(kpts)):
                    x, y, v = int(kpts[kp_idx][0]), int(kpts[kp_idx][1]), kpts[kp_idx][2]
                    if v == 2:  # Only draw keypoints with visibility == 2
                        # Draw filled circle for each keypoint using unique color
                        color = keypoint_colors[kp_idx % 22]  # Cycle colors if more than 22 keypoints
                        cv2.circle(canvas, 
                                   (x + start_x, y + start_y),  # Position adjusted for canvas
                                   radius=3,  # Radius of the circle
                                   color=color,  # Color for this keypoint
                                   thickness=-1)  # Filled circle

        # Plot bounding boxes on the canvas
        if len(bboxes) > 0 and len(bboxes[i]) > 0:
            boxes = bboxes[i]
            cls = classes[i]  # Get class labels for this image
            
            if boxes.shape[-1] == 5:  # rotated bounding boxes (xywhr)
                is_obb = True
                boxes = ops.xywhr2xyxyxyxy(boxes)
            else:  # regular bounding boxes (xywh)
                is_obb = False
                boxes = ops.xywh2xyxy(boxes)
            
            # Scale normalized coordinates to pixel values
            if boxes.max() <= 5.1:  # if normalized with tolerance 0.1
                boxes[..., [0, 2]] *= w
                boxes[..., [1, 3]] *= h
            
            # Adjust coordinates for canvas position
            boxes[..., [0, 2]] += start_x
            boxes[..., [1, 3]] += start_y
            
            # Draw each box with its class color
            for j, box in enumerate(boxes.astype(np.int64).tolist()):
                class_idx = int(cls[j])  # Get class index for current box
                color = colors(class_idx)  # Get color based on class index
                
                if is_obb:
                    # For rotated boxes, use polygon drawing
                    pts = np.array(box).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [pts], True, color, thickness=1)
                    
                    # Add class label near the box
                    x_min = min(p[0] for p in box[::2])
                    y_min = min(p[1] for p in box[1::2])
             
                else:
                    # For regular boxes, use rectangle drawing
                    cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, thickness=1)
                    
              

        # Add sample info text
        info = sample_infos[i]
        text = (f"Rotation: {info['rotation_angle']:.2f}, Scale: {info['scale_val']:.2f}, "
                f"FlipY: {info['flipy']}, Center: ({int(info['sample_center'].x)}, {int(info['sample_center'].y)})")
        file_text = f"File: {info['img_file']}"

        # Calculate position for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        file_text_size = cv2.getTextSize(file_text, font, font_scale, thickness)[0]

        text_x = (canvas_size - text_size[0]) // 2
        text_y = start_y // 2  # Positioned above the image
        file_text_x = (canvas_size - file_text_size[0]) // 2
        file_text_y = text_y + text_size[1] + 20  # Below the first line of text

        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(canvas, file_text, (file_text_x, file_text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Save each image on the canvas as a separate file
        img_path = save_dir / f"val_batch{ni}_img{i}.jpg"
        cv2.imwrite(str(img_path), canvas)

    return None


def plot_callbackval(self, batch, ni):
    train_dir = self.save_dir / "val"
    train_dir.mkdir(exist_ok=True)
    e = (len(os.listdir(train_dir))) * self.args.plot_interval
    batch_dir = train_dir / f"e{e}_b{ni}_sample"
    batch_dir.mkdir(exist_ok=True)
    
    # Callback to plot and save each training sample
    plot_val_samples(
        batch['img'],
        batch['batch_idx'],
        batch['sample_infos'],
        batch['im_file'],
        batch_dir,
        ni
    )
