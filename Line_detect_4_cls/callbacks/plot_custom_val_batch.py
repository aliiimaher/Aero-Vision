import cv2
import numpy as np
import torch
from pathlib import Path
import os
def plot_val_samples(images, batch_idx, sample_infos, paths, save_dir, ni):
    """Creates individual plots of training sample images with keypoints from sample_infos."""
    # Extract keypoints from sample_infos
    keypoints = []
    for info in sample_infos:
        keypoints.append(info['keypoints'].numpy())  # Assuming keypoints in sample_info are tensors

    # Call the function to plot images with keypoints
    plot_images_with_keypoints(
        images,
        batch_idx,
        keypoints,
        sample_infos,
        paths,
        save_dir,
        ni
    )

def plot_images_with_keypoints(
    images: torch.Tensor,
    batch_idx: torch.Tensor,
    keypoints: list,
    sample_infos: list,
    paths: list,
    save_dir: Path,
    ni: int
):
    """Plot individual images with keypoints and save them."""
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    bs, _, h, w = images.shape
    bs = min(bs, 16)

    if np.max(images[0]) <= 1:
        images *= 255

    # Define new canvas size
    canvas_width = 1800
    canvas_height = 1800
    
    for i in range(bs):
        img_with_keypoints = images[i].transpose(1, 2, 0).astype(np.uint8)

        # Create a white canvas
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
        
        # Calculate position to place the image at the center
        start_y = (canvas_height - h) // 2
        start_x = (canvas_width - w) // 2
        
        # Place the image on the canvas
        canvas[start_y:start_y+h, start_x:start_x+w] = img_with_keypoints

        # Plot keypoints and lines
        if len(keypoints) > 0 and len(keypoints[i]) > 0:
            img_keypoints = keypoints[i]
            
            if len(img_keypoints) > 0:
                img_keypoints[..., 0] = img_keypoints[..., 0] * w + start_x
                img_keypoints[..., 1] = img_keypoints[..., 1] * h + start_y

                for kpts in img_keypoints:
                    for kp_idx in range(len(kpts) - 1):
                        x1, y1, v1 = int(kpts[kp_idx][0]), int(kpts[kp_idx][1]), kpts[kp_idx][2]
                        x2, y2, v2 = int(kpts[kp_idx + 1][0]), int(kpts[kp_idx + 1][1]), kpts[kp_idx + 1][2]
                        if v1 > 0 and v2 > 0:
                            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add sample info text above the image
        info = sample_infos[i]
        text = (f"Rotation: {info['rotation_angle']:.2f}, Scale: {info['scale_val']:.2f}, "
                f"FlipY: {info['flipy']}, Center: ({int(info['sample_center'].x)}, {int(info['sample_center'].y)})")
        
        # Add filename on a new line
        file_text = f"File: {info['img_file']}"

        # Calculate text size and position for centering
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        file_text_size = cv2.getTextSize(file_text, font, font_scale, thickness)[0]
        
        text_x = (canvas_width - text_size[0]) // 2
        text_y = start_y // 2  # Halfway between top and image start
        
        file_text_x = (canvas_width - file_text_size[0]) // 2
        file_text_y = text_y + text_size[1] + 20  # 20 pixels below the first line
        
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(canvas, file_text, (file_text_x, file_text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Save the resulting image
        cv2.imwrite(str(save_dir / f"val_batch{ni}_img{i}.jpg"), canvas)

def plot_callbackval(self, batch, ni):
    val_dir = self.save_dir / f"val"
    val_dir.mkdir(exist_ok=True)
    e = (len(os.listdir(val_dir))//2) * self.args.val_interval
    batch_dir = val_dir / f"e{e}_b{ni}_sample"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    plot_val_samples(
        batch["img"],
        batch["batch_idx"],
        batch["sample_infos"],
        batch["im_file"],
        batch_dir,
        ni
    )