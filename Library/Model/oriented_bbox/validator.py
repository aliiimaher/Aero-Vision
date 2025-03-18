import torch
from ultralytics.models.yolo.obb import OBBValidator
from pathlib import Path
from dataset import OrientedBBoxDataset
from torch.utils.data import DistributedSampler, RandomSampler
from ultralytics.data.build import InfiniteDataLoader
import matplotlib as mpl
import albumentations as A
import numpy as np
import os
import cv2
import sys
sys.path.insert(0, os.path.abspath('/home/4TDrive/Yousefi'))
from ultralytics.utils.plotting import output_to_target
from library.Model.object_detection.validator import get_custom_transforms , collate_fn
from ultralytics.utils.plotting import output_to_target, Annotator, colors
from typing import Callable, Dict, List, Optional, Union
from ultralytics.utils.checks import is_ascii
from albumentations.pytorch import ToTensorV2
from ultralytics.utils import ops
import math

class CustomAnnotator(Annotator):
    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        """
        Draws a bounding box to image with label and a line connecting the midpoints of the width.

        Args:
            box (list or tuple): The bounding box coordinates [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            rotated (bool, optional): Whether the box is rotated (OBB).
        """
        
        
        # Debugging the box input
        print(f"Original box: {box}")
        def get_shorter_edges(box):
            """
            این تابع دو ضلع کوتاه‌تر را پیدا می‌کند و وسط آن‌ها را به‌عنوان عرض باندینگ باکس در نظر می‌گیرد.
            """
            # تبدیل نقاط به آرایه NumPy
            box = np.array(box, dtype=np.float32)

            # محاسبه‌ی طول چهار ضلع باندینگ باکس
            edges = [
                (np.linalg.norm(box[0] - box[1]), (box[0], box[1])),  # ضلع بالا
                (np.linalg.norm(box[1] - box[2]), (box[1], box[2])),  # ضلع راست
                (np.linalg.norm(box[2] - box[3]), (box[2], box[3])),  # ضلع پایین
                (np.linalg.norm(box[3] - box[0]), (box[3], box[0]))   # ضلع چپ
            ]

            # مرتب‌سازی بر اساس طول و انتخاب دو ضلع کوتاه‌تر
            edges = sorted(edges, key=lambda x: x[0])[:2]

            return edges
        # Ensure box is a list or tuple of numbers
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if isinstance(box, (list, tuple)):
            try:
                if isinstance(box[0], (list, tuple)):
                    box = [(float(b[0]), float(b[1])) for b in box]  # Convert each coordinate pair to a tuple of floats
                else:
                    box = [float(b) for b in box]

                print(f"Flattened box: {box}")
            except Exception as e:
                print(f"Error while flattening box: {e}")
                raise
        else:
            raise TypeError(f"box must be a list or tuple, not {type(box)}")

        if self.pil or not is_ascii(label):
            if rotated:
                p1 = box[0]
                self.draw.polygon(box, width=self.lw, outline=color)
            else:
                p1 = (box[0][0], box[0][1])
                self.draw.polygon(box, outline=color, width=self.lw)  # در حالت عادی، باندینگ باکس به‌صورت چندضلعی رسم می‌شود.

            # پیدا کردن دو ضلع کوتاه‌تر
            shorter_edges = get_shorter_edges(box)

            # محاسبه‌ی نقاط میانی برای دو ضلع کوتاه‌تر
            mid1 = ((shorter_edges[0][1][0] + shorter_edges[0][1][1]) / 2).astype(int)
            mid2 = ((shorter_edges[1][1][0] + shorter_edges[1][1][1]) / 2).astype(int)

            # رسم خط بین وسط دو ضلع کوتاه‌تر
            self.draw.line([tuple(mid1), tuple(mid2)], fill=color, width=self.lw)

            if label:
                w, h = self.font.getsize(label)
                outside = p1[1] >= h
                if p1[0] > self.im.size[0] - w:
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
               

        else:  # OpenCV
            if rotated:
                p1 = [int(b[0]) for b in box]
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)
            else:
                p1, p2 = (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1]))
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)

            # پیدا کردن دو ضلع کوتاه‌تر
            shorter_edges = get_shorter_edges(box)

            # محاسبه‌ی نقاط میانی برای دو ضلع کوتاه‌تر
            mid1 = ((shorter_edges[0][1][0] + shorter_edges[0][1][1]) / 2).astype(int)
            mid2 = ((shorter_edges[1][1][0] + shorter_edges[1][1][1]) / 2).astype(int)

            # رسم خط بین وسط دو ضلع کوتاه‌تر
            cv2.line(self.im, tuple(mid1), tuple(mid2), color, thickness=self.lw, lineType=cv2.LINE_AA)

            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
                h += 3
                outside = p1[1] >= h
                if p1[0] > self.im.shape[1] - w:
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                

def plot_custom_images(
    images: Union[torch.Tensor, np.ndarray],
    batch_idx: Union[torch.Tensor, np.ndarray],
    cls: Union[torch.Tensor, np.ndarray],
    bboxes: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.float32),
    confs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    masks: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.uint8),
    kpts: Union[torch.Tensor, np.ndarray] = np.zeros((0, 51), dtype=np.float32),
    paths: Optional[List[str]] = None,
    fname: str = "images.jpg",
    names: Optional[Dict[int, str]] = None,
    on_plot: Optional[Callable] = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.1,
) -> Optional[np.ndarray]:
    """
    Plot image grid with labels, bounding boxes, masks, and keypoints.

    Args:
        images: Batch of images to plot. Shape: (batch_size, channels, height, width).
        batch_idx: Batch indices for each detection. Shape: (num_detections,).
        cls: Class labels for each detection. Shape: (num_detections,).
        bboxes: Bounding boxes for each detection. Shape: (num_detections, 4) or (num_detections, 5) for rotated boxes.
        confs: Confidence scores for each detection. Shape: (num_detections,).
        masks: Instance segmentation masks. Shape: (num_detections, height, width) or (1, height, width).
        kpts: Keypoints for each detection. Shape: (num_detections, 51).
        paths: List of file paths for each image in the batch.
        fname: Output filename for the plotted image grid.
        names: Dictionary mapping class indices to class names.
        on_plot: Optional callback function to be called after saving the plot.
        max_size: Maximum size of the output image grid.
        max_subplots: Maximum number of subplots in the image grid.
        save: Whether to save the plotted image grid to a file.
        conf_thres: Confidence threshold for displaying detections.

    Returns:
        np.ndarray: Plotted image grid as a numpy array if save is False, None otherwise.

    Note:
        This function supports both tensor and numpy array inputs. It will automatically
        convert tensor inputs to numpy arrays for processing.
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None

            if len(bboxes):
                boxes = bboxes[idx]
                conf = confs[idx] if confs is not None else None  # check for confidence presence (label vs pred)
                if len(boxes):
                    if boxes[:, :4].max() <= 5.1:  # if normalized with tolerance 0.1
                        boxes[..., [0, 2]] *= w  # scale to pixels
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5  # xywhr
                boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        CustomAnnotator.box_label(box, label, color=color, rotated=is_obb)

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    annotator.text((x, y), f"{c}", txt_color=color, box_style=True)

            # Plot keypoints
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 5.01 or kpts_[..., 1].max() <= 5.01:  # if normalized with tolerance .01
                        kpts_[..., 0] *= w  # scale to pixels
                        kpts_[..., 1] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)

            # Plot masks
            if len(masks):
                if idx.shape[0] == masks.shape[0]:  # overlap_masks=False
                    image_masks = masks[idx]
                else:  # overlap_masks=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)

                im = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        try:
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                        except Exception:
                            pass
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)  # save
    if on_plot:
        on_plot(fname)


class CustomOBBValidator(OBBValidator):
    def __init__(self, *args, plot_callbackval=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_callbackval = plot_callbackval

    def get_dataloader(self, images_dir, batch_size, rank, mode):
        parent_path, last_folder_name = os.path.split(images_dir)
        dataset = OrientedBBoxDataset(
            data_dir=parent_path,
            transform=get_custom_transforms(),
            use_keypoint_clipping=False,
            use_aligner=True,
        )
        
        if rank == -1:  # Single GPU or CPU
            sampler = RandomSampler(dataset) #if mode == "train" else None
        else:  # Distributed training
            sampler = DistributedSampler(dataset) #if mode == "train" else None
        batch_size = batch_size // (2 if mode == "val" else 1)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size * 2,
            shuffle=(sampler is None),  # Shuffle if no sampler is used
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=24  # Adjust according to your system
        )
 
    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        val_dir = self.save_dir / f"val"
        val_dir.mkdir(exist_ok=True)
        e = (len(os.listdir(val_dir))//2) * self.args.val_interval
        batch_dir = val_dir / f"e{e}_b{ni}_pred"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(len(batch["img"])):
            individual_batch = {
                "img": batch["img"][i:i+1],
                "im_file": batch["im_file"][i:i+1],
            }
            
            individual_preds = output_to_target(preds[i:i+1], max_det=self.args.max_det)
            
            # Plot the custom images with bounding boxes
            img_with_predictions = plot_custom_images(
                individual_batch["img"],
                *individual_preds,
                paths=individual_batch["im_file"],
                fname=batch_dir / f"val_batch{ni}_img{i}_pred.jpg",
                names=self.names,
                on_plot=self.on_plot,
                save=False  # Don't save immediately, we'll handle it below
            )
            
            if img_with_predictions is None:
                continue  # Handle the case where the image was not generated
            
            # Create a white canvas of 1800x1800 pixels
            canvas = np.full((1800, 1800, 3), 255, dtype=np.uint8)

            # Get the dimensions of the image
            h, w, _ = img_with_predictions.shape
            
            # Calculate starting positions to center the image on the canvas
            start_y = (canvas.shape[0] - h) // 2
            start_x = (canvas.shape[1] - w) // 2
            
            # Place the image on the canvas
            canvas[start_y:start_y+h, start_x:start_x+w] = img_with_predictions

            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(batch_dir / f"val_batch{ni}_img{i}.jpg"), canvas_bgr)
 

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples using the custom plot_callback."""
        if self.plot_callbackval:
            self.plot_callbackval(self, batch, ni)
        else:
            super().plot_val_samples(batch, ni)
