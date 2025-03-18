import torch
from ultralytics.models.yolo.detect import DetectionValidator
from dataset import PhysicalCellsObjectDetectionDataset
from ultralytics.utils.plotting import output_to_target, Annotator, colors
from typing import Callable, Dict, List, Optional, Union
from ultralytics.utils.checks import is_ascii
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DistributedSampler, RandomSampler
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.utils import ops
import matplotlib as mpl
import albumentations as A
import numpy as np
import os
import math
import cv2

class CustomAnnotator(Annotator):
    def box_label(self, box, label="", color=(0, 255, 0), txt_color=(255, 255, 255), rotated=False, cls=None, conf=None):
        # Existing implementation remains unchanged
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, torch.Tensor):
            box = box.tolist()

        if not rotated:
            x1, y1, x2, y2 = box

        if conf is not None:
            conf_normalized = np.clip(conf, 0, 1)
            color_mapped = mpl.cm.winter(conf_normalized)
            color = (int(color_mapped[2] * 255), int(color_mapped[1] * 255), int(color_mapped[0] * 255))

        if cls == 1:  
            if self.pil:
                self.draw.line([(x1, y1), (x2, y2)], fill=color, width=self.lw)
            else:
                cv2.line(self.im, (x1, y1), (x2, y2), color, self.lw, lineType=cv2.LINE_AA)

        elif cls == 0:  
            if self.pil:
                self.draw.line([(x1, y2), (x2, y1)], fill=color, width=self.lw)
            else:
                cv2.line(self.im, (x1, y2), (x2, y1), color, self.lw, lineType=cv2.LINE_AA)

        else:  
            if self.pil or not is_ascii(label):
                if rotated:
                    self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)
                else:
                    p1 = (x1, y1)
                    self.draw.rectangle(box, width=self.lw, outline=color)
                if label:
                    w, h = self.font.getsize(label)
                    outside = p1[1] >= h
                    if p1[0] > self.im.size[0] - w:
                        p1 = self.im.size[0] - w, p1[1]
                    self.draw.rectangle(
                        (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                        fill=color,
                    )
                    self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
            else:
                if rotated:
                    p1 = [int(b) for b in box[0]]
                    cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)
                else:
                    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
                if label:
                    w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
                    h += 3
                    outside = p1[1] >= h
                    if p1[0] > self.im.shape[1] - w:
                        p1 = self.im.shape[1] - w, p1[1]
                    p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                    cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                    cv2.putText(
                        self.im,
                        label,
                        (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                        0,
                        self.sf,
                        txt_color,
                        thickness=self.tf,
                        lineType=cv2.LINE_AA,
                    )


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
    conf_thres: float = 0.01,
) -> Optional[np.ndarray]:
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    
    # Ensure the images are in RGB format
    if images.shape[1] == 3:  # Check if channels are in RGB
        images = images[:, [2, 1, 0], :, :]  # Convert from BGR to RGB if needed

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        mosaic[y: y + h, x: x + w, :] = images[i].transpose(1, 2, 0)

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = CustomAnnotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders

        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            
            if len(bboxes):
                boxes = bboxes[idx]
                # conf = confs[idx] if confs is not None else None

                conf = confs[idx] if confs is not None else None
                classes = cls[idx].astype("int")

                idx = np.argsort(conf)
                boxes = boxes[idx]
                conf = conf[idx]
                classes = classes[idx]

                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:
                        boxes[..., [0, 2]] *= w
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:
                        boxes[..., :4] *= scale
                    boxes[..., 0] += x
                    boxes[..., 1] += y
                    is_obb = boxes.shape[-1] == 5
                    boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                    
                    for j, box in enumerate(boxes.astype(np.int64).tolist()):
                        c = classes[j]
                        color = colors(c)
                        if conf is not None and conf[j] > conf_thres:
                            label = f"{conf[j]:.2f}"
                            conf_normalized = (conf[j] - conf_thres) / (1 - conf_thres)
                            conf_clipped = np.clip(conf_normalized, 0, 1)
                            color_mapped = mpl.cm.winter(conf_clipped)
                            color = (int(color_mapped[2] * 255), int(color_mapped[1] * 255), int(color_mapped[0] * 255))
                            annotator.box_label(box, "", color=color, rotated=is_obb, cls=c, conf=conf[j])

    # Save or return the image
    if not save:
        return np.asarray(annotator.im)  # Return the image if not saving
    annotator.im.save(fname)  # Save the image if saving
    if on_plot:
        on_plot(fname)

    # Return the image as an array after saving
    return np.asarray(annotator.im)



def get_custom_transforms():
    return A.Compose([
        A.OneOf([
            A.Blur(p=0.5),
            A.MedianBlur(p=0.5),
        ], p=0.7),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.5), contrast_limit=0.5, p=0.5
            ),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=60, p=0.5),
            A.ToGray(p=0.5),
        ], p=0.5),
        A.GaussNoise(var_limit=(50, 200)),
        A.ImageCompression(
            quality_lower=15,
            quality_upper=20,
            p=0.5,
        ),
        A.GaussianBlur(
            blur_limit=(3, 15),
            p=0.5,
        ),
        A.CoarseDropout(
            max_holes=10,
            max_height=100,
            max_width=100,
            min_holes=1,
            min_height=10,
            min_width=10,
            fill_value=(0, 0, 0),
            always_apply=True , # Ensures CoarseDropout is applied to every image
            p=0.2,
        ),
        ToTensorV2(),  # Convert to tensor at the end
    ])


def collate_fn(batch):
    if not batch:
        return {}
    
    new_batch = {}
    keys = batch[0][0].keys()
    values = list(zip(*[list(b[0].values()) for b in batch]))
    
    for i, k in enumerate(keys):
        value = values[i]
        if k == "img":
            value = torch.stack(value, 0)
        if k in {"bboxes", "cls"}:
            value = torch.cat(value, 0)
        new_batch[k] = value
    
    new_batch["batch_idx"] = torch.cat([torch.full([len(b[0]["cls"])], i) for i, b in enumerate(batch)]).to(batch[0][0]['img'].device)
    new_batch["cls"] = new_batch["cls"][:, None]
    new_batch["sample_infos"] = [b[1] for b in batch]
    return new_batch


class CustomObjectDetectionValidator(DetectionValidator):
    def __init__(self, *args, plot_callbackval=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_callbackval = plot_callbackval

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode='val'):
        parent_path, _ = os.path.split(dataset_path)
        dataset = PhysicalCellsObjectDetectionDataset(
            data_dir=parent_path,
            transform=self.get_transforms(),
            use_keypoint_clipping=False,
        )

        if rank == -1:  # Single GPU or CPU
            sampler = RandomSampler(dataset) if mode == "val" else None
        else:  # Distributed training
            sampler = DistributedSampler(dataset) if mode == "val" else None
        batch_size = batch_size // (2 if mode == "val" else 1)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size * 2,
            shuffle=(sampler is None),  # Shuffle if no sampler is used
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4  # Adjust according to your system
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