import torch
from ultralytics.models.yolo.detect import DetectionValidator
from dataset import PhysicalCellsObjectDetectionDataset
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DistributedSampler, RandomSampler
from ultralytics.data.build import InfiniteDataLoader
import albumentations as A
import cv2
import numpy as np
from pathlib import Path
import matplotlib as mpl
import os
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils import ops
from typing import Callable, Dict, List, Optional, Union
from ultralytics.models.yolo.pose import PoseValidator
import math
def output_to_target(output, max_det=300):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets = []
    kpts_list = []
    for i, o in enumerate(output):
        # Extract boxes, confidence, class, and keypoints
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        # Extract keypoints if they exist (assuming 17 keypoints with x,y,conf for each)
        if o.shape[1] > 6:  # Check if keypoints exist
            kpts = o[:max_det, 6:].cpu()  # Get all keypoints
            kpts_list.append(kpts)
        else:
            kpts_list.append(torch.zeros((len(box), 23)))  # Default empty keypoints
            
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, ops.xyxy2xywh(box), conf), 1))
    
    targets = torch.cat(targets, 0).numpy()
    kpts = torch.cat(kpts_list, 0).numpy()
    
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1], kpts
def get_train_transforms():
    return A.Compose([
        A.Blur(p=0.1),
        ToTensorV2(),
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
        elif k in {"bboxes", "cls", "keypoints"}:
            value = torch.cat(value, 0)
        new_batch[k] = value

    new_batch["batch_idx"] = torch.cat(
        [torch.full([len(b[0]["cls"])], i) for i, b in enumerate(batch)]
    ).to(batch[0][0]['img'].device)
    new_batch["cls"] = new_batch["cls"][:, None]
    new_batch["sample_infos"] = [b[1] for b in batch]
    
    return new_batch

class CustomPoseValidator(PoseValidator):
    def __init__(self, *args, plot_callbackval=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_callbackval = plot_callbackval

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode='val'):
        parent_path, _ = os.path.split(dataset_path)
        dataset = PhysicalCellsObjectDetectionDataset(
            data_dir=parent_path,
            transform=self.get_transforms(),
        )
        
        sampler = RandomSampler(dataset) if rank == -1 else DistributedSampler(dataset)
        
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=8,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=24
        )

    def plot_val_samples(self, batch, ni):
        if self.plot_callbackval:
            self.plot_callbackval(self, batch, ni)
        else:
            super().plot_val_samples(batch, ni)
'''
    def plot_predictions(self, batch, preds, ni):
        val_dir = self.save_dir / f"val"
        val_dir.mkdir(exist_ok=True)
        e = (len(os.listdir(val_dir)) // 2) * self.args.val_interval
        batch_dir = val_dir / f"e{e}_b{ni}_pred"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(len(batch["img"])):
            individual_batch = {
                "img": batch["img"][i:i+1],
                "im_file": batch["im_file"][i:i+1],
            }
            
            # Get predictions including keypoints
            batch_idx, cls, boxes, confs, kpts = output_to_target(preds[i:i+1], max_det=self.args.max_det)
            
            img_with_predictions = plot_images_with_keypoints(
                individual_batch["img"],
                batch_idx,
                cls,
                boxes,
                confs,
                kpts=kpts,  # Pass keypoints
                paths=individual_batch["im_file"],
                fname=batch_dir / f"val_batch{ni}_img{i}_pred.jpg",
                names=self.names,
                on_plot=self.on_plot,
                save=False
            )
            
            if img_with_predictions is None:
                continue
            
            canvas = np.full((1800, 1800, 3), 255, dtype=np.uint8)
            h, w, _ = img_with_predictions.shape
            start_y = (canvas.shape[0] - h) // 2
            start_x = (canvas.shape[1] - w) // 2
            canvas[start_y:start_y+h, start_x:start_x+w] = img_with_predictions
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(batch_dir / f"val_batch{ni}_img{i}.jpg"), canvas_bgr)

from PIL import Image
def plot_images_with_keypoints(
    images: Union[torch.Tensor, np.ndarray],
    batch_idx: Union[torch.Tensor, np.ndarray],
    cls: Union[torch.Tensor, np.ndarray],
    bboxes: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.float32),
    confs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    masks: Union[torch.Tensor, np.ndarray] = np.zeros(0, dtype=np.uint8),
    kpts: Union[torch.Tensor, np.ndarray] = np.zeros((0, 22), dtype=np.float32),
    paths: Optional[List[str]] = None,
    fname: str = "images.jpg",
    names: Optional[Dict[int, str]] = None,
    on_plot: Optional[Callable] = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.7,
) -> Optional[np.ndarray]:
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    
    if images.shape[1] == 3:
        images = images[:, [2, 1, 0], :, :]

    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = int(np.ceil(bs ** 0.5))
    if np.max(images[0]) <= 1:
        images *= 255

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        mosaic[y:y + h, x:x + w, :] = images[i].transpose(1, 2, 0)

    scale = max_size / ns / max(h, w)
    if scale < 1:
        h, w = math.ceil(scale * h), math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, (int(w * ns), int(h * ns)))

    fs = int((h + w) * ns * 0.01)
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)

        if len(cls) > 0:
            idx = batch_idx == i
            boxes = bboxes[idx]
            conf = confs[idx] if confs is not None else None
            classes = cls[idx].astype("int")
            kpts_batch = kpts[idx] if kpts is not None and kpts.size > 0 else None

            if len(boxes):
                if conf is not None:
                    idx_sorted = np.argsort(conf)
                    boxes = boxes[idx_sorted]
                    classes = classes[idx_sorted]
                    if kpts_batch is not None:
                        kpts_batch = kpts_batch[idx_sorted]

                if boxes[:, :4].max() <= 5.1:
                    boxes[..., [0, 2]] *= w
                    boxes[..., [1, 3]] *= h
                elif scale < 1:
                    boxes[..., :4] *= scale
                
                boxes[..., 0] += x
                boxes[..., 1] += y
                boxes = ops.xywh2xyxy(boxes) if boxes.shape[-1] == 4 else ops.xywhr2xyxyxyxy(boxes)

                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    if conf is not None and conf[j] > conf_thres:
                        annotator.rectangle(box, outline=color, width=2)

         
                # Draw keypoints if available
                if kpts_batch is not None and len(kpts_batch):
                    if kpts_batch[..., 0].max() <= 3:
                        kpts_batch[..., 0] *= w
                        kpts_batch[..., 1] *= h
                    kpts_batch[..., 0] += x
                    kpts_batch[..., 1] += y

                    # Convert PIL image to numpy array for OpenCV operations
                    img_array = np.array(annotator.im)
                    
                    for j, kp_set in enumerate(kpts_batch):
                        if conf is not None and conf[j] <= conf_thres:
                            continue
                            
                        for kp_idx in range(0, len(kp_set), 3):
                            x_kp, y_kp, conf_kp = (
                                int(kp_set[kp_idx]),
                                int(kp_set[kp_idx + 1]),
                                kp_set[kp_idx + 2]
                            )
                            # Draw only keypoints with confidence > 0.7
                            if conf_kp > 0.3:  
                                color = tuple(map(int, colors(kp_idx // 3)))
                                cv2.circle(
                                    img_array,
                                    (x_kp, y_kp),
                                    radius=3,
                                    color=color,
                                    thickness=-1
                                )
                    
                    # Convert back to PIL Image
                    annotator.im = Image.fromarray(img_array)


    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)
    if on_plot:
        on_plot(fname)
    return np.asarray(annotator.im)
'''
