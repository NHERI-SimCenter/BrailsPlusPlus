import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import supervision as sv
import os
import glob
import pickle

import torch
import torchvision

import brails

from typing import List, Optional, Dict, Any
from .groundingdino.util.inference import Model
from .segment_anything import sam_model_registry, SamPredictor
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from dataclasses import dataclass


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


def verify_and_download_models(download_url, filepath):
    if (not os.path.isfile(filepath)):
        model_path = filepath
        print('Loading default segmentation model file to the pretrained folder...')
        torch.hub.download_url_to_file(download_url,
                                       model_path, progress=False)

# build GroundingDINO and SAM


def build_models(device="cuda:0"):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DEVICE FOUND: {DEVICE}')
    if (not os.path.exists('tmp/models')):
        os.makedirs('tmp/models')
    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = brails.__file__.replace(
        '__init__.py', '') + "processors/vlm_segmenter/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "tmp/models/groundingdino_swint_ogc.pth"
    GROUNDING_DINO_CHECKPOINT_URL = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'
    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "tmp/models/sam_vit_h_4b8939.pth"
    SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    verify_and_download_models(
        GROUNDING_DINO_CHECKPOINT_URL, GROUNDING_DINO_CHECKPOINT_PATH)
    verify_and_download_models(SAM_CHECKPOINT_URL, SAM_CHECKPOINT_PATH)

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=device)
    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](
        checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    return grounding_dino_model, sam_predictor

# Prompting SAM with detected boxes


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

# def combine_mask_by_label(labels, masks, label_options):
#     unique_labels = np.unique(label_options)
#     class_masks = np.zeros((len(unique_labels), masks.shape[-2], masks.shape[-1]))
#     for label, mask in zip(labels, masks):
#         mask = np.sum(np.array(mask), axis = 0)
#         class_idx = np.argwhere(unique_labels == label).item()
#         class_masks[class_idx] += mask
#         class_masks[class_idx] = np.clip(class_masks[class_idx], a_min = 0, a_max = 1)
#     return class_masks


def show_binary_mask(mask, ax, label_code):
    # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask = mask * label_code
    ax.imshow(mask, interpolation=None)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                 facecolor=(0, 0, 0, 0), lw=2))


def detect(image: Image.Image, labels: List[str], threshold: float = 0.3,
           detector_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(
        model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(
        ".") else label+"." for label in labels]

    results = object_detector(
        image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


def run_on_one_image(


    img_source, output_dir, grounding_dino_model, sam_predictor, CLASSES,
    BOX_THRESHOLD=0.35, TEXT_THRESHOLD=0.25, NMS_THRESHOLD=0.8, visualize=False
):
    SOURCE_IMAGE_PATH = img_source
    img_name = SOURCE_IMAGE_PATH.split("/")[-1][:-4]
    CLASS_TO_CODE = {curr_class: idx+1 for idx,
                     curr_class in enumerate(CLASSES)}

    # load image
    im = Image.open(SOURCE_IMAGE_PATH)
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections_raw = detect(im, CLASSES, BOX_THRESHOLD, None)
    xyxy = []
    confidence = []
    class_ids = []
    mask_labels = []
    classes = [cl + '.' if not cl.endswith('.') else cl for cl in CLASSES]
    for det in detections_raw:
        xyxy.append(det.box.xyxy)
        confidence.append(det.score)
        class_ids.append(classes.index(det.label))
        mask_labels.append(det.label[:-1])
    detections = sv.Detections.empty()
    detections.xyxy = np.array(xyxy, dtype=np.float32)
    detections.confidence = np.array(confidence, dtype=np.float32)
    detections.class_id = np.array(class_ids)

    """
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )
    mask_labels = [CLASSES[class_id] for _, _, _, class_id, _, _ in detections]
    """

    # NMS post process
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    output_dict = {"coord": detections.xyxy,
                   "confidence": detections.confidence,
                   "class": detections.class_id}

    # convert detections to masks
    masks = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    detections.mask = masks

    pixel_mask = np.zeros(image.shape[:2], dtype=int)
    # sort by area of mask in descending order
    sorted_idx = np.argsort(np.sum(masks, axis=(1, 2)))[::-1]
    for idx in sorted_idx:
        mask = masks[idx]
        class_label = mask_labels[idx]
        pixel_mask[mask] = CLASS_TO_CODE[class_label]

    output_mask = Image.fromarray(pixel_mask.astype(np.uint8))
    output_mask.save(os.path.join(output_dir, f"{img_name}_mask.png"))

    mask_path = os.path.join(output_dir, f"{img_name}_mask.obj")
    with open(mask_path, "wb") as fp:
        pickle.dump({"mask": pixel_mask, "code": CLASS_TO_CODE}, fp)
        fp.close()

    if (visualize):
        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(
            scene=image.copy(), detections=detections)
        cv2.imwrite(os.path.join(
            output_dir, f"{img_name}_mask_overlap.png"), annotated_image)
    return pixel_mask, mask_path
