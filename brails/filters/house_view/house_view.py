# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 The Regents of the University of California
#
# This file is part of BRAILS.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Brian Wang

# minor minor mods: fmk

import brails.types.image_set as brails_image_set
from brails.filters.filter import Filter

import torch
import numpy as np
import os
from PIL import Image

from pathlib import Path
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import supervision as sv

@dataclass
class FilterBoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class FilterDetectionResult:
    score: float
    label: str
    box: FilterBoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'FilterDetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=FilterBoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

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
    results = [FilterDetectionResult.from_dict(result) for result in results]

    return results



class HouseView(Filter):


  def __init__(self, input_data: dict):
    
    self.text_prompt = "single house in middle of image without frontview occlusion"
    self.box_treshhold = 0.35
    self.text_treshhold = 0.25


  def _bound_one_image(self, IMAGE_PATH, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, model, device):    
    '''
      Performs cropping of target house on one image(not sure which function can better restructure into pipeline)
      '''

    # detect objects
    image = Image.open(IMAGE_PATH)
    detections_raw = detect(image, [TEXT_PROMPT], BOX_TRESHOLD, None)
    xyxy = []
    confidence = []
    class_ids = []
    labels = []
    for det in detections_raw:
        xyxy.append(det.box.xyxy)
        confidence.append(det.score)
        labels.append(det.label[:-1])
    detections = sv.Detections.empty()
    detections.xyxy = np.array(xyxy, dtype=np.float32)
    detections.confidence = np.array(confidence, dtype=np.float32)

    tgt = {
      "img_name": IMAGE_PATH.split('/')[-1],
      "img_source":Image.open(IMAGE_PATH),
      "boxes": torch.Tensor(xyxy),
      "labels": labels
    }
    return tgt

  def _crop_and_save_img(self, tgt, output_dir, random = False):    
    '''
      Given cropping information from bound_one_image, perform cropping and save cropped image
      Inputs
      - tgt: dictionary from bound_one_image, that stores img-related info and bounding boxes of houses
      - output_dir: target folder to save image
      '''
    
    boxes, labels = tgt["boxes"], tgt["labels"]
    img_name, img = tgt['img_name'], tgt['img_source']
    W, H = img.size
    
    assert len(boxes) == len(labels), "boxes and labels must have same length"
    if(len(boxes) == 0): #no boxes because boxes_logits < threshold
      print(f'{img_name} has no boxes')
      return False, (img_name, len(boxes))
    
    # draw boxes and masks
    if(len(boxes) > 1 and not random):
      box_areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes] #choose the house with largest foreground area
      box_idx = np.argmax(box_areas)
    else:
      box_idx = np.random.randint(len(boxes))
      
    box, label = boxes[box_idx], labels[box_idx]
    # from 0..1 to 0..W, 0..H
    # box = box * torch.Tensor([W, H, W, H])
    # # from xywh to xyxy
    # box[:2] -= box[2:] / 2 #box center = (box[0] + w/2, box[1] + h/2)
    # box[2:] += box[:2] #bot_right = (x0 + w, y0 + h)
    # draw
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    
    #get more background for house
    x0, y0 = max(1, x0-40), max(1, y0-40)
    x1, y1 = min(W-1, x1+40), min(H-1, y1+40)
    
    crop = img.crop((x0, y0, x1, y1))
    crop.save(os.path.join(output_dir, img_name), 'PNG')
      
    return True, (img_name, len(boxes))

  def filter1(self, image_path,  output_dir):
    
    model = load_model(self.CONFIG_PATH, self.WEIGHTS_PATH)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    crop_dict = self._bound_one_image(image_path, self.text_prompt, self.box_treshhold, self.text_treshhold, model, device)
    self._crop_and_save_img(crop_dict, output_dir, random = False)

  def filter(self, input_images: brails_image_set.ImageSet,  output_dir: str):

    
    def isImage(im):
      return im.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    
    #
    # ensure consistance in dir_path, i.e remove ending / if given and make directory
    #
    
    dir_path = Path(output_dir)
    os.makedirs(f'{dir_path}',exist_ok=True)

    #
    # filter and create image set
    #

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    output_images = brails_image_set.ImageSet()
    output_images.dir_path = dir_path    

    input_dir = input_images.dir_path
    for key, im in input_images.images.items():
      print(key,im)
      if isImage(im.filename):
          
        image = os.path.join(input_dir, im.filename)
        print(image)        
        
        # eventually do in parallel
        #batch_images.append(image)
        #batch_keys.append(key)
        #batch_features.append(im.features)
        crop_dict = self._bound_one_image(image, self.text_prompt, self.box_treshhold, self.text_treshhold, model = None, device = device)
        self._crop_and_save_img(crop_dict, output_dir, random = False)
        img = brails_image_set.Image(im.filename, im.properties)
        output_images.add_image(key, img)

    return output_images

      

