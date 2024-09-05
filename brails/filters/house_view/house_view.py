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

from brails.types.image_set import ImageSet
from brails.filters.filter import Filter

import torch
import numpy as np
import os
import groundingdino
from PIL import Image
from .groundingdino.util.inference import load_model, load_image, predict
from pathlib import Path


class HouseView(Filter):

    def __init__(self, input_data: dict):

        self.text_prompt = "single house in middle of image without frontview occlusion"
        self.box_treshhold = 0.35
        self.text_treshhold = 0.25

        self.WEIGHTS_PATH = "./tmp/groundingdino_swint_ogc.pth"
        # self.CONFIG_PATH = os.path.join(os.path.abspath(__file__), "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        # path_groundingdino = os.path.dirname(groundingdino.__file__)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.CONFIG_PATH = os.path.join(curr_dir, 'groundingdino/config/GroundingDINO_SwinT_OGC.py')
        self.verify_and_download_models()

    def verify_and_download_models(self):
        GROUNDING_DINO_CHECKPOINT_PATH = "tmp/groundingdino_swint_ogc.pth"
        GROUNDING_DINO_CHECKPOINT_URL = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'
        if (not os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH)):
            print('Loading houseview filter model checkpoint...')
            torch.hub.download_url_to_file(GROUNDING_DINO_CHECKPOINT_URL,
                                        GROUNDING_DINO_CHECKPOINT_PATH, progress=False)

    def _bound_multiple_images(self, IMAGE_PATH_LIST, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, model, device):
        '''
          Method to get house bounding boxes for a batch of images
          Inputs
          - IMAGE_PATH_LIST: path to images
          - TEXT_PROMPT: text prompt related to target object 
          - BOX_THRESHOLD / TEXT_THRESHOLD: threshold to reject/accept target bounding box proposals
          '''

        image_list = []
        for IMAGE_PATH in IMAGE_PATH_LIST:
            image_source, image = load_image(IMAGE_PATH)
            image_list.append(image)
        image_list = torch.stack(image_list).to(torch.device("cuda:0"))
        #   print(f'image_list shape = {image_list.shape}, type = {type(image_list)}')

        tgt_list = []
        for i, image in enumerate(image_list):
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                device=device
            )
        labels = [f"{phrase} {logit:.2f}" for phrase,
                  logit in zip(phrases, logits)]
        tgt = {
            "img_name": IMAGE_PATH_LIST[i].split("/")[-1],
            "img_source": Image.open(IMAGE_PATH_LIST[i]),
            "boxes": boxes,
            "labels": labels
        }
        tgt_list.append(tgt)
        return tgt_list

    def _bound_one_image(self, IMAGE_PATH, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, model, device):
        '''
          Same functionality as above method, but performs on one image(not sure which function can better restructure into pipeline)
          '''

        image_source, image = load_image(IMAGE_PATH)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=device
        )

        labels = [f"{phrase} {logit:.2f}" for phrase,
                  logit in zip(phrases, logits)]
        img_path = []
        tgt = {
            "img_name": IMAGE_PATH.split('/')[-1],
            "img_source": Image.open(IMAGE_PATH),
            "boxes": boxes,
            "labels": labels
        }
        return tgt

    def _crop_and_save_img(self, tgt, output_dir, random=False):
        '''
          Given cropping information from bound_one_image, perform cropping and save cropped image
          Inputs
          - tgt: dictionary from bound_one_image, that stores img-related info and bounding boxes of houses
          - output_dir: target folder to save image
          '''

        boxes, labels = tgt["boxes"], tgt["labels"]
        img_name, img = tgt['img_name'], tgt['img_source']
        W, H = img.size

        assert len(boxes) == len(
            labels), "boxes and labels must have same length"
        if (len(boxes) == 0):  # no boxes because boxes_logits < threshold
            print(f'{img_name} has no boxes')
            return False, (img_name, len(boxes))

        # draw boxes and masks
        if (len(boxes) > 1 and not random):
            # choose the house with largest foreground area
            box_areas = [box[2] * box[3] for box in boxes]
            box_idx = np.argmax(box_areas)
        else:
            box_idx = np.random.randint(len(boxes))

        box, label = boxes[box_idx], labels[box_idx]
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2  # box center = (box[0] + w/2, box[1] + h/2)
        box[2:] += box[:2]  # bot_right = (x0 + w, y0 + h)
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # get more background for house
        x0, y0 = max(1, x0-40), max(1, y0-40)
        x1, y1 = min(W-1, x1+40), min(H-1, y1+40)

        crop = img.crop((x0, y0, x1, y1))
        crop.save(os.path.join(output_dir, img_name), 'PNG')

        return True, (img_name, len(boxes))

    def filter1(self, image_path,  output_dir):

        model = load_model(self.CONFIG_PATH, self.WEIGHTS_PATH)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        crop_dict = self._bound_one_image(
            image_path, self.text_prompt, self.box_treshhold, self.text_treshhold, model, device)
        self._crop_and_save_img(crop_dict, output_dir, random=False)

    def filter(self, input_images: ImageSet,  output_dir: str):

        def isImage(im):
            return im.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

        #
        # ensure consistance in dir_path, i.e remove ending / if given and make directory
        #

        dir_path = Path(output_dir)
        os.makedirs(f'{dir_path}', exist_ok=True)

        #
        # filter and create image set
        #

        model = load_model(self.CONFIG_PATH, self.WEIGHTS_PATH)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        output_images = ImageSet()
        output_images.dir_path = dir_path

        input_dir = input_images.dir_path
        for key, im in input_images.images.items():
            print(key, im)
            if isImage(im.filename):
                image = os.path.join(input_dir, im.filename)
                print(image)

                # eventually do in parallel
                # batch_images.append(image)
                # batch_keys.append(key)
                # batch_features.append(im.features)
                crop_dict = self._bound_one_image(
                    image, self.text_prompt, self.box_treshhold, self.text_treshhold, model, device)
                self._crop_and_save_img(crop_dict, output_dir, random=False)
                output_images.add_image(key, im, im.properties)

        return output_images
