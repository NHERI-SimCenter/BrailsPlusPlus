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


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import sys
import zipfile
import json
from .clip.clip import load, tokenize
from .clip.model import build_model
from .clip.utils import compute_similarity, aggregate_predictions
from brails.types.image_set import ImageSet
from typing import Optional, Dict


#import utils
class CLIPClassifier:
    def __init__(self, task, input_dict: Optional[dict] =None):
        #initializr model, predict, retrain() -> don't support train() (from scratch) for now
        if(input_dict != None):
            self.modelArch = input_dict['model_arch']
        else:
            self.modelArch = "ViT-B/32"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.preds = None
        self.batch_size = 20
        self.template = "a photo of a {}" 

    def predict(self, images: ImageSet, modelPath='tmp/models/VIT-B-32.pth', 
                classes=None, text_prompts = None):
        assert classes==None or (classes!=None and text_prompts!=None), 'customized classes provide customized prompts (text_prompts can not be None)'
        if(classes != None):
            assert len(text_prompts) % len(classes) == 0, "number of text prompts should be equal across classes (i.e. number of text prompts should be multiples of num classes)"
        if(classes!=None):
            self.classes, self.text_prompts = classes, text_prompts
        
        self.modelPath = modelPath
        
        def isImage(im):
            return im.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        
        #download CLIP model & load to device
        if(os.path.exists(modelPath)):
            model, data_transforms = load(modelPath, self.device, modelPath) #load model
        else:
            download_root = os.path.dirname(modelPath)
            os.makedirs(download_root, exist_ok = True)
            model, data_transforms = load(self.modelArch, self.device, download_root = download_root) #download model if not found
        model.eval()

        #tokenize text prompts
        text_input = torch.cat([tokenize(self.template.format(c)) for c in self.text_prompts]).to(self.device)
        prompts_per_class = len(self.text_prompts) // len(self.classes)
        
        preds = {}
        batch_imgs = []
        batch_keys = []
        data_dir = images.dir_path
        if not os.path.isdir(data_dir):
            print('ImageClassifier fails as ', data_dir, ' is not a directory')

        for idx, (key, im) in enumerate(images.images.items()):
            if isImage(im.filename):
                image = Image.open(os.path.join(data_dir, im.filename)).convert("RGB")
                image = data_transforms(image).float()
                image = image.to(self.device)
                image = image.unsqueeze(0)
                batch_imgs.append(image)
                batch_keys.append(key)  

            #batch processing (batch size = 20)
            if((idx != 0 and idx % self.batch_size == 0) or idx == len(images.images)-1):
                image_input = torch.cat(batch_imgs)
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    text_features = model.encode_text(text_input)
                #For k prompts/class, C classes, N images, compute matrix of dimension N x (kC)
                similarity = compute_similarity(image_features, text_features)
                #aggregate matrix of Nxkc into N x C matrix, then select the label with largest score as prediction
                batch_preds, _ = aggregate_predictions(similarity, agg_method = "max", gap = prompts_per_class)
                for (k, p) in zip(batch_keys, batch_preds):
                    preds[k]=self.classes[p.item()]
                
                batch_imgs, batch_keys = [], []
        return preds

if __name__ == '__main__':
    pass
