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
# Fei Pan


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
from brails.types.image_set import ImageSet
from typing import Optional, Dict
from .utils import load_predictions_from_json, prompt_and_save_caption

class GPT:
    def __init__(self, api_key, input_dict: Optional[dict] =None):
        #initializr model, predict, retrain() -> don't support train() (from scratch) for now
        if(input_dict != None):
            self.modelArch = input_dict['model_arch']
            self.prompt_str = input_dict['prompt_str']
            self.classes = input_dict['classes']
            assert 'gpt' in self.modelArch
        else:
            self.modelArch = "gpt-4o"
        self.API_KEY = api_key
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.batchSize = None
        self.nepochs = None
        self.trainDataDir = None
        self.imgDir = None
        self.lossHistory = None
        self.preds = None

    def predict(self, images: ImageSet, prompt_str = None, classes=None, save_response = True):
        def isImage(im):
            return im.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        
        def valid_prompt(classes, prompts):
            if(classes == None and prompts == None):return True
            for gt_class in classes:
                if(gt_class not in prompts):
                    return False
            return True
            
        if((classes == None and prompt_str!=None) or (classes != None and prompt_str==None)):
            raise Exception('if text prompt is provided, classes are needed to convert gpt response to predicted labels')
        elif(not valid_prompt(classes, prompt_str)):
            raise Exception('text prompts must contain class labels to correctly map gpt response to predicted labels')
        elif(prompt_str!=None and classes!=None): #update variables if not none
            self.prompt_str, self.classes = prompt_str, classes

        data_dir = images.dir_path
        json_dir = os.path.join(data_dir, 'gpt_response')
        if not os.path.isdir(data_dir):
            print('GPT fails as ', data_dir, ' is not a directory')
        if(save_response and not os.path.exists(json_dir)):
            os.makedirs(json_dir)
        preds = {}
        for idx, (key, im) in enumerate(images.images.items()):
            if isImage(im.filename):
                image_path = os.path.join(data_dir, im.filename)
                output_file = None if not save_response else os.path.join(data_dir, 'gpt_response', im.filename.replace('png', 'json').replace('jpg', 'json'))
                if(not os.path.exists(output_file)):
                    response = prompt_and_save_caption(image_path, self.prompt_str, self.API_KEY, self.modelArch, output_file, max_tokens = 300)
                else:
                    response = json.load(open(output_file))
                batch_preds = load_predictions_from_json(response, img_keys = key, options = self.classes)
                preds.update(batch_preds)
        return preds


if __name__ == '__main__':
    pass
