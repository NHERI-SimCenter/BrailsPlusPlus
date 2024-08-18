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


import os
import copy
import sys
import glob
from typing import Any, Callable, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from .grounded_sam_utils import run_on_one_image, build_models 
from brails.types.image_set import ImageSet
from typing import Optional, Dict      

class VLMSegmenter:  
    def __init__(self, input_dict: Optional[dict] = None):        
        """
        The class constructor sets up the path prompts or whatever.
        
        Args
            input_data: dict Optional. The init function looks into dict for values needed,
        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.grounding_dino_model, self.sam_predictor = build_models(self.device)
        if(input_dict!=None):
            self.classes = input_dict['classes']
        else:
            self.classes = ["terrain", "road", "tree", "door", "window", "facade"]
    def predict(self, images: ImageSet, modelPath=None, classes = None, output_dir = "tmp/images/street", visualize = True):

        """
        The method predicts the stuff 
        
        Args
            images: ImageSet The set of images for which a prediction is required

        Return
            saved_paths: paths to the saved masks
        """
        def isImage(im):
            return im.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

        assert modelPath==None or len(modelPath)==2, "both GroundingDINO and SAM checkpoint should be provided"
        self.classes = classes if classes != None else self.classes
        if(modelPath!=None):
            try:
                dino_state_dict = torch.load(modelPath[0])
                sam_state_dict = torch.load(modelPath[1])
                self.grounding_dino_model.load_state_dict(dino_state_dict).eval()
                self.sam_predictor.load_state_dict(sam_state_dict).eval()
            except:
                sys.exit('Error occurred during loading model. Please check your provided models align with model size during constructor call')        
        
        # Run the image through the segmentation model:
        if(not os.path.exists(output_dir)):
            os.makedirs(output_dir, exist_ok = True)

        saved_paths = {}
        data_dir = images.dir_path
        for idx, (key, im) in enumerate(images.images.items()):
            if isImage(im.filename):
                img_path = os.path.join(data_dir, im.filename)
                mask, mask_path = run_on_one_image(
                    img_path, output_dir, self.grounding_dino_model, self.sam_predictor, self.classes, visualize = visualize)
                saved_paths[key] = mask_path
                im.mask = mask

        print(f'mask saved at {output_dir} as png')
        return saved_paths

if __name__ == '__main__':
    pass

