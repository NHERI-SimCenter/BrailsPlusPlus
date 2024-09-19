# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 The Regents of the University of California
#
# This file is part of BRAILS++
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
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 04-11-2024


from brails.processors.image_classifier.image_classifier import ImageClassifier
from brails.types.image_set import ImageSet
from typing import Optional, Dict

import torch
import os

class OccupancyClassifier(ImageClassifier):

    """
    The OccupancyClassifier classifier attempts to predict occupancy types of 
    buildings as 1 of 2 types: Residential or Other. The classification is done
    by the ImageClassifier class. This class is a wrapper that just sets up the
    inputs for that class.

    Variables
       model_path: str Path to the model file
    
    Methods:
       predict(ImageSet): To return the predictions for the set of images 
                          provided

    """
    
    def __init__(self, input_data: Optional[dict]=None): 
        
        """
        The class constructor sets up the path to the trained model file. If no
        model is provided, the class downloads a default from the web for this 
        and subsequent use.
        
        Args
            input_data: dict Optional. The init function looks for a 
            'model_path' key to set model_path.
        """
        
        # If input_data, provided check if it contains 'modelPath' key and 
        # also pass on to base class:
        if not input_data == None:
            super().__init__(input_data)
            model_path = input_data['modelPath']
        else:
            model_path = None

        # If no model_file is provided, use one previously downloaded or got 
        # get it if not existing:
    
        if model_path == None:
            os.makedirs('tmp/models',exist_ok=True)
            model_path = 'tmp/models/OccupancyClassifier_v1.pth'
            if not os.path.isfile(model_path):
                print('\n\nLoading default occupancy classifier model file to'+
                      ' tmp/models folder...')
                url = ('https://zenodo.org/record/7272099/files/'
                'trained_model_occupancy_v1.pth')
                torch.hub.download_url_to_file(url, model_path, progress=True)
                print('Default occupancy classifier model loaded')
            else: 
                print(f"\nDefault occupancy classifier model at {model_path}" +
                      ' loaded')
        else:
            print('\nInferences will be performed using the custom model at' +
                  f' {model_path}')
     
        self.model_path = model_path
        self.classes = ['Other','Residential']  
        
    def predict(self, images: ImageSet)->dict:

        """
        The method that predicts building occupancy type.
        
        Args
            images: ImageSet containing the set of images for which a 
                    prediction is required

        Returns
            dict: Dictionary with keys set to image names in ImageSet.images 
                  and values set to the predicted occupancy types
        """
        
        imageClassifier = ImageClassifier()
        return imageClassifier.predict(images, model_path = self.model_path, classes = self.classes)
        
    def retrain(self, dataDir, batchSize=8, nepochs=100, plotLoss=True):
        imageClassifier = ImageClassifier()
        imageClassifier.retrain(self.modelPath,dataDir,batchSize,nepochs,
                                plotLoss)

if __name__ == '__main__':
    pass