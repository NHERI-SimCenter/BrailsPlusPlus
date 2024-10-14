"""Class object to use and retrain the roof type classifier model."""
#
# Copyright (c) 2024 The Regents of the University of California
#
# This file is part of BRAILS++.
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
# Barbaros Cetiner
# Last updated:
# 10-11-2024

import os
from typing import Optional
import torch
from brails.processors.image_classifier.image_classifier import ImageClassifier
from brails.types.image_set import ImageSet


class RoofShapeClassifier(ImageClassifier):
    """
    RoofShapeClassifier predicts roof shapes from aerial imagery.

    This classifier serves as a wrapper around the ImageClassifier class,
    setting up the necessary inputs for roof type classification.

    Attributes__
        model_path (str): Path to the model file.
        classes (List[str]): List of roof type classes that will be predicted.

    Methods__
        predict(images: ImageSet) -> Dict[str, str]:
            Returns predictions for the set of images provided.
        retrain(data_dir, batch_size=8, nepochs=100, plot_loss=True)
            Retrains the roof type classifier model with the provided data.
    """

    def __init__(self, input_data: Optional[dict] = None):
        """
        Initialize the RoofShapeClassifier.

        If no model path is provided in input_data, the class downloads a
        default model. If prediction classes are not provided om input_data,
        roof type precitions are made for flat, galble, and hip roofs.

        Args__
            input_data (Optional[dict]): A dictionary containing initialization
            parameters.
        """
        # If input_data is provided, check if it contains 'modelPath' key:

        if input_data is not None:
            super().__init__(input_data)
            model_path = input_data['modelPath']
        else:
            model_path = None

        # if no model_file provided, use the one previously downloaded or
        # download it if not existing:
        if model_path is None:
            os.makedirs('tmp/models', exist_ok=True)
            model_path = 'tmp/models/roofTypeClassifier_v1.pth'
            if not os.path.isfile(model_path):
                print(
                    '\n\nLoading default roof classifier model file to '
                    'tmp/models folder...')
                torch.hub.download_url_to_file(
                    'https://zenodo.org/record/7271554/files/'
                    'trained_model_rooftype.pth',
                    model_path,
                    progress=True)
                print('Default roof classifier model loaded')
            else:
                print(
                    f"\nDefault roof classifier model in {model_path} loaded")
        else:
            print(
                '\nInferences will be performed using the custom model in'
                f' {model_path}')

        self.model_path = model_path
        self.classes = ['Flat', 'Gable', 'Hip']

    def predict(self, images: ImageSet) -> dict:
        """
        Predict the roof shape for the given images.

        Args__
            images (ImageSet): The set of images for which predictions are
                required.

        Returns__
            dict: A dictionary mapping image keys to their predicted roof
                shapes. The keys correspond to the same keys used in
                ImageSet.images, and the values are the predicted roof shape
                (Flat, Gable, or Hip).
        """
        imageClassifier = ImageClassifier()
        return imageClassifier.predict(images,
                                       model_path=self.model_path,
                                       classes=self.classes)

    def retrain(self, data_dir, batch_size=8, nepochs=100, plot_loss=True):
        """
        Retrains the roof type classifier model with the provided data.

        Args__
            data_dir (str): The directory containing training data.
            batch_size (int): The number of samples per batch. Default is 8.
            nepochs (int): The number of epochs for training. Default is 100.
            plot_loss (bool): Whether to plot the loss during training. Default
                is True.
        """
        imageClassifier = ImageClassifier()
        imageClassifier.retrain(self.model_path,
                                data_dir,
                                batch_size,
                                nepochs,
                                plot_loss)


if __name__ == '__main__':
    pass
