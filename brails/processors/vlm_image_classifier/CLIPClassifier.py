"""Class object to call CLIP for image classification."""
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
# Brian Wang
# Barbaros Cetiner
#
# Last updated:
# 10-11-2024


import torch
import os
from PIL import Image
from .clip.clip import load, tokenize
from .clip.utils import compute_similarity, aggregate_predictions
from brails.types.image_set import ImageSet
from typing import Optional, List, Dict


class CLIPClassifier:
    """
    A classifier that utilizes CLIP model to predict attributes from images.

    This class is designed to load a CLIP model, process input images,
    and make predictions for the entered textual prompts. It supports
    customizable classes and prompts to enhance prediction accuracy.

    Attributes__
        model_arch (str): The architecture of the model to be used. Available
            model architectures are 'ViT-B/32' (default), RN50', 'RN101',
            'RN50x4','RN50x16', 'RN50x64', 'ViT-B/16', 'ViT-L/14', and
            'ViT-L/14@336px'.
        device (torch.device): The device (CPU or GPU) used for computations.
        preds (Optional[Dict[str, str]]): A dictionary to hold predictions.
        batch_size (int): The number of images processed in a single batch.
        template (str): A template for formatting text prompts.

    Args__
        task (str): The task for which the classifier is being used.
        input_dict (Optional[dict]): A dictionary containing model architecture
            and other configuration parameters.
    """

    def __init__(self, task: str, input_dict: Optional[dict] = None):
        """
        Initialize CLIPClassifier.

        Args__
            task (str): The task for which the classifier is being used.
            input_dict (Optional[dict]): A dictionary containing model
                architecture and other configuration parameters.
        """
        if (input_dict is not None):
            self.model_arch = input_dict['model_arch']
        else:
            self.model_arch = "ViT-B/32"

        # Set default model path based on model architecture:
        self.model_path = f'tmp/models/{self.model_arch.replace("/", "-")}.pth'

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.preds = None
        self.batch_size = 20
        self.template = "a photo of a {}"

    def predict(self,
                images: ImageSet,
                model_path: str = '',
                classes: Optional[List[str]] = None,
                text_prompts: Optional[List[str]] = None
                ) -> Dict[str, str]:
        """
        Predicts classes for the given images using the CLIP model.

        Args__
            images (ImageSet): An object containing the images to classify.
            model_path (str): The path to the pre-trained model.
            classes (Optional[List[str]]): A list of class names.
            text_prompts (Optional[List[str]]): A list of text prompts
                corresponding to the classes.

        Returns__
            Dict[str, str]: A dictionary mapping image keys to their predicted
                classes.

        Raises__
            AssertionError: If the conditions on classes and text_prompts are
                not met.
        """
        assert (
            classes is None or
            (classes is not None and text_prompts is None)
        ), 'Customized classes provide customized prompts (text_prompts cannot'
        ' be None)'
        if (classes is not None):
            assert (
                len(text_prompts) % len(classes) == 0
            ), 'Number of text prompts should be equal across classes (i.e.,'
            ' the number of text prompts should be a multiple of the number'
            ' of classes)'

        if (classes is not None):
            self.classes, self.text_prompts = classes, text_prompts

        if model_path:
            self.model_path = model_path
        else:
            model_path = self.model_path

        def isImage(im: str) -> bool:
            return im.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

        # Download CLIP model & load to device:
        if (os.path.exists(model_path)):
            model, data_transforms = load(
                model_path, self.device, model_path)  # load model
        else:
            download_root = os.path.dirname(model_path)
            os.makedirs(download_root, exist_ok=True)
            # Download model if not found:
            model, data_transforms = load(
                self.model_arch, self.device, download_root=download_root)
        model.eval()

        # Tokenize text prompts:
        text_input = torch.cat([tokenize(self.template.format(c))
                               for c in self.text_prompts]).to(self.device)
        prompts_per_class = len(self.text_prompts) // len(self.classes)

        preds = {}
        batch_imgs = []
        batch_keys = []
        data_dir = images.dir_path
        if not os.path.isdir(data_dir):
            print('ImageClassifier failed as ', data_dir, ' is not a valid '
                  'directory')

        for idx, (key, im) in enumerate(images.images.items()):
            if isImage(im.filename):
                image = Image.open(os.path.join(
                    data_dir, im.filename)).convert("RGB")
                image = data_transforms(image).float()
                image = image.to(self.device)
                image = image.unsqueeze(0)
                batch_imgs.append(image)
                batch_keys.append(key)

            # Batch processing (batch size = 20):
            if ((idx != 0 and idx % self.batch_size == 0) or
                    idx == len(images.images)-1):
                image_input = torch.cat(batch_imgs)
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    text_features = model.encode_text(text_input)
                # Compute similarity matrix. For k prompts/class, C classes,
                # N images, compute matrix of dimension N x (kC):
                similarity = compute_similarity(image_features, text_features)

                # Aggregate matrix of Nxkc into N x C matrix, then select the
                # label with largest score as prediction:
                batch_preds, _ = aggregate_predictions(
                    similarity, agg_method="max", gap=prompts_per_class)
                for (k, p) in zip(batch_keys, batch_preds):
                    preds[k] = self.classes[p.item()]

                batch_imgs, batch_keys = [], []
        return preds


if __name__ == '__main__':
    pass
