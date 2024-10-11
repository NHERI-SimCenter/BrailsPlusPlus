"""Class object to predict roof types from aerial imagery using CLIP."""
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
# Fei Pan
# Barbaros Cetiner
#
# Last updated:
# 10-11-2024

from typing import Optional
from brails.processors.vlm_image_classifier.CLIPClassifier import \
    CLIPClassifier


class RoofShapeVLM(CLIPClassifier):
    """
    RoofShapeVLM classifier attempts to predict roof shapes using CLIP model.

    Attributes__
        input_dict (Optional[dict]): A dictionary containing prompts and class
            information.
        text_prompts (list): A list of textual prompts for the classifier.
        classes (list): A list of roof shape classes.

    Methods__
        predict(ImageSet): Returns the predictions for the set of images
            provided.
    """

    def __init__(self, input_dict: Optional[dict] = None):
        """
        Initialize RoofShapeVLM classifier with specified prompts and classes.

        Args__
            input_dict (Optional[dict]): A dictionary containing the prompts
                and classes. If None, default prompts and classes are used.
        """
        super().__init__(task="roofshape", input_dict=input_dict)
        self.input_dict = input_dict
        if (self.input_dict is not None):
            self.text_prompts = self.input_dict['prompts']
            self.classes = self.input_dict['classes']
        else:
            self.text_prompts = [
                'Identify rooftops with a ridge running along the top',
                'flat roof, roof with one flat section',
                'hip roof'
            ]
            self.classes = ['Gable', 'Flat', 'Hip']

    # inherit method from CLIPClassifier
    # def predict(self, image: ImageSet):

    #     """
    #     The method predicts the roof shape.

    #     Args
    #         images: ImageSet The set of images for which a prediction is
    #           required

    #     Returns
    #         dict: The keys being the same keys used in ImageSet.images, the
    #           values being the predicted roof shape
    #     """

    #     return
