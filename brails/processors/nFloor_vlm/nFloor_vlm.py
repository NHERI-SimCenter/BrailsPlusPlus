"""Class object to predict number of floors from street imagery using CLIP."""
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


class NFloorVLM(CLIPClassifier):
    """
    NFloorVLM classifier predicts number of floors in a building using CLIP.

    This class is designed to initialize the classifier with text prompts and
    classes related to the number of floors and make predictions from
    street-level imagery.

    Attributes__
        text_prompts (List[str]): A list of text prompts for floor
            classifications.
        classes (List[int]): A list of classes representing the number of
            floors (default [1, 2, 3])

    Args__
        input_dict (Optional[dict]): A dictionary containing prompts and
            classes for customization.
    """

    def __init__(self, input_dict: Optional[dict] = None):
        """
        Initialize the NFloorVLM classifier.

        Args__
            input_dict (Optional[dict]): A dictionary containing values needed
                for initialization, such as prompts and classes. If not
                provided, default prompts and classes will be used.
        """
        super().__init__(task="roofshape", input_dict=input_dict)
        self.input_dict = input_dict
        if (self.input_dict is not None):
            self.text_prompts = self.input_dict['prompts']
            self.classes = self.input_dict['classes']
        else:
            # each class should have equal amount of text prompts
            self.text_prompts = [
                'one story house', 'bungalow', 'flat house',
                'single-story side split house', 'two story house',
                'two story townhouse', 'side split house', 'raised ranch',
                'three story house', 'three story house', 'three story house',
                'three-decker'
            ]
            self.classes = [1, 2, 3]

    # inherit method from CLIPClassifier
    # def predict(self, image: ImageSet):

    #     """
    #     The method predicts the roof shape.

    #     Args
    #         images: ImageSet The set of images for which a prediction is
    #            required

    #     Returns
    #         dict: The keys being the same keys used in ImageSet.images, the
    #            values being the predicted roof shape
    #     """

    #     return
