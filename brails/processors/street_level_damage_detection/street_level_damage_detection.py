"""Class object to predict building damage from street imagery using CLIP."""
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
#
# Last updated:
# 02-05-2025

from brails.processors.vlm_image_classifier.CLIPClassifier import \
    CLIPClassifier
from brails.types.image_set import ImageSet


class DamageDetection_StreetLevel(CLIPClassifier):
    """
    A classifier that determines if a building is damaged using CLIP.

    This class initializes the classifier with text prompts and predefined
    damage-related classes to make predictions from street-level imagery.

    Attributes:
        text_prompts (list[str]):
            A list of text prompts used for damage classification.
        classes (list[str]):
            A list of class labels representing the damage states.
    """

    def __init__(self, input_dict: dict):
        """
        Initialize the DamageDetection_StreetLevel classifier.

        Args:
            input_dict (dict):
                A dictionary containing initialization parameters,
                such as model architecture.
        """
        super().__init__(input_dict=input_dict)
        self.input_dict = input_dict

    def predict(self, images: ImageSet):
        """
        Predicts whether buildings in the given images are damaged using CLIP.

        This method sets predefined damage classification classes and their
        corresponding text prompts before calling the parent class' `predict`
        method.

        Args:
            images (ImageSet):
                An object containing the images to classify.

        Returns:
            dict[str, str]:
                A dictionary mapping image keys to their predicted damage state
                ('Damaged' or 'Not damaged').

        Raises:
            TypeError or ValueError:
                If the conditions on classes and text_prompts are not met in
                the parent method.
        """
        self.classes = ['Damaged', 'Not damaged']
        self.text_prompts = ['building that is damaged',
                             'building that is not damaged']
        predictions = super().predict(images, classes=self.classes,
                                      text_prompts=self.text_prompts)

        return predictions
