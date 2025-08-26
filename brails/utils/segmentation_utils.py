# Copyright (c) 2024 The Regents of the University of California
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
# Barbaros Cetiner
#
# Last updated:
# 08-19-2025

"""
This module provides utilities for visualizing segmentation results.

.. autosummary::

      SegmentationUtils
"""

import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


class SegmentationUtils:
    """
    Utility class for creating visualizations of segmentation masks on images.

    This class provides static methods to overlay a segmentation mask onto a
    base image and to convert class-labeled masks to RGB images.

    Please use the following statement to import :class:`SegmentationUtils`.

    .. code-block:: python

        from brails.utils import SegmentationUtils
    """

    @staticmethod
    def create_overlaid_image(
        base_image: Union[str, Path],
        mask_image: np.ndarray,
        output_dir: Union[str, Path]
    ) -> None:
        """
        Overlay a segmentation mask on a base image and save the result.

        The segmentation mask is blended with the original image using a fixed
        transparency (alpha) to produce a visually interpretable output.

        Args:
            base_image (str or Path):
                Path to the original image file.
            mask_image (np.ndarray):
                A 2D array of class labels representing the segmentation mask.
            output_dir (str or Path):
                Directory where the overlaid image will be saved. Will be
                created if it does not exist.

        Returns:
            None

        Example:
            This example assumes that a valid image file named
            ``example_image.jpg`` exists in the current working directory.

            >>> import numpy as np
            >>> from brails.utils import SegmentationUtils
            >>> mask = np.zeros((100, 100), dtype=int)
            >>> mask[25:75, 25:75] = 1
            >>> SegmentationUtils.create_overlaid_image(
            ...     base_image='example_image.jpg',
            ...     mask_image=mask,
            ...     output_dir='overlaid_images'
            ... )
        """
        mask_rgb = SegmentationUtils.decode_segmap(mask_image, nc=4)

        image = Image.open(base_image)  # Replace with your image file
        base_image_np = np.array(image)

        mask_image_resized = np.array(Image.fromarray(mask_rgb))

        # Define alpha (transparency) for blending the mask with the base image
        alpha = 0.5

        # Create an overlay image by blending the mask and base image
        # Convert mask and base image to float for blending:
        base_image_float = base_image_np.astype(float)
        mask_image_float = mask_image_resized.astype(float)

        # Blend images
        combined_image_float = (alpha * mask_image_float +
                                (1 - alpha) * base_image_float)

        # Clip values to [0, 255] and convert back to uint8:
        combined_image_np = np.clip(
            combined_image_float, 0, 255).astype(np.uint8)

        # Convert back to PIL Image and save:
        imname = base_image.split('/')[-1]
        ext = imname.split('.')[-1]
        imout_name = imname.replace(ext, '') + '_segmented.' + ext
        os.makedirs(output_dir, exist_ok=True)
        path = Path(output_dir)

        combined_image_pil = Image.fromarray(combined_image_np)
        combined_image_pil.save(path / imout_name)

    @staticmethod
    def decode_segmap(image: np.ndarray, nc: int = 5) -> np.ndarray:
        """
        Convert a class-labeled mask to an RGB image.

        Args:
            image (np.ndarray):
                2D array of class indices representing the mask.
            nc (int, optional):
                Number of classes. Default is 5.

        Returns:
            np.ndarray:
                RGB image (H, W, 3) representation of the mask.

        Example:
            >>> import numpy as np
            >>> from brails.utils import SegmentationUtils
            >>> mask = np.zeros((50, 50), dtype=int)
            >>> mask[10:30, 10:30] = 2
            >>> rgb_mask = SegmentationUtils.decode_segmap(mask)
            >>> rgb_mask.shape
            (50, 50, 3)
        """
        label_colors = np.array([
            (0, 0, 0),        # background
            (255, 0, 0),      # roof
            (255, 165, 0),    # facade
            (0, 0, 255),      # window
            (175, 238, 238)   # door
        ])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for label in range(nc):
            idx = image == label
            r[idx] = label_colors[label, 0]
            g[idx] = label_colors[label, 1]
            b[idx] = label_colors[label, 2]

        return np.stack([r, g, b], axis=2)
