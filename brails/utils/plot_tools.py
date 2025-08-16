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
# 08-16-2025

"""
This module defines utilities for creating visually-appealing figures.

.. autosummary::

    PlotTools
"""

import math
import os
import random
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
from PIL import Image

from brails.types.image_set import ImageSet


class PlotTools:
    """
    A utility class for generating visually-appealing image grids.

    The :class:`PlotTools` class provides static methods to visualize model
    predictions alongside images, enabling intuitive and structured displays
    for both qualitative evaluation and presentation purposes. It is designed
    to support image sets managed through the :class:`ImageSet` class and
    prediction outputs from classification models.

    To import the :class:`PlotTools` class, use:

    .. code-block:: python

        from brails.utils import PlotTools
    """

    @staticmethod
    def plot_images(
        imageset: ImageSet,
        rows: int = 2,
        cols: int = 3,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Plot a random subset of images from an image set in a grid layout.

        Args:
            imageset (ImageSet):
                A BRAILS ``ImageSet`` object containing the images to be
                displayed.
            rows (int, optional):
                Number of rows in the display grid. Defaults to 2.
            cols (int, optional):
                Number of columns in the display grid. Defaults to 3.
            random_seed (Optional[int], optional):
                Seed for reproducible random sampling. Defaults to None.

        Returns:
            None: This function only displays the images using ``matplotlib``.

        Raises:
            ValueError:
                If the number of available images is smaller than
                ``rows * cols``.

        Example:
        >>> google_satellite = importer.get_class('GoogleSatellite')()
        >>> images_satellite = google_satellite.get_images(
        ...     small_inventory,
        ...     'tmp/satellite/'
        ... )
        >>> PlotTools.plot_images(
        ...     images_satellite,
        ...     rows=3,
        ...     cols=4,
        ...     random_seed=42
        ... )
        # Output:
        # A matplotlib figure is displayed with a 2x3 grid of randomly
        # selected images from the ImageSet. Each subplot shows one image
        # with its filename as the title.
        """
        images_list = [os.path.join(imageset.dir_path, im.filename)
                       for im in imageset.images.values()]

        if random_seed is not None:
            random.seed(random_seed)

        n_images = rows * cols
        if len(images_list) < n_images:
            raise ValueError(
                f'Not enough images. Need at least {n_images}, got '
                '{len(images_list)}.'
            )

        # Randomly sample images:
        selected_images = random.sample(images_list, n_images)

        # Create the plot:
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = axes.flatten()

        for ax, img_path in zip(axes, selected_images):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(os.path.basename(img_path), fontsize=8)
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f'Error: {type(e).__name__}\n{os.path.basename(img_path)}',
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray")
                ax.axis("off")

        # Hide any unused axes:
        for ax in axes[len(selected_images):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_predictions(
        images: ImageSet,
        predictions: Dict[str, Union[int, float, str]],
        attribute_name: str,
        num_samples: Union[int, str] = 'all',
        crop_image: bool = True
    ) -> None:
        """
        Display a set of images along with their corresponding predictions.

        This method randomly samples a specified number of images from the
        provided ``images`` object, optionally crops the images, resizes them,
        and then displays them in a grid with the associated predictions.

        Args:
            images (ImageSet):
                A custom object that holds image paths and metadata. It should
                have an attribute ``images`` (a list of image objects) and
                ``dir_path`` (a string representing the directory path where
                images are located).
            predictions (dict):
                A dictionary where the keys are image identifiers (e.g., file
                names or unique keys) and the values are the predicted labels
                (e.g., binary or categorical values).
            attribute_name (str):
                The name of the attribute being predicted, which is displayed
                in the title for each image.
            num_samples (int or str, optional)
                The number of images to display. If set to ``'all'``, all
                images in the set will be displayed. Defaults to ``'all'``.
            crop_image (bool, optional):
                If set to ``True``, the images will be cropped (top 1/6 and
                bottom 1/4 removed). If set to ``False``, the images will be
                displayed without cropping. Defaults to ``True``.

        Returns:
            None
                This method directly displays the plot of images and their
                predictions, and does not return any value.

        Notes:
        - The images are resized to fit the grid, adjusting for the maximum
          image height.
        - Axes are hidden for a cleaner presentation, and the prediction value
          is displayed as the title for each image.
        - The method handles both cropping and resizing while maintaining the
          image aspect ratio.

        Example:
            PlotTools.show_predictions(
                images=my_image_set,
                predictions=my_predictions,
                attribute_name="Class",
                num_samples=5,
                crop_image=True,
            )
        """
        images_files = {}
        for key in images.images:
            images_files[key] = images.dir_path + \
                '/' + images.images[key].filename

        if isinstance(num_samples, str) and num_samples.lower() == 'all':
            num_samples = len(images_files)
        else:
            num_samples = min(num_samples, len(images_files))

        sampled_keys = random.sample(list(images_files.keys()), num_samples)

        # Create a panel of plots
        nrows = math.ceil(len(sampled_keys) / 4)
        fig, axes = plt.subplots(nrows, 4, figsize=(3 * 3, 4 * nrows))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Adjust layout
        fig.tight_layout(pad=1)

        image_dimensions = {}
        max_height = 0

        is_str = (0 in predictions.values() and 1 in predictions.values()) or \
            all(isinstance(value, str) for value in predictions.values())

        for key in sampled_keys:
            # Get the image path dynamically:
            image_path = images_files[key]

            # Get the dimensions of the image:
            with Image.open(image_path) as img:
                width, height = img.size
                image_dimensions[key] = (width, height)

            # Update the max height in one line
            max_height = max(max_height, height)

        # Plot each image with a caption:
        for i, key in enumerate(sampled_keys):
            img = Image.open(images_files[key])  # Open the image

            if crop_image:
                top_crop = height // 6
                bottom_crop = height - height // 4
                cropped_img = img.crop((0, top_crop, width, bottom_crop))
            else:
                cropped_img = img

            # Resize the cropped image to the max_height while keeping aspect
            # ratio:
            new_height = max_height
            new_width = int((new_height / cropped_img.height)
                            * cropped_img.width)  # Maintain aspect ratio
            resized_img = cropped_img.resize((new_width, new_height))

            axes[i].imshow(resized_img)
            value = predictions[key]
            if is_str:
                value = "No" if value == 0 else "Yes" if value == 1 else value
            else:
                value = str(value)
            axes[i].set_title(attribute_name + ': ' + value,
                              fontsize=10)  # Set the caption
            axes[i].axis('off')  # Hide axes for cleaner presentation

        # Hide unused subplots:
        for j in range(len(sampled_keys), len(axes)):
            axes[j].axis('off')

        plt.show()
