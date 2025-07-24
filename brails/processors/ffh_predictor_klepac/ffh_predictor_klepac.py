# Copyright (c) 2025 The Regents of the University of California
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
# Steven Klepac
# Arthriya Subgranon
# Barbaros Cetiner
#
# Last updated:
# 07-24-2025

"""
This module defines a class for retrieving tunnel data from NTI.

.. autosummary::

    FFHPredictorKlepac
"""

import os
import pickle
import time
from typing import Optional, Dict

import torch
import cv2
import numpy as np
from shapely import Polygon
from tqdm import tqdm

from brails.types.image_set import ImageSet


class FFHPredictorKlepac:
    """
    Class to estimate the First Floor Height (FFH) of buildings from images.

    This class manages loading a Detectron2-based model (either a default or
    user-provided) to perform roof classification and estimate the first floor
    elevation using image inputs.

    Features:
    - Downloads and caches a default model if no model path is provided.
    - Supports preprocessing images (cropping and resizing) before prediction.
    - Calculates FFH by analyzing house and door polygons detected in model
      outputs.
    - Provides an interface to perform batch predictions on a collection of
      images.

    Args:
        input_data (Optional[dict]): Optional initialization parameters,
            including: 'modelPath': path to a custom model file to use instead
            of the default.

    Usage example:
        predictor = FFHPredictorKlepac(input_data={'modelPath':
                                                   'path/to/custom_model.pkl'})
        results = predictor.predict(image_set)
    """

    def __init__(self, input_data: Optional[dict] = None):
        """
        Initialize the FFHPredictorKlepac.

        If no model path is provided in input_data, the class downloads a
        default model. If prediction classes are not provided om input_data,
        roof type precitions are made for flat, galble, and hip roofs.

        Args:
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
            model_path = 'tmp/models/klepac_ffh_predictor.pkl'
            if not os.path.isfile(model_path):
                print(
                    '\n\nLoading default roof classifier model file to '
                    'tmp/models folder...')
                torch.hub.download_url_to_file(
                    'https://www.dropbox.com/scl/fi/qi35w6x4wqtr8k9dmpq2o/'
                    'klepac_ffh_predictor.pkl?rlkey=wa95bdqzvsrqsg92gdrxs6ykm&'
                    'st=7h2nbaw9&dl=1',
                    model_path,
                    progress=True
                )
                print('Default roof classifier model loaded')
            else:
                print(
                    f"\nDefault roof classifier model in {model_path} loaded")
        else:
            print(
                '\nInferences will be performed using the custom model in'
                f' {model_path}')

        self.model_path = model_path

    def _calculate_ffh(self, outputs) -> Optional[float]:
        """
        Calculate the First Floor Height (FFH) in ft using a Detectron2 model.

        Assumptions:
        - Class 0 corresponds to houses.
        - Class 1 corresponds to doors.
        - Doors are 80 inches tall (used for pixel-to-feet ratio estimation).

        Args:
            outputs:
                Detectron2 output dict containing 'instances' with 'pred_boxes'
                and 'pred_classes'.

        Returns:
            Optional[float]:
                FFH in feet if calculable, else None.
        """
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes.tensor.numpy()
        pred_classes = instances.pred_classes.numpy()

        houses = []
        doors = []

        for box, cls in zip(pred_boxes, pred_classes):
            x1, y1, x2, y2 = box.astype(float)
            polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            if cls == 0:
                houses.append(polygon)
            elif cls == 1:
                doors.append(polygon)

        if not houses:
            return None

        # Choose the largest house polygon:
        house = max(houses, key=lambda p: p.area)

        # Bottom Y coordinate of the house polygon:
        house_bottom = max(house.exterior.xy[1])

        if not doors:
            # No doors detected, cannot calculate FFH:
            return None

        # Find door with highest bottom coordinate inside the house polygon:
        door_candidates = [door for door in doors if door.within(house)]

        if not door_candidates:
            # Doors detected, but none within the house polygon:
            return None

        best_door = max(door_candidates, key=lambda d: max(d.exterior.xy[1]))

        door_bottom = max(best_door.exterior.xy[1])
        door_top = min(best_door.exterior.xy[1])
        door_height_px = door_bottom - door_top

        # Pixel-to-feet ratio (assuming door height = 80 inches):
        feet_per_pixel = (80 / 12) / door_height_px

        # FFH in pixels = difference between bottom of house & bottom of door:
        ffh_pixels = house_bottom - door_bottom

        # Convert FFH to feet:
        ffh_feet = ffh_pixels * feet_per_pixel

        return ffh_feet

    def _crop_center_square(self, img: np.ndarray) -> np.ndarray:
        """
        Crops an image to a centered square based on the shorter side.

        If the height is greater than the width, the crop is performed
        vertically. If the width is greater than the height, the crop is
        performed horizontally.

        Args:
            img (np.ndarray):
                Input image as a NumPy array of shape (H, W, C).

        Returns:
            np.ndarray:
                Cropped square image of shape (min(H, W), min(H, W), C).
        """
        height, width = img.shape[:2]

        if height == width:
            return img.copy()

        if height > width:
            margin = (height - width) // 2
            return img[margin:margin + width, :]
        else:
            margin = (width - height) // 2
            return img[:, margin:margin + height]

    def _preprocess_image(
        self,
        img: np.ndarray,
        target_size: int = 800
    ) -> np.ndarray:
        """
        Crop image to a centered square and resize it to specified dimensions.

        Args:
            img (np.ndarray):
                Input image as a NumPy array of shape (H, W, C).
            target_size (int, optional):
                Target height and width for the resized image. Defaults to 800.

        Returns:
            np.ndarray:
                Preprocessed image of shape (target_size, target_size, C).
        """
        square_img = self._crop_center_square(img)
        resized_img = cv2.resize(square_img, (target_size, target_size),
                                 interpolation=cv2.INTER_CUBIC)
        return resized_img

    def predict(self, images: ImageSet) -> Dict[str, Optional[float]]:
        """
        Perform first floor elevation predictions on a set of images.

        This method loads a pre-trained model from `self.model_path` and uses
        it to predict the first floor height (FFH) for each image in the
        provided `images` collection. The images are preprocessed before
        inference, and the FFH is calculated from model outputs.

        Args:
        images (ImageSet)
            An object containing image data with the following attributes:
            - `dir_path` (str): Path to the directory containing the image
               files.
            - `images` (dict): Dictionary where keys are image identifiers and
              values are image metadata objects that must have a `filename`
              attribute.

        Returns:
        Dict[str, Optional[float]]
            A dictionary mapping each image key to its predicted first floor
            height (float), or `None` if prediction was not possible
            (e.g., image file missing).

        Raises:
        NotADirectoryError
            If `images.dir_path` is not a valid directory path.
        """
        with open(self.model_path, 'rb') as f:
            predictor = pickle.load(f)

        print("\nPerforming first floor elevation predictions...")

        # Start program timer:
        start_time = time.time()

        data_dir = images.dir_path
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"Predict method failed. '{data_dir}' "
                                     "is not a directory")

        predictions = {}
        for key, im in tqdm(images.images.items()):
            im_path = os.path.join(data_dir, im.filename)
            if os.path.isfile(im_path):
                image = cv2.imread(im_path)
                image_processed = self._preprocess_image(image)
                outputs = predictor(image_processed)
                predictions[key] = self._calculate_ffh(outputs)
            else:
                predictions[key] = None

        # End program timer and display execution time:
        end_time = time.time()
        hours, rem = divmod(end_time-start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal execution time: {int(hours):02}:{int(minutes):02}:"
              f"{seconds:05.2f}")
