"""Class object to use and retrain the floor detector model."""
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
# 10-08-2024

import os
import time
import warnings
from tqdm import tqdm

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import torch

from brails.types.image_set import ImageSet
from lib.infer_detector import Infer
from .lib.train_detector import Detector

warnings.filterwarnings("ignore")


class NFloorDetector():
    """
    A class to manage and configure the system for detecting floors.

    This class provides an organized structure to store system parameters and
    configurations required for training and inference tasks in a floor
    detection system. It initializes a nested dictionary to hold various
    configurations and parameters for different stages of processing.

    Attributes_
        system_dict (Dict[str, Dict[str, Any]]): A dictionary to store system
        configurations and parameters, initialized with nested dictionaries for
        training and inference settings.

    Methods_
        __init__(self, input_data: Optional[Dict[str, Any]] = None):
            Initializes the NFloorDetector instance with default configurations
            and optionally with provided input data.

        set_fixed_params(self):
            Configures fixed parameters required for the floor detection
            system.
        train(self, comp_coeff: int = 3, top_only: bool = False,
              optim: str = "adamw", lr: float = 1e-4, nepochs: int = 25,
              ngpu: int = 1) -> None:
            Sets up parameters for training the floor detection model.
    """

    def __init__(self, input_data: dict = None) -> None:
        """
        Initialize the NFloorDetector instance.

        Args__
            input_data (Optional[dict]): An optional dictionary that can be
                    used to initialize the system parameters. If not provided,
                    the system is initialized with default values.

        Initializes__
            self.system_dict (dict): A nested dictionary structure with
                predefined keys for storing training and inference
                configurations. The default structure includes placeholders
                for training data and models, as well as inference parameters.
        """
        self.system_dict = {}
        self.system_dict["train"] = {}
        self.system_dict["train"]["data"] = {}
        self.system_dict["train"]["model"] = {}
        self.system_dict["infer"] = {}
        self.system_dict["infer"]["params"] = {}

        self._set_fixed_params()

    def _set_fixed_params(self) -> None:
        """
        Set fixed parameters for training and model configuration.

        This method initializes predefined values in the system dictionary for
        training data and model parameters. It sets the training and validation
        dataset names, the class labels, and specific model hyperparameters
        such as validation interval, save interval, early stopping minimum
        delta, and early stopping patience.

        The following parameters are set:
            - Training dataset name: "train"
            - Validation dataset name: "valid"
            - Class labels: ["floor"]
            - Validation interval: 1
            - Save interval: 5
            - Early stopping minimum delta: 0.0
            - Early stopping patience: 0

        This method does not return any values; it modifies the instance's
        `system_dict` in place.
        """
        self.system_dict["train"]["data"]["trainSet"] = "train"
        self.system_dict["train"]["data"]["validSet"] = "valid"
        self.system_dict["train"]["data"]["classes"] = ["floor"]
        self.system_dict["train"]["model"]["valInterval"] = 1
        self.system_dict["train"]["model"]["saveInterval"] = 5
        self.system_dict["train"]["model"]["esMinDelta"] = 0.0
        self.system_dict["train"]["model"]["esPatience"] = 0

    def train(self,
              comp_coeff: int = 3,
              top_only: bool = False,
              optim: str = "adamw",
              learning_rate: float = 1e-4,
              nepochs: int = 25,
              ngpu: int = 1) -> None:
        """
        Train a floor detection model.

        Args__
            comp_coeff (int): Coefficient for comparison, default is 3.
            top_only (bool): If True, only the top layers are trained, default
                is False.
            optim (str): Optimizer type, default is "adamw".
            lr (float): Learning rate, default is 1e-4.
            nepochs (int): Number of epochs for training, default is 25.
            ngpu (int): Number of GPUs to use, default is 1.

        Updates__
            self.system_dict (Dict[str, Dict[str, Any]]): Updates the model
            configuration parameters for training in the system dictionary.
        """
        # Initialize the Object Detector:
        gtf = Detector()

        classes_list = self.system_dict["train"]["data"]["classes"]
        batch_size = self.system_dict["train"]["data"]["batchSize"]
        num_workers = self.system_dict["train"]["data"]["nWorkers"]

        gtf.set_train_dataset(self.system_dict["train"]["data"]["rootDir"],
                              "",
                              "",
                              self.system_dict["train"]["data"]["trainSet"],
                              classes_list=classes_list,
                              batch_size=batch_size,
                              num_workers=num_workers)

        gtf.set_val_dataset(self.system_dict["train"]["data"]["rootDir"],
                            "",
                            "",
                            self.system_dict["train"]["data"]["validSet"])

        # Define the model architecture:
        model_architecture = f"efficientdet-d{comp_coeff}.pth"
        self.system_dict["train"]["model"]["compCoeff"] = comp_coeff

        gtf.set_model(model_name=model_architecture,
                      num_gpus=ngpu,
                      freeze_head=top_only)
        self.system_dict["train"]["model"]["topOnly"] = top_only
        self.system_dict["train"]["model"]["nGPU"] = ngpu

        # Set model hyperparameters:
        es_min_delta = self.system_dict["train"]["model"]["esMinDelta"]
        es_patience = self.system_dict["train"]["model"]["esPatience"]

        gtf.set_hyperparams(optimizer=optim,
                            lr=learning_rate,
                            es_min_delta=es_min_delta,
                            es_patience=es_patience)
        self.system_dict["train"]["model"]["optim"] = optim
        self.system_dict["train"]["model"]["lr"] = learning_rate

        # Train the model:
        val_interval = self.system_dict["train"]["model"]["valInterval"]
        save_interval = self.system_dict["train"]["model"]["saveInterval"]

        gtf.train(num_epochs=nepochs,
                  val_interval=val_interval,
                  save_interval=save_interval)
        self.system_dict["train"]["model"]["numEpochs"] = nepochs

    def retrain(self,
                optim: str = "adamw",
                learning_rate: float = 1e-4,
                nepochs: int = 25,
                ngpu: int = 1) -> None:
        """
        Retrain the object detection model with specified parameters.

        Args__
            optim (str): Optimizer to use for training (default is "adamw").
            lr (float): The learning rate for the optimizer (default is 1e-4).
            nepochs (int): Number of epochs to train the model (default is 25).
            ngpu (int): The number of GPUs to use for training (default is 1).

        This method configures the training parameters, sets the training and
        validation datasets, initializes the model architecture, and starts
        the re-training process for the default number of floors predictor in
        BRAILS++.
        """
        self.system_dict["train"]["model"]["compCoeff"] = 4
        self.system_dict["train"]["model"]["topOnly"] = False

        # Create the Object Detector Object
        gtf = Detector()

        classes_list = self.system_dict["train"]["data"]["classes"]
        batch_size = self.system_dict["train"]["data"]["batchSize"]
        num_workers = self.system_dict["train"]["data"]["nWorkers"]
        gtf.set_train_dataset(self.system_dict["train"]["data"]["rootDir"],
                              "",
                              "",
                              self.system_dict["train"]["data"]["trainSet"],
                              classes_list=classes_list,
                              batch_size=batch_size,
                              num_workers=num_workers)

        gtf.set_val_dataset(self.system_dict["train"]["data"]["rootDir"],
                            "",
                            "",
                            self.system_dict["train"]["data"]["validSet"])

        # Define the Model Architecture
        coeff = self.system_dict["train"]["model"]["compCoeff"]

        model_path = os.path.join(
            'pretrained_weights', f"efficientdet-d{coeff}.pth")

        os.makedirs('pretrained_weights', exist_ok=True)
        if not os.path.isfile(model_path):
            print('Loading default floor detector model file to the pretrained'
                  ' folder...')
            torch.hub.download_url_to_file(
                'https://zenodo.org/record/4421613'
                '/files/efficientdet-d4_trained.pth',
                model_path,
                progress=False)

        freeze_head = self.system_dict["train"]["model"]["topOnly"]
        gtf.set_model(model_name=f"efficientdet-d{coeff}.pth",
                      num_gpus=ngpu,
                      freeze_head=freeze_head)
        self.system_dict["train"]["model"]["numEpochs"] = nepochs
        self.system_dict["train"]["model"]["nGPU"] = ngpu

        # Set Model Hyperparameters
        es_min_delta = self.system_dict["train"]["model"]["esMinDelta"]
        es_patience = self.system_dict["train"]["model"]["esPatience"]
        gtf.set_hyperparams(optimizer=optim,
                            lr=learning_rate,
                            es_min_delta=es_min_delta,
                            es_patience=es_patience)
        self.system_dict["train"]["model"]["optim"] = optim
        self.system_dict["train"]["model"]["lr"] = learning_rate

        # Train the model:
        val_interval = self.system_dict["train"]["model"]["valInterval"]
        save_interval = self.system_dict["train"]["model"]["saveInterval"]
        gtf.train(num_epochs=nepochs,
                  val_interval=val_interval,
                  save_interval=save_interval)

    def predict(
        self,
        images: ImageSet,
        model_path: str = 'tmp/models/efficientdet-d4_nfloorDetector.pth'
    ) -> dict:
        """
        Predict the number of floors in buildings from the given images.

        Args__
            images (ImageSet): ImageSet object containing the collection of
                images to be analyzed.
            modelPath (str): The file path to the pre-trained model. If the
                default path is used, the model will be downloaded (default is
                'tmp/models/efficientdet-d4_nfloorDetector.pth').

        Returns__
            predictions (dict): Number of floors predictions with the keys
                being the same keys used in ImageSet.images.

        This method processes the images provided, loads the specified model,
        and performs inference to determine the number of floors in each
        building. It handles the setup of the inference environment, manages
        model loading, and provides a report on the execution time.

        It also includes functions for polygon creation, intersection checks,
        and managing thresholds during inference to ensure accurate
        predictions.

        The results of the predictions are also stored in the instance's system
        dictionary under the key 'predictions'.
        """
        gpu_enabled = torch.cuda.is_available()

        image_list = [os.path.join(images.dir_path, image.filename)
                      for _, image in images.images.items()]
        image_keys = list(images.images.keys())

        self.system_dict["infer"]["images"] = image_list
        self.system_dict["infer"]["modelPath"] = model_path
        self.system_dict["infer"]["gpuEnabled"] = gpu_enabled
        self.system_dict["infer"]['predictions'] = []

        print('\nDetermining the number of floors for each building...')

        def install_default_model(model_path: str) -> None:
            if model_path == 'tmp/models/efficientdet-d4_nfloorDetector.pth':
                os.makedirs('tmp/models', exist_ok=True)

                if not os.path.isfile(model_path):
                    print('Loading default floor detector model file to '
                          'tmp/models folder...')
                    torch.hub.download_url_to_file(
                        'https://zenodo.org/record/4421613/files'
                        '/efficientdet-d4_trained.pth',
                        model_path,
                        progress=False)
                    print('Default floor detector model loaded')
                else:
                    print(
                        f"Default floor detector model in {model_path} loaded")
            else:
                print('Inferences will be performed using the custom model at '
                      f'{model_path}')

        def create_polygon(bounding_box: list) -> Polygon:
            """
            Create a polygon from the given bounding box coordinates.

            Args__
                bounding_box (list): A sequence of four coordinates
                    representing the bounding box in the format
                    [x1, y1, x2, y2].

            Returns__
                Polygon: A polygon object created from the bounding box
                    coordinates.
            """
            polygon = Polygon([(bounding_box[0], bounding_box[1]),
                               (bounding_box[0], bounding_box[3]),
                               (bounding_box[2], bounding_box[3]),
                               (bounding_box[2], bounding_box[1])
                               ])
            return polygon

        def intersect_polygons(poly1: Polygon, poly2: Polygon) -> float:
            """
            Calculate the overlap ratio between two polygons.

            This function determines the intersection area between two polygons
            and calculates the overlap ratio as a percentage of the area of the
            first polygon.

            Args__
                poly1 (Polygon): The first polygon to compare.
                poly2 (Polygon): The second polygon to compare.

            Returns__
                float: The overlap ratio as a percentage of the area of poly1.
                       Returns 0 if there is no intersection or if either
                       polygon has an area of zero.
            """
            if poly1.intersects(poly2):
                poly_area = poly1.intersection(poly2).area
                if poly1.area != 0 and poly2.area != 0:
                    overlap_ratio = poly_area/poly1.area*100
                else:
                    overlap_ratio = 0
            else:
                overlap_ratio = 0
            return overlap_ratio

        def check_threshold_level(boxes_poly: list) -> tuple:
            """
            Check if threshold level needs changing from poly overlap ratios.

            Args__
                boxes_poly (list): A list of polygon objects.

            Returns__
                tuple: A tuple containing two boolean values:
                    - threshold_change (bool): Indicates if a change in the
                        threshold has occurred.
                    - threshold_increase (bool): Indicates if the threshold
                        has increased.
            """
            threshold_change = False
            threshold_increase = False

            if not boxes_poly:
                threshold_change = True
            else:
                false_detect = np.zeros(len(boxes_poly))
                for k in range(len(boxes_poly)):
                    overlap_ratio = np.array(
                        [intersect_polygons(p, boxes_poly[k])
                         for p in boxes_poly], dtype=float
                    )
                    false_detections = [
                        idx for idx, val in enumerate(overlap_ratio)
                        if val > 75
                    ]

                    false_detect[k] = len(false_detections[1:])

                threshold_change = any(false_detect > 2)
                if threshold_change:
                    threshold_increase = True

            return threshold_change, threshold_increase

        def compute_derivative(cent_boxes: np.ndarray) -> np.ndarray:
            """
            Compute the derivative of the center boxes.

            Args__
                cent_boxes (np.ndarray): An array of shape (n, 2) where n is
                    the number of boxes and each row represents the (x, y)
                    coordinates of the box centers.

            Returns__
                dy_over_dx (np.ndarray): A 2D array containing the derivatives
                between each pair of boxes.
            """
            n_boxes = cent_boxes.shape[0]
            dy_over_dx = np.zeros((n_boxes, n_boxes)) + 10

            for k in range(n_boxes):
                for m in range(n_boxes):
                    dx = abs(cent_boxes[k, 0] - cent_boxes[m, 0])
                    dy = abs(cent_boxes[k, 1] - cent_boxes[m, 1])
                    if k != m:
                        dy_over_dx[k, m] = dy / dx

            return dy_over_dx

        install_default_model(self.system_dict["infer"]["modelPath"])

        # Start program timer:
        start_time = time.time()

        # Create and define the inference model:
        classes = ["floor"]

        print("\nPerforming floor detections...")
        gtf_infer = Infer()
        gtf_infer.load_model(self.system_dict["infer"]["modelPath"],
                             classes,
                             use_gpu=self.system_dict["infer"]["gpuEnabled"])

        predictions = {}
        for img_no, im_path in enumerate(tqdm(image_list)):
            # Perform iterative inference:
            img = cv2.imread(im_path)
            img = cv2.resize(img, (640, 640))
            cv2.imwrite("input.jpg", img)
            _, _, boxes = gtf_infer.predict("input.jpg", threshold=0.2)
            boxes_poly = [create_polygon(bbox) for bbox in boxes]

            multiplier = 1
            while check_threshold_level(boxes_poly)[0]:
                if check_threshold_level(boxes_poly)[1]:
                    conf_threshold = 0.2 + multiplier*0.1
                    if conf_threshold > 1:
                        break
                else:
                    conf_threshold = 0.2 - multiplier*0.02
                    if conf_threshold == 0:
                        break
                _, _, boxes = gtf_infer.predict(
                    "input.jpg", threshold=conf_threshold)
                multiplier += 1
                boxes_poly = [create_polygon(bbox) for bbox in boxes]

            # Postprocessing:
            boxes_poly = [create_polygon(bbox) for bbox in boxes]

            nested_boxes = np.zeros((10*len(boxes)), dtype=int)
            counter = 0
            for bbox_poly in boxes_poly:
                overlap_ratio = np.array(
                    [intersect_polygons(p, bbox_poly)
                     for p in boxes_poly], dtype=float)
                ind = [idx for idx, val in enumerate(
                    overlap_ratio) if val > 75][1:]
                nested_boxes[counter:counter+len(ind)] = ind
                counter += len(ind)
            nested_boxes = np.unique(nested_boxes[:counter])

            counter = 0
            for box_ind in nested_boxes:
                del boxes[box_ind-counter]
                counter += 1

            n_boxes = len(boxes)

            boxes_poly = []
            boxes_extended_poly = []
            cent_boxes = np.zeros((n_boxes, 2))
            for k in range(n_boxes):
                bbox = boxes[k]
                temp_poly = create_polygon(bbox)
                boxes_poly.append(temp_poly)
                xcoord, ycoord = temp_poly.centroid.xy
                cent_boxes[k, :] = np.array([xcoord[0],
                                             ycoord[0]])
                boxes_extended_poly.append(create_polygon(
                    [0.9*bbox[0], 0, 1.1*bbox[2], len(img)-1]))

            stacked_ind = []
            for bbox in boxes_extended_poly:
                overlap_ratio = np.array(
                    [intersect_polygons(p, bbox)
                     for p in boxes_extended_poly], dtype=float)
                stacked_ind.append(
                    [idx for idx, val in enumerate(overlap_ratio) if val > 10])

            unique_stacks0 = [list(x) for x in set(tuple(x)
                                                   for x in stacked_ind)]

            dy_over_dx = compute_derivative(cent_boxes)
            stacks = np.where(dy_over_dx > 1.3)

            counter = 0
            unique_stacks0 = [[] for i in range(n_boxes)]
            for k in range(n_boxes):
                while counter < len(stacks[0]) and k == stacks[0][counter]:
                    unique_stacks0[k].append(stacks[1][counter])
                    counter += 1

            unique_stacks0 = [list(x) for x in set(tuple(x)
                                                   for x in unique_stacks0)]

            if len(unique_stacks0) == 1 or len(unique_stacks0) == 1:
                nfloors = len(unique_stacks0[0])
            else:
                lbound = len(img)/5
                ubound = 4*len(img)/5
                middle_poly = Polygon(
                    [(lbound, 0), (lbound, len(img)),
                     (ubound, len(img)), (ubound, 0)])
                overlap_ratio = np.empty(len(unique_stacks0))
                for k in range(len(unique_stacks0)):
                    poly = unary_union([boxes_poly[x]
                                       for x in unique_stacks0[k]])
                    overlap_ratio[k] = (
                        intersect_polygons(poly, middle_poly))

                ind_keep = np.argsort(-overlap_ratio)[0:2]
                stack4address = []
                for k in range(2):
                    if overlap_ratio[ind_keep[k]] > 10:
                        stack4address.append(ind_keep[k])
                if len(stack4address) != 0:
                    nfloors = max(len(unique_stacks0[x])
                                  for x in stack4address)
                else:
                    nfloors = len(unique_stacks0[0])

            predictions[image_keys[img_no]] = nfloors

        self.system_dict["infer"]['predictions'] = predictions

        # End program timer and display execution time:
        end_time = time.time()
        hours, rem = divmod(end_time-start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal execution time: {int(hours):02}:{int(minutes):02}:"
              f"{seconds:05.2f}")

        # Cleanup the root folder:
        if os.path.isfile("input.jpg"):
            os.remove("input.jpg")
        if os.path.isfile("output.jpg"):
            os.remove("output.jpg")

        return predictions
