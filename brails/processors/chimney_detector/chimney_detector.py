"""Class object to use and retrain the chimney detector model."""
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
# 11-20-2024

import os
import time
import warnings
from tqdm import tqdm

import cv2
import torch

from brails.types.image_set import ImageSet
from .lib.infer_detector import Infer
from .lib.train_detector import Detector

warnings.filterwarnings("ignore")


class ChimneyDetector():
    """
    A class to manage and configure the system for detecting chimneys.

    This class provides an organized structure to store system parameters and
    configurations required for training and inference tasks in a chimney
    detection system. It initializes a nested dictionary to hold various
    configurations and parameters for different stages of processing.

    Attributes:
        system_dict (dict):
            A dictionary to store system configurations and parameters,
            initialized with nested dictionaries for training and inference
            settings.

    Methods:
        train(self, comp_coeff: int = 3, top_only: bool = False,
              optim: str = "adamw", lr: float = 1e-4, nepochs: int = 25,
              ngpu: int = 1) -> None:
            Sets up parameters for training the chimney detection model.
    """

    def __init__(self, input_data: dict = None) -> None:
        """
        Initialize the ChimneyDetector instance.

        Args:
            input_data (dict):
                An optional dictionary that can be used to initialize the
                system parameters. If not provided, the system is initialized
                with default values.

        Initializes_:
            self.system_dict (dict):
                A nested dictionary structure with predefined keys for storing
                training and inference configurations. The default structure
                includes placeholders for training data and models, as well as
                inference parameters.
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
            - Class labels: ["chimney"]
            - Validation interval: 1
            - Save interval: 5
            - Early stopping minimum delta: 0.0
            - Early stopping patience: 0

        This method does not return any values; it modifies the instance's
        `system_dict` in place.
        """
        self.system_dict["train"]["data"]["trainSet"] = "train"
        self.system_dict["train"]["data"]["validSet"] = "valid"
        self.system_dict["train"]["data"]["classes"] = ["chimney"]
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
        Train a chimney detection model.

        Args:
            comp_coeff (int):
                Coefficient for comparison, default is 3.
            top_only (bool):
                If True, only the top layers are trained, default is False.
            optim (str):
                Optimizer type, default is "adamw".
            lr (float):
                Learning rate, default is 1e-4.
            nepochs (int):
                Number of epochs for training, default is 25.
            ngpu (int):
                Number of GPUs to use, default is 1.

        Updates:
            self.system_dict (dict[str, dict[str, Any]]):
                Updates the model configuration parameters for training in the
                system dictionary.
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
        Retrain the chimney detection model with specified parameters.

        Args:
            optim (str):
                Optimizer to use for training (default is "adamw").
            lr (float):
                The learning rate for the optimizer (default is 1e-4).
            nepochs (int):
                Number of epochs to train the model (default is 25).
            ngpu (int):
                The number of GPUs to use for training (default is 1).

        This method configures the training parameters, sets the training and
        validation datasets, initializes the model architecture, and starts
        the re-training process for the default chimney detector in
        BRAILS++.
        """
        self.system_dict["train"]["model"]["compCoeff"] = 4
        self.system_dict["train"]["model"]["topOnly"] = False

        # Initialize the Object Detector class:
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
        model_name = 'efficientdet-d4_chimneyDetector.pth'
        model_path = os.path.join('pretrained_weights', model_name)

        os.makedirs('pretrained_weights', exist_ok=True)
        if not os.path.isfile(model_path):
            print('Loading default chimney detector model file to the '
                  'pretrained folder...')
            torch.hub.download_url_to_file(
                'https://zenodo.org/record/5775292'
                '/files/efficientdet-d4_chimneyDetector.pth',
                model_path,
                progress=False)

        freeze_head = self.system_dict["train"]["model"]["topOnly"]
        gtf.set_model(model_name=model_name,
                      num_gpus=ngpu,
                      freeze_head=freeze_head)
        self.system_dict["train"]["model"]["numEpochs"] = nepochs
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

    def predict(
        self,
        images: ImageSet,
        model_path: str = 'tmp/models/efficientdet-d4_chimneyDetector.pth'
    ) -> dict:
        """
        Predict the existence of chimneys in buildings from the given images.

        Args:
            images (ImageSet):
                ImageSet object containing the collection of images to be
                analyzed.
            modelPath (str):
                The file path to the pre-trained model. If the default path is
                used, the model will be downloaded (default is
                'tmp/models/efficientdet-d4_chimneyDetector.pth').

        Returns_:
            predictions (dict):
                Existence of chimneys with the keys being the same keys used in
                ImageSet.images.

        This method processes the images provided, loads the specified model,
        and performs inference to determine the existence of chimneys in each
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

        print('\nChecking the existence of chimneys for each building...')

        def install_default_model(model_path: str) -> None:
            if model_path == 'tmp/models/efficientdet-d4_chimneyDetector.pth':
                os.makedirs('tmp/models', exist_ok=True)

                if not os.path.isfile(model_path):
                    print('Loading default chimney detector model file to '
                          'tmp/models folder...')
                    torch.hub.download_url_to_file(
                        'https://zenodo.org/record/5775292/files'
                        '/efficientdet-d4_chimneyDetector.pth',
                        model_path,
                        progress=False)
                    print('Default chimney detector model loaded')
                else:
                    print(
                        f"Default chimney detector model in {model_path} "
                        'loaded')
            else:
                print('Inferences will be performed using the custom model in '
                      f'{model_path}')

        install_default_model(self.system_dict["infer"]["modelPath"])

        # Start program timer:
        start_time = time.time()

        # Create and define the inference model:
        classes = ["chimney"]

        print("\nPerforming chimney detections...")
        gtf_infer = Infer()
        gtf_infer.load_model(self.system_dict["infer"]["modelPath"],
                             classes,
                             use_gpu=self.system_dict["infer"]["gpuEnabled"])

        predictions = {}
        for img_no, im_path in enumerate(tqdm(image_list)):
            if os.path.isfile(im_path):
                img = cv2.imread(im_path)
                cv2.imwrite("input.jpg", img)
                _, _, boxes = gtf_infer.predict(
                    "input.jpg", threshold=0.35)
                if len(boxes) >= 1:
                    predictions[image_keys[img_no]] = 1
                else:
                    predictions[image_keys[img_no]] = 0
            else:
                predictions[image_keys[img_no]] = None

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
