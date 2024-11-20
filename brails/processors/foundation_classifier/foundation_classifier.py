"""Class object to predict if a building is elevated."""
# Copyright (c) 2024 The Regents of the University of California
#
# This file is part of BRAILS++
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
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 11-19-2024

import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from brails.types.image_set import ImageSet
from .models.resnet_applied import resnet50
from .utils.Datasets import Foundation_Type_Testset
from .csail_segmentation_tool.csail_segmentation import MaskBuilding


class FoundationElavationClassifier:
    """
    Classifier for predicting if a building is elevated.

    This class utilizes a pretrained ResNet50 model for binary classification
    of images. It can preprocess images, apply optional building segmentation
    masks, and perform predictions.

    Attributes:
        checkpoint (str):
            Path to the model checkpoint. Defaults to the best pretrained
            model if not provided.
        mask_buildings (bool):
            Whether to apply building segmentation masks to input images.
        load_masks (bool):
            Whether to use pre-generated segmentation masks.
        work_dir (str):
            Directory for storing intermediate files and results.
        print_res (bool):
            Whether to print prediction results during inference.
        classes (list[str]):
            List of class labels ('Non-elevated', 'Elevated').
        checkpoints_dir (str):
            Directory for storing model checkpoints.
        model_file (str):
            Path to the loaded or downloaded model checkpoint.
        model_dir (str):
            Directory for segmentation model components.
        device (str):
            Device to run computations ('cuda' or 'cpu').
        test_transforms (transforms.Compose):
            Image transformations pipeline for preprocessing.

    Methods:
        predict(images: ImageSet) -> dict:
            Predicts whether buildings in the given set of images are elevated
            or non-elevated. Returns a dictionary mapping image IDs to their
            predicted classes.
    """

    def __init__(self, input_data: dict | None = None):
        """
        Initialize the FoundationElevationClassifier.

        Args:
            input_data(dict, optional):
                A dictionary containing input parameters. If None, default
                values are used. Expected keys are:
                - 'checkpoint' (str):
                    Path to the checkpoint file. Defaults to the best
                    pretrained model if not provided.
                - 'mask_buildings' (bool):
                    If True, masks non-building parts of the image
                    (may slow down processing). Defaults to False.
                - 'load_masks' (bool):
                    If True, uses pre-generated segmentation masks. Otherwise,
                    generates masks on the fly if 'mask_buildings' is True.
                    Defaults to False.
                - 'work_dir' (str):
                    Directory to save intermediate files and results.
                    Defaults to 'tmp'.
                - 'print_res' (bool):
                    If True, prints results during processing. Defaults to
                    False.
        """
        if input_data is None:
            input_data = {}

        self.checkpoint = input_data.get('checkpoint', '')
        self.mask_buildings = input_data.get('maskBuildings', False)
        self.load_masks = input_data.get('loadMasks', False)
        self.work_dir = input_data.get('workDir', 'tmp')
        self.print_res = input_data.get('printRes', False)
        self.classes = ['Non-elevated', 'Elevated']

        # Ensure checkpoints directory exists:
        self.checkpoints_dir = os.path.join(self.work_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Load or download model checkpoint:
        self.model_file = self._load_model_checkpoint()

        # Define segmentation model paths:
        self.model_dir, encoder_path, decoder_path = \
            self._setup_segmentation_model()

        # Ensure segmentation model files are downloaded:
        self._download_segmentation_files(encoder_path, decoder_path)

        # Set device to use to perform tensor calculations:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Define image transforms to use in performing predictions:
        self.test_transforms = self._initialize_transforms()

    def _load_model_checkpoint(self) -> str:
        """
        Load the model checkpoint for foundation elevation classification.

        If a checkpoint is specified in `self.checkpoint`, it will be used.
        Otherwise, the method attempts to load the best pretrained model
        from the default location. If the pretrained model does not exist
        locally, it is downloaded.

        Returns:
            str:
                Path to the model checkpoint file.
        """
        if self.checkpoint:
            return self.checkpoint

        # Use the best pretrained model if no checkpoint is provided:
        weight_file_path = os.path.join(
            self.checkpoints_dir, 'best_masked.pkl')
        if not os.path.isfile(weight_file_path):
            print('Downloading pretrained model checkpoint...')
            torch.hub.download_url_to_file(
                'https://zenodo.org/record/4145934/files/best_masked.pkl',
                weight_file_path
            )
        return weight_file_path

    def _setup_segmentation_model(self) -> tuple:
        """
        Set up the segmentation model paths for encoder and decoder components.

        Creates the required directories for the segmentation model if they
        do not already exist. Returns the paths for the model directory,
        encoder, and decoder.

        Returns:
            tuple:
                A tuple containing:
                    - str: Path to the model directory.
                    - str: Path to the encoder file.
                    - str: Path to the decoder file.

        Notes:
            - Paths are constructed relative to `self.work_dir`.
            - Model files are expected to follow a specific naming convention.
        """
        model_name = 'ade20k-resnet50dilated-ppm_deepsup'
        model_dir = os.path.join(
            self.work_dir, 'csail_segmentation_tool', 'csail_seg', model_name)
        os.makedirs(model_dir, exist_ok=True)

        encoder_path = f'{model_name}/encoder_epoch_20.pth'
        decoder_path = f'{model_name}/decoder_epoch_20.pth'

        return model_dir, encoder_path, decoder_path

    def _download_segmentation_files(self,
                                     encoder_path: str,
                                     decoder_path: str):
        """
        Download encoder and decoder files for the segmentation model.

        Downloads the encoder and decoder files from the remote server if they
        do not already exist in the specified local directory.

        Args:
            encoder_path(str): Relative path to the encoder file.
            decoder_path(str): Relative path to the decoder file.

        Raises:
            Any exceptions raised during file download will propagate.
        """
        local_encoder_path = os.path.join(
            self.model_dir, 'encoder_epoch_20.pth')
        local_decoder_path = os.path.join(
            self.model_dir, 'decoder_epoch_20.pth')

        if not os.path.isfile(local_encoder_path):
            print('Downloading encoder file...')
            torch.hub.download_url_to_file(
                'http://sceneparsing.csail.mit.edu/model/pytorch/'
                f'{encoder_path}',
                local_encoder_path
            )

        if not os.path.isfile(local_decoder_path):
            print('Downloading decoder file...')
            torch.hub.download_url_to_file(
                'http://sceneparsing.csail.mit.edu/model/pytorch/'
                f'{decoder_path}',
                local_decoder_path
            )

    def _initialize_transforms(self) -> transforms.Compose:
        """
        Initialize the image transformation pipeline for preprocessing.

        Applies a sequence of transformations to prepare input images for the
        model. This includes resizing, normalizing pixel values, and optionally
        masking non-building areas if `mask_buildings` is enabled.

        Returns:
            transforms.Compose:
                A composed set of transformations to be applied to input
                images.

        Notes:
            - If `mask_buildings` is True and `load_masks` is False, a building
              mask is generated dynamically and inserted as the first step in
              the transformation pipeline.
            - Normalization uses standard ImageNet mean and standard deviation
              values.
        """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transforms_list = [transforms.Resize(
            (224, 224)), transforms.ToTensor(), normalize]

        if self.mask_buildings and not self.load_masks:
            transforms_list.insert(0, MaskBuilding(
                self.device, model_dir=self.model_dir))

        return transforms.Compose(transforms_list)

    def predict(self, images: ImageSet):
        """
        Predict whether a building is elevated from street-level imagery.

        Args:
            images(ImageSet):
                An `ImageSet` object containing image file paths and their
                metadata.

        Returns:
            dict:
                A dictionary mapping image IDs to predicted classes. The keys
                are derived from the `ImageSet` metadata, and the values are
                the predicted class labels (e.g., 'elevated', 'non-elevated').

        Raises:
            NotADirectoryError:
                If the directory specified in `images.dir_path` does not exist.

        Notes:
            - The model performs binary classification for each input image.
            - Class predictions and confidence scores are displayed if
              `self.print_res` is set to `True`.
            - Ensure the `ImageSet` object provides valid image paths and
              metadata.
        """
        def is_image(im):
            """
            Check if a given filename corresponds to an image file.

            Args:
                im(str):
                    The filename or file path to check.

            Returns:
                bool:
                    `True` if the file has an image extension(e.g., .png, .jpg,
                    .jpeg, .bmp), otherwise `False`.
            """
            return im.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

        data_dir = images.dir_path
        if not os.path.isdir(data_dir):
            raise NotADirectoryError('FoundationElavationClassifier failed. '
                                     f'{data_dir} is not a directory')

        image_files_dict = {}
        image_files = []
        for key, im in images.images.items():
            im_path = os.path.join(data_dir, im.filename)
            if is_image(im.filename) and os.path.isfile(im_path):
                image_files_dict[im.filename] = key
                image_files.append(im_path)
        print(image_files_dict)
        print(image_files)

        # Initialize the dataset and DataLoader:
        dataset = Foundation_Type_Testset(image_files,
                                          transform=self.test_transforms,
                                          mask_buildings=self.mask_buildings,
                                          load_masks=self.load_masks)

        test_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=0)

        # Load the model:
        model = resnet50(low_dim=1)
        if self.device == 'cpu':
            state_dict = torch.load(self.model_file,
                                    map_location=torch.device(self.device))
        else:
            state_dict = torch.load(self.model_file)

        # Handle DataParallel loading issues:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if any('module' in name for name in unexpected):
            # Remapping to remove effects of DataParallel:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v  # remove 'module.' of dataparallel
            model.load_state_dict(new_state_dict, strict=False)

        # Ensure no unexpected keys remain:
        if len(missing) or len(unexpected):
            print(f'Missing or unexpected keys: {missing},{unexpected}')
            print('This should not happen. Check if checkpoint is correct')

        model.eval()
        model = model.to(self.device)

        # Run inference:
        pred = {}
        with torch.no_grad():
            for image, filename in test_loader:
                image = image.to(self.device)
                prediction = model(image.float())
                score = torch.sigmoid(prediction).cpu().numpy()[0][0]
                prediction_bin = int(score >= 0.5)  # Binary classification
                prediction_class = self.classes[prediction_bin]
                image_path = filename[0]
                pred[image_files_dict[image_path]] = prediction_class
                prob = score if score >= 0.5 else 1.0 - score
                if self.print_res:
                    print(
                        f'Image :  {image_path}     '
                        f'Class : {prediction_class} '
                        f'({str(round(prob*100,2))}%)')

        return pred


if __name__ == '__main__':
    pass
