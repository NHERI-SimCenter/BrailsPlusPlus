"""Class object to predict era of construction of buildings."""
#
# Copyright (c) 2022 The Regents of the University of California
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
# Sascha Hornauer
#
# Last updated:
# 11-20-2024

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from torchvision.transforms import transforms
from tqdm import tqdm

from brails.types.image_set import ImageSet
from .lib.datasets import YearBuiltFolder

matplotlib.use('agg')
sm = nn.Softmax()


class YearBuiltClassifier():
    """
    A classifier for predicting the construction era of buildings.

    This model classifies images of buildings into one of several predefined
    construction eras based on the year they were built. It leverages a deep
    learning model to predict the building era based on visual features. The
    model is loaded from a checkpoint, which can either be pre-trained or
    provided by the user.

    Attributes:
        checkpoint (str):
            Path to the model checkpoint. Defaults to an empty string, which
            uses a pre-trained model.
        work_dir (str):
            Directory where model files and outputs are stored. Defaults to
            'tmp'.
        print_res (bool):
            If True, prints additional debug information. Defaults to False.
        classes (list):
            List of construction era class labels (e.g., [1960, 1975, 1985,
            1995, 2005, 2015]).
        device (str):
            The device on which the model will run, either 'cpu' or 'cuda'
            (GPU).
        checkpoints_dir (str):
            Directory for saving and loading model checkpoints.
        model_file (str):
            Path to the model file.
        model (torch.nn.Module):
            The loaded PyTorch model.
        test_transforms (torchvision.transforms.Compose):
            The transformation pipeline for input preprocessing.

    Methods:
        predict(images: ImageSet):
            Predicts the construction era for a set of input images.
        evaluate_to_stats(testloader: torch.utils.data.DataLoader):
            Evaluates the model on a dataset and collects predictions and
            probabilities.
        construct_confusion_matrix_image(classes: list, con_mat: np.ndarray):
            Constructs a confusion matrix heatmap for evaluation.
    """

    def __init__(self, input_data: dict | None = None):
        """
        Initialize the YearBuiltClassifier.

        This classifier predicts the era of construction for buildings based
        on their features. It loads a pretrained model (or a user-provided
        checkpoint) and configures the necessary environment for inference.

        Args:
            input_data (dict, optional):
                Configuration options for the classifier. Expected keys:
                    - 'checkpoint' (str):
                        Path to the model checkpoint. Defaults to the
                        pretrained version.
                    - 'workDir' (str):
                        Directory for saving files. Defaults to 'tmp'.
                    - 'printRes' (bool):
                        If True, prints additional information during
                        initialization. Defaults to False.
        Raises:
            RuntimeError:
                If the model checkpoint cannot be loaded or does not match the
                expected format.
        """
        if input_data is None:
            input_data = {}

        self.checkpoint = input_data.get('checkpoint', '')
        self.work_dir = input_data.get('workDir', 'tmp')
        self.print_res = input_data.get('printRes', False)

        # Define prediction classes corresponding to Pre-1970, 1970-197,
        # 1980-1989, 1990-1999, 2000-2009, and Post-2010:
        self.classes = [1960, 1975, 1985, 1995, 2005, 2015]

        # Set device to use to perform tensor calculations:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Ensure checkpoints directory exists:
        self.checkpoints_dir = os.path.join(self.work_dir, 'models')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Define model file path:
        weight_file_path = os.path.join(self.checkpoints_dir,
                                        'yearBuiltv0.1.pth')

        # Load the model file
        self.model_file = self._load_model_checkpoint(weight_file_path)

        # Load the model to the appropriate device
        if self.device == 'cpu':
            self.model = torch.load(self.model_file,
                                    map_location=torch.device(self.device))
        else:
            self.model = torch.load(self.model_file)

        # Define preprocessing transforms:
        self.test_transforms = self._initialize_transforms()

        # Get the number of prediction classes in the loaded model:
        self.num_classes = self._validate_model()

        # Set the model to evaluation mode and send it to the appropriate
        # device:
        self.model.eval()
        self.model = self.model.to(self.device)

    def _load_model_checkpoint(self, weight_file_path: str) -> str:
        """
        Load the model checkpoint.

        Args:
            weight_file_path (str):
                Path to the default model checkpoint.

        Returns:
            str:
                Path to the model file.
        """
        if self.checkpoint:
            return self.checkpoint

        if not os.path.isfile(weight_file_path):
            print('\nDownloading default model checkpoint to '
                  f'{self.checkpoints_dir}...')
            torch.hub.download_url_to_file(
                'https://zenodo.org/record/4310463/files/model_best.pth',
                weight_file_path,
                progress=False
            )
            print("Default era of construction classifier model loaded.")
        else:
            print('Default model checkpoint found at '
                  f'{self.checkpoints_dir}.')
        return weight_file_path

    def _initialize_transforms(self) -> transforms.Compose:
        """
        Initialize the transformation pipeline for input preprocessing.

        Returns:
            transforms.Compose:
                The preprocessing pipeline.
        """
        return transforms.Compose([
            transforms.Resize((550, 550)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        print('\nDetermining the era of construction for each building...')

    def _validate_model(self) -> int:
        """
        Validate the loaded model and determine the number of output classes.

        Returns:
            int:
                The number of output classes.

        Raises:
            RuntimeError:
                If the model is incompatible or incorrectly formatted.
        """
        try:
            return self.model.classifier1[4].out_features
        except AttributeError as e:
            raise RuntimeError(
                'The model checkpoint does not match the expected format. '
                f'Please verify the checkpoint file. Error: {e}'
            )

    def predict(self, images: ImageSet):
        """
        Predict construction era for a set of images provided in an ImageSet.

        Args:
            images (ImageSet):
                An object containing a collection of images, where
                `images.dir_path` specifies the directory of images and
                `images.images` contains a dictionary of image objects with
                filenames.

        Raises:
            NotADirectoryError:
                If the provided directory path does not exist or is not a
                directory.

        Returns:
            dict:
                A dictionary where keys are the image identifiers from
                `images.images`, and values are the predicted class labels
                (construction era) for each image.
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
            raise NotADirectoryError('YearBuiltClassifier failed. '
                                     f'{data_dir} is not a directory')

        image_files_dict = {}
        image_files = []
        for key, im in images.images.items():
            im_path = os.path.join(data_dir, im.filename)
            if is_image(im.filename) and os.path.isfile(im_path):
                image_files_dict[im.filename] = key
                image_files.append(im_path)

        # Initialize the dataset and DataLoader:
        dataset = YearBuiltFolder(image_files,
                                  transforms=self.test_transforms,
                                  classes=range(self.num_classes),
                                  calc_perf=False)

        # Do not change the batchsize:
        test_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0)

        print("Performing construction era classifications...")
        predictions_data = self.evaluate_to_stats(test_loader)
        pred = {}
        for prediction in tqdm(predictions_data):
            image_path = str(prediction['filename']).replace('\\', '/')
            prediction_int = (prediction['prediction'][0])
            prediction_class = self.classes[prediction_int]
            pred[image_files_dict[image_path]] = prediction_class
            prob = prediction['probability']
            if self.print_res:
                print(
                    f'Image :  {image_path}     '
                    f'Class : {prediction_class} '
                    f'({str(round(prob*100,2))}%)')

        return pred

    def evaluate_to_stats(self,
                          testloader: torch.utils.data.DataLoader
                          ) -> list[dict[str, [str | int | float]]]:
        """
        Evaluate the model on a dataset & collect predictions & probabilities.

        Args:
            testloader (torch.utils.data.DataLoader):
                The DataLoader for the test dataset.

        Returns:
            list[dict[str, [str | int | float]]]:
                A list of dictionaries containing predictions, probabilities,
                and optionally ground truth.
        """
        self.model.eval()

        # Ensure ground truth is not calculated by default:
        self.calc_perf = False

        predictions = []

        # Validate batch size:
        batch_size = testloader.batch_size
        if batch_size != 1:
            raise NotImplementedError('This method supports only a batch size '
                                      'of 1. Larger batch sizes are not '
                                      'compatible.'
                                      )

        with torch.no_grad():
            for inputs, label, filename in testloader:
                # Move inputs to the specified device:
                inputs = inputs.to(self.device)
                inputs = Variable(inputs)

                # Perform model inference:
                output_1, output_2, output_3, output_concat = \
                    self.model(inputs)
                outputs_combined = output_1 + output_2 + output_3 + \
                    output_concat

                # Apply softmax to get probabilities:
                output_probs = sm(outputs_combined.data)
                _, predicted_class = torch.max(output_probs, 1)

                # Extract prediction probability for the predicted class:
                prediction = predicted_class[0].flatten().cpu().numpy()
                probability = output_probs[0][prediction][0].item()

                # Extract the filename for logging:
                filename_str = filename[0]

                # Append prediction results to the list
                prediction_entry = {
                    'filename': filename_str,
                    'prediction': prediction,
                    'probability': probability,
                }

                if self.calc_perf:
                    prediction_entry['ground truth'] = label.cpu().numpy()

        return predictions

    def construct_confusion_matrix_image(classes: list,
                                         con_mat: np.ndarray) -> plt.Figure:
        """
        Construct a confusion matrix heatmap image.

        Args:
            classes (list):
                List of class labels.
            con_mat (np.ndarray):
                Confusion matrix.

        Returns:
            plt.Figure:
                Matplotlib figure containing the confusion matrix heatmap.
        """
        # Normalize the confusion matrix
        con_mat_norm = np.around(con_mat.astype(
            'float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

        # Create a figure
        figure, ax = plt.subplots(figsize=(8, 8))

        # Display the normalized confusion matrix as a heatmap
        cax = ax.matshow(con_mat_norm, cmap=plt.cm.Blues)

        # Add colorbar for reference
        plt.colorbar(cax)

        # Add labels to the axes
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Add annotations to each cell
        for i in range(con_mat_norm.shape[0]):
            for j in range(con_mat_norm.shape[1]):
                ax.text(j, i, str(con_mat_norm[i, j]),
                        ha="center", va="center", color="black")

        # Set axis labels
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Adjust layout for clarity
        plt.tight_layout()

        # Draw the canvas
        figure.canvas.draw()

        return figure
