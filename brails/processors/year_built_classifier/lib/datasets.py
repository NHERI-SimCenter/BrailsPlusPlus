"""Class object that prepares data for the year built classifier."""
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
# Sascha Hornauer
# Barbaros Cetiner
#
# Last updated:
# 11-20-2024

import os
from collections import defaultdict

import numpy as np
from scipy import signal
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class YearBuiltFolder(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing images.

    Attributes:
        img_paths (list[str]):
            List of image file paths.
        filenames (list[str]):
            List of image file names.
        labels (Union[list[str], list[np.ndarray]]):
            List of labels corresponding to images.
        calc_perf (bool):
            Whether to calculate performance metrics.
        soft_labels (bool):
            Whether soft labels are assigned.
        classes (np.ndarray):
            Array of unique class labels.
        class_weights (dict[str, float]):
            Weights for each class based on their frequencies.
        train_weights (list[float]):
            Weights for individual samples for training.
        transforms (Any):
            Image transformations applied during loading.

    Methods:
        __len__():
            Returns the total number of samples in the dataset.
        __getitem__(index: int) -> tuple[torch.Tensor,torch.Tensor|list],str]:
            Returns the image, label, and image path for the given index.
        loader(path: str) -> Image.Image:
            Loads and returns an image from the specified path.
    """

    def __init__(self,
                 image_folder: str | list[str],
                 soft_labels: bool = False,
                 gaussian_std: float = 1.5,
                 transforms: transforms.Compose | None = None,
                 classes: np.ndarray | None = None,
                 calc_perf: bool = False):
        """
        Initialize YearBuiltFolder class.

        Args:
            image_folder (str or list[str]):
                Path to an image folder, a single image, or a list of image
                paths.
            soft_labels (bool, optional):
                Whether to assign soft labels using Gaussian distribution.
                Defaults to False.
            gaussian_std (float, optional):
                Standard deviation for Gaussian soft label calculation.
                Only relevant if `soft_labels` is True. Defaults to 1.5.
            transforms (transforms.Compose or None):
                Image transformations to apply. Defaults to None.
            classes (Optional[np.ndarray], optional):
                Array of predefined class labels. If None, classes are inferred
                from the dataset. Defaults to None.
            calc_perf (bool, optional):
                Whether to calculate performance metrics. If True, labels will
                be generated based on folder names. Defaults to False.

        """
        self.transforms = transforms

        self.img_paths = []
        self.filenames = []
        self.labels = []
        self.calc_perf = calc_perf
        self.soft_labels = soft_labels

        # Determine file list based on input type:
        if isinstance(image_folder, list):  # A list of image paths
            # The following format is consistent with os.walk output:
            files = [os.path.split(i)[1] for i in image_folder]
            file_list = [(os.path.split(image_folder[0])[0], [], files.copy())]

        elif isinstance(image_folder, str):  # An image
            if not os.path.isdir(image_folder):
                if os.path.isfile(image_folder):
                    # The following format is consistent with os.walk output:
                    file_list = [(os.path.split(image_folder)[0], [], [
                        os.path.split(image_folder)[1]])]
                else:
                    raise FileNotFoundError('Error: Image folder or file '
                                            f'{image_folder} not found.')

            else:  # A directory path
                file_list = os.walk(image_folder, followlinks=True)

        else:
            raise ValueError('Error: incorrect image input. Can either have'
                             'list or string for the list or path of image(s)')

        class_counts = defaultdict(lambda: 0)

        for root, _, fnames in sorted(file_list):
            for fname in sorted(fnames):

                if 'jpg' in fname or 'png' in fname:
                    img_path = os.path.join(root, fname)

                    self.filenames.append(fname)
                    self.img_paths.append(img_path)

                    if calc_perf:
                        label = root.split(os.path.sep)[-1]
                        self.labels.append(label)
                        # count labels
                        class_counts[label] = class_counts[label] + 1

        if classes is None:
            self.classes = np.unique(self.labels)
        else:
            self.classes = classes
        # Calculate train weights for weighted sampling
        self.class_weights = {}
        self.train_weights = []

        for _class in self.classes:
            self.class_weights[_class] = sum(
                np.array([label == _class for label in self.labels]))

        for entry in self.labels:
            self.train_weights.append(1 / self.class_weights[entry])

        ################################
        # Optionally, create soft labels by pre-calculating unimodal gaussians
        # and assigning them according to the class label:

        if soft_labels:
            res = 100
            class_n = len(self.classes)
            window = signal.gaussian(
                res*class_n, std=(res/10)*(class_n-1)*float(gaussian_std))

            center_id = int((res*class_n)/2)

            samples = []
            for i in range(-class_n//2, class_n//2+1):
                pos_of_int = window[min(
                    max(center_id+res*i, 0), window.shape[0]-1)]
                samples.append(pos_of_int)

            samples = np.array(samples)

            label_lookup = defaultdict()
            for class_id in self.classes:
                class_soft_labels = []

                max_class_id = class_n
                for i in range(max_class_id):
                    sample_id = (i+3)-np.flatnonzero(self.classes == class_id)

                    if sample_id < 0 or sample_id >= max_class_id:
                        class_soft_labels.append(0.0)
                    else:
                        class_soft_labels.append(samples[sample_id].squeeze())

                class_soft_labels = np.array(class_soft_labels)

                label_lookup[class_id] = class_soft_labels

            year_built_softlabels = [label_lookup[label]
                                     for label in self.labels]

            self.labels = year_built_softlabels

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.filenames)

    def loader(self, path: str) -> Image.Image:
        """
        Load an image from the specified path and converts it to RGB format.

        Args:
            path (str): The file path of the image to load.

        Returns:
            Image.Image: The loaded image in RGB format.
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self,
                    index: int
                    ) -> tuple[torch.Tensor, [torch.Tensor | list], str]:
        """
        Retrieve the image, target label, and file path for the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, [torch.Tensor | list], str]:
                - The transformed image as a tensor.
                - The target label as a tensor (if `calc_perf` is True) or
                  an empty list.
                - The file path of the image.
        """
        path = self.img_paths[index]
        img = self.loader(path)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.calc_perf:
            label = self.labels[index]
            # The label is either a probability distribution or the class
            # number:
            if self.soft_labels:
                target = label
                target = torch.FloatTensor(target)
            else:
                target = [np.flatnonzero(self.classes == label)]
                target = torch.LongTensor(target).squeeze()

        else:
            target = []

        return img, target, path
