"""Class object that prepares data for the foundation elevation classifier."""
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
# Sascha Hornauer
#
# Last updated:
# 11-19-2024

from __future__ import print_function, division

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


eps = np.finfo(float).eps  # A small number to avoid zeros


class Foundation_Type_Testset(Dataset):
    """
    A custom dataset class for loading images and optional masks .

    This class handles the loading of images from a given directory or a list
    of image paths. It can also load associated segmentation masks for
    buildings if specified. The dataset supports transformations applied to the
    images and provides the option to mask non-building areas.

    Attributes:
        transform (callable, optional):
            Optional transform to be applied to each image. Default is None.
        mask_buildings (bool, optional):
            Whether to mask out non-building areas in the image. Default is
            False.
        load_masks (bool, optional):
            Whether to load pre-existing masks for the images. Default is
            False.
        img_paths (list):
            List of image paths.
        mask_paths (list):
            List of mask paths.
        filenames (list):
            List of filenames paths.
    """

    def __init__(self,
                 image_folder: str | list,
                 transform=None,
                 mask_buildings: bool = False,
                 load_masks: bool = False):
        """
        Initialize the dataset with image paths and optional mask loading.

        Args:
            image_folder (str or list):
                Path to a directory or a list of image files.
            transform (callable, optional):
                Optional transform to be applied on the image.
            mask_buildings (bool, optional):
                Whether to mask out non-building areas.
            load_masks (bool, optional):
                Whether to load pre-existing masks from the directory.
        """
        self.transform = transform
        self.mask_buildings = mask_buildings
        self.load_masks = load_masks

        self.img_paths = []
        self.mask_paths = []
        self.filenames = []

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

        # Collect image and mask paths:
        for root, _, fnames in sorted(file_list):
            for fname in sorted(fnames):
                if 'jpg' in fname or 'png' in fname:
                    if 'mask' in fname:
                        continue
                    img_path = os.path.join(root, fname)

                    if self.load_masks:
                        _, file_extension = os.path.splitext(img_path)
                        mask_filename = fname.replace(
                            file_extension, '-mask.png')
                        mask_path = os.path.join(root, mask_filename)
                        if not os.path.isfile(mask_path):
                            print(f'No mask for {fname}. Skipping')
                            continue
                        self.mask_paths.append(mask_path)
                    self.filenames.append(fname)
                    self.img_paths.append(img_path)

    def __len__(self):
        """
        Return the number of images in the dataset.

        This method is used to determine the length of the dataset, i.e., how
        many images are available to load. It counts the number of valid
        image paths in the dataset.

        Returns:
            int:
                The number of images in the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        """
        Retrieve image and filename by index, applying transformations if any.

        Args:
            idx (int):
                Index of the image.

        Returns:
            tuple:
                Transformed image and corresponding filename.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_paths[idx]
        image = Image.open(img_name).convert('RGB')

        # Apply building masking if needed:
        if self.mask_buildings and self.load_masks:
            image = np.array(image)
            mask_filename = self.mask_paths[idx]
            mask = Image.open(mask_filename)
            mask = np.array(mask)
            # Filter building labels
            mask[np.where((mask != 25) & (mask != 1))] = 0
            image[mask == 0, :] = 0
            image = Image.fromarray(np.uint8(image))

        # Apply any given transformation:
        if (self.transform):
            image = self.transform(image)

        fname = self.filenames[idx]
        return (image, fname)


class Foundation_Type_Binary(Dataset):
    def __init__(self,
                 image_folder,
                 transform=None,
                 mask_buildings=False,
                 load_masks=False):

        self.transform = transform
        self.classes = ['Raised', 'Not Raised']
        self.img_paths = []
        self.mask_paths = []
        labels = []
        self.mask_buildings = mask_buildings
        self.load_masks = load_masks

        assert os.path.isdir(image_folder), \
            f'Image folder {image_folder} not found or not a path'

        for root, _, fnames in sorted(os.walk(image_folder, followlinks=True)):
            for fname in sorted(fnames):
                if 'jpg' in fname or 'png' in fname:
                    if 'mask' in fname:
                        continue
                    img_path = os.path.join(root, fname)

                    _, file_extension = os.path.splitext(img_path)
                    mask_filename = fname.replace(file_extension, '-mask.png')
                    mask_path = os.path.join(root, mask_filename)
                    if not os.path.isfile(mask_path):
                        print('No mask for {}. Skipping'.format(fname))
                        continue

                    labels.append(os.path.dirname(
                        img_path).split(os.path.sep)[-1])
                    self.img_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        self.train_labels = np.zeros(len(labels))

        for class_id in ['5001', '5005', '5002', '5003']:
            idx = np.where(np.array(labels) == class_id)[0]
            self.train_labels[idx] = 0
        for class_id in ['5004', '5006']:  # Piles Piers and Posts
            idx = np.where(np.array(labels) == class_id)[0]
            self.train_labels[idx] = 1

        # Train weights for optional weighted sampling
        self.train_weights = np.ones(len(self.train_labels))
        self.train_weights[self.train_labels == 0] = np.sum(
            self.train_labels == 0) / len(self.train_labels)
        self.train_weights[self.train_labels == 1] = np.sum(
            self.train_labels == 1) / len(self.train_labels)
        self.train_weights = 1-self.train_weights

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_paths[idx]
        image = Image.open(img_name).convert('RGB')

        if self.mask_buildings and self.load_masks:
            image = np.array(image)
            mask_filename = self.mask_paths[idx]
            mask = Image.open(mask_filename)
            mask = np.array(mask)
            # Filter building labels
            mask[np.where((mask != 25) & (mask != 1))] = 0
            image[mask == 0, :] = 0
            image = Image.fromarray(np.uint8(image))

        class_id = torch.FloatTensor([self.train_labels[idx]])

        if (self.transform):
            image = self.transform(image)
        return (image, class_id, idx)
