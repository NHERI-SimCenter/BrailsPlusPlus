"""Class objects to train and use image segmentation models."""
# -*- coding: utf-8 -*-
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
#
# Last updated:
# 11-20-2024


import os
import csv
import copy
import time
import sys
from pathlib import Path
from typing import Any
from PIL import Image
import torch
import torchvision.models.segmentation as models
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score  # , roc_auc_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors


class DatasetBinary(VisionDataset):
    """
    A PyTorch dataset class for loading binary masks paired with images.

    Args:
        root (str):
            The root directory containing the dataset.
        imageFolder (str):
            The name of the folder containing the images.
        maskFolder (str):
            The name of the folder containing the binary masks.
        transforms (transforms.Compose or None, optional):
            A composition of transforms to apply to both the images and masks.
            If None, no transform will be applied. Default is None.

    Attributes:
        image_names (list[Path]):
            A sorted list of file paths to the images in the dataset.
        mask_names (list[Path]):
            A sorted list of file paths to the binary masks in the dataset.

    Methods:
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(index: int):
            Retrieves a sample (image and corresponding binary mask) at the
            specified index.
    """

    def __init__(self,
                 root: str,
                 imageFolder: str,
                 maskFolder: str,
                 transforms: transforms.Compose | None = None) -> None:
        """
        Initialize the DatasetBinary class.

        Initialize the dataset by verifying the existence of the image and
        mask directories, and loading the file paths.

        Args:
            root (str):
                Root directory of the dataset.
            imageFolder (str):
                Folder name containing the images.
            maskFolder (str): F
            older name containing the binary masks.
            transforms (transforms.Compose or None, optional):
                A transform to apply to both the images and masks.

        Raises:
            OSError: If either the image or mask folder does not exist.
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / imageFolder
        mask_folder_path = Path(self.root) / maskFolder

        # Ensure the existence of the image and mask directories:
        if not image_folder_path.exists():
            raise OSError(f'{image_folder_path} does not exist!')
        if not mask_folder_path.exists():
            raise OSError(f'{mask_folder_path} does not exist!')

        # Load sorted paths for images and masks:
        self.image_names = sorted(image_folder_path.glob("*"))
        self.mask_names = sorted(mask_folder_path.glob("*"))

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int:
                The number of samples in the dataset.
        """
        return len(self.image_names)

    def __getitem__(self,
                    index: int) -> dict[Image, Image]:
        """
        Retrieve the sample (image and corresponding mask) at specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict:
                A dictionary with the following keys:
                    - "image" (PIL.Image.Image): The image at the specified
                      index.
                    - "mask" (PIL.Image.Image): The corresponding binary mask.

        If transforms are provided, they will be applied to both the image and
        the mask.
        """
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]

        # Open the image and mask files:
        with open(image_path, 'rb') as im_file, \
                open(mask_path, 'rb') as mask_file:
            image = Image.open(im_file)
            mask = Image.open(mask_file)

            sample = {"image": image, "mask": mask}

            # Apply transforms if defined:
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])

            return sample


class DatasetRGB(VisionDataset):
    """
    A PyTorch dataset class for loading binary masks paired with images.

    Args:
        root (str):
            The root directory containing the dataset.
        imageFolder (str):
            The name of the folder containing the images.
        maskFolder (str):
            The name of the folder containing the binary masks.
        transforms (transforms.Compose or None, optional):
            A function or transform to apply to both the images and masks.
            If None, no transformation will be applied. Default is None.

    Attributes:
        image_names (list[Path]):
            A sorted list of file paths to the images in the dataset.
        mask_names (list[Path]):
            A sorted list of file paths to the binary masks in the dataset.
    """

    def __init__(self,
                 root: str,
                 imageFolder: str,
                 maskFolder: str,
                 transforms: transforms.Compose | None = None) -> None:
        """
        Initialize the DatasetRGB class.

        Initializes the dataset by verifying the existence of the image and
        mask directories, and loading the file paths of the images and masks.

        Args:
            root (str):
                Root directory of the dataset.
            imageFolder (str):
                Folder name containing the images.
            maskFolder (str):
                Folder name containing the binary masks.
            transforms (transforms.Compose or None, optional):
                A function/transform to apply to both images and masks. Default
                is None.

        Raises:
            OSError:
                If either the image or mask folder does not exist.
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / imageFolder
        mask_folder_path = Path(self.root) / maskFolder

        # Check if the specified folders exist:
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist!")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist!")

        # Sort the paths for consistent ordering:
        self.image_names = sorted(image_folder_path.glob("*"))
        self.mask_names = sorted(mask_folder_path.glob("*"))

    def __len__(self) -> int:
        """
        Return the number of samples (image-mask pairs) in the dataset.

        Returns:
            int:
                The total number of samples in the dataset (i.e., number of
                image-mask pairs).
        """
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        """
        Retrieve a sample from the dataset at a given index.

        Retrieves a sample from the dataset at the specified index. The sample
        consists of an image and its corresponding binary mask. The image is
        loaded as a PIL image and the mask is converted  to a tensor.

        Args:
            index (int):
                The index of the sample to retrieve.

        Returns:
            dict:
                A dictionary with the following keys:
                    - "image" (PIL.Image.Image): The image at the specified
                      index.
                    - "mask" (torch.Tensor): The corresponding binary mask as a
                      tensor.

        If a transformation (`transforms`) is provided, it will be applied to
        both the image and the mask. The mask will be converted to a tensor of
        type `torch.long` after applying the transformation.
        """
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]

        # Open the image and mask files:
        with open(image_path, "rb") as im_file, \
                open(mask_path, "rb") as mask_file:
            image = Image.open(im_file)
            mask = Image.open(mask_file)

            # Prepare the sample dictionary:
            sample = {"image": image, "mask": mask}

            # Apply transformations if they are provided:
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = torch.tensor(
                    np.array(sample["mask"], dtype=np.uint8), dtype=torch.long)

            return sample


class ImageSegmenter:
    """
    A class to manage and train an image segmentation model.

    This class provides methods to download train models using transfer
    learning, retrain existing models, and make predictions using these models.
    The class supports using different model architectures and performs
    operations on GPU if available.

    Attributes:
        model_arch (str):
            The model architecture (e.g., "deeplabv3_resnet101").
        device (torch.device):
            The device for computation, either "cuda:0" for GPU or "cpu".
        batch_size (int, optional):
            The batch size for training, initialized to None.
        nepochs (int, optional):
            The number of epochs for training, initialized to None.
        train_data_dir (str, optional):
            The directory containing training data, initialized to None.
        classes (list[str], optional):
            List of class names for classification, initialized to None.
        loss_history (list[float], optional):
            History of loss values during training, initialized to None.
    """

    def __init__(self, model_arch="deeplabv3_resnet101"):
        """
        Initialize the ImageClassifier instance, set model data.

        Args:
            model_arch (str, optional):
                The architecture of the model. Defaults to
                'deeplabv3_resnet101'. Valid options include:
                - "deeplabv3_resnet50"
                - "deeplabv3_resnet101"
                - "fcn_resnet50"
                - "fcn_resnet101"
            Additional model architectures can be added as needed.

        Raises:
            ValueError:
                If the model architecture is invalid.
        """
        # Validate model architecture:
        valid_architectures = ['deeplabv3_resnet50', 'deeplabv3_resnet101',
                               'fcn_resnet50', 'fcn_resnet101']
        if model_arch.lower() in valid_architectures:
            self.model_arch = model_arch.lower()
        else:
            raise ValueError(f"Invalid model architecture '{model_arch}'. "
                             'Valid options are: '
                             f"{', '.join(valid_architectures)}.")

        # Set device to GPU if available, else CPU:
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize the remaining attributes:
        self.batch_size = None
        self.nepochs = None
        self.train_data_dir = None
        self.classes = None
        self.loss_history = None

    def train(self,
              train_data_dir: str,
              classes: list[str],
              batch_size: int = 2,
              nepochs: int = 100,
              es_tolerance: int = 10,
              plot_loss: bool = True) -> None:
        """
        Train a segmentation model using the specified training input.

        Args:
            train_data_dir (str):
                Directory containing the training data. It should have
                subdirectories for each class.
            classes (list[str]):
                List of class names. The classes should correspond to the
                subdirectories in `train_data_dir`.
            batch_size (int, optional):
                Number of samples per batch. Defaults to 2.
            nepochs (int, optional):
                Number of epochs for training. Defaults to 100.
            es_tolerance (int, optional):
                Number of epochs to wait for improvement before early stopping.
                Defaults to 10.
            plot_loss (bool, optional):
                Whether to plot the training loss curve. Defaults to True.

        Raises:
            FileNotFoundError:
                If the `train_data_dir` is not found.
            ValueError:
                If `classes` is empty or if the number of epochs is invalid.

        Returns:
            None
        """
        nclasses = len(classes)
        if nclasses > 1:
            nlayers = nclasses + 1
        else:
            nlayers = nclasses

        if self.model_arch.lower() == "deeplabv3_resnet50":
            model = models.deeplabv3_resnet50(pretrained=True, progress=True)
            model.classifier = models.deeplabv3.DeepLabHead(2048, nlayers)
        elif self.model_arch.lower() == "deeplabv3_resnet101":
            model = models.deeplabv3_resnet101(pretrained=True, progress=True)
            model.classifier = models.deeplabv3.DeepLabHead(2048, nlayers)
        elif self.model_arch.lower() == "fcn_resnet50":
            model = models.fcn_resnet50(pretrained=True, progress=True)
            model.classifier = models.fcn.FCNHead(2048, nlayers)
        elif self.model_arch.lower() == "fcn_resnet101":
            model = models.fcn_resnet101(pretrained=True, progress=True)
            model.classifier = models.fcn.FCNHead(2048, nlayers)

        self.train_data_dir = train_data_dir
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.classes = classes

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10
        es_counter = 0
        val_loss_history = []

        # Define Optimizer
        modelOptim = torch.optim.Adam(model.parameters(), lr=1e-4)

        if nclasses == 1:
            # Define Loss Function
            lossFnc = torch.nn.MSELoss(reduction='mean')

            # Set Training and Validation Datasets
            dataTransforms = transforms.Compose([transforms.ToTensor()])

            segdata = {
                x: DatasetBinary(root=Path(train_data_dir) / x,
                                 imageFolder="images",
                                 maskFolder="masks",
                                 transforms=dataTransforms)
                for x in ["train", "valid"]
            }
        else:
            # Define Loss Function
            lossFnc = torch.nn.CrossEntropyLoss()

            # Set Training and Validation Datasets
            dataTransforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225]),
            ])

            segdata = {
                x: DatasetRGB(root=Path(train_data_dir) / x,
                              imageFolder="images",
                              maskFolder="masks",
                              transforms=dataTransforms)
                for x in ["train", "valid"]
            }

        dataLoaders = {
            x: DataLoader(segdata[x],
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=0)
            for x in ["train", "valid"]
        }

        # Set Training Device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Create and Initialize Training Log File
        perfMetrics = {"f1-score": f1_score}
        fieldnames = ['epoch', 'train_loss', 'valid_loss'] + \
            [f'train_{m}' for m in perfMetrics.keys()] + \
            [f'valid_{m}' for m in perfMetrics.keys()]
        with open(os.path.join('log.csv'), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        # Train
        startTimer = time.time()
        for epoch in range(1, nepochs+1):
            print('-' * 60)
            print(f"Epoch: {epoch}/{nepochs}")
            batchsummary = {a: [0] for a in fieldnames}

            for phase in ["train", "valid"]:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                # Iterate over data.
                for sample in tqdm(iter(dataLoaders[phase]), file=sys.stdout):
                    inputs = sample['image'].to(device)
                    masks = sample['mask'].to(device)
                    # zero the parameter gradients
                    modelOptim.zero_grad()

                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = lossFnc(outputs['out'], masks)
                        y_pred = outputs['out'].data.cpu().numpy().ravel()
                        y_true = masks.data.cpu().numpy().ravel()

                        for name, metric in perfMetrics.items():
                            if name == 'f1-score':
                                # Use a classification threshold of 0.1
                                if nclasses == 1:
                                    batchsummary[f'{phase}_{name}'].append(
                                        metric(y_true > 0, y_pred > 0.1))
                                else:
                                    f1Classes = np.zeros(nclasses)
                                    nPixels = np.zeros(nclasses)
                                    for classID in range(nclasses):
                                        f1Classes[classID] = metric(
                                            y_true == classID, y_pred[
                                                classID*len(y_true):
                                                (classID+1)*len(y_true)
                                            ] > 0.1)
                                        nPixels[classID] = np.count_nonzero(
                                            y_true == classID)
                                    f1weights = nPixels/(np.sum(nPixels))
                                    f1 = np.matmul(f1Classes, f1weights)
                                    batchsummary[f'{phase}_{name}'].append(f1)
                            else:
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(y_true.astype('uint8'), y_pred))

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            modelOptim.step()
                batchsummary['epoch'] = epoch
                epoch_loss = loss
                batchsummary[f'{phase}_loss'] = epoch_loss.item()
            for field in fieldnames[3:]:
                batchsummary[field] = np.mean(batchsummary[field])
            print((f'train loss: {batchsummary["train_loss"]: .4f}, '
                   f'valid loss: {batchsummary["valid_loss"]: .4f}, '
                   f'train f1-score: {batchsummary["train_f1-score"]: .4f}, '
                   f'valid f1-score: {batchsummary["valid_f1-score"]: .4f}, '))
            with open(os.path.join('log.csv'), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(batchsummary)
                # deep copy the model
                if phase == 'valid' and epoch_loss < best_loss:
                    es_counter = 0
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'valid':
                    es_counter += 1
                    val_loss_history.append(epoch_loss)
            if es_counter >= es_tolerance:
                print('Early termination criterion satisfied.')
                break
            print()

        time_elapsed = time.time() - startTimer
        print('-' * 60)
        print('Training completed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print(f'Lowest validation loss: {best_loss: .4f}')
        print('Training complete.')

        # Load best model weights:
        model.load_state_dict(best_model_wts)

        # Save the best model:
        os.makedirs('tmp/models', exist_ok=True)
        torch.save(model, 'tmp/models/trained_seg_model.pth')
        self.model_path = 'tmp/models/trained_seg_model.pth'

        # Plot the training curves of validation accuracy vs. number
        #  of training epochs for the transfer learning method and
        #  the model trained from scratch
        plothist = [h.cpu().numpy() for h in val_loss_history]
        self.loss_history = plothist

        if plot_loss:
            plt.plot(range(1, len(plothist)+1), plothist)
            plt.title("Validation Losses vs. Number of Training Epochs")
            plt.xlabel("Training Epochs")
            plt.ylabel("Validation Losses")
            # plt.ylim((0.4,1.))
            plt.xticks(np.arange(1, len(plothist)+1, 1.0))
            plt.show()

    def predict(self,
                imdir: str,
                classes: list[str],
                model_path: str = 'tmp/models/trained_seg_model.pth'
                ) -> dict[str, str]:
        """
        Segment images in the specified directory using a pre-trained model.

        Args:
            imdir (str):
                The directory containing the images to be predicted.
            classes (list[str]):
                List of class names corresponding to the model's output
                classes.
            model_path (str, optional):
                Path to the trained model file. Defaults to
                'tmp/models/trained_seg_model.pth'.

        Returns:
            dict[str, str]:
                A dictionary where keys are image filenames and values are
                predicted class labels.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
            NotADirectoryError: If the specified image directory is not a
                directory.
            ValueError: If the image directory is empty or no valid images are
                found.
        """
        self.model_path = model_path
        img = Image.open(imdir)
        # self.classes = sorted(classes)

        # Load the evaluation model:
        modelEval = torch.load(self.model_path)
        modelEval.eval()

        # Run the image through the segmentation model:
        device = self.device

        nclasses = len(classes)
        if nclasses == 1:
            trf = transforms.Compose([transforms.ToTensor()])
        else:
            trf = transforms.Compose([transforms.Resize(640),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])

        inp = trf(img).unsqueeze(0).to(device)
        scores = modelEval.to(device)(inp)['out']
        if nclasses == 1:
            pred = scores.detach().cpu().squeeze().numpy()
            mask = (pred > 0.1).astype('uint8')
            plt.imshow(img)
            plt.title('Image Input')
            plt.show()
            plt.imshow(mask, cmap='Greys')
            plt.title('Model Prediction')
            plt.show()
        else:
            pred = torch.argmax(scores.squeeze(), dim=0).detach().cpu().numpy()
            mask = []
            plt.imshow(img)
            for cl in range(1, nclasses+1):
                colorlist = ['red', 'blue', 'darkorange', 'darkgreen',
                             'crimson', 'lime', 'cyan', 'darkviolet',
                             'saddlebrown']
                mask.append((pred == cl).astype(np.uint8))
                ccmap = colors.ListedColormap([colorlist[cl-1]])
                data_masked = np.ma.masked_where(mask[cl-1] == 0, mask[cl-1])
                plt.imshow(data_masked, interpolation='none',
                           vmin=0, alpha=0.5, cmap=ccmap)
            plt.title('Model Prediction')
            plt.show()
        self.pred = mask


if __name__ == '__main__':
    pass
