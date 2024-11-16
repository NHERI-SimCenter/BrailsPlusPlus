"""Class object to train and use image classification models."""
#
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
# Last updated:
# 11-16-2024

import time
import os
import copy
import zipfile
from typing import Union, Dict, List, Tuple

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from brails.types.image_set import ImageSet


MODEL_PROPERTIES = {
    'convnext_t': {'model': "models.convnext_tiny(weights='IMAGENET1K_V1')",
                   'input_size':
                   models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms.
                   keywords['resize_size']},
    'convnext_s': {'model': "models.convnext_small(weights='IMAGENET1K_V1')",
                   'input_size':
                   models.ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms.
                   keywords['resize_size']},
    'convnext_b': {'model': "models.convnext_base(weights='IMAGENET1K_V1')",
                   'input_size':
                   models.ConvNeXt_Base_Weights.IMAGENET1K_V1.transforms.
                   keywords['resize_size']},
    'convnext_l': {'model': "models.convnext_large(weights='IMAGENET1K_V1')",
                   'input_size':
                   models.ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms.
                   keywords['resize_size']},
    'efficientnetv2_s': {'model':
                         "models.efficientnet_v2_s(weights='IMAGENET1K_V1')",
                         'input_size':
                         models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.
                         transforms.keywords['resize_size']},
    'efficientnetv2_m': {'model':
                         "models.efficientnet_v2_m(weights='IMAGENET1K_V1')",
                         'input_size':
                         models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.
                         transforms.keywords['resize_size']},
    'efficientnetv2_l': {'model':
                         "models.efficientnet_v2_l(weights='IMAGENET1K_V1')",
                         'input_size':
                         models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.
                         transforms.keywords['resize_size']},
    'regnet_16': {'model': "models.regnet_x_16gf(weights='IMAGENET1K_V2')",
                  'input_size': models.RegNet_X_16GF_Weights.IMAGENET1K_V2.
                  transforms.keywords['resize_size']},
    'regnet_32': {'model': "models.regnet_x_32gf(weights='IMAGENET1K_V2')",
                  'input_size': models.RegNet_X_32GF_Weights.IMAGENET1K_V2.
                  transforms.keywords['resize_size']},
    'resnet_50': {'model': "models.resnet50(weights='IMAGENET1K_V2')",
                  'input_size': models.ResNet50_Weights.IMAGENET1K_V2.
                  transforms.keywords['resize_size']},
    'resnet_101': {'model': "models.resnet101(weights='IMAGENET1K_V2')",
                   'input_size': models.ResNet101_Weights.IMAGENET1K_V2.
                   transforms.keywords['resize_size']},
    'resnet_152': {'model': "models.resnet152(weights='IMAGENET1K_V2')",
                   'input_size': models.ResNet152_Weights.IMAGENET1K_V2.
                   transforms.keywords['resize_size']},
    'vit_h14': {'model': "models.vit_b_16(weights='IMAGENET1K_SWAG_E2E_V1')",
                'input_size': models.ViT_B_16_Weights.
                IMAGENET1K_SWAG_E2E_V1.transforms.keywords['resize_size']},
    'vit_b16': {'model': "models.vit_h_14(weights='IMAGENET1K_SWAG_E2E_V1')",
                'input_size': models.ViT_H_14_Weights.
                IMAGENET1K_SWAG_E2E_V1.transforms.keywords['resize_size']},
}


class ImageClassifier():
    """
    A class to manage and train an image classification model.

    This class provides methods to download train models using transfer
    learning, retrain existing models, and make predictions using these models.
    The class supports using different model architectures and performs
    operations on GPU if available.

    Attributes__
    - device (torch.device): The device (GPU or CPU) on which the model will be
        trained and evaluated.
    - implemented_architectures (List[str]): List of supported model
        architectures.
    """

    def __init__(self):
        """
        Initialize the class instance and sets up essential attributes.

        Attributes:
            device (torch.device):
                Specifies the device to be used for computations. It will use
                a GPU (CUDA) if available; otherwise, it defaults to the CPU.
            implemented_architectures (list):
                A list of supported model architectures, derived from the keys
                in the `MODEL_PROPERTIES` dictionary.

        Notes:
        - The `torch.device` setting ensures compatibility with the hardware
          available on the system.
        - Ensure that `MODEL_PROPERTIES` is defined and contains information
          about the supported model architectures before using this class.
        """
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.implemented_architectures = list(MODEL_PROPERTIES.keys())

    def _download_default_dataset(self):
        """
        Download and prepare the default dataset for training.

        This is a private method and is not intended to be accessed directly.
        Implement this method to handle dataset downloading and setup.
        """
        # URL of the default dataset
        url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'

        # Define the path where the downloaded zip file will be saved
        zipdir = os.path.join('tmp', url.split('/')[-1])

        datadir = 'tmp/hymenoptera_data'

        if not (os.path.exists(zipdir) and os.path.exists(datadir)):
            print('Downloading default dataset...')

            # Send an HTTP GET request to download the dataset
            req = requests.get(url)

            # Create the 'tmp' directory if it does not exist
            os.makedirs('tmp', exist_ok=True)

            # Write the content of the response (dataset) to a file
            with open(zipdir, 'wb') as output_file:
                output_file.write(req.content)

            # Extract the contents of the zip file to the 'tmp' directory
            with zipfile.ZipFile(zipdir, 'r') as zip_ref:
                zip_ref.extractall('tmp')

            print(f'Default dataset is available in {datadir}')

        else:
            print(f'Default dataset found in {datadir}')

    def _train_model(self,
                     model: nn.Module,
                     device: torch.device,
                     dataloaders: Dict[str, DataLoader],
                     criterion: nn.Module,
                     optimizer: optim.Optimizer,
                     num_epochs: int,
                     es_tolerance: int
                     ) -> Tuple[nn.Module, List[float]]:
        """
        Train a model for a specified number of epochs.

        This function performs training and validation phases over a given
        number of epochs, evaluates model performance, and implements
        early stopping based on validation accuracy.

        Args:
            model (nn.Module):
                The model to be trained.
            device (torch.device):
                The device (GPU or CPU) on which the model and data are
                located.
            dataloaders (dict[str, DataLoader]):
                A dictionary containing 'train' and 'val' DataLoaders for
                training and validation phases.
            criterion (nn.Module):
                The loss function used to compute the loss.
            optimizer (optim.Optimizer):
                The optimizer used to update model parameters.
            num_epochs (int):
                The number of epochs to train the model.
            es_tolerance (int):
                The number of epochs with no improvement after which training
                will be stopped early.

        Returns:
            Tuple[nn.Module, List[float]]:
                - The trained model with the best weights.
                - A list of validation accuracies recorded at each epoch during
              training.

        Prints:
        - Training and validation loss and accuracy for each epoch.
        - The best validation accuracy achieved.
        - The elapsed time of the training process.

        Example:
        >>> model, val_acc_history = _train_model(
        ...     model=my_model,
        ...     device=torch.device('cuda:0'),
        ...     dataloaders={'train': train_loader, 'val': val_loader},
        ...     criterion=nn.CrossEntropyLoss(),
        ...     optimizer=optim.Adam(my_model.parameters(), lr=0.001),
        ...     num_epochs=25,
        ...     es_tolerance=5
        ... )
        """
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        es_counter = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double(
                ) / len(dataloaders[phase].dataset)

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: '
                      f'{epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    es_counter = 0
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    es_counter += 1
                    val_acc_history.append(epoch_acc)
            if es_counter >= es_tolerance:
                print('Early termination criterion satisfied.')
                break
            print()

        time_elapsed = time.time() - since
        print(f'Best val Acc: {best_acc:4f}'.format(best_acc))
        print(f'Elapsed time: {time_elapsed // 60:.0f}m '
              f'{time_elapsed % 60:.0f}s')
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    def _set_parameter_requires_grad(self,
                                     model: nn.Module,
                                     feature_extracting: bool
                                     ) -> nn.Module:
        """
        Set `requires_grad` attribute of the parameters of a PyTorch model.

        If `feature_extracting` is set to `True`, the model's parameters will
        have `requires_grad` set to `False`, freezing the layers so they are
        not updated during training. Otherwise, all parameters will be set to
        `requires_grad = True`, making them trainable.

        Args:
            model (nn.Module):
                The model whose parameters' `requires_grad` attribute will be
                modified.
            feature_extracting (bool):
                If `True`, freezes all parameters of the model. If `False`,
                makes all parameters trainable.

        Returns:
            nn.Module:
                The modified model with the appropriate `requires_grad`
                settings.

        Example:
        >>> model = models.resnet18(pretrained=True)
        >>> model = _set_parameter_requires_grad(model,
                                                 feature_extracting=True)
        """
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
        return model

    def _data_augmenter(self,
                        model_inp_size: int
                        ) -> Dict[str, transforms.Compose]:
        """
        Create data transformations for training and validation datasets.

        Args:
            model_inp_size (int):
                The input size to which the images will be resized for the
                model.

        Returns:
            Dict[str, transforms.Compose]:
                A dictionary containing the data transformations for the
                'train' and 'val' datasets.
                - 'train': Resizes images, applies random horizontal flipping,
                    converts them to tensors, and normalizes using ImageNet
                    statistics.
                - 'val': Resizes images, converts them to tensors, and
                normalizes using ImageNet statistics.

        Example:
        >>> transforms_dict = data_transformer(224)
        >>> train_transforms = transforms_dict['train']
        >>> val_transforms = transforms_dict['val']
        """
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(model_inp_size),
                transforms.CenterCrop(model_inp_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(model_inp_size),
                transforms.CenterCrop(model_inp_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
        }
        return data_transforms

    def train(
        self,
        model_arch: str = 'efficientnetv2_s',
        train_data_dir: str = 'tmp/hymenoptera_data',
        batch_size: int = 32,
        nepochs: Union[int, List[int]] = 100,
        es_tolerance: int = 10,
        plot_accuracy: bool = True
    ) -> None:
        """
        Train a model using transfer learning.

        Parameters__
        - model_arch (str): The architecture of the model to use
            (e.g., 'efficientnetv2_s').
        - train_data_dir (str): The directory where the training and validation
            data is located.
        - batch_size (int): The number of samples per batch.
        - nepochs (Union[int, List[int]]): Number of epochs for initial
            training and fine-tuning.
            If an integer, it will be split into two halves for initial
            training and fine-tuning.
            If a list of two integers, it will use the two values as epochs for
            initial training and fine-tuning respectively.
        - es_tolerance (int): Number of epochs with no improvement after which
            training will be stopped early.
        - plot_accuracy (bool): Whether to plot the validation accuracy
            against the number of training epochs.

        Returns__
        - None: This method does not return any value.

        Raises__
        - ValueError: If `nepochs` is not an integer or a list of two integers.
        - NotImplementedError: model_arch is not defined

        Example__
        >>> trainer = ImageClassifier()
        >>> trainer.train(
        ...     model_arch='resnet50',
        ...     train_data_dir='path/to/data',
        ...     batch_size=64,
        ...     nepochs=[10, 20],
        ...     es_tolerance=5,
        ...     plot_accuracy=True
        ... )
        New classifier head trained using transfer learning.
        Fine-tuning the model...
        Training complete.
        """

        def initialize_model(modelname, num_classes, feature_extract):
            # Load model:
            # modelname = ''.join(filter(str.isalnum, model_name.lower()))
            model_ft = eval(MODEL_PROPERTIES[modelname]['model'])
            model_ft = self._set_parameter_requires_grad(model_ft,
                                                         feature_extract)

            if "resnet" in modelname or "regnet" in modelname:
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, num_classes)

            elif "efficientnetv2" in modelname or "convnext" in modelname:
                num_ftrs = model_ft.classifier[-1].in_features
                model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)

            elif "vit" in modelname:
                num_ftrs = model_ft.heads.head.in_features
                model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
            else:
                raise NotImplementedError("Model name or architecture not "
                                          "defined!")

            return model_ft

        self.model_arch = model_arch.lower()
        self.train_data_dir = train_data_dir
        self.batch_size = batch_size
        self.es_tolerance = es_tolerance

        # Check if the training data directory is set to the default value
        if self.train_data_dir == 'tmp/hymenoptera_data':
            self._download_default_dataset()
        classes = os.listdir(os.path.join(self.train_data_dir, 'train'))
        self.classes = sorted(classes)
        num_classes = len(self.classes)

        # Get the number of epochs for initial training and finetuning:
        if isinstance(nepochs, int):
            nepochs_it = round(nepochs/2)
            nepochs_ft = nepochs - nepochs_it
        elif isinstance(nepochs, list) and len(nepochs) >= 2:
            nepochs_it = nepochs[0]
            nepochs_ft = nepochs[1]
        else:
            raise ValueError('Incorrect nepochs entry. Number of epochs should'
                             ' be defined as an integer or a list of two '
                             'integers!')

        self.nepochs = [nepochs_it, nepochs_ft]

        # Initialize the model for this run
        model_ft = initialize_model(
            self.model_arch, num_classes, feature_extract=False)

        self.model_inp_size = MODEL_PROPERTIES[self.model_arch]['input_size']

        # Data augmentation & normalization for training and normalization for
        # validation:
        data_transforms = self._data_augmenter(self.model_inp_size)

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(
            self.train_data_dir, x), data_transforms[x]) for x in
            ['train', 'val']}

        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=self.batch_size, shuffle=True,
            num_workers=0) for x in ['train', 'val']}

        # Send the model to GPU
        model_ft = model_ft.to(self.device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with
        # requires_grad is True.
        params_to_update = model_ft.parameters()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=0.001)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist = self._train_model(model_ft, self.device,
                                           dataloaders_dict, criterion,
                                           optimizer_ft, num_epochs=nepochs_it,
                                           es_tolerance=self.es_tolerance)
        print('New classifier head trained using transfer learning.')

        # Initialize the non-pretrained version of the model used for this run
        print('\nFine-tuning the model...')
        model_ft = self._set_parameter_requires_grad(model_ft,
                                                     feature_extracting=False)
        final_model = model_ft.to(self.device)
        final_optimizer = optim.Adam(
            final_model.parameters(), lr=0.0001)
        final_criterion = nn.CrossEntropyLoss()
        _, final_hist = self._train_model(final_model, self.device,
                                          dataloaders_dict, final_criterion,
                                          final_optimizer,
                                          num_epochs=nepochs_ft,
                                          es_tolerance=self.es_tolerance)
        print('Training complete.')
        os.makedirs('tmp/models', exist_ok=True)
        torch.save(final_model, 'tmp/models/trained_model.pth')
        self.model_path = 'tmp/models/trained_model.pth'

        # Plot the training curves of validation accuracy vs. number
        #  of training epochs for the transfer learning method and
        #  the model trained from scratch

        plothist = [h.cpu().numpy() for h in hist] + [h.cpu().numpy()
                                                      for h in final_hist]
        self.accuracy = plothist
        if plot_accuracy:
            plt.title("Validation Accuracy vs. Number of Training Epochs")
            plt.xlabel("Training Epochs")
            plt.ylabel("Validation Accuracy")
            plt.plot(range(1, len(plothist)+1), plothist)
            plt.ylim((0.4, 1.))
            plt.xticks(np.arange(1, len(plothist)+1, 1.0))
            plt.show()

    def retrain(self,
                model_arch: str = 'efficientnetv2_s',
                model_path: str = 'tmp/models/trained_model.pth',
                train_data_dir: str = 'tmp/hymenoptera_data',
                batch_size: int = 32,
                nepochs: Union[int, List[int]] = 100,
                es_tolerance: int = 10,
                plot_accuracy: bool = True
                ) -> None:
        """
        Retrain existing model using training dataset and hyperparameters.

        Parameters__
        - model_path (str): Path to the pre-trained model to be fine-tuned.
            Default is 'tmp/models/trained_model.pth'.
        - train_data_dir (str): Directory containing the training and
            validation datasets. Default is 'tmp/hymenoptera_data'.
        - model_inp_size (int): Input size for the model, used for resizing
            images in the dataset. Default is 384.
        - batch_size (int): Batch size for data loading. Default is 32.
        - nepochs (Union[int, List[int]]): Number of epochs for training.
            Should be an integer for retraining. Default is 100.
        - es_tolerance (int): Early stopping tolerance; the number of epochs
            without improvement before stopping. Default is 10.
        - plot_accuracy (bool): Whether to plot the validation accuracy over
            epochs. Default is True.

        Returns__
        - None

        Raises__
        - ValueError: If `nepochs` is not provided as an integer during
            retraining.

        Procedure__
        1. Loads the model from the specified path.
        2. Applies data augmentation and normalization to the training dataset
            and normalization to the validation dataset.
        3. If the default training data directory is used, downloads a sample
            dataset.
        4. Prepares PyTorch `DataLoader` objects for the training and
            validation datasets.
        5. Sends the model to the GPU and fine-tunes it using Stochastic
            Gradient Descent (SGD) optimization.
        6. After training, the model is saved at the specified path.
        7. If `plot_accuracy` is `True`, plots the validation accuracy versus
            training epochs.

        Example__
        >>> classifier = ImageClassifier()
        >>> classifier.retrain(model_path='model.pth',
                               train_data_dir='my_data',
                               nepochs=50)
        """
        self.model_path = model_path
        self.train_data_dir = train_data_dir
        self.batch_size = batch_size
        self.es_tolerance = es_tolerance

        try:
            self.model_inp_size = MODEL_PROPERTIES[model_arch]['input_size']
        except KeyError:
            raise NotImplementedError(f"The model architecture '{model_arch}' "
                                      "is not implemented")

        # Data augmentation & normalization for training and normalization for
        # validation:
        data_transforms = self._data_augmenter(self.model_inp_size)

        # Check if the training data directory is set to the default value
        if self.train_data_dir == 'tmp/hymenoptera_data':
            self._download_default_dataset()
        classes = os.listdir(os.path.join(self.train_data_dir, 'train'))
        self.classes = sorted(classes)

        if isinstance(nepochs, int):
            self.nepochs = [0, nepochs]
        else:
            raise ValueError('Incorrect nepochs entry. For retraining, number '
                             'of epochs should be defined as an integer')

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(
            self.train_data_dir, x), data_transforms[x])
            for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=self.batch_size, shuffle=True,
            num_workers=0) for x in ['train', 'val']}

        # Send the model to GPU
        model_init = torch.load(self.model_path)
        model = self._set_parameter_requires_grad(
            model_init, feature_extracting=False)
        final_model = model.to(self.device)
        final_optimizer = optim.SGD(
            final_model.parameters(), lr=0.0001, momentum=0.9)
        final_criterion = nn.CrossEntropyLoss()
        print(
            '\nRetraining the model using the data located in '
            f'{self.train_data_dir} folder...')
        _, final_hist = self._train_model(
            final_model, self.device, dataloaders_dict, final_criterion,
            final_optimizer, num_epochs=nepochs,
            es_tolerance=self.es_tolerance)
        print('Training complete.')
        torch.save(final_model, self.model_path)

        # Plot the training curves of validation accuracy vs. number
        #  of training epochs for the transfer learning method and
        #  the model trained from scratch
        plothist = [h.cpu().numpy() for h in final_hist]
        self.accuracy = plothist

        if plot_accuracy:
            plt.plot(range(1, len(plothist)+1), plothist)
            plt.title("Validation Accuracy vs. Number of Training Epochs")
            plt.xlabel("Training Epochs")
            plt.ylabel("Validation Accuracy")
            # plt.ylim((0.4,1.))
            plt.xticks(np.arange(1, len(plothist)+1, 1.0))
            plt.show()

    def predict(self,
                images: ImageSet,
                model_arch: str = 'efficientnetv2_s',
                model_path: str = 'tmp/models/trained_model.pth',
                classes: List[str] = ['Ants', 'Bees']
                ) -> Dict[str, str]:
        """
        Predict the class of each image in the provided image set.

        Args__
            images (ImageSet): An object containing the directory path and
                image data.
            model_arch (str): The architecture of the model to be used for
                prediction.
            model_path (str, optional): Path to the pre-trained model. Defaults
                to 'tmp/models/trained_model.pth'.
            classes (List[str], optional): List of class labels. Defaults to
                ['Ants', 'Bees'].

        Raises__
            NotImplementedError: If the specified model architecture is not
                found in MODEL_PROPERTIES.
            NotADirectoryError: If the directory containing the images is not
                valid.

        Returns__
            Dict[str, str]: A dictionary where the keys are image names and
                the values are the predicted classes.
        """
        self.model_path = model_path
        self.classes = sorted(classes)

        try:
            self.model_inp_size = MODEL_PROPERTIES[model_arch]['input_size']
        except KeyError:
            raise NotImplementedError(f"The model architecture '{model_arch}' "
                                      "is not implemented")

        def image_loader(image_name):
            loader = transforms.Compose([
                transforms.Resize(self.model_inp_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            image = Image.open(image_name).convert("RGB")
            image = loader(image).float()
            image = image.unsqueeze(0)
            return image.to(self.device)

        def is_image(im):
            return im.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and \
                os.path.isfile(im)

        model = torch.load(self.model_path, map_location=self.device)
        model.eval()

        preds = {}
        data_dir = images.dir_path
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"ImageClassifier failed. '{data_dir}' "
                                     "is not a directory")

        for key, im in images.images.items():
            if is_image(im.filename):
                image = image_loader(os.path.join(data_dir, im.filename))
                _, pred = torch.max(model(image), 1)
                preds[key] = classes[pred.item()]
            else:
                preds[key] = None

        # self.preds = preds
        return preds


if __name__ == '__main__':
    pass
