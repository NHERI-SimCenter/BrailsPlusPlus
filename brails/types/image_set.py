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
# Frank McKenna

#
# Last updated:
# 06-05-2025

"""
This module contains classes for managing and manipulating image sets.

Classes:
    - Image: A class to represent an individual image.
    - ImageSet: A class for handling collections of images.

.. autosummary::
    Image
    ImageSet
"""

import os
from typing import Optional, Dict, Any, Union, List


class Image:
    """
    Represents an image and its associated metadata.

    Attributes:
        filename (str):
            The name of the file containing the image.
        properties (dict):
                A dictionary containing metadata or properties related to the
                image, such as camera settings, location, or depth maps.

    Methods:
        set_image(filename):
            Update the filename associated with the image.
        update_properties(additional_properties):
            Merge new key-value pairs into the existing image properties.
        print_info():
            Print the filename and any associated properties to the console.
    """

    def __init__(
        self,
        filename: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an Image instance.

        Args:
            filename (str):
                The name of the image file.
            properties (Optional[Dict[str, Any]]):
                Optional dictionary of image properties.
        """
        self.filename = filename
        self.properties = properties if properties is not None else {}

    def update_properties(self, additional_properties: Dict[str, Any]) -> None:
        """
        Update the image's properties.

        Args:
            additional_properties (Dict[str, Any]):
                Key-value pairs to update image properties.
        """
        self.properties.update(additional_properties)

    def print_info(self) -> None:
        """Print the image filename and properties, if any."""
        if not self.properties:
            print('filename:', self.filename)
        else:
            print('filename:', self.filename, 'properties:', self.properties)

    def set_image(self, filename: str) -> None:
        """
        Update the filename for the image.

        Args:
            filename (str): New filename for the image.
        """
        self.filename = filename


class ImageSet:
    """
    A collection of Image objects.

    Attributes:
        dir_path (str):
            Path to the directory containing image files.
        images (Dict[Union[str, int], Image]):
            Dictionary of Image objects keyed by user-defined identifiers.

    Methods:
        set_directory(path_to_dir, include_existing_images,
                      limited_to_extension_types):
            Set the directory path and optionally load existing images.
        add_image(key, filename, properties):
            Add a new Image to the collection.
        get_image(key):
            Retrieve an Image by its key.
        print_info():
            Print the directory path and details of all stored images.
    """

    def __init__(self):
        """Initialize an empty ImageSet."""
        self.dir_path = ''
        self.images: Dict[Union[str, int], Image] = {}

    def add_image(
        self,
        key: Union[str, int],
        filename: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create and add a new Image to the ImageSet.

        Args:
            key (Union[str, int]):
                Identifier for the image.
            filename (str):
                Name of the image file.
            properties (Optional[Dict[str, Any]]):
                Optional metadata for the image.

        Returns:
            bool:
                True if image was added; False if the key already exists.
        """
        if properties is None:
            properties = {}

        if key not in self.images:
            image = Image(filename, properties)
            self.images[key] = image
            return True

        return False

    def get_image(self, key: Union[str, int]) -> Optional[Image]:
        """
        Retrieve an image by key.

        Args:
            key (Union[str, int]):
                Identifier for the image.

        Returns:
            Optional[Image]:
                The image if found; otherwise, None.
        """
        return self.images.get(key)

    def print_info(self) -> None:
        """Print the image directory and details of each image in the set."""
        print("directory:", self.dir_path)
        if self.images:
            print("images (num images:", len(self.images), ")\n")
            for key, image in self.images.items():
                if not image.properties:
                    print('key:', key, 'filename:', image.filename)
                else:
                    print('key:', key, 'filename:', image.filename,
                          'properties:', image.properties)
        print('\n')

    def set_directory(
        self,
        path_to_dir: str,
        include_existing_images: bool = False,
        limited_to_extension_types: Optional[List[str]] = None
    ) -> bool:
        """
        Set the image directory and optionally load existing images.

        Args:
            path_to_dir (str):
                Path to the directory containing image files.
            include_existing_images (bool):
                Whether to add existing image files in the directory.
            limited_to_extension_types (Optional[List[str]]):
                If set, only include files with these extensions.

        Returns:
            bool:
                True if the directory is valid and set; False otherwise.
        """
        if not os.path.isdir(path_to_dir):
            print(f'Warning: the supplied directory {path_to_dir} is not a '
                  'valid directory')
            return False

        self.dir_path = path_to_dir

        # If instructed to include current images in the directory:
        # - Retrieve the list of files.
        # - For each file, construct the full path.
        # - Include the file if there is no extension filter, or if its
        # extension matches the allowed list.
        if include_existing_images:

            count = 0
            entries = os.listdir(self.dir_path)
            for entry in entries:

                # Get the full path of the entry:
                full_path = os.path.join(self.dir_path, entry)

                # Check if the entry is a file:
                if os.path.isfile(full_path):

                    if (
                        limited_to_extension_types is None
                        or os.path.splitext(full_path)[1] in
                        limited_to_extension_types
                    ):

                        count += 1
                        self.images[count] = Image(entry)
        return True
