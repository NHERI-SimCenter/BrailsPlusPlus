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
# 08-15-2025

"""
This module contains classes for managing and manipulating sets of image.

Classes:
    - Image: A class to represent an individual image.
    - ImageSet: A class for handling collections of images.

.. autosummary::
    Image
    ImageSet
"""

import os
from typing import Any, Dict, Iterator, List, Optional, Union


class Image:
    """
    Represents an image and its associated metadata.

    To import the :class:`Image` class, use:

    .. code-block:: python

        from brails.types.image_set import Image

    Attributes:
        filename (str):
            The name of the file containing the image.
        properties (dict):
                A dictionary containing metadata or properties related to the
                image, such as camera settings, location, or depth maps.

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
                Optional dictionary of image properties. Defaults to an empty
                dictionary if not provided.

        Examples:
            >>> img = Image('cat.png')
            >>> img.filename
            'cat.png'
            >>> img.properties
            {}

            >>> img = Image('dog.jpg', {'width': 800, 'height': 600})
            >>> img.properties
            {'width': 800, 'height': 600}
        """
        self.filename = filename
        self.properties = properties if properties is not None else {}

    def update_properties(self, additional_properties: Dict[str, Any]) -> None:
        """
        Update image properties.

        Args:
            additional_properties (Dict[str, Any]):
                Key-value pairs to update image properties.

        Example:
            >>> img = Image(
            ...     'gstrt_4769427063_-12213443753.jpg',
            ...     {'width': 640}
            ... )
            >>> img.update_properties({'height': 480, 'cam_elevation': 12.1})
            >>> img.properties
            {'width': 640, 'height': 480, 'cam_elevation': 12.1}
        """
        self.properties.update(additional_properties)

    def print_info(self) -> None:
        """
        Print the image filename and properties.

        If no properties are set, only the filename will be printed.

        Examples:
            >>> img = Image('gstrt_4769427063_-12213443753.jpg')
            >>> img.print_info()
            filename: 'gstrt_4769427063_-12213443753.jpg'

            >>> img.update_properties({'width': 640, 'height': 480})
            >>> img.print_info()
            filename: 'gstrt_4769427063_-12213443753.jpg'
            properties: {'width': 640, 'height': 480}
        """
        if not self.properties:
            print('filename:', self.filename)
        else:
            print('filename:', self.filename, 'properties:', self.properties)

    def update_filename(self, filename: str) -> None:
        """
        Update the filename for the image.

        Args:
            filename (str): New filename for the image.

        Example:
            >>> img = Image('building.jpg')
            >>> img.update_filename('gstrt_4769427063_-12213443753.jpg')
            >>> img.filename
            'gstrt_4769427063_-12213443753.jpg'
        """
        self.filename = filename


class ImageSet:
    """
    A collection of Image objects.

    The :class:`ImageSet` class can be imported using the syntax below:

    .. code-block:: python

        from brails.types.image_set import ImageSet

    Attributes:
        dir_path (str):
            Path to the directory containing image files.
        images (Dict[Union[str, int], Image]):
            Dictionary of :class:`Image` objects keyed by user-defined
            identifiers.
    """

    def __init__(self):
        """Initialize an empty ImageSet."""
        self.dir_path = ''
        self.images: Dict[Union[str, int], Image] = {}

    def __len__(self) -> int:
        """
        Return the number of images in the set.

        Examples:
            >>> img_set = ImageSet()
            >>> len(img_set)
            0

            >>> img_set.add_image(1, "facade.jpg")
            True
            >>> len(img_set)
            1
        """
        return len(self.images)

    def __iter__(self) -> Iterator[Image]:
        """
        Iterate over the images in the set.

        Yields:
            Image: Each :class:`Image` object in the set.

        Examples:
            >>> img_set = ImageSet()
            >>> img_set.add_image(1, "roof.jpg")
            >>> img_set.add_image(2, "facade.jpg")
            True
            >>> for img in img_set:
            ...     print(img.filename)
            roof.jpg
            facade.jpg
        """
        return iter(self.images.values())

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
                ``True`` if image was added; ``False`` if the key already
                exists.

        Examples:
            >>> img_set = ImageSet()
            >>> img_set.add_image(
            ...     1,
            ...     'building_front.jpg',
            ...     {'width': 1920, 'height': 1080}
            ... )
            True

            >>> img_set.add_image(1, 'street_view.png')  # Duplicate key
            False
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
                The :class:`Image` if found; otherwise, ``None``.

        Examples:
            >>> img_set = ImageSet()
            >>> img_set.add_image('main', 'roof_detail.jpg')
            True

            >>> img = img_set.get_image('main')
            >>> img.filename
            'roof_detail.jpg'
        """
        return self.images.get(key)

    def print_info(self) -> None:
        """
        Print information about the image set.

        Displays:
            - The directory path (if set).
            - The number of images in the set.
            - The key, filename, and any properties for each image.

        Examples:
            >>> img_set = ImageSet()
            >>> img_set.add_image('north_view', 'north_building.jpg')
            >>> img_set.add_image(
            ...     'east_view',
            ...     'east_building.jpg',
            ...     {'location': (36.0142, -75.6679)}
            ... )
            True
            >>> img_set.print_info()
            Directory:
            Total number of images: 2
            List of Images
            ----------------
            - key: north_view, filename: north_building.jpg
            - key: east_view, filename: east_building.jpg,
            properties: {'location': (36.0142, -75.6679)}
        """
        print('Directory:', self.dir_path)
        if self.images:
            print(f'Total number of images: {len(self.images)}\n')
            print('List of Images\n----------------')
            for key, image in self.images.items():
                if not image.properties:
                    print(f'- key: {key}, filename: {image.filename}')
                else:
                    print(f'- key: {key}, filename: {image.filename}, ',
                          f'properties: {image.properties}')
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
                If ``True``, add existing image files in the directory
                to the set using numeric keys starting at 1.
            limited_to_extension_types (Optional[List[str]]):
                If provided, only include files whose extensions match
                one of the allowed types (e.g., ``['.jpg', '.png']``).

        Returns:
            bool:
                ``True`` if the directory exists and was set;
                ``False`` if the directory does not exist.

        Examples:
            Assuming the path ``/data/images`` exists:

            >>> img_set = ImageSet()
            >>> img_set.set_directory(
            ...     '/data/images',
            ...     include_existing_images=True,
            ...     limited_to_extension_types=['.jpg', '.png']
            )
            True

            Assuming the path ``/DOES-NOT-EXIST`` does not exist:

            >>> img_set.set_directory('/DOES-NOT-EXIST')
            Warning: the specified path /DOES-NOT-EXIST is not a valid
            directory
            False
        """
        if not os.path.isdir(path_to_dir):
            print(f'Warning: the specified path {path_to_dir} is not a '
                  'valid directory')
            return False

        self.dir_path = path_to_dir

        if include_existing_images:
            # Collect all integer keys from the current images dictionary:
            existing_int_keys = [k for k in self.images.keys()
                                 if isinstance(k, int)]

            # Determine the starting index for new images
            # If there are no existing integer keys, start at 0:
            index = max(existing_int_keys, default=-1) + 1

            # Prepare a list of allowed extensions in lowercase if specified
            # Otherwise, allow all file types:
            extensions_allowed = [e.lower() for e in
                                  limited_to_extension_types] if \
                limited_to_extension_types else None

            # List all entries (files and directories) in the target directory:
            entries = os.listdir(self.dir_path)
            for entry in entries:

                # Get the full path of the entry:
                full_path = os.path.join(self.dir_path, entry)

                # Only proceed if the entry is a file (skip directories):
                if os.path.isfile(full_path):
                    # Extract the file extension in lowercase:
                    ext = os.path.splitext(full_path)[1].lower()

                    # Check if the file extension is allowed. If no
                    # restrictions are set, all files are allowed:
                    if (
                        limited_to_extension_types is None or
                        ext in extensions_allowed
                    ):
                        # Add the file as an Image object to the images
                        # dictionary:
                        self.images[index] = Image(entry)
                        index += 1
        return True
