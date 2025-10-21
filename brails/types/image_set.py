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
# 10-21-2025

"""
This module contains classes for managing and manipulating sets of image.

Classes:
    - Image: A class to represent an individual image.
    - ImageSet: A class for handling collections of images.

.. autosummary::
    Image
    ImageSet
"""

import hashlib
from typing import Any, Dict, Iterator, List, Optional, Union
import os

class Image:
    """
    Represents an image and its associated metadata.

    To import the :class:`Image` class, use:

    .. code-block:: python

        from brails.types.image_set import Image

    Parameters:
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

    def hash_image(self):
        """
        Generate a unique hash for image based on its filename and properties.
        
        This method concatenates the image's filename and its associated 
        properties, converts them to strings, and computes an MD5 hash. The
        resulting hexadecimal string can be used for efficient duplicate 
        detection, for example, to identify identical images with the same 
        filenames and attributes.
        
        Returns:
            str: Hexadecimal string representing the MD5 hash of the image.
        
        Example:
            >>> image1 = Image(
            ...     filename='roof_001.jpg',
            ...     properties={'resolution': '1024x768', 'format': 'JPEG'}
            ... )
            >>> image2 = Image(
            ...     filename='roof_002.jpg',
            ...     properties={'resolution': '1024x768', 'format': 'JPEG'}
            ... )
            >>> hash1 = image1.hash_image()
            >>> print(hash1)
            30e317bf41042c6f1d20d66599234139
            >>> hash2 = image2.hash_image()
            >>> print(hash2)
            b03f805070d9fbcc9fc82fa217f16b90
            >>> hash1 == hash2
            False
        """
        filename_str = str(self.filename)
        feat_str = str(self.properties)
        return hashlib.md5((filename_str + feat_str).encode()).hexdigest()

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


class ImageSet:
    """
    A collection of Image objects.

    The :class:`ImageSet` class can be imported using the syntax below:

    .. code-block:: python

        from brails.types.image_set import ImageSet

    Parameters:
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

            >>> img = Image('facade.jpg')
            >>> img_set.add_image(1, img)
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
            >>> img1 = Image('roof.jpg')
            >>> img2 = Image('facade.jpg')
            >>> img_set.add_image(1, img1)
            >>> img_set.add_image(2, img2)
            True
            >>> for img in img_set:
            ...     print(img.filename)
            roof.jpg
            facade.jpg
        """
        return iter(self.images.values())

    def add_image(self, image_id: Union[str, int], image: Image) -> bool:
        """
        Add an image to the image set.

        Args:
            image_id (Union[str, int]):
                Unique key to identify the image in the set.
            image (Image):
                The image instance to be added.
    
        Returns:
            bool:
                ``True`` if the image was successfully added.
                ``False`` if the ``image_id`` already exists in the image set.
    
        Raises:
            TypeError:
                If ``image`` is not an instance of the :class:`Image` class.

        Examples:
            >>> img_set = ImageSet()
            >>> img1 = Image(
            ...     'building_front.jpg',
            ...     {'width': 1920, 'height': 1080}
            ... )
            >>> img_set.add_image(1, img1)
            True

            >>> img2 = Image('street_view.png')
            >>> img_set.add_image(1, img2)  # Duplicate key
            Image with id 1 already exists. Image was not added.
            False
        """
        if not isinstance(image, Image):
            raise TypeError("Expected an instance of Image.")

        if image_id in self.images:
            print(f'Image with id {image_id} already exists. Image was not '
                  'added.')
            return False

        self.images[image_id] = image
        return True

    def combine(
            self, 
            imageset_to_combine: 'ImageSet', 
            key_map: dict = None
        ) -> dict:
        """
        Combine two ImageSet objects into a new merged ImageSet, avoiding duplicate images.
    
        Images are compared using their hashed pixel or metadata representation.
        Duplicate images (identical data) are skipped.
        Images from imageset2 can have their keys remapped using key_map, and
        any resulting key conflicts are automatically resolved by assigning new unique IDs.
    
 
        Args:
            imageset_to_combine (ImageSet):  
                The ImageSet whose images will be merged into this ImageSet.  
            key_map (dict, optional):  
                A dictionary mapping original keys from ``imageset_to_combine``
                to new keys in the merged ImageSet. Example: ``{"img_01": 
                "new_img_A", "img_02": "new_img_B"}``.  If not provided, keys
                are used as-is.
    
        Returns:
            dict:  
                A mapping from each original key in ``imageset_to_combine`` to
                its final key in the combined image set (after applying 
                ``key_map`` and resolving duplicates or key conflicts).
    
        Example:

            The following example demonstrates how two ``ImageSet`` objects
            are merged. The base image set initially contains one image 
            ('img_01'). The secondary set has two images: one unique 
            ('fileB.jpg') and one duplicate of the existing file ('fileA.jpg').
            A key mapping is provided so 'img_02' in the secondary set becomes
            'img_04' in the merged set. 
            
            Please note that, after combining the two sets, only the unique
            image ('fileB.jpg') is added with the new key 'img_04'. The 
            final inventory contains two images, 'img_01' and 'img_04', 
            confirming that 1) the merge successfully preserved unique entries,
            and 2) resolved key conflicts automatically.
    
            >>> base_set = ImageSet()
            >>> _ = base_set.add_image('img_01', Image('fileA.jpg'))
            >>> other_set = ImageSet()
            >>> _ = other_set.add_image('img_02', Image('fileB.jpg'))
            >>> _ = other_set.add_image('img_03', Image('fileA.jpg'))
            >>> key_map = {'img_02': 'img_04', 'img_03': 1}
            >>> merged_keys = base_set.combine(other_set, key_map)
            >>> print(merged_keys)
            {'img_02': 'img_04'}
            >>> base_set.print_info()
            Directory: 
            Total number of images: 2
            List of Images
            ----------------
            - key: img_01, filename: fileA.jpg
            - key: img_04, filename: fileB.jpg
        """
        # Build hash lookup for existing images:
        existing_hashes = {
            image.hash_image(): key for key, image in self.images.items()
        }
    
        # Determine next available numeric ID:
        next_id = self._get_next_numeric_id()
    
        # Track key mapping from imageset_to_combine → self.images:
        merged_key_map = {}
    
        for orig_key, image in imageset_to_combine.images.items():
            image_hash = image.hash_image()
    
            # Skip duplicates based on hash:
            if image_hash in existing_hashes:
                continue
    
            # Apply key mapping if provided:
            mapped_key = key_map.get(orig_key, orig_key) if key_map else orig_key
            new_key = mapped_key
    
            # Ensure key uniqueness:
            while new_key in self.images:
                new_key = next_id
                next_id += 1
    
            # Add image and record mapping:
            self.add_image(new_key, image)
            merged_key_map[orig_key] = new_key
            existing_hashes[image_hash] = new_key
    
        return merged_key_map

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
            >>> img_in = Image('roof_detail.jpg')
            >>> img_set.add_image('main', img_in)
            True

            >>> img_out = img_set.get_image('main')
            >>> img_out.filename
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
            >>> img1 = Image('north_building.jpg')
            >>> img2 = Image(
            ...     'east_building.jpg',    
            ...     {'location': (36.0142, -75.6679)}
            ... )
            >>> img_set.add_image('north_view', img1)
            >>> img_set.add_image('east_view', img2)
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

    def remove_image(self, image_id: Union[str, int]) -> bool:
        """
        Remove an image from the image set.

        Args:
            image_id (Union[str, int]):
                Unique key to identify the image in the set.

        Returns:
            bool: ``True`` if image was removed, ``False`` otherwise.

        Example:
            >>> img_set = ImageSet()
            >>> img1 = Image(
            ...     'east_building.jpg',    
            ...     {'width': 1920, 'height': 1080}
            ... )
            >>> img_set.add_image('img1', img1)
            True
            >>> img_set.remove_image('img1')
            True
            >>> len(img_set)
            0
        """
        if image_id in self.images:
            del self.images[image_id]
            return True
        else:
            return False

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

    def _get_next_numeric_id(self) -> int:
        """
        Compute the next available numeric image ID in the image set.
    
        Returns:
            int:
                The next available numeric ID (max numeric key + 1).
                Returns 0 if the image set is empty, contains no numeric keys,
                or cannot be accessed.
    
        Notes:
            - Non-numeric keys are ignored.
            - If image set access or key conversion fails, the function
              safely falls back to returning 0.
            - This function is typically used to generate sequential
              numeric identifiers for new images.
        """
        try:
            keys = getattr(self.images, 'keys', lambda: [])()
            numeric_ids = [int(k) for k in keys if str(k).isdigit()]
            return (max(numeric_ids) + 1) if numeric_ids else 0
        except Exception:
            return 0