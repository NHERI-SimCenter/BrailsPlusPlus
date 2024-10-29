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
# Frank McKenna
# Barbaros Cetiner
#
# Last updated:
# 10-28-2024

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
# import sys  ERROR HANDLING NEED TO DISCUSSS


class Image:
    """
    A class to represent an image.

    Attributes:
          filename (str): filename of image
          properties (dict): properties of image if known, i.e., camera
              location, depth map

    Methods:
        set_image: Set the filename for an Image.
        update_properties: Update image properties.
        print_info: Print Image properties.
    """

    def __init__(self, filename: str, properties: dict = None):
        """
        Initialize an Image.

        Args:
            filename (str): the name of file containing image
            properties (dict): image properties (default is an empty dict)
        """
        self.filename = filename
        self.properties = properties if properties is not None else {}

    def set_image(self, filename: str):
        """
        Set the filename for an Image.

        Args:
            filename (str): New filename.
        """
        self.filename = filename

    def update_properties(self, additional_properties: dict):
        """
        Update Image properties.

        Args:
        additional_properties (dict): additional properties to update the
            current Image properties
        """
        self.properties.update(additional_properties)

    def print_info(self):
        """Print Image properties."""
        if not self.properties:
            print('filename: ', self.filename)
        else:
            print('filename: ', self.filename,
                  ' properties: ', self.properties)


class ImageSet:
    """
    A class representing a set of images.

    Attributes:
        dir_path (str):
              A path to image directory if images all in same location.
        images (dict):
              A dict of all images, key is the id for the image, value is an
              Image

     Methods:
        set_directory: Set the directory path and load existing images.
        add_image: Create and add a new image to the ImageSet.
        get_image: Retrieve an image by its key.
        print_info: Print details of the image set.
    """

    def __init__(self):
        """Initialize an Image set."""
        self.dir_path = ''
        self.images = {}

    def set_directory(
        self,
        path_to_dir: str,
        include_existing_images: bool = False,
        limited_to_extension_types=None,
    ):
        """
        Set the directory path.

        Args:
            path_to_dir (str):
                 The path to the directory.
            include_existing_images(bool):
                 Boolean to indicate if all files in path_to_dir should be
                 added in self.images.
            limited_to_extension_types (list):
                 if include_existing_images will only include files with
                 specific extensions in list, default is None, i.e.
                 include all files

        Returns:
            bool:
                 True if the directory exists and is valid, False otherwise.
        """
        # check valid dir, if true set path:
        if os.path.isdir(path_to_dir):
            self.dir_path = path_to_dir

        else:

            # sys.exit('ERROR: the supplied dir: ',
            #         path_to_dir,
            #         ' is not a valid directory')
            print('Warning: the supplied directory ',
                  path_to_dir, ' is not a valid directory')
            return False

        # if asked to include current images in dir,
        # get list of files and for each file, create full path
        # and add if no limit on file extensions or if the file extension
        # matches one provided.

        if include_existing_images:

            count = 0
            entries = os.listdir(self.dir_path)
            for entry in entries:

                # Get the full path of the entry
                full_path = os.path.join(self.dir_path, entry)

                # Check if the entry is a file
                if os.path.isfile(full_path):

                    if (
                        limited_to_extension_types is None
                        or os.path.splitext(full_path)[1] in
                        limited_to_extension_types
                    ):

                        count += 1
                        image = Image(entry)
                        self.images[count] = image

        return True

    def add_image(self, key: str | int,
                  filename: str,
                  properties: dict = None):
        """
        Create and add a new image to the ImageSet.

        Args:
            key (str|int): The identifier for the image.
            filename (str): The name of the file containing the image.
            properties (dict): Image properties (default is an empty dict).

        Returns:
            bool: True if Image is added, False if Image already exists.
        """
        if properties is None:
            properties = {}

        if key not in self.images:
            image = Image(filename, properties)
            self.images[key] = image
            return True

        return False

    def get_image(self, key: str | int):
        """
        Get an image.

        Args:
            key (str|int): The identifier for the image.

        Returns:
            Image:
                 The image or None if no image with key
        """
        return self.images.get(key)
        #
        # check exists and not there .. TO DO
        #

        # self.images[key] = image

    def print_info(self):
        """Print the image set."""
        # print("\nImageSet : NAME WOULD BE NICE")
        print("directory:", self.dir_path)
        if len(self.images) != 0:
            print("images (num images: ", len(self.images), ")\n")

            for key, image in self.images.items():
                if not image.properties:
                    print('key: ', key, 'filename: ', image.filename)
                else:
                    print('key: ', key, 'filename: ', image.filename,
                          ' properties: ', image.properties)
        print('\n')
