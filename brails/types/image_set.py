# Written: fmk, 3/24
# Licensce: BSD-2

"""
This module defines clesses related to image sets

.. autosummary::

    ImageSet
"""

import os


class ImageSet:
    """
    A class representing a set of images.

    Attributes:
        dir_path (str):
              A path to image directory if images all in same location.
        images (dict):
              A dict of all images, key is the id for the image, value is local file name

     Methods:
        __init__:
        set_directory(self,path_to_dir, include_existing_images, limited_to_extensions)
        add_image(self, id, file_name)
        print(self)
    """

    def __init__(self):
        """
        Initialize an Image set
        """

        self.dir_path = ""
        self.images = {}

    def set_directory(
        self,
        path_to_dir,
        include_existing_images: bool = False,
        limited_to_extension_types=None,
    ):
        """
        To set the dir_path.

        Args:

            path_to_dir (str):
                 The path to the directory.
            include_exsiting_images(bool):
                 Boolean to indicate if all files in path_to_dir should be added in self.images
            limited_to_extension_type (list):
                 if include_existing_images will only include files with specific extensions in
                 list, default is None, i.e. include all files

        Returns:
            bool:
                 True if the directory exists and is valid, False otherwise.
        """

        # check valid dir, if true set path
        if os.path.isdir(path_to_dir):
            self.dir_path = path_to_dir
        else:
            print("ERROR: the supplied dir: ", path_to_dir, " is not a valid directory")
            return False

        # if asked to include current images in dir,
        # get list of files and for each file, create full path
        # and add ff no limit on file extension or file extension equals
        # one provided then add image

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
                        or os.path.splitext(full_path)[1] in limited_to_extension_types
                    ):

                        count += 1
                        self.images[count] = entry

        return True

    def print(self):
        """
        To print the image set

        """
        print("directory:", self.dir_path)
        if len(self.images) != 0:
            print("images (num images: ", len(self.images), ")\n")
            for key, value in self.images.items():
                print("\t", key, ":", value)
