# -*- coding: utf-8 -*-
#
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
# 04-11-2024

# Written: fmk, 3/24

"""
This module defines clesses related to image sets

.. autosummary::
    ImageSet
"""

import os

# import sys  ERROR HANDLING NEED TO DISCUSSS

class Image:
    
    """
    A class to represent an image
        
    Attributes:
          filename (str): filename of image
          properties (dict): properties of image if known, i.e. camera location, depth map,
        
    Methods:
        __init__:
        set_image(self, filename: str)
        update_properties(self, additional_properties: dict)
        print(self)
    """
        
    def __init__(self, filename:str, properties:dict = {}):
        """
            Initialize an Image set
            
            Args:
            filename (str):
              the name of file containing image
            proprties (dict):
              the image properties, by defaullt empty dict
        """        
            
        self.filename = filename 
        self.properties = properties
        
        
    def set_image(self, filename:str):
        """
        To set the filename.
                
        Args:
            filename (str):
       """
            
        self.filename = filename
            
            
    def update_propertiies(self, additional_properties: dict):
        """
            To update the properties
            
            Args:
            additional_proprties (str):
               additional properties ti update the current properties with
        """
            
        self.properties.update(additional_properties)


    def print(self):
        """
            To print the Image.
        """
        if not self.properties:
            print('filename: ',self.filename)
        else:
            print('filename: ',self.filename, ' properties: ', self.properties)                        
            

                
class ImageSet:
    """
    A class representing a set of images.

    Attributes:
        dir_path (str):
              A path to image directory if images all in same location.
        images (dict):
              A dict of all images, key is the id for the image, value is an Image

     Methods:
        __init__:
        set_directory(self,path_to_dir, include_existing_images, limited_to_extensions)
        add_image(self, id, Image)
        get_image(self, id)
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

            # sys.exit('ERROR: the supplied dir: ', path_to_dir, ' is not a valid directory')
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
                        image = Image(entry)
                        self.images[count] = image

        return True

    def add_image(self, key, filename: str, properties: dict = {}):
        """
        To create and add a new image too the set

        Args:

            key:
             the assets key, type could be string or int
            image:
             the image to be added

        Returns:
            bool:
                 True if the Image is added, False if already there
        """

        #
        # create a new image and add .. need to check if files there!
        #

        if key not in self.images:
            image = Image(filename, properties)
            self.images[key] = image
            return True
        
        else:
            
            return False

        
    def get_image(self, key):
        """
        To get an image

        Args:
            key:
               the key corresponding to the image to be returned

        Returns:
            Image:
                 The image or None if no image with key
        """

        return self.images.get(key)
    
        #
        # check exists and not there .. TO DO
        #

        self.images[key] = image        

    def print(self):
        
        """
        To print the image set

        """
        print("ImageSet : NAME WOULD BE NICE")        
        print("directory:", self.dir_path)
        if len(self.images) != 0:
            print("images (num images: ", len(self.images), ")\n")
            
            for key, image in self.images.items():
                if not image.properties:
                    print('key: ', key, 'filename: ',image.filename)
                else:
                    print('key: ', key, 'filename: ',image.filename, ' properties: ', image.properties)                                        

