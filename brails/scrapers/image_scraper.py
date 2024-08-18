# Writtten: fmk (abstract class)  03/24
#           Barbaros Cetiner (subclass needed methods: _polygon_area and create_asset_inventory)
# license: BSD-3 (see LICENSCE.txt file: https://github.com/NHERI-SimCenter/BrailsPlusPlus)

"""
This module defines abstract FootprintScraper class

.. autosummary::

    ImageScraper
"""

from abc import ABC, abstractmethod

from brails.types.image_set import ImageSet
from brails.types.asset_inventory import AssetInventory

class ImageScraper(ABC):
    """
    Abstract base class representing a class that obtains images given an AssetInventory

      Methods:
         get_images(inventory): An abstract method to return an ImageSet given an AssetInventory
    """

    def __init__(self, name):
        self.name = name


    @abstractmethod
    def get_images(self, inventory: AssetInventory, dir: str) -> ImageSet:
        """
        An abstract class that must be implemented by subclasses.

        This method will be used by the caller to obtain the images for assets in an area.

        Args:
              inventory (AssetInventory):
                   The AssetInventory.

        Returns:
              Image_Set:
                    An image_Set for the assets in the inventory.

        """
        pass

