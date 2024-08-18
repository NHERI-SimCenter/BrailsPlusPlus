from abc import ABC, abstractmethod
from brails.types.image_set import ImageSet

"""
This module defines abstract filter class

.. autosummary::

    Filter
"""

class Filter(ABC):
    """
    Abstract base class representing a class that filters an ImageSet

      Methods:
         __init__(dict): Constructor
         get_footprints(location): An abstract method to return the footprint given a location
    """

    
    def __init__(self, input_data: dict):
        self.input_data = input_data
    
    @abstractmethod
    def filter(self, images: ImageSet, dir_path: str) ->ImageSet:
        """
        An abstract class that must be implemented by subclasses.

        This method will be used by the caller to obtain a filtered ImageSet

        Args:
              image_set (ImageSet):
                   The input ImageSet to be filtered
              dir_path
                   The path to output dir where filtered images are to be placed        

        Returns:
              ImageSet:
                    The filtered set of images

        """
        pass
