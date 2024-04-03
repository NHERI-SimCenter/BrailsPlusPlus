from abc import ABC, abstractmethod

from brails.types.building_inventory import AssetInventory

class RegionBoundary(ABC):
   """                                                                                                                                                 
    Abstract base class representing a class that obtains a polygon surrounding region a region
   
     Methods:
        get_Boundary(): To return the boundary points
   
   """
   
   @abstractmethod
   def get_Boundary(self) -> tuple:
      
      """An abstract class that must be implemented by subclasses.
      
      This method will be used by the caller to obtain the boundary.
      
      Returns:
            tuple: An immutable list of points defining the boundary of the region.
   
      """
      pass


