from abc import ABC, abstractmethod
from brails.types.asset_inventory import AssetInventory


class InferenceEngine(ABC):
    """
    Abstract base class representing a class that adds exra features to assets in the AssetInventory

      Methods:
         infer(inventory): An abstract method to add the extra feature

    """

    @abstractmethod
    def infer(self, input_inventory: AssetInventory) ->AssetInventory: 
        """
        Infer new features for the Assets in an Asset Inventory

        Args:
          input_inventory (AssetInventory): the inventory

        Returns:
           AssetInventory: a new asset inventory with additional features
          
        """
        pass
