# Written: fmk, 3/24
# License: BSD-2

"""
This module defines clesses related to aset ineventories

.. autosummary::

    AssetInventory
    Asset
"""


class AssetInventory:
    """
    A class representing a Asset Inventory.

    Attributes:
        inventory (dict): The inventory stored in a dict accessed by asset_id

     Methods:
        __init__: Constructor that just creats an empty inventory
        add_asset(id, coordinates): to add an asset to the inventory
        add_asset_features(asset_id, features): to append new features to the asset
        get_asset_coordinates(asset_id): to get features of a particular assset
        get_asset_features(asset_id): to coordinatesof a particular assset
    """

    def __init__(self):
        """
        Initialize a Asset Inventory by setting inventory to an empty dict.
        """

        self.inventory = {}

    class Asset:
        """
        A data structure for an asset that holds coordinates and features.

        Attributes:
            coordinates (list):
                  A two-dimensional array of coordinates [[x1, y1],[x2, y2],..[xn,yn]]
            features (dict):
                  The features (attributes) of an asset.

        Methods:
        """

        def __init__(self, asset_id, coordinates, features={}):
            """
            Initialize a Asset Inventory by setting inventory to an empty dict.

            Args:
                asset_id (str):
                    the id of the asset
                coordinates (list):
                    coordinates of the asset
                features (dict):
                    feature dict, if none provided empty dict assumed
            """

            is_two_d = True
            if not isinstance(coordinates, list):
                is_two_d = False
            else:
                for item in coordinates:
                    if not isinstance(item, list):
                        is_two_d = False

            if is_two_d is True:
                self.coordinates = coordinates
            else:
                print(
                    " Error Asset.__init__ cordinates passed for asset ",
                    asset_id,
                    " is not a 2d list",
                )
                self.coordinates = []

            self.features = features

        def add_features(self, additional_features: dict):
            """
            Update the existing features in an asset

            Args:
               additional_features (dict):
                   new features to merge into asset
            """

            self.features.update(additional_features)

    def print_inventory(self):
        """
        To print the asset inventory
        """

        print(self.__class__.__name__)
        print("Inventory stored in: ", self.inventory.__class__.__name__)
        for key, value in self.inventory.items():
            print("key: ", key, " value: ", value)

    def add_asset(self, asset_id: int, coordinates: list, features={}) -> bool:
        """
        To initialize an Asset and add to inventory

        Args:
            asset_id (str):
                  The unique asset id.
            coordinates (list):
                  A two-dimensional list representing the coordinates,
                  [[x1, y1][x2, y2],..,[xN, yN]]

        Returns:
            bool:
                  True if asset was addded, False otherwise.
        """

        asset = self.inventory.get(asset_id, None)

        if asset is not None:
            print(
                "ERROR: AssetInventory.add_asset_feature: asset with id",
                id,
                " already exists",
            )
            return False

        # create asset and add using id as the key
        asset = self.Asset(asset_id, coordinates, features)
        self.inventory[id] = asset

        return True

    def add_asset_features(self, asset_id, new_features: dict):
        """
        Add a asset feature to a asset.

        Args:
           id (str):
                 The unique asset id.
           feature (dict):
                 A dict of features to add for a asset

        Returns:
           bool:
                 Success (True) or Failure (False)
        """

        asset = self.inventory.get(asset_id, None)
        if asset is None:
            print(
                "ERROR: AssetInventory.add_asset_feature: no asset exists with id",
                asset_id,
            )
            return False

        return asset.add_features(new_features)

    def get_asset_features(self, asset_id):
        """
        Get features of a particular asset.

        Args:
            id (str):
                  The unique asset id.

        Returns:
            tuple:
                A tuple containing a boolean value indicating whether the processing was
                successful and the dict of asset features if asset present
        """

        asset = self.inventory.get(asset_id, None)
        if asset is None:
            return False, {}

        return True, asset.features

    def get_asset_coordinates(self, asset_id):
        """
        Add a asset feature to a asset.

        Args:
            asset_id (str):
                 The unique asset id.

        Returns:
            tuple:
                 A tuple containing a boolean value indicating whether the processing was
                 successful and the dict of asset features if asset present

        """

        asset = self.inventory.get(asset_id, None)
        if asset is None:
            return False, []

        return True, asset.coordinates
