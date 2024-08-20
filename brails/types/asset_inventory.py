# Written: fmk, 3/24
# License: BSD-2

"""
This module defines clesses related to aset ineventories

.. autosummary::

    AssetInventory
    Asset
"""

import random


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

    def print(self):
        
        print('\t coords: ' , self.coordinates);
        print('\t features: ' , self.features);        



class AssetInventory:
    """
    A class representing a Asset Inventory.

    Attributes:
        inventory (dict): The inventory stored in a dict accessed by asset_id

     Methods:
        __init__: Constructor that just creats an empty inventory
        print(): to print the inventory
        add_asset(id, coordinates): to add an asset to the inventory with just a list of coordinates
        add_asset(id, Asset): to add an asset to the inventory     
        add_asset_features(asset_id, features): to append new features to the asset
        get_asset_coordinates(asset_id): to get features of a particular assset
        get_asset_features(asset_id): to coordinatesof a particular assset
        get_random_sample(size, seed): to get a smaller subset
        get_footprints(): to return a list of footprints
        get_random_footprints(): to return a random sample of the footprints
        get_geojson(): To return the contents as a geojson dict
    """

    def __init__(self):
        """
        Initialize a Asset Inventory by setting inventory to an empty dict.
        """

        self.inventory = {}

    def print(self):
        """
        To print the asset inventory
        """

        print(self.__class__.__name__)
        print("Inventory stored in: ", self.inventory.__class__.__name__)
        for key, asset in self.inventory.items():
            print("key: ", key, "asset:")
            asset.print()

    def add_asset(self, asset_id: int, coordinates: list) -> bool:
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

        existing_asset = self.inventory.get(asset_id, None)

        if existing_asset is not None:
            print(
                "ERROR: AssetInventory.add_asset_feature: asset with id",
                id,
                " already exists",
            )
            return False

        # create asset and add using id as the key
        asset = Asset(asset_id, coordinates, features)
        self.inventory[asset_id] = asset

        return True


    def add_asset(self, asset_id: int, asset: Asset) -> bool:
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

        existing_asset = self.inventory.get(asset_id, None)

        if existing_asset is not None:
            print(
                "ERROR: AssetInventory.add_asset_feature: asset with id",
                id,
                " already exists",
            )
            return False

        # create asset and add using id as the key
        self.inventory[asset_id] = asset

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

    def get_random_sample(self, number, seed=None): 
        """
        Method to return a smaller AssetInvenntory of randomly selected assets

        Args:
            number (int):
                 The number of assets to be in smaller inventory
            seed (int):
                 The seed for generator, if None provided no seed.

        Returns:
           AssetInventory
                 A smaller inventory of randomly selected assets
        """

        result = AssetInventory()
        if (seed is not None):
            random.seed(seed)

        list_random_keys = random.sample(self.inventory.keys(), number)
        for key in list_random_keys:
            result.add_asset(key, self.inventory[key])
            
        return result

    def get_footprints(self) ->list:
        
        """
        Method to return the footprints of the assets in the invetory

        Args:
        
        Returns:
           list
                 The asset.coordinates of each asset
           keys
                 The asset keys
        """
        
        result_footprints = []
        result_keys = []
        for key, asset in self.inventory.items():
            result_footprints.append(asset.coordinates)
            result_keys.append(key)

        return result_footprints, result_keys

    def get_random_footprints(self, number, seed=None) ->list:
        """
        Method to return the footprints of a number of randomly selected assets in the inventory.

        Args:
            number (int):
                 The number of asset coordinates
            seed (int):
                 The seed for generator, if None provided no seed.
        
        Returns:
           list
                 The asset.coordinates of the random set of assets chosen
        """                

        result_footprints = []
        result_keys = []        
        if (seed is not None):
            random.seed(seed)

        list_random_keys = random.sample(self.inventory.keys(), number)
        for key in list_random_keys:
            asset = self.inventory[key]
            result_footprints.append(asset.coordinates)
            result_keys.append(key)

        return result_footprints, result_keys
        

    def get_geojson(self) ->dict:
        
        """
        Method to return the assets in a GeoJSON dictionary

        Args:
        
        Returns:
           dict: GeoJSON dictionary
        
        """

        geojson = {'type':'FeatureCollection', 
                   "crs": {"type": "name", "properties": 
                           {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                   'features':[]}
        
        for key, asset in self.inventory.items():
            if len(asset.coordinates) == 1:
                   point_feature = {
                       "type": "Feature",
                       "geometry": {
                           "type": "Point",
                           "coordinates": [lon, lat]
                       },
                    "properties": {
                        "name": name
                    }
                   }
                   
                   geometry = {"type":"Point",
                               "coordinates": [asset.coordinates[0][0], asset.ccordinates[0][1]]
                               }
            else:
                geometry = {'type': 'Polygon',
                            'coordinates': asset.coordinates
                            }                
                
                feature = {'type':'Feature',
                           'properties':asset.features,
                           'geometry':geometry
                           }
                if 'type' in asset.features:
                    feature['type'] = asset.features['type']

                geojson['features'].append(feature)
                # here we could put in NA! for imputation and ensure all features have same set of keys!!

        return geojson    
