# Written: fmk, 3/24
# License: BSD-2

"""
This module defines clesses related to aset ineventories

.. autosummary::

    AssetInventory
    Asset
"""

import random
import csv
import numpy as np
import pandas as pd

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

    def add_features(self, additional_features: dict, overwrite=True):
        """
            Update the existing features in an asset
        
            Args:
               additional_features (dict):
                   new features to merge into asset
            """

        if overwrite==True:
            self.features.update(additional_features)
        else:
            additional_features.update(self.features)
            self.features = additional_features

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
        add_asset(asset_id, Asset): to add an asset to the inventory     
        add_asset_coordinates(asset_id, coordinates): to add an asset to the inventory with just a list of coordinates
        add_asset_features(asset_id, features): to append new features to the asset
        add_asset_features_from_csv(file_path, id_column): To add asset features from a csv file
        remove_asset(asset_id): to remove an asset to the inventory     
        get_asset_coordinates(asset_id): to get features of a particular assset
        get_asset_features(asset_id): to coordinatesof a particular assset
        get_random_sample(size, seed): to get a smaller subset
        get_footprints(): to return a list of footprints
        get_random_footprints(): to return a random sample of the footprints
        get_geojson(): To return the contents as a geojson dict
        get_asset_ids(): To return the asset ids as a list
        read_from_csv(file_path, keep_existing, str_type, id_column): To read inventory dataset from a csv table
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

    def add_asset_coordinates(self, asset_id: int, coordinates: list) -> bool:
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
               "ERROR: AssetInventory.add_asset_feature: asset with id {} already exists".format(asset_id),
            )
            return False

        # create asset and add using id as the key
        asset = Asset(asset_id, coordinates)
        self.inventory[asset_id] = asset

        return True


    def add_asset(self, asset_id: int, asset: Asset) -> bool:
        """
        To initialize an Asset and add to inventory

        Args:
            asset_id (str):
                  The unique asset id.
            asset (Asset):
                  An asset.

        Returns:
            bool:
                  True if asset was addded, False otherwise.
        """
        existing_asset = self.inventory.get(asset_id, None)

        if existing_asset is not None:
            print(
                "ERROR: AssetInventory.add_asset_feature: asset with id",
                asset_id,
                " already exists",
            )
            return False

        # create asset and add using id as the key
        self.inventory[asset_id] = asset

        return True    

    def remove_asset(self, asset_id: int) ->bool:
        """
        To remove an Asset

        Args:
            asset_id (str):
                  The unique asset id.

        Returns:
            bool:
                  True if asset was removed, False otherwise.
        """
        del self.inventory[asset_id]

        return True


    def add_asset_features(self, asset_id, new_features: dict, overwrite=True):
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

        return asset.add_features(new_features, overwrite)

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
                # sy - completing incompete code
                geometry = {"type":"Point",
                            "coordinates": [[asset.coordinates[0][0], asset.coordinates[0][1]]]
                           }

                point_feature = {"type": "Feature",
                                  "geometry": geometry,
                                  "properties": asset.features
                                }

                geojson['features'].append(point_feature)

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


    def get_asset_ids(self) ->list:
        
        """
        Method to return the asset ids

        Args:
        
        Returns:
           list: asset ids
        
        """

        return list(self.inventory.keys())


    def read_from_csv(self, file_path, keep_existing, str_type="building", id_column=None) -> bool:
        """
        Read inventory data from a CSV file and add it to the inventory.

        Args:
            file_path (str):
                  The path to the CSV file
            keep_existing (bool):
                  If False, the inventory will be initialized
            str_type (str):
                  "building" or "bridge"
            id_column (str):
                  The name of column that contains id values. If None, new indicies will be assigned

        Returns:
            bool:
                  True if assets were addded, False otherwise.
        """

        def is_float(element: any) -> bool:
            #If you expect None to be passed:
            if element is None: 
                return False
            try:
                float(element)
                return True
            except ValueError:
                return False
            pass


        if keep_existing:
            if len(self.inventory)==0:
                print("No existing inventory found. Creating a new inventory")
                id_counter = 1
            else:
                id_counter = max(self.inventory.keys()) + 1 # we don't want a duplicate the id
        else:
            self.inventory={}
            id_counter = 1

        # Attempt to open the file
        try: 
            with open(file_path, mode="r") as csvfile:
                csv_reader = csv.DictReader(csvfile)
                rows = list(csv_reader)
        except FileNotFoundError:
            raise Exception("The file {} does not exist.".format(csvfile))

        # Check if latitude/longitude exist
        lat = ['latitude','lat']
        lon = ['longitude','lon']
        key_names = csv_reader.fieldnames
        lat_id = np.where([x.lower() in lat for x in key_names])[0]
        lon_id = np.where([x.lower() in lon for x in key_names])[0]
        if len(lat_id)==0 :            
            raise Exception("The key 'Latitude' or 'Lat' (case insensitive) not found. Please specify the building coordinate.")
        if len(lon_id)==0:            
            raise Exception("The key 'Longitude' or 'Lon' (case insensitive) not found. Please specify the building coordinate.")
        lat_key = key_names[lat_id[0]]
        lon_key = key_names[lon_id[0]]

        for bldg_features in rows:
            for i,key in enumerate(bldg_features):

                # converting to a number
                val = bldg_features[key]
                if val.isdigit():
                    bldg_features[key]=int(val)
                elif is_float(val):
                    bldg_features[key]=float(val)

            coordinates = [[bldg_features[lat_key], bldg_features[lon_key]]]
            bldg_features.pop(lat_key)
            bldg_features.pop(lon_key)
            
            #TODO: what should the types be?
            if 'type' in bldg_features.keys():
                if bldg_features['type'] not in ["building","bridge"]:
                    raise Exception("The csv file {file_path} cannot have a column named 'type'")
            else:
                bldg_features['type']=str_type

            # is the id provided by user?
            if id_column==None:
                # if not we assin the id
                id = id_counter
            else:
                if id_column not in bldg_features.keys() :            
                    raise Exception("The key '{}' not found in {}".format(id_column, file_path))
                id = bldg_features[id_column]

            asset = Asset(id, coordinates, bldg_features)
            self.add_asset(id, asset)
            id_counter += 1

        return True    


    def add_asset_features_from_csv(self, file_path, id_column) -> bool:
        """
        Read inventory data from a CSV file and add it to the inventory.

        Args:
            file_path (str):
                  The path to the CSV file
            id_column (str):
                  The name of column that contains id values. If None, new indicies will be assigned

        Returns:
            bool:
                  True if assets were addded, False otherwise.
        """

        def is_float(element: any) -> bool:
            #If you expect None to be passed:
            if element is None: 
                return False
            try:
                float(element)
                return True
            except ValueError:
                return False
            pass

        try: # Attempt to open the file
            with open(file_path, mode="r") as csvfile:
                csv_reader = csv.DictReader(csvfile)
                rows = list(csv_reader)
        except FileNotFoundError:
            raise Exception("The file {} does not exist.".format(csvfile))

        for bldg_features in rows:
            for i,key in enumerate(bldg_features):
                # converting to number
                val = bldg_features[key]
                if val.isdigit():
                    bldg_features[key]=int(val)
                elif is_float(val):
                    bldg_features[key]=float(val)

            if id_column not in bldg_features.keys() :            
                raise Exception("The key '{}' not found in {}".format(id_column, file_path))
            id = bldg_features[id_column]

            self.add_asset_features(id, bldg_features)

        return True    

    def get_dataframe(self, n_possible_worlds=1, features_possible_worlds=[]) -> bool:
        """
        Read inventory data from a CSV file and add it to the inventory.

        Args:
            n_possible_worlds (int):
                  Number of possible worlds
            features_possible_worlds (list of str):
                  Indicate the features with multiple possible worlds

        Returns:
            bool:
                  True if assets were addded, False otherwise.
        """

        features_json=self.get_geojson()['features']
        bldg_properties = [(self.inventory[i].features | {"index":i}) for dummy,i in enumerate(self.inventory)]


        # [bldg_features['properties'] for bldg_features in features_json]

        nbldg = len(bldg_properties)

        if n_possible_worlds==1:
            bldg_properties_df = pd.DataFrame(bldg_properties)

        else:
            # First enumerate assets to see which columns have multiple worlds

            vector_columns = set()
            for entry in bldg_properties:
                vector_columns.update([key for key, value in entry.items() if isinstance(value, list)])

            flat_data = []
            for entry in bldg_properties:
                row = {key: value for key, value in entry.items() if (key not in vector_columns)} # stays the same
                for key in vector_columns:
                    value = entry[key]
                    if isinstance(value, list):
                        if not len(value)==n_possible_worlds:
                            raise ValueError("The specified # of possible worlds are {} but {} constains {} realizations in {}".format(n_possible_worlds, key, len(value), entry))

                        for i in range(n_possible_worlds):
                            row[f'{key}_{i+1}'] = value[i] 
                    else:
                        for i in range(n_possible_worlds):
                            row[f'{key}_{i+1}'] = value

                flat_data.append(row)

            bldg_properties_df = pd.DataFrame(flat_data)

        bldg_properties_df.drop(columns=['type'],inplace=True)

        #  get centoried
        lat_values = [None] * nbldg
        lon_values = [None] * nbldg
        for idx in range(nbldg):
            polygon_coordinate = features_json[idx]['geometry']['coordinates']
            latitudes = [coord[0] for coord in polygon_coordinate]
            longitudes = [coord[1] for coord in polygon_coordinate]
            lat_values[idx] = sum(latitudes) / len(latitudes)
            lon_values[idx] = sum(longitudes) / len(longitudes)


        # to be used for spatial interpolation
        # lat_values = [features_json[idx]['geometry']['coordinates'][0][0] for idx in range(nbldg)]
        # lon_values = [features_json[idx]['geometry']['coordinates'][0][1] for idx in range(nbldg)]
        bldg_geometries_df = pd.DataFrame()
        bldg_geometries_df["Lat"]=lat_values
        bldg_geometries_df["Lon"]=lon_values
        bldg_geometries_df["index"]=bldg_properties_df["index"]

        bldg_properties_df = bldg_properties_df.set_index("index")
        bldg_geometries_df = bldg_geometries_df.set_index("index")

        return bldg_properties_df, bldg_geometries_df, nbldg
