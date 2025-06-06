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
# Barbaros Cetiner
# Frank McKenna
#
# Last updated:
# 06-05-2025

"""
This module defines classes associated with asset inventories.

.. autosummary::

    AssetInventory
    Asset
"""

import csv
import json
import logging
import random
from copy import deepcopy
from datetime import datetime
from typing import Union, List, Tuple, Dict, Any, Optional

try:
    # Python 3.8+
    from importlib.metadata import version
except ImportError:
    # For Python <3.8, use the backport
    from importlib_metadata import version

import numpy as np
import pandas as pd
from shapely import box
from shapely.geometry import shape

from brails.utils.input_validator import InputValidator
from brails.utils.spatial_join_methods.base import SpatialJoinMethods


def clean_floats(obj: Any) -> Any:  # TODO: Check if this is needed
    """
    Recursively convert float values that are mathematically integers to int.

    This function traverses a nested structure (e.g., dict, list, JSON-like
    object) and converts any float that is numerically equivalent to an integer
    into an int, improving the readability and cleanliness of the output,
    especially for serialization.

    Args:
        obj (Any):
            A JSON-like object (dict, list, or primitive value) to process.

    Returns:
        Any:
            The input object with eligible floats converted to integers.
    """
    if isinstance(obj, dict):
        return {k: clean_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_floats(v) for v in obj]
    elif isinstance(obj, float) and obj.is_integer():
        return int(obj)
    else:
        return obj


class Asset:
    """
    A data structure for an asset that holds it coordinates and features.

    Attributes:
        asset_id (str|int):: Unique identifier for the asset.
        coordinates (list[List[float]]): A list of coordinate pairs
            [[lon1, lat1], [lon2, lat2], ..., [lonN, latN]].
        features (dict[str, any]): A dictionary of features (attributes) for
            the asset.

    Methods:
        add_features(additional_features: dict[str, any],
            overwrite: bool = True): Update the existing features in the asset.
        print_info(): Print the coordinates and features of the asset.
    """

    def __init__(
        self,
        asset_id: Union[str, int],
        coordinates: List[List[float]],
        features: Dict[str, Any] = None,
    ) -> None:
        """
        Initialize an Asset with an asset ID, coordinates, and features.

        Args:
            asset_id (str|int): The unique identifier for the asset.
            coordinates (list[list[float]]): A two-dimensional list
                representing the geometry of the asset in [[lon1, lat1],
                [lon2, lat2], ..., [lonN, latN]] format.
            features (dict[str, Any], optional): A dictionary of features.
                Defaults to an empty dict.
        """
        coords_check, output_msg = InputValidator.validate_coordinates(
            coordinates)
        if coords_check:
            self.coordinates = coordinates
        else:
            print(
                f'{output_msg} Setting coordinates for asset '
                f'{asset_id} to an empty list'
            )
            self.coordinates = []

        self.features = features if features is not None else {}

    def add_features(
            self,
            additional_features: Dict[str, Any],
            overwrite: bool = True
    ) -> Tuple[bool, int]:
        """
        Update the existing features in the asset.

        Args:
            additional_features (dict[str, any]): New features to merge into
                the asset's features.
            overwrite (bool, optional): Whether to overwrite existing features.
                Defaults to True.
        """
        n_pw = 1

        if overwrite:
            # Overwrite existing features with new ones:
            self.features.update(additional_features)

            # count # possible worlds
            for key, val in additional_features.items():
                if isinstance(val, list):
                    if (n_pw == 1) or (n_pw == len(val)):
                        n_pw = len(val)
                    else:
                        print(
                            f'WARNING: # possible worlds was {n_pw} but now '
                            f'is {len(val)}. Something went wrong.'
                        )
                        n_pw = len(val)

            updated = True

        else:
            # Only update with new keys, do not overwrite existing keys:
            updated = False
            for key, val in additional_features.items():
                if key not in self.features:
                    # write
                    self.features[key] = val

                    # count # possible worlds
                    if isinstance(val, list):
                        if (n_pw == 1) or (n_pw == len(val)):
                            n_pw = len(val)
                        else:
                            print(
                                f'WARNING: # possible worlds was {n_pw} but '
                                f'now is {len(val)}. Something went wrong.'
                            )
                            n_pw = len(val)

                    updated = True

        return updated, n_pw

    def remove_features(self, feature_list: List[str]) -> bool:
        """
        Update the existing features in the asset.

        Args:
            feature_list (dict[str, any]): List of features to be removed

        Return:
            bool: True if features are removed
        """
        for key in list(feature_list):
            self.features.pop(key, None)

        return True

    def print_info(self) -> None:
        """Print the coordinates and features of the asset."""
        print("\t Coordinates: ", self.coordinates)
        print("\t Features: ", self.features)


class AssetInventory:
    """
    A class representing a Asset Inventory.

    Attributes:
        inventory (dict): The inventory stored in a dict accessed by asset_id

     Methods:
        add_asset(asset_id, Asset): Add an asset to the inventory.
        add_asset_coordinates(asset_id, coordinates): Add an asset to the
            inventory with just a list of coordinates.
        add_asset_features(asset_id, features, overwrite): Append new features
            to the asset.
        add_model_predictions(predictions, feature_key): Add model predictions
            to assets under a specified feature key.
        change_feature_names(feature_name_mapping): Rename feature names in an
            AssetInventory using user-specified mapping.
        convert_polygons_to_centroids(): Convert polygon geometries in the
            inventory to their centroid points.
        get_asset_coordinates(asset_id): Get coordinates of a particular
            assset.
        get_asset_features(asset_id): Get features of a particular assset.
        get_asset_ids(): Return the asset ids as a list.
        get_coordinates(): Return a list of footprints.
        get_extent(buffer): Calculate the geographical extent of the inventory.
        get_geojson(): Return inventory as a geojson dict.

        get_random_sample(size, seed): Get subset of the inventory.
        write_to_geojson(): Write inventory to file in GeoJSON format. Also
                            return inventory as a geojson dict.
        join(inventory_to_join, method="get_points_in_polygons"): Perform a
            spatial join with another AssetInventory using the specified
            method.
        print_info(): Print the asset inventory.
        remove_asset(asset_id): Remove an asset to the inventory.
        remove_features(feature_list): Remove features from the inventory.
        read_from_csv(file_path, keep_existing, str_type, id_column): Read
            inventory dataset from a csv table
        add_asset_features_from_csv(file_path, id_column): Add asset features
            from a csv file.

    """

    def __init__(self) -> None:
        """Initialize AssetInventory with an empty inventory dictionary."""
        self.inventory = {}
        self.n_pw = 1

    def add_asset(self, asset_id: Union[str, int], asset: Asset) -> bool:
        """
        Add an Asset to the inventory.

        Args:
            asset_id (str | int):
                The unique identifier for the asset.
            asset (Asset):
                The asset to be added.

        Returns:
            bool:
                True if the asset was added successfully, False otherwise.

        Raises:
            TypeError:
                If `asset` is not an instance of `Asset`.
        """
        if not isinstance(asset, Asset):
            raise TypeError("Expected an instance of Asset.")

        if asset_id in self.inventory:
            print(f'Asset with id {asset_id} already exists. Asset was not '
                  'added')
            return False

        self.inventory[asset_id] = asset
        return True

    def add_asset_coordinates(
        self,
        asset_id: Union[str, int],
        coordinates: List[List[float]]
    ) -> bool:
        """
        Add an Asset to the inventory by adding its coordinate information.

        Args:
            asset_id (str|int):
                The unique identifier for the asset.
            coordinates (list[list[float]]):
                A two-dimensional list
                representing the geometry in [[lon1, lat1], [lon2, lat2], ...,
                [lonN, latN]] format.

        Returns:
            bool: True if the asset was added successfully, False otherwise.
        """
        existing_asset = self.inventory.get(asset_id, None)

        if existing_asset is not None:
            print(f'Asset with id {asset_id} already exists. Coordinates were '
                  'not added')
            return False

        # Create asset and add using id as the key:
        asset = Asset(asset_id, coordinates)
        self.inventory[asset_id] = asset

        return True

    def add_asset_features(
        self,
        asset_id: Union[str, int],
        new_features: Dict[str, Any],
        overwrite: bool = True
    ) -> bool:
        """
        Add new asset features to the Asset with the specified ID.

        Args:
            asset_id (str|int):
                The unique identifier for the asset.
            new_features (dict):
                A dictionary of features to add to the asset.
            overwrite (bool):
                Whether to overwrite existing features with the
                same keys. Defaults to True.

        Returns:
            bool: True if features were successfully added, False if the asset
                does not exist or the operation fails.
        """
        asset = self.inventory.get(asset_id, None)
        if asset is None:
            print(f'No existing Asset with id {asset_id} found. Asset '
                  'features not added.')
            return False

        status, n_pw = asset.add_features(new_features, overwrite)
        if n_pw == 1:
            pass
        elif (n_pw == self.n_pw) or (self.n_pw == 1):
            self.n_pw = n_pw
        else:
            print(f'WARNING: # possible worlds was {self.n_pw} but is now '
                  f'{n_pw}. Something went wrong.')
            self.n_pw = n_pw
        return status

    def add_model_predictions(
        self,
        predictions: Dict[Any, Any],
        feature_key: str
    ) -> None:
        """
        Add model predictions to the inventory.

        This method goes through the inventory and updates each item by adding
        the corresponding model prediction as a new feature under the specified
        key. Items without a matching prediction are left unchanged.

        Args:
            predictions (dict):
                A dictionary where keys correspond to inventory items and
                values represent the predicted features to be added.
            feature_key (str):
                The key under which the predictions will be stored as a new
                feature in each inventory item.

        Raises:
            TypeError:
                If `predictions` is not a dictionary or `feature_key` is not a
                string.
            ValueError:
                If none of the keys in `predictions` exist in `self.inventory`.

        Example:
            self.add_model_predictions(predictions={1:'gable',
                                                    3:'flat',
                                                    12:'hip'},
                                       feature_key='roof_type')
        """
        # Validate predictions input:
        if not isinstance(predictions, dict):
            raise TypeError("Expected 'predictions' to be a dictionary.")

        # Ensure at least one key in predictions matches inventory:
        common_keys = set(predictions.keys()) & set(self.inventory.keys())
        if not common_keys:
            raise ValueError("None of the keys in 'predictions' exist in "
                             "'self.inventory'.")

        # Validate feature_key input:
        if not isinstance(feature_key, str):
            raise TypeError("Expected 'feature_key' to be a string.")

        # Update inventory items with corresponding predictions:
        for key, val in self.inventory.items():
            if key in predictions:
                val.add_features({feature_key: predictions.get(key)})

    def change_feature_names(
            self,
            feature_name_mapping: Dict[str, str]
    ) -> None:
        """
        Rename feature names in an AssetInventory using user-specified mapping.

        Args:
            feature_name_mapping (dict):
                A dictionary where keys are the original feature names and
                values are the new feature names.

        Raises:
            TypeError:
                If the mapping is not a dictionary or contains invalid
                key-value pairs.
        """
        # Validate that feature_name_mapping is a dictionary:
        if not isinstance(feature_name_mapping, dict):
            raise TypeError(
                "The 'feature_name_mapping' must be a dictionary.")

        # Validate that all keys and values are strings:
        for original_name, new_name in feature_name_mapping.items():
            if not isinstance(original_name, str) or not isinstance(new_name,
                                                                    str):
                raise TypeError(
                    'Both original and new names of features must be '
                    f'strings. Invalid pair: ({original_name}, '
                    f'{new_name})'
                )

        # Iterate over each asset in the AssetInventory:
        for asset in self.inventory.values():
            # Iterate over the defined name mappings in feature_name_mapping:
            for original_name, new_name in feature_name_mapping.items():
                # If the original_name for a feature name exists in the asset's
                # features:
                if original_name in asset.features:
                    # Rename the feature by popping the old key and adding it
                    # under the new key:
                    asset.features[new_name] = asset.features.pop(
                        original_name)

    def convert_polygons_to_centroids(self) -> None:
        """
        Convert polygon geometries in the inventory to their centroid points.

        Iterates through the asset inventory and replaces the coordinates of
        each polygon or linestring geometry with the coordinates of its
        centroid. Point geometries are left unchanged.

        This function is useful for spatial operations that require point
        representations of larger geometries (e.g., matching, distance
                                              calculations).

        Notes:
            - Polygon coordinates are wrapped in a list to ensure proper
              GeoJSON structure.
            - Linestrings are treated as such unless the geometry is invalid or
              ambiguous.

        Modifies:
            self.inventory (dict):
                Updates the `coordinates` field of each asset in-place by
                replacing polygons and linestrings with their centroid.
        """
        for key, asset in self.inventory.items():
            if InputValidator.is_point(asset.coordinates):
                continue

            elif InputValidator.is_linestring(asset.coordinates):
                geometry = {"type": "LineString",
                            "coordinates": asset.coordinates}
            else:
                if InputValidator.is_polygon(asset.coordinates):
                    geometry = {"type": "Polygon",
                                "coordinates": [asset.coordinates]}
                else:
                    geometry = {"type": "LineString",
                                "coordinates": asset.coordinates}

            centroid = shape(geometry).centroid
            asset.coordinates = [[centroid.x, centroid.y]]

    # def remove_asset_features(self, asset_id: str | int, feature_list: list) -> tuple[bool, dict]:
    #     """
    #     Get features of a particular asset.

    #     Args:
    #         asset_id (str|int): The unique identifier for the asset.
    #         feature_list (list): The list of features to be removed

    #     Returns:
    #         bool: True if features were removed, False otherwise.
    #     """
    #     asset = self.inventory.get(asset_id, None)
    #     asset.remove_features(feature_list)

    #    return True

    def get_asset_coordinates(
            self,
            asset_id: Union[str, int]
    ) -> Tuple[bool, List]:
        """
        Get the coordinates of a particular asset.

        Args:
            asset_id (str | int):
                The unique identifier for the asset.

        Returns:
            tuple[bool, list]]:
                A tuple where the first element is a boolean
                indicating whether the asset was found, and the second element
                is a list of coordinate pairs in the format [[lon1, lat1],
                [lon2, lat2], ..., [lonN, latN]] if the asset is present.
                Returns an empty list if the asset is not found.
        """
        asset = self.inventory.get(asset_id, None)
        if asset is None:
            return False, []

        return True, asset.coordinates

    def get_asset_features(
            self,
            asset_id: Union[str, int]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Get features of a particular asset.

        Args:
            asset_id (str|int):
                The unique identifier for the asset.

        Returns:
            tuple[bool, Dict]:
                A tuple where the first element is a boolean
                indicating whether the asset was found, and the second element
                is a dictionary containing the asset's features if the asset
                is present. Returns an empty dictionary if the asset is not
                found.
        """
        asset = self.inventory.get(asset_id, None)
        if asset is None:
            return False, {}

        return True, asset.features

    def get_asset_ids(self) -> List[Union[str, int]]:
        """
        Retrieve the IDs of all assets in the inventory.

        Returns:
            list[str | int]:
                A list of asset IDs, which may be strings or
                integers.
        """
        return list(self.inventory.keys())

    def get_coordinates(
            self
    ) -> Tuple[List[List[List[float]]], List[Union[str, int]]]:
        """
        Get geometry (coordinates) and keys of all assets in the inventory.

        Returns:
            tuple[list[list[list[float, float]]], list[str | int]]: A tuple
                containing:
                - A list of coordinates for each asset, where each coordinate
                    is represented as a list of [longitude, latitude] pairs.
                - A list of asset keys corresponding to each Asset.
        """
        coordinates = [asset.coordinates for asset in self.inventory.values()]
        asset_ids = list(self.inventory.keys())

        return coordinates, asset_ids

    def get_random_sample(
        self,
        nsamples: int,
        seed: Optional[Union[int, float, str, bytes, bytearray]] = None
    ) -> 'AssetInventory':
        """
        Generate a smaller AssetInventory with a random selection of assets.

        This method randomly selects `nsamples` assets from the existing
        inventory and returns a new AssetInventory instance containing only
        these sampled assets. The randomness can be controlled using an
        optional `seed` for reproducibility.

        Args:
            nsamples (int):
                The number of assets to randomly sample from the inventory.
                Must be a positive integer not exceeding the total number of
                assets.
            seed (int | float | str | bytes | bytearray | None, optional):
                A seed value for the random generator to ensure
                reproducibility. If None, the system default (current system
                time) is used.

        Returns:
            AssetInventory:
                A new AssetInventory instance containing the randomly selected
                subset of assets.

        Raises:
            ValueError:
                If `nsamples` is not a positive integer or exceeds the number
                of assets in the inventory.
        """
        if not isinstance(nsamples, int) or nsamples <= 0:
            ValueError('Number of samples must be a positive integer.')

        if nsamples > len(self.inventory):
            raise ValueError('Number of samples cannot exceed the number of '
                             'assets in the inventory')

        if seed is not None:
            random.seed(seed)

        random_keys = random.sample(list(self.inventory.keys()), nsamples)

        result = AssetInventory()
        for key in random_keys:
            result.add_asset(key, self.inventory[key])

        return result

    def get_extent(self, buffer: Union[str, List[int]] = 'default') -> box:
        """
        Calculate the geographical extent of the inventory.

        Args:
            buffer (str or list[int]):
                A string or a list of 4 integers.
                - 'default' applies preset buffer values.
                - 'none' applies zero buffer values.
                - A list of 4 integers defines custom buffer values for each
                  edge of the bounding box in the order minlon, maxlon, minlat,
                  maxlat.

        Returns:
            shapely.geometry.box:
                A Shapely polygon representing the extent of the inventory,
                with buffer applied.

        Raises:
            ValueError: If the buffer input is invalid.
        """
        # Check buffer input:
        buffer_levels = buffer.lower()

        if buffer == 'default':
            buffer_levels = [0.0002, 0.0001, 0.0002, 0.0001]
        elif buffer == 'none':
            buffer_levels = [0, 0, 0, 0]
        elif (
            isinstance(buffer, list)
            and len(buffer) == 4
            and all(isinstance(x, int) for x in buffer)
        ):
            buffer_levels = buffer.copy()
        else:
            raise ValueError('Invalid buffer input. Valid options for the '
                             "buffer input are 'default', 'none', or a list of"
                             ' 4 integers.')

        # Determine the geographical extent of the inventory:
        minlon, maxlon, minlat, maxlat = 180, -180, 90, -90
        for asset in self.inventory.values():
            for lon, lat in asset.coordinates:
                minlon, maxlon = min(minlon, lon), max(maxlon, lon)
                minlat, maxlat = min(minlat, lat), max(maxlat, lat)

        # Create a Shapely polygon from the determined extent:
        return box(
            minlon - buffer_levels[0],
            minlat - buffer_levels[1],
            maxlon + buffer_levels[2],
            maxlat + buffer_levels[3],
        )

    def get_geojson(self) -> Dict[str, Any]:
        """
        Generate a GeoJSON representation of the assets in the inventory.

        The function constructs a valid GeoJSON `FeatureCollection`, where each
        asset is represented as a `Feature`. Each feature contains:
        - A `geometry` field (Point, LineString, or Polygon) based on the
          asset's coordinates.
        - A `properties` field containing asset-specific metadata.

        Additionally, the GeoJSON output includes:
        - A timestamp (`generated`) indicating when the data was created.
        - The `BRAILS` package version (if available).
        - A Coordinate Reference System (`crs`) definition set to "CRS84".

        Returns:
            dict:
                A dictionary formatted as a GeoJSON `FeatureCollection`
                containing all assets in the inventory.
        Note:
            Assets without geometry are excluded from the generated GeoJSON
            representation.
        """
        try:
            brails_version = version("BRAILS")
        except Exception:
            brails_version = "NA"

        geojson = {
            "type": "FeatureCollection",
            "generated": str(datetime.now()),
            "brails_version": brails_version,
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": [],
        }

        for key, asset in self.inventory.items():
            geometry = None
            if InputValidator.is_point(asset.coordinates):
                geometry = {"type": "Point",
                            "coordinates": asset.coordinates[0]}
            elif InputValidator.is_linestring(asset.coordinates):
                geometry = {
                    "type": "LineString",
                    "coordinates": asset.coordinates}
            elif InputValidator.is_polygon(asset.coordinates):
                geometry = {"type": "Polygon",
                            "coordinates": [asset.coordinates]}

            if geometry:
                geojson["features"].append(
                    {"type": "Feature",
                     "properties": asset.features,
                     "geometry": geometry
                     }
                )

        return geojson

    def join(
        self,
        inventory_to_join: 'AssetInventory',
        method: str = 'get_points_in_polygons'
    ) -> None:
        """
        Merge with another AssetInventory using specified spatial join method.

        Args:
            inventory_to_join (AssetInventory):
                The inventory to be joined with the current one.
            method (str):
                The spatial join method to use. Defaults to 'get_points_in_
                polygons'. The method defines how the join operation is
                executed between inventories.

        Raises:
            TypeError:
                - If `inventory_to_join` is not an instance of
                  `AssetInventory`.
                - If `method` is not a string.

        Returns:
            None: This method modifies the `AssetInventory` instance in place.
        """
        # Ensure inventory_to_join is of type AssetInventory:
        if not isinstance(inventory_to_join, AssetInventory):
            raise TypeError('Inventory input specified for join needs to be an'
                            'AssetInventory')

        # Ensure method is a valid string:
        if not isinstance(method, str):
            raise TypeError('Join method should be a valid string')

        # Perform the spatial join using the specified method:
        self = SpatialJoinMethods.execute(method,
                                          self,
                                          inventory_to_join)

    def print_info(self) -> None:
        """
        Print summary information about the AssetInventory.

        This method outputs the name of the class, the type of data structure
        used to store the inventory, and basic information about each asset
        in the inventory, including its key and features.

        Returns:
            None
        """
        print(self.__class__.__name__)
        print("Inventory stored in: ", self.inventory.__class__.__name__)
        for key, asset in self.inventory.items():
            print("Key: ", key, "Asset:")
            asset.print_info()

    def remove_asset(self, asset_id: Union[str, int]) -> bool:
        """
        Remove an Asset from the inventory.

        Args:
            asset_id (str|int):
                The unique identifier for the asset.

        Returns:
            bool:
                True if asset was removed, False otherwise.
        """
        del self.inventory[asset_id]

        return True

    def remove_features(self, feature_list: List[str]) -> bool:
        """
        Remove specified features from all assets in the inventory.

        Args:
            feature_list (list[str]):
                The unique identifier for the asset.

        Returns:
            bool:
                True if features were removed, False otherwise.
        """
        for _, asset in self.inventory.items():
            asset.remove_features(feature_list)

        return True

    def write_to_geojson(self, output_file: str = "") -> Dict:
        """
        Write inventory to a GeoJSON file and return the GeoJSON dictionary.

        This method generates a GeoJSON representation of the asset inventory,
        writes it to the specified file path (if provided), and returns the
        GeoJSON object.

        Args:
            output_file (str, optional):
                Path to the output GeoJSON file. If empty, no file is written.

        Returns:
            Dict:
                A dictionary containing the GeoJSON FeatureCollection.
        """
        geojson = self.get_geojson()

        print(f'GEOJSON {geojson}')

        # Modify the GeoJSON features to include an 'id' field:
        # This is the variation of GeoJSON that is used in R2D

        print(f'FILE: {output_file}')

        for i, ft in enumerate(geojson["features"]):
            if "id" in ft['properties']:
                id = ft['properties'].pop("id")
            else:
                id = str(i)
            ft['id'] = str(id)

        # Write the created GeoJSON dictionary into a GeoJSON file:
        if output_file:
            with open(output_file, "w", encoding="utf-8") as file_out:
                json.dump(clean_floats(geojson), file_out, indent=2)

        return geojson

    def read_from_csv(
        self,
        file_path: str,
        keep_existing: bool,
        str_type: str = "building",
        id_column: Optional[str] = None
    ) -> bool:
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
                  The name of column that contains id values. If None, new
                  indices will be assigned

        Returns:
            bool:
                  True if assets were addded, False otherwise.
        """

        def is_float(element: Any) -> bool:
            # If you expect None to be passed:
            if element is None:
                return False
            try:
                float(element)
                return True
            except ValueError:
                return False
            pass

        if keep_existing:
            if len(self.inventory) == 0:
                print('No existing inventory found. Reading in the new '
                      'inventory from the file.')
                id_counter = 1
            else:
                # we don't want a duplicate the id
                id_counter = max(self.inventory.keys()) + 1
        else:
            self.inventory = {}
            id_counter = 1

        # Attempt to open the file
        try:
            with open(file_path, mode="r") as csvfile:
                csv_reader = csv.DictReader(csvfile)
                rows = list(csv_reader)
        except FileNotFoundError:
            raise Exception("The file {} does not exist.".format(file_path))

        # Check if latitude/longitude exist
        lat = ["latitude", "lat"]
        lon = ["longitude", "lon", "long"]
        key_names = csv_reader.fieldnames
        lat_id = np.where([y.lower() in lat for y in key_names])[0]
        lon_id = np.where([x.lower() in lon for x in key_names])[0]
        if len(lat_id) == 0:
            raise Exception(
                "The key 'Latitude' or 'Lat' (case insensitive) not found. "
                'Please specify the building coordinate.'
            )
        if len(lon_id) == 0:
            raise Exception(
                "The key 'Longitude' or 'Lon' (case insensitive) not found. "
                'Please specify the building coordinate.'
            )
        lat_key = key_names[lat_id[0]]
        lon_key = key_names[lon_id[0]]

        for bldg_features in rows:
            for i, key in enumerate(bldg_features):
                # converting to a number
                val = bldg_features[key]
                if val.isdigit():
                    bldg_features[key] = int(val)
                elif is_float(val):
                    bldg_features[key] = float(val)

            # coordinates = [[bldg_features[lat_key], bldg_features[lon_key]]]
            coordinates = [[bldg_features[lon_key], bldg_features[lat_key]]]

            bldg_features.pop(lat_key)
            bldg_features.pop(lon_key)

            # TODO: what should the types be?
            if "type" in bldg_features.keys():
                if bldg_features["type"] not in ["building", "bridge"]:
                    raise Exception(f"The csv file {file_path} cannot have a "
                                    "column named 'type'")
            else:
                bldg_features["type"] = str_type

            # is the id provided by user?
            if id_column is None:
                # if not we assin the id
                id = id_counter
            else:
                if id_column not in bldg_features.keys():
                    raise Exception(
                        "The key '{}' not found in {}".format(
                            id_column, file_path)
                    )
                id = bldg_features[id_column]

            asset = Asset(id, coordinates, bldg_features)
            self.add_asset(id, asset)
            id_counter += 1

        return True

    def add_asset_features_from_csv(
        self,
        file_path: str,
        id_column: Union[str, None]
    ) -> bool:
        """
        Read inventory data from a CSV file and add it to the inventory.

        Args:
            file_path (str):
                  The path to the CSV file
            id_column (str):
                  The name of column that contains id values. If None, new
                  indicies will be assigned

        Returns:
            bool:
                  True if assets were addded, False otherwise.
        """
        try:  # Attempt to open the file
            with open(file_path, mode="r") as csvfile:
                csv_reader = csv.DictReader(csvfile)
                rows = list(csv_reader)
        except FileNotFoundError:
            raise Exception("The file {} does not exist.".format(csvfile))

        for bldg_features in rows:
            for i, key in enumerate(bldg_features):
                # converting to number
                val = bldg_features[key]
                if val.isdigit():
                    bldg_features[key] = int(val)
                elif InputValidator.is_float(val):
                    bldg_features[key] = float(val)

            if id_column not in bldg_features.keys():
                raise Exception(
                    "The key '{}' not found in {}".format(id_column, file_path)
                )
            id = bldg_features[id_column]

            self.add_asset_features(id, bldg_features)

        return True

    def get_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Convert the asset inventory into two structured DataFrames and count.

        This method processes the internal asset inventory and returns:
        - A DataFrame containing the features of each asset, with support for
          multiple possible worlds (realizations).
        - A DataFrame containing centroid coordinates (latitude and longitude)
          for spatial operations.
        - The total number of assets in the inventory.

        The method flattens feature lists into separate columns if multiple
        possible worlds exist. It also derives centroid coordinates from the
        GeoJSON geometry for each asset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, int]:
                - Asset feature DataFrame (indexed by asset ID).
                - Asset geometry DataFrame with 'Lat' and 'Lon' columns.
                - Total number of assets (int).

        Raises:
            ValueError:
                If a feature list's length does not match the expected number
                of possible worlds.
        """
        n_possible_worlds = self.get_n_pw()

        asset_json = self.get_geojson()
        features_json = asset_json["features"]
        bldg_properties = [
            (self.inventory[i].features | {"index": i})
            for dummy, i in enumerate(self.inventory)
        ]

        # [bldg_features['properties'] for bldg_features in features_json]

        nbldg = len(bldg_properties)

        if n_possible_worlds == 1:
            bldg_properties_df = pd.DataFrame(bldg_properties)

        else:
            # First enumerate assets to see which columns have multiple worlds

            vector_columns = set()
            for entry in bldg_properties:
                vector_columns.update(
                    [key for key, value in entry.items() if isinstance(
                        value, list)]
                )

            flat_data = []
            for entry in bldg_properties:
                row = {
                    key: value
                    for key, value in entry.items()
                    if (key not in vector_columns)
                }  # stays the same
                for key in vector_columns:
                    value = entry[key]
                    if isinstance(value, list):
                        if not len(value) == n_possible_worlds:
                            raise ValueError(
                                'The specified # of possible worlds are '
                                f'{n_possible_worlds} but {key} contains '
                                f'{len(value)} realizations in {entry}'
                            )

                        for i in range(n_possible_worlds):
                            row[f"{key}_{i+1}"] = value[i]
                    else:
                        for i in range(n_possible_worlds):
                            row[f"{key}_{i+1}"] = value

                flat_data.append(row)

            bldg_properties_df = pd.DataFrame(flat_data)

        if "type" in bldg_properties_df.columns:
            bldg_properties_df.drop(columns=["type"], inplace=True)

        #  get centoried
        lat_values = [None] * nbldg
        lon_values = [None] * nbldg

        for idx in range(nbldg):
            polygon_coordinate = features_json[idx]["geometry"]["coordinates"]

            if isinstance(polygon_coordinate[0], list):
                if isinstance(polygon_coordinate[0][0], list):
                    # multipolygon
                    latitudes = [coord[0][1] for coord in polygon_coordinate]
                    longitudes = [coord[0][0] for coord in polygon_coordinate]
                else:
                    # polygon or point
                    latitudes = [coord[1] for coord in polygon_coordinate]
                    longitudes = [coord[0] for coord in polygon_coordinate]
            else:
                # point?
                latitudes = [polygon_coordinate[1]]
                longitudes = [polygon_coordinate[0]]
            lat_values[idx] = sum(latitudes) / len(latitudes)
            lon_values[idx] = sum(longitudes) / len(longitudes)

        # to be used for spatial interpolation
        # lat_values = [features_json[idx]['geometry']['coordinates'][0][0] for idx in range(nbldg)]
        # lon_values = [features_json[idx]['geometry']['coordinates'][0][1] for idx in range(nbldg)]
        bldg_geometries_df = pd.DataFrame()
        bldg_geometries_df["Lat"] = lat_values
        bldg_geometries_df["Lon"] = lon_values
        bldg_geometries_df["index"] = bldg_properties_df["index"]

        bldg_properties_df = bldg_properties_df.set_index("index")
        bldg_geometries_df = bldg_geometries_df.set_index("index")

        return bldg_properties_df, bldg_geometries_df, nbldg

    def get_world_realization(self, id: int = 0) -> 'AssetInventory':
        """
        Extract a single realization (possible world) from an inventory.

        This method generates a new `AssetInventory` instance where all
        features containing multiple possible worlds are reduced to a single
        realization specified by `id`. Features that are not lists are copied
        as-is.

        Args:
            id (int, default=0):
                The index of the realization to extract. Must be within the
                range of available possible worlds (0-based indexing).

        Returns:
            AssetInventory:
                A new inventory object representing the selected realization.

        Raises:
            Exception:
                - If `id > 0` but the inventory only contains a single
                  realization.
                - If any feature has fewer realizations than the specified
                  `id`.
        """
        new_inventory = deepcopy(self)

        if self.n_pw == 1 and id > 0:
            raise Exception(
                'Cannot retrive different realizations as the inventory '
                'contains only a single realization. Consider setting id=0'
            )

        for i in self.get_asset_ids():
            flag, features = self.get_asset_features(i)
            for key, val in features.items():
                if isinstance(val, list):
                    if len(val) > id:
                        new_inventory.add_asset_features(
                            i, {key: val[id]}, overwrite=True
                        )
                    elif len(val) == id:
                        errmsg = (
                            f'The world index {id} should be smaller than the '
                            f'existing number of worlds {len(val)}, as the '
                            'index starts from zero.'
                        )
                        raise Exception(errmsg)

                    else:
                        errmsg = (
                            f'The world index {id} should be smaller than the '
                            f'existing number of worlds, e.g. asset id {i}, '
                            f'feature {key} contains only {len(val)} '
                            'realizations.'
                        )
                        raise Exception(errmsg)

        return new_inventory

    def update_world_realization(
        self,
        id: int,
        world_realization: 'AssetInventory'
    ) -> None:
        """
        Update the current AssetInventory with a specific world realization.

        This method integrates feature values from a single-world
        `world_realization` inventory into the current multi-world inventory
        by updating the realization at the specified index `id`.

        Args:
            id (int):
                The index of the world (realization) to update. Must be less
                than `self.n_pw`.
            world_realization (AssetInventory):
                An AssetInventory instance representing a single realization
                of the world. All features in this inventory must be scalar
                (i.e., not lists).

        Raises:
            Exception:
                - If the specified `id` is not within the valid range of
                  realizations.
                - If `world_realization` contains features with multiple
                  realizations.
        """
        if self.n_pw == id:
            errmsg = (
                f'The world index {id} should be smaller than the existing '
                f'number of worlds {self.n_pw}, and the index starts from '
                'zero.'
            )
            raise Exception(errmsg)

        elif self.n_pw < id:
            errmsg = (
                f'The world index {id} should be smaller than the existing '
                f'number of worlds {self.n_pw}.'
            )
            raise Exception(errmsg)

        for i in world_realization.get_asset_ids():
            flag, features = self.get_asset_features(i)
            flag_new, features_new = world_realization.get_asset_features(i)

            for key, val in features_new.items():

                if isinstance(val, list):
                    errmsg = (
                        'world_realization should not contain multiple '
                        'possible worlds.'
                    )
                    raise Exception(errmsg)

                try:
                    original_value = self.get_asset_features(i)[1][key]
                except KeyError:
                    # create a new key-value pair if it did not exist.
                    original_value = val

                # initalize
                if isinstance(original_value, list):
                    new_value = original_value
                else:
                    new_value = [original_value]*self.n_pw

                # udpate
                new_value[id] = val

                # if identical, shrink it
                if (len(set(new_value)) == 1):
                    new_value = new_value[0]

                # overwrite existing ones.
                self.add_asset_features(
                    i, {key: new_value}, overwrite=True
                )
        return

    def get_n_pw(self) -> int:  # move to Asset
        """
        Get the number of possible worlds (realizations) in the inventory.

        Returns:
            int:
                The number of possible worlds stored in the inventory.
        """
        return self.n_pw

    def get_multi_keys(self) -> Tuple[List[str], List[str]]:  # move to Asset
        """
        Identify features that contain multiple realizations across assets.

        Iterates through all assets and returns two lists:
        - Keys associated with multi-valued features (i.e., lists).
        - All unique feature keys present in the inventory.

        Returns:
            Tuple[List[str], List[str]]:
                - A list of keys corresponding to multi-realization features.
                - A complete list of all unique feature keys in the inventory.
        """
        multi_keys = []
        all_keys = []

        for i in self.get_asset_ids():
            flag, features = self.get_asset_features(i)

            for key, val in features.items():
                if isinstance(val, list) and key not in multi_keys:
                    multi_keys.append(key)
                if key not in all_keys:
                    all_keys.append(key)

        return multi_keys, all_keys
