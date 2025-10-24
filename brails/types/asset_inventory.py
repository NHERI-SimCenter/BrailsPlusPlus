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
# 08-11-2025

"""
This module defines classes associated with asset inventories.

.. autosummary::

    AssetInventory
    Asset
"""

import os
import csv
import json
import random
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from collections.abc import Iterable

try:
    # Python 3.8+
    from importlib.metadata import version
except ImportError:
    # For Python <3.8, use the backport
    from importlib_metadata import version

import pandas as pd
from shapely import box
from shapely.geometry import shape, LineString, Polygon

from brails.utils.input_validator import InputValidator
from brails.utils.spatial_join_methods.base import SpatialJoinMethods


def clean_floats(obj: Any) -> Any:
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
    """A spatial asset with geometry coordinates, and attributes.

    To import the :class:`Asset` class, use:

    .. code-block:: python

        from brails.types.asset_inventory import Asset


    Attributes:
        asset_id (str or int):
            Unique identifier for the asset.
        coordinates (list[list[float]]):
            Geometry coordinates of the asset, typically as a list of [x, y]
            pairs.
        features (dict[str, Any], optional):
            Additional attributes/features of the asset. Defaults to ``None``.
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
            asset_id (str or int):
                The unique identifier for the asset.
            coordinates (list[list[float]]):
                A two-dimensional list representing the geometry of the asset
                in ``[[lon1, lat1], [lon2, lat2], ..., [lonN, latN]]`` format.
            features (dict[str, Any], optional):
                A dictionary of features. Defaults to an empty dict.
        """
        coords_check, output_msg = InputValidator.validate_coordinates(
            coordinates)
        if coords_check:
            self.coordinates = coordinates
        else:
            print(
                f'{output_msg} for {asset_id}; defaulting to an empty list.'
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
            additional_features (dict[str, Any]):
                New features to merge into the asset's features.
            overwrite (bool, optional):
                Whether to overwrite existing features. Defaults to ``True``.

        Returns:
            tuple[bool, int]:
                A tuple containing two values:

                - updated (bool): ``True`` if any features were added or
                  updated, ``False`` otherwise.
                - n_pw (int): Number of possible worlds.

        Examples:
            >>> asset = Asset(
            ...     asset_id='123',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4180, 37.7755]]
            ... )
            >>> updated, n_pw = asset.add_features({'roof_type': 'gable'})
            >>> print(updated, n_pw)
            True 1
            >>> print(asset.features)
            {'roof_type': 'gable'}

            >>> updated, n_pw = asset.add_features(
            ...     {'possible_heights': [10, 15, 20], 'roof_type': 'hip'},
            ...     overwrite=True
            ... )
            >>> print(updated, n_pw)
            True 3
            >>> print(asset.features)
            {'roof_type': 'hip', 'possible_heights': [10, 15, 20]}

            >>> # When overwrite=False, existing keys are not updated
            >>> updated, n_pw = asset.add_features(
            ...     {'roof_type': 'flat', 'color': 'red'},
            ...     overwrite=False
            ... )
            >>> print(updated, n_pw)
            True 1
            >>> print(asset.features)
            {'roof_type': 'hip',
            'possible_heights': [10, 15, 20],
            'color': 'red'}
        """

        def update_possible_worlds(n_pw, val):

            if isinstance(val, list) and key != 'Households': # Households are currently deterministic
                if (n_pw != 1) and (n_pw != len(val)):
                    print(
                        f'WARNING: # possible worlds was {n_pw} but now '
                        f'is {len(val)}. Something went wrong.'
                    )
                return len(val)
            else:
                return n_pw

        n_pw = 1

        if overwrite:
            # Overwrite existing features with new ones:
            self.features.update(additional_features)

            for key, val in additional_features.items():
                n_pw = update_possible_worlds(n_pw, val)

            updated = True

        else:
            # Only update with new keys, do not overwrite existing keys:
            updated = False
            for key, val in additional_features.items():
                if key not in self.features:
                    # write
                    self.features[key] = val

                    n_pw = update_possible_worlds(n_pw, val)

                    updated = True

        return updated, n_pw

    def get_centroid(self) -> List[List[Optional[float]]]:
        """
        Get the centroid of the asset geometry.

        Returns:
            list[list[float]]:
                ``[[x, y]]`` if centroid can be calculated, ``[[None, None]]``
                otherwise.

        Examples:
            >>> asset = Asset(
            ...     asset_id='123',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4190, 37.7750]]
            ... )
            >>> asset.get_centroid()
            [[-122.41919999999999, 37.774950000000004]]

            >>> empty_asset = Asset(asset_id='empty', coordinates=[])
            Coordinates input is empty for empty; defaulting to an empty list.
            >>> empty_asset.get_centroid()
            [[None, None]]
        """
        coords = self.coordinates
        if not coords:
            return [[None, None]]

        try:
            if InputValidator.is_point(coords):
                return coords
            elif InputValidator.is_linestring(coords):
                geometry = LineString(coords)
            elif InputValidator.is_polygon(coords):
                geometry = Polygon(coords)
            else:
                return [[None, None]]

            centroid = geometry.centroid
            return [[centroid.x, centroid.y]]

        except Exception:
            return [[None, None]]

    def remove_features(self, features_to_remove: Iterable[str]) -> bool:
        """
        Remove specified features from the asset.

        Args:
            features_to_remove(Iterable[str]):
                An iterable of feature keys to remove from the Asset features.
                Accepts iterable types such as ``list``, ``tuple``, ``set``, or
                ``dict_keys``.

        Returns:
            bool: ``True`` if at least one feature was removed; ``False``
            otherwise.

        Raises:
            TypeError: If ``features_to_remove`` is not an iterable of strings.

        Example:
            >>> asset = Asset(
            ...     asset_id='123',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4180, 37.7755]]
            ... )
            >>> asset.features = {'color': 'red', 'height': 10}
            >>> asset.remove_features(['color'])
            True
            >>> asset.features
            {'height': 10}
        """
        if (not isinstance(features_to_remove, Iterable) or
                not all(isinstance(k, str) for k in features_to_remove)):
            raise TypeError(
                'features_to_remove must be an iterable of strings.'
            )

        removed_count = 0
        for feature_key in features_to_remove:
            if feature_key in self.features:
                del self.features[feature_key]
                removed_count += 1

        return removed_count > 0

    def print_info(self) -> None:
        """
        Print the asset's coordinates and feature attributes.

        This method outputs the spatial coordinates and all associated
        feature key-value pairs of the asset to the console.

        Returns:
            None

        Example:
            >>> asset = Asset(
            ...     asset_id='123',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4180, 37.7755]],
            ...     features={'roof_type': 'gable'}
            ... )
            >>> asset.print_info()
            Coordinates:  [[-122.4194, 37.7749], [-122.4180, 37.7755]]
            Features:  {'roof_type': 'gable'}
        """
        print("Coordinates: ", self.coordinates)
        features_json = json.dumps(self.features, indent=2)
        print(f"Features: {features_json}")


class AssetInventory:
    """
    A class representing a collection of Assets managed as an inventory.

    This class provides methods to add, manipulate, join, write and query
    a collaction of :`class:Asset` objects.

    To import the :class:`AssetInventory` class, use:

    .. code-block:: python

        from brails.types.asset_inventory import AssetInventory
    """

    def __init__(self) -> None:
        """Initialize AssetInventory with an empty inventory dictionary."""
        self.inventory: Dict = {}
        self.n_pw = 1

    def add_asset(self, asset_id: Union[str, int], asset: Asset) -> bool:
        """
        Add an Asset to the inventory.

        Args:
            asset_id(str or int):
                The unique identifier for the asset.
            asset(Asset):
                The asset to be added.

        Returns:
            bool: ``True`` if the asset was added successfully, ``False``
            otherwise.

        Raises:
            TypeError: If ``asset`` is not an instance of :class:`Asset`.

        Examples:
            >>> asset = Asset(
            ...     asset_id='001',
            ...     coordinates=[[-122.4194, 37.7749]],
            ...     features={'type': 'building'}
            ... )
            >>> inventory = AssetInventory()
            >>> success = inventory.add_asset('001', asset)
            >>> print(success)
            True

            >>> # Adding the same asset_id again will fail
            >>> success = inventory.add_asset('001', asset)
            >>> print(success)
            False
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
        Add an ``Asset`` to the inventory by adding its coordinate information.

        Args:
            asset_id(str or int):
                The unique identifier for the asset.
            coordinates(list[list[float]]):
                A two-dimensional list
                representing the geometry in ``[[lon1, lat1], [lon2, lat2],
                ..., [lonN, latN]]`` format.

        Returns:
            bool: ``True`` if the asset was added successfully, ``False``
            otherwise.

        Examples:
            >>> inventory = AssetInventory()
            >>> coords = [[-122.42, 37.77], [-122.43, 37.78], [-122.44, 37.79]]
            >>> success = inventory.add_asset_coordinates('A123', coords)
            >>> print(success)
            True

            >>> # Attempt to add the same asset_id again
            >>> success = inventory.add_asset_coordinates('A123', coords)
            >>> print(success)
            False
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
        Add new asset features to the Asset with the specified ``asset_id``.

        Args:
            asset_id(str or int):
                The unique identifier for the asset.
            new_features(dict):
                A dictionary of features to add to the asset.
            overwrite(bool):
                Whether to overwrite existing features with the same keys.
                Defaults to ``True``.

        Returns:
            bool:
                ``True`` if features were successfully added, ``False`` if the
                asset does not exist or the operation fails.

        Examples:
            >>> inventory = AssetInventory()
            >>> inventory.add_asset_coordinates(
            ...     'A123',
            ...     [[-122.4194, 37.7749], [-122.4180, 37.7755]]
            ... )
            True
            >>> features = {'height': 10, 'material': 'concrete'}
            >>> success = inventory.add_asset_features('A123', features)
            >>> print(success)
            True

            >>> # Add features without overwriting existing keys
            >>> more_features = {'material': 'steel', 'color': 'red'}
            >>> success = inventory.add_asset_features(
            ...     'A123',
            ...     more_features,
            ...     overwrite=False
            ... )
            >>> print(success)
            True
            >>> asset = inventory.inventory['A123']
            >>> print(asset.features)
            {'height': 10, 'material': 'concrete', 'color': 'red'}
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
            predictions(dict):
                A dictionary where keys correspond to inventory items and
                values represent the predicted features to be added.
            feature_key(str):
                The key under which the predictions will be stored as a new
                feature in each inventory item.

        Returns:
            None

        Raises:
            TypeError:
                If ``predictions`` is not a dictionary or ``feature_key`` is
                not a string.
            ValueError:
                If none of the keys in ``predictions`` exist in the inventory.

        Example:
            Suppose the inventory contains assets with IDs 1, 3, and 12. To add
            predicted roof types for these assets:

            >>> predictions = {1: 'gable', 3: 'flat', 12: 'hip'}
            >>> inventory.add_model_predictions(
                    predictions,
                    feature_key='roof_type'
                )

            After this call, each asset with an ID in ``predictions`` will have
            a new feature ``roof_type`` set to the corresponding predicted
            value.

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
        Rename feature names in ``AssetInventory`` via user-specified mapping.

        Args:
            feature_name_mapping(dict):
                A dictionary where keys are the original feature names and
                values are the new feature names.

        Raises:
            TypeError:
                If the mapping is not a dictionary or contains invalid
                key-value pairs.

        Example:
            First create an :class:`AssetInventory` with two assets with IDs
            ``asset1`` and ``asset2``.

            >>> inventory = AssetInventory()
            >>> asset1 = Asset(
            ...     asset_id='asset1',
            ...     coordinates=[
            ...         [-122.4194, 37.7749],
            ...         [-122.4194, 37.7849],
            ...         [-122.4094, 37.7849],
            ...         [-122.4094, 37.7749],
            ...         [-122.4194, 37.7749]
            ...     ],
            ...     features={'old_name': 100, 'unchanged': 50}
            ... )
            >>> inventory.add_asset('asset1', asset1)
            >>> asset2 = Asset(
            ...     asset_id='asset2',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4094, 37.7849]],
            ...     features={'old_name': 200}
            ... )
            >>> inventory.add_asset('asset2', asset2)

            Then, change the names of the features of these assets.

            >>> inventory.change_feature_names({'old_name': 'new_name'})
            >>> inventory.inventory['asset1'].features
            {'new_name': 100, 'unchanged': 50}
            >>> inventory.inventory['asset2'].features
            {'new_name': 200}
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
        representations of larger geometries(e.g., matching, distance
        calculations).

        Note:
            - Polygon coordinates are wrapped in a list to ensure proper
              GeoJSON structure.
            - Linestrings are converted to points at their centroid unless the
              geometry is invalid or ambiguous.

        Modifies:
            self.inventory (dict):
                Updates the ``coordinates`` field of each asset in-place by
                replacing polygons and linestrings with their centroid.

        Example:
            >>> inventory = AssetInventory()
            >>> inventory.add_asset_coordinates(
            ...     'asset1',
            ...     [
            ...         [-122.4194, 37.7749],
            ...         [-122.4194, 37.7849],
            ...         [-122.4094, 37.7849],
            ...         [-122.4094, 37.7749],
            ...         [-122.4194, 37.7749]
            ...     ]  # Polygon
            ... )
            >>> inventory.add_asset_coordinates(
            ...     'asset2',
            ...     [
            ...         [-122.4194, 37.7749],
            ...         [-122.4094, 37.7849]
            ...     ]  # LineString
            ... )
            >>> inventory.convert_polygons_to_centroids()
            >>> inventory.get_asset_coordiates('asset1')
            [[-122.4144, 37.7799]]
            >>> inventory.get_asset_coordiates('asset2')
            [[-122.4144, 37.7799]]
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

    def get_asset_coordinates(
            self,
            asset_id: Union[str, int]
    ) -> Tuple[bool, List]:
        """
        Get the coordinates of a particular asset.

        Args:
            asset_id(str or int):
                The unique identifier for the asset.

        Returns:
            tuple[bool, list]:
                - **bool** – Indicates whether the asset was found.
                - **list** – A list of coordinate pairs in the format
                  ``[[lon1, lat1], [lon2, lat2], ..., [lonN, latN]]`` if
                  found, or an empty list if the asset does not exist.
        Example:
            >>> inventory = AssetInventory({
            ...     "A101": Asset(
            ...     coordinates=[[30.123, -97.456], [30.124, -97.457]]
            ...     ),
            ...     "B202": Asset(
            ...     coordinates=[[40.789, -74.123], [40.790, -74.124]]
            ...     )
            ... })
            >>> inventory.get_asset_coordinates("A101")
            (True, [[30.123, -97.456], [30.124, -97.457]])
            >>> inventory.get_asset_coordinates("Z999")
            (False, [])
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
            asset_id(str or int):
                The unique identifier for the asset.

        Returns:
            tuple[bool, Dict]:
                A tuple where the first element is a boolean indicating whether
                the asset was found, and the second element is a dictionary
                containing the asset's features if the asset is present.
                Returns an empty dictionary if the asset is not found.

        Examples:
            >>> inventory = AssetInventory()
            >>> asset = Asset(
            ...     asset_id='001',
            ...     coordinates=[[-122.4194, 37.7749]],
            ...     features={'height': 10, 'material': 'concrete'}
            ... )
            >>> inventory.add_asset('001', asset)
            True
            >>> found, features = inventory.get_asset_features('001')
            >>> found
            True
            >>> features
            {'height': 10, 'material': 'concrete'}

            >>> found, features = inventory.get_asset_features('nonexistent')
            >>> found
            False
            >>> features
            {}
        """
        asset = self.inventory.get(asset_id, None)
        if asset is None:
            return False, {}

        return True, asset.features

    def get_asset_ids(self) -> List[Union[str, int]]:
        """
        Retrieve the IDs of all assets in the inventory.

        Returns:
            list[str or int]:
                A list of asset IDs, which may be strings or
                integers.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset_coordinates(
            ...     'asset1',
            ...     [[-122.4194, 37.7749], [-122.4180, 37.7750]]
            ... )
            >>> _ = inventory.add_asset_coordinates(
            ...     2,
            ...     [[-74.0060, 40.7128], [-74.0055, 40.7130]]
            ... )
            >>> inventory.get_asset_ids()
            ['asset1', 2]
        """
        return list(self.inventory.keys())

    def get_coordinates(
            self
    ) -> Tuple[List[List[List[float]]], List[Union[str, int]]]:
        """
        Get geometry(coordinates) and keys of all assets in the inventory.

        Returns:
            tuple[list[list[list[float, float]]], list[str or int]]:
                A tuple containing:

                - A list of coordinates for each asset, where each coordinate
                  is represented as a list of [longitude, latitude] pairs.
                - A list of asset keys corresponding to each asset.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset_coordinates(
            ...     'asset1',
            ...     [[-122.4194, 37.7749], [-122.4180, 37.7750]]
            ... )
            >>> _ = inventory.add_asset_coordinates(
            ...     'asset2',
            ...     [[-74.0060, 40.7128], [-74.0055, 40.7130]]
            ... )
            >>> coords, keys = inventory.get_coordinates()
            >>> coords
            [[[-122.4194, 37.7749], [-122.4180, 37.7750]],
             [[-74.0060, 40.7128], [-74.0055, 40.7130]]]
            >>> keys
            ['asset1', 'asset2']
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
        Generate a smaller asset inventory with a random selection of assets.

        This method randomly selects ``nsamples`` assets from the existing
        inventory and returns a new :class:`AssetInventory` instance containing
        only these sampled assets. The randomness can be controlled using an
        optional ``seed`` for reproducibility.

        Args:
            nsamples(int):
                The number of assets to randomly sample from the inventory.
                Must be a positive integer not exceeding the total number of
                assets.
            seed(int or float or str or bytes or bytearray or None, optional):
                A seed value for the random generator to ensure
                reproducibility. If None, the system default (current system
                time) is used.

        Returns:
            AssetInventory:
                A new :class:`AssetInventory` instance containing the randomly
                selected subset of assets.

        Raises:
            ValueError:
                If ``nsamples`` is not a positive integer or exceeds the number
                of assets in the inventory.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset('asset1', Asset(
            ...     asset_id='asset1',
            ...     coordinates=[[-122.4194, 37.7749]],
            ...     features={'type': 'building'}
            ... ))
            >>> _ = inventory.add_asset('asset2', Asset(
            ...     asset_id='asset2',
            ...     coordinates=[[-74.0060, 40.7128]],
            ...     features={'type': 'bridge'}
            ... ))
            >>> _ = inventory.add_asset('asset3', Asset(
            ...     asset_id='asset3',
            ...     coordinates=[[2.3522, 48.8566]],
            ...     features={'type': 'tower'}
            ... ))
            >>> sample_inventory = inventory.get_random_sample(
            ...     nsamples=2,
            ...     seed=42
            ... )
            >>> sorted(sample_inventory.get_asset_ids())
            ['asset1', 'asset3']
        """
        if not isinstance(nsamples, int) or nsamples <= 0:
            raise ValueError('Number of samples must be a positive integer.')

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

    def get_extent(self, buffer: Union[str, List[float]] = 'default') -> box:
        """
        Calculate the geographical extent of the inventory.

        Args:
            buffer(str or list[float]): A string or a list of 4 floats.

                - ``'default'`` applies preset buffer values.
                - ``'none'`` applies zero buffer values.
                - A list of 4 floats defines custom buffer values for each
                  edge of the bounding box in the order:
                  [minlon buffer, minlat buffer, maxlon buffer, maxlat buffer].

        Returns:
            shapely.geometry.box:
                A Shapely polygon representing the extent of the inventory,
                with buffer applied.

        Raises:
            ValueError: If the ``buffer`` input is invalid.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset('asset1', Asset(
            ...     asset_id='asset1',
            ...     coordinates=[[-122.4194, 37.7749]],
            ...     features={'type': 'building'}
            ... ))
            >>> _ = inventory.add_asset('asset2', Asset(
            ...     asset_id='asset2',
            ...     coordinates=[[-74.0060, 40.7128]],
            ...     features={'type': 'bridge'}
            ... ))

            >>> # Get extent with default buffer:
            >>> extent_default = inventory.get_extent(buffer='default')
            >>> print(extent_default.bounds)
            (-122.4196, 37.7748, -74.0058, 40.712900000000005)

            >>> # Get extent with no buffer:
            >>> extent_none = inventory.get_extent(buffer='none')
            >>> print(extent_none.bounds)
            (-122.4194, 37.7749, -74.006, 40.7128)

            >>> # Get extent with a custom buffer:
            >>> extent_custom = inventory.get_extent(
            ...     buffer=[0.1, 0.1, 0.1, 0.1]
            ... )
            >>> print(extent_custom.bounds)
            (-122.51939999999999, 37.6749, -73.906, 40.8128)
        """
        # Check buffer input:
        error_msg = ("Invalid buffer input. Valid options for the buffer input"
                     "are 'default', 'none', or a list of 4 integers.")

        if isinstance(buffer, str):
            if buffer.lower() == 'default':
                buffer_levels = [0.0002, 0.0001, 0.0002, 0.0001]
            elif buffer.lower() == 'none':
                buffer_levels = [0, 0, 0, 0]
            else:
                raise ValueError(error_msg)
        elif (
            isinstance(buffer, list)
            and len(buffer) == 4
            and all(isinstance(x, (int, float)) for x in buffer)
        ):
            buffer_levels = buffer.copy()
        else:
            raise ValueError(error_msg)

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

        The function constructs a valid GeoJSON ``FeatureCollection``, where
        each asset is represented as a ``Feature``. Each feature contains:

        - A ``geometry`` field defining a ``Point``, ``LineString``, or
          ``Polygon`` based on the asset's coordinates.
        - A ``properties`` field containing asset-specific metadata.

        Additionally, the GeoJSON output includes:

        - A timestamp indicating when the data was created.
        - The ``BRAILS`` package version (if available).
        - A Coordinate Reference System (``crs``) definition set to ``CRS84``.

        Returns:
            dict:
                A dictionary formatted as a GeoJSON ``FeatureCollection``
                containing all assets in the inventory.
        Note:
            Assets without geometry are excluded from the generated GeoJSON
            representation.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset_coordinates(
            ...     'asset1',
            ...     coordinates=[[-122.4194, 37.7749]]
            ... )
            >>> _ = inventory.add_asset_coordinates(
            ...     'asset2',
            ...     coordinates=[[-74.0060, 40.7128], [-73.935242, 40.730610]]
            ... )
            >>> inventory_geojson = inventory.get_geojson()
            >>> print(inventory_geojson)
            {'type': 'FeatureCollection',
             'generated': '2025-08-11 02:49:47.520953',
             'brails_version': '4.0',
             'crs': {'type': 'name',
                     'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
             'features': [{'type': 'Feature', 'properties': {},
                           'geometry': {'type': 'Point',
                                        'coordinates': [-122.4194, 37.7749]}},
                          {'type': 'Feature', 'properties': {},
                           'geometry': {
                               'type': 'LineString',
                               'coordinates': [[-74.006, 40.7128],
                                               [-73.935242, 40.73061]]
                               }
                           }
                          ]
             }
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
                properties = asset.features | {"id": key}

                geojson["features"].append(
                    {
                        "type": "Feature",
                        "properties": properties,
                        "geometry": geometry
                    }
                )

        return geojson

    def join(
        self,
        inventory_to_join: 'AssetInventory',
        method: str = 'GetPointsInPolygons'
    ) -> None:
        """
        Merge with another AssetInventory using specified spatial join method.

        Args:
            inventory_to_join(AssetInventory):
                The inventory to be joined with the current one.
            method(str):
                The spatial join method to use. Defaults to
                ``GetPointsInPolygons``. The ``method`` defines how the join
                operation is executed between inventories.

        Raises:
            TypeError: If ``inventory_to_join`` is not an instance of
                :class:`AssetInventory` or if ``method`` is not a string.

        Returns:
            None:
                This method modifies the :class:`AssetInventory` instance in
                place.

        Example:
            >>> polygon1_asset = Asset('polygon1', [
            ...     [-122.40, 37.75],
            ...     [-122.40, 37.76],
            ...     [-122.39, 37.76],
            ...     [-122.39, 37.75],
            ...     [-122.40, 37.75]
            ... ])
            >>> polygon2_asset = Asset('polygon2', [
            ...     [-122.38, 37.77],
            ...     [-122.38, 37.78],
            ...     [-122.37, 37.78],
            ...     [-122.37, 37.77],
            ...     [-122.38, 37.77]
            ... ])
            >>> poly_inventory = AssetInventory()
            >>> _ = poly_inventory.add_asset(
            ...     asset_id='polygon1',
            ...     asset=polygon1_asset
            ... )
            >>> _ = poly_inventory.add_asset(
            ...     asset_id='polygon2',
            ...     asset=polygon2_asset
            ... )
            >>> poly_inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
            Key:  polygon1 Asset:
                Coordinates:  [[-122.4, 37.75], [-122.4, 37.76],
                               [-122.39, 37.76], [-122.39, 37.75],
                               [-122.4, 37.75]]
                Features:  {}
            Key:  polygon2 Asset:
                Coordinates:  [[-122.38, 37.77], [-122.38, 37.78],
                               [-122.37, 37.78], [-122.37, 37.77],
                               [-122.38, 37.77]]
                Features:  {}
            >>> # This point lies within polygon1's boundaries:
            >>> point_inventory = AssetInventory()
            >>> _ = point_inventory.add_asset(
            ...     asset_id = 'point1',
            ...     asset = Asset(
            ...         'point1',
            ...         [[-122.395, 37.755]],
            ...         features={'FFE':6.8}
            ...     )
            ... )
            >>> poly_inventory.join(point_inventory)
            Joining inventories using GetPointsInPolygons method...
            Identified a total of 1 matched points.
            Inventories successfully joined.
            >>> poly_inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
            Key:  polygon1 Asset:
                Coordinates:  [[-122.4, 37.75], [-122.4, 37.76],
                               [-122.39, 37.76], [-122.39, 37.75],
                               [-122.4, 37.75]]
                Features:  {'FFE': 6.8}
            Key:  polygon2 Asset:
                Coordinates:  [[-122.38, 37.77], [-122.38, 37.78],
                               [-122.37, 37.78], [-122.37, 37.77],
                               [-122.38, 37.77]]
                Features:  {}
        """
        # Ensure inventory_to_join is of type AssetInventory:
        if not isinstance(inventory_to_join, AssetInventory):
            raise TypeError('Inventory input specified for join needs to be an'
                            ' AssetInventory')

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

        This method outputs the name of the class , the type of data structure
        used to store the inventory, and basic information about each asset
        in the inventory, including its ``key`` and ``features``.

        Returns:
            None

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset(
            ...     asset_id='building1',
            ...     asset=Asset(
            ...         'building1',
            ...         [[-122.40, 37.75], [-122.40, 37.76],
            ...          [-122.39, 37.76], [-122.39, 37.75],
            ...          [-122.40, 37.75]],
            ...         features={'height': 12.5}
            ...     )
            ... )
            >>> _ = inventory.add_asset(
            ...     asset_id='building2',
            ...     asset=Asset(
            ...         'building2',
            ...         [[-122.38, 37.77], [-122.38, 37.78],
            ...          [-122.37, 37.78], [-122.37, 37.77],
            ...          [-122.38, 37.77]],
            ...         features={'height': 8.0}
            ...     )
            ... )
            >>> inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
            Key:  building1 Asset:
                Coordinates:  [[-122.4, 37.75], [-122.4, 37.76],
                               [-122.39, 37.76], [-122.39, 37.75],
                               [-122.4, 37.75]]
                Features:  {'height': 12.5}
            Key:  building2 Asset:
                Coordinates:  [[-122.38, 37.77], [-122.38, 37.78],
                               [-122.37, 37.78], [-122.37, 37.77],
                               [-122.38, 37.77]]
                Features:  {'height': 8.0}
        """
        print(self.__class__.__name__)
        print("Inventory stored in: ", self.inventory.__class__.__name__)
        for key, asset in self.inventory.items():
            print("Key: ", key, "Asset:")
            asset.print_info()

    def remove_asset(self, asset_id: Union[str, int]) -> bool:
        """
        Remove an asset from the inventory.

        Args:
            asset_id(str or int):
                The unique identifier for the asset.

        Returns:
            bool: ``True`` if asset was removed, ``False`` otherwise.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset(
            ...     asset_id='building1',
            ...     asset=Asset(
            ...         'building1',
            ...         [[-122.40, 37.75], [-122.40, 37.76],
            ...          [-122.39, 37.76], [-122.39, 37.75],
            ...          [-122.40, 37.75]],
            ...         features={'height': 12.5}
            ...     )
            ... )
            >>> inventory.remove_asset('building1')
            True
            >>> inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
        """
        if asset_id in self.inventory:
            del self.inventory[asset_id]
            return True
        else:
            return False

    def remove_features(self, features_to_remove: Iterable[str]) -> bool:
        """
        Remove specified features from all assets in the inventory.

        Args:
            features_to_remove(Iterable[str]):
                An iterable of feature keys to remove from each :class:`Asset`.
                Accepts types such as ``list``, ``tuple``, ``dict_keys``, etc.

        Returns:
            bool:
                ``True`` if at least one feature was removed from any asset,
                ``False`` otherwise.

        Raises:
            TypeError: If ``features_to_remove`` is not an iterable of strings.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset(
            ...     asset_id='building1',
            ...     asset=Asset(
            ...         'building1',
            ...         [[-122.40, 37.75], [-122.40, 37.76],
            ...          [-122.39, 37.76], [-122.39, 37.75],
            ...          [-122.40, 37.75]],
            ...         features={'height': 12.5, 'floors': 3}
            ...     )
            ... )
            >>> _ = inventory.add_asset(
            ...     asset_id='building2',
            ...     asset=Asset(
            ...         'building2',
            ...         [[-122.38, 37.77], [-122.38, 37.78],
            ...          [-122.37, 37.78], [-122.37, 37.77],
            ...          [-122.38, 37.77]],
            ...         features={'height': 8.0, 'floors': 2}
            ...     )
            ... )
            >>> inventory.remove_features(['floors'])
            True
            >>> inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
            Key:  building1 Asset:
                Coordinates:  [[-122.4, 37.75], [-122.4, 37.76],
                               [-122.39, 37.76], [-122.39, 37.75],
                               [-122.4, 37.75]]
                Features:  {'height': 12.5}
            Key:  building2 Asset:
                Coordinates:  [[-122.38, 37.77], [-122.38, 37.78],
                               [-122.37, 37.78], [-122.37, 37.77],
                               [-122.38, 37.77]]
                Features:  {'height': 8.0}
        """
        if (not isinstance(features_to_remove, Iterable)
                or not all(isinstance(f, str) for f in features_to_remove)):
            raise TypeError(
                'features_to_remove must be an iterable of strings.'
            )

        results = [asset.remove_features(features_to_remove)
                   for asset in self.inventory.values()]
        return any(results)

    def write_to_geojson(self, output_file: str = "") -> Dict:
        """
        Write inventory to a GeoJSON file and return the GeoJSON dictionary.

        This method generates a GeoJSON representation of the asset inventory,
        writes it to the specified file path (if provided), and returns the
        GeoJSON object.

        Args:
            output_file(str, optional):
                Path to the output GeoJSON file. If empty, no file is written.

        Returns:
            Dict:
                A dictionary containing the GeoJSON ``FeatureCollection``.

        Examples:
            Define an AssetInventory consisting of a single asset.

            >>> inventory = AssetInventory()
            >>> inventory.add_asset(
            ...     asset_id='asset1',
            ...     asset=Asset(
            ...         'asset1',
            ...         [
            ...             [-122.40, 37.75],
            ...             [-122.40, 37.76],
            ...             [-122.39, 37.76],
            ...             [-122.39, 37.75],
            ...             [-122.40, 37.75]
            ...         ],
            ...         features={'name': 'Building A', 'roofHeight': 22.6}
            ...     )
            ... )

            Write the AssetInventory data into a GeoJSON dictionary.

            >>> geojson_dict = inventory.write_to_geojson()
            >>> print(geojson_dict['features'])
            [{'type': 'Feature', 'properties': {'name': 'Building A',
            'roofHeight': 22.6}, 'geometry': {'type': 'Polygon', 'coordinates':
            [[[-122.4, 37.75], [-122.4, 37.76], [-122.39, 37.76],
            [-122.39, 37.75], [-122.4, 37.75]]]}, 'id': '0'}]

            Write the AssetInventory data in a GeoJSON dictionary and a file
            named  ``output.geojson``.

            >>> geojson_written = inventory.write_to_geojson('output.geojson')
            Wrote 1 assets to {../output.geojson}
        """
        geojson = self.get_geojson()

        # Ensure each feature has an 'id' field in the top-level feature object
        # This is the variation of GeoJSON that is used in R2D:
        for index, feature in enumerate(geojson['features']):
            feature_id = feature['properties'].pop("id", str(index))
            feature['id'] = str(feature_id)

        # If a file name is provided, write the created GeoJSON dictionary into
        # a GeoJSON file:
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(clean_floats(geojson), f, indent=2)
                print(
                    f"Wrote {len(geojson['features'])} assets to "
                    f'{os.path.abspath(output_file)}'
                )
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
            file_path(str):
                  The path to the CSV file
            keep_existing(bool):
                  If ``False``, the inventory will be initialized
            str_type(str):
                  ``building`` or ``bridge``
            id_column(str):
                  The name of column that contains id values. If ``None``, new
                  indices will be assigned

        Returns:
            bool:
                  ``True`` if assets were addded, ``False`` otherwise.
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
        lat_id = [i for i, y in enumerate(key_names) if y.lower() in lat]
        lon_id = [i for i, x in enumerate(key_names) if x.lower() in lon]
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

            # TODO: Avoid hardcoding types here; consider using dynamic type
            # handling or type hints instead.
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

    def read_from_geojson(
        self,
        file_path: str,
        id_column: Optional[str] = None,
        keep_existing: bool = False
    ) -> bool:
        """
        Read inventory data from a GeoJSON file and add it to the inventory.

        This method loads an asset inventory from a provided GeoJSON file path.
        The GeoJSON should be a FeatureCollection where each Feature represents
        an asset with geometry and properties.

        Args:
            file_path (str):
                The path to the GeoJSON file
            id_column (str, optional):
                The name of the parameter containing asset IDs. If ``None``,
                the asset IDs will be assigned automatically.
            keep_existing (bool):
                If False, the inventory will be cleared before loading.
                If True, new assets will be added to existing inventory.
                Defaults to False.

        Returns:
            bool:
                True if assets were added successfully, False otherwise.

        Raises:
            Exception:
                If the file doesn't exist, is not valid JSON, or doesn't
                conform to the expected GeoJSON schema.
        """
        if not keep_existing:
            self.inventory = {}

        # Attempt to open and parse the file
        try:
            with open(file_path, mode="r", encoding="utf-8") as geojson_file:
                data = json.load(geojson_file)
        except FileNotFoundError:
            raise Exception(f"The file {file_path} does not exist.")
        except json.JSONDecodeError as e:
            raise Exception(f"The file {file_path} is not valid JSON: {e}")

        # Validate basic GeoJSON structure
        if not isinstance(data, dict):
            raise Exception("GeoJSON must be a dictionary object.")

        if data.get("type") != "FeatureCollection":
            raise Exception("GeoJSON must be a FeatureCollection.")

        if "features" not in data:
            raise Exception("GeoJSON FeatureCollection must contain 'features'.")

        features = data["features"]
        if not isinstance(features, list):
            raise Exception("GeoJSON 'features' must be a list.")

        # Process each feature
        id_counter = 1
        if keep_existing and len(self.inventory) > 0:
            # Start ID counter after existing assets
            existing_ids = [k for k in self.inventory.keys()
                          if isinstance(k, int)]
            if existing_ids:
                id_counter = max(existing_ids) + 1

        for i, feature in enumerate(features):
            # Validate feature structure
            if not isinstance(feature, dict):
                print(f"Warning: Skipping invalid feature at index {i} "
                      f"(not a dictionary)")
                continue

            if feature.get("type") != "Feature":
                print(f"Warning: Skipping feature at index {i} "
                      f"(type is not 'Feature')")
                continue

            # Extract geometry
            geometry = feature.get("geometry")
            if not geometry:
                print(f"Warning: Skipping feature at index {i} "
                      f"(no geometry)")
                continue

            # Convert geometry to coordinates format expected by Asset class
            coordinates = []
            geom_type = geometry.get("type")
            geom_coords = geometry.get("coordinates")

            if not geom_coords:
                print(f"Warning: Skipping feature at index {i} "
                      f"(no coordinates in geometry)")
                continue

            try:
                if geom_type == "Point":
                    # Point: coordinates are [lon, lat]
                    coordinates = [geom_coords]
                elif geom_type == "LineString":
                    # LineString: coordinates are [[lon1, lat1], [lon2, lat2], ...]
                    coordinates = geom_coords
                elif geom_type == "Polygon":
                    # Polygon: coordinates are [[[lon1, lat1], [lon2, lat2], ...]]
                    # We take the exterior ring (first array)
                    coordinates = geom_coords[0]
                else:
                    print(f"Warning: Skipping feature at index {i} "
                          f"(unsupported geometry type: {geom_type})")
                    continue
            except (IndexError, TypeError) as e:
                print(f"Warning: Skipping feature at index {i} "
                      f"(invalid coordinates structure: {e})")
                continue

            # Extract properties (features)
            properties = feature.get("properties", {})
            if not isinstance(properties, dict):
                print(f"Warning: Feature at index {i} has invalid properties "
                      f"(not a dictionary), using empty properties")
                properties = {}

            # Determine asset ID
            asset_id = None

            # First, check if there's an 'id' field at the feature level
            if id_column is None:
                id_key = "id"
            else:
                id_key = id_column

            if id_key in feature:
                asset_id = feature[id_key]
            # Then check if there's an 'id' in properties
            elif id_key in properties:
                asset_id = properties[id_key]  # Remove from properties
            else:
                # Use auto-generated ID
                asset_id = id_counter
                id_counter += 1

            # Convert string IDs to int if they represent integers
            if isinstance(asset_id, str) and asset_id.isdigit():
                asset_id = int(asset_id)

            # Create and add the asset
            try:
                asset = Asset(asset_id, coordinates, properties)
                success = self.add_asset(asset_id, asset)
                if not success:
                    print(f"Warning: Asset with ID {asset_id} already exists, "
                          f"skipping feature at index {i}")
            except Exception as e:
                print(f"Warning: Failed to create asset from feature at "
                      f"index {i}: {e}")
                continue

        return True

    def add_asset_features_from_csv(
        self,
        file_path: str,
        id_column: Union[str, None]
    ) -> bool:
        """
        Read inventory data from a CSV file and add it to the inventory.

        Args:
            file_path(str):
                  The path to the CSV file
            id_column(str):
                  The name of column that contains id values. If ``None``, new
                  indicies will be assigned

        Returns:
            bool: ``True`` if assets were addded, ``False`` otherwise.
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

        - A ``DataFrame`` containing the features of each asset, with support
          for multiple possible worlds(realizations).
        - A ``DataFrame`` containing centroid coordinates(longitude and
          latitude) for spatial operations.
        - The total number of assets in the inventory.

        The method flattens feature lists into separate columns if multiple
        possible worlds exist. It also derives centroid coordinates from the
        GeoJSON geometry for each asset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, int]:

                - Asset feature ``DataFrame`` (indexed by asset ID).
                - Asset geometry ``DataFrame`` with 'Lat' and 'Lon' columns.
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
            {**self.inventory[i].features, "index": i}
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
        Extract a single realization(possible world) from an inventory.

        This method generates a new :class:`AssetInventory` instance where all
        features containing multiple possible worlds are reduced to a single
        realization specified by `id`. Features that are not lists are copied
        as-is .

        Args:
            id(int, default=0):
                The index of the realization to extract. Must be within the
                range of available possible worlds(0-based indexing).

        Returns:
            AssetInventory:
                A new inventory object representing the selected realization.

        Raises:
            Exception: If ``id > 0`` but the inventory only contains a single
                       realization.
            Exception: If any feature has fewer realizations than the specified
                       ``id``.
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
        ``world_realization`` inventory into the current multi-world inventory
        by updating the realization at the specified index ``id``.

        Args:
            id(int):
                The index of the world(realization) to update. Must be less
                than ``self.n_pw``.
            world_realization(AssetInventory):
                An :class:`AssetInventory` instance representing a single
                realization of the world. All features in this inventory must
                be scalar (i.e., not lists).

        Raises:
            Exception:
                - If the specified ``id`` is not within the valid range of
                  realizations.
                - If ``world_realization`` contains features with multiple
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

        - Keys associated with multi-valued features(i.e., lists).
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
