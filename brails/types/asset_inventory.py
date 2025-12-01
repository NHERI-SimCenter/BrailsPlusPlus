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
# 10-22-2025

"""
This module defines classes associated with asset inventories.

.. autosummary::

    AssetInventory
    Asset
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections.abc import Hashable, Iterable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from importlib.metadata import PackageNotFoundError, version

import pandas as pd
from shapely import box
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection, shape, mapping
from shapely.geometry.base import BaseGeometry  # noqa: TC002

from brails.utils.clean_floats import clean_floats
from brails.types.housing_unit_inventory import HousingUnitInventory
from brails.utils.geo_tools import GeoTools
from brails.utils.input_validator import InputValidator
from brails.utils.spatial_join_methods.base import SpatialJoinMethods


class Asset:
    """A spatial asset with geometry coordinates, and attributes.

    To import the :class:`Asset` class, use:

    .. code-block:: python

        from brails.types.asset_inventory import Asset


    Attributes:
        asset_id (str or int):
            Unique identifier for the asset.
        coordinates (list[list[float]] | list[list[list[float]]]):
            The geometry of the asset. Supports:
            - ``[[x, y]]`` (Point)
            - ``[[x, y], [x, y], ...]`` (LineString, Polygon)
            - ``[[[x, y], ...], ...]`` (MultiLineString, MultiPolygon)
        features (dict[str, Any], optional):
            Additional attributes/features of the asset. Defaults to ``None``.
    """

    def __init__(
        self,
        asset_id: str | int,
        coordinates: list[list[float]],
        features: dict[str, Any] | None = None,
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
        coords_check, output_msg = InputValidator.validate_coordinates(coordinates)
        if coords_check:
            self.coordinates = coordinates
        else:
            print(f'{output_msg} for {asset_id}; defaulting to an empty list.')
            self.coordinates = []

        self.features = features if features is not None else {}

    def add_features(
        self,
        additional_features: dict[str, Any],
        vector_features: list[str] | None = None,
        *,
        overwrite: bool = True,
    ) -> tuple[bool, int]:
        """
        Update the existing features in the asset.

        Args:
            additional_features (dict[str, Any]):
                New features to merge into the asset's features.
            vector_features (list[str], optional):
                A list of feature names that store their values in vectors.
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
        if vector_features is None:
            vector_features = []

        def update_possible_worlds(n_pw: int, key: str, val: Any) -> int:  # noqa: ANN401
            """Check a feature and update the number of possible worlds."""
            is_probabilistic = False

            if key in vector_features:
                # These are only probabilistic if they are a list of lists.
                if (
                    isinstance(val, list)
                    and len(val) > 0
                    and isinstance(val[0], list)
                ):
                    is_probabilistic = True

            elif isinstance(val, list):
                is_probabilistic = True

            # If the feature is probabilistic, update n_pw
            if is_probabilistic:
                if (n_pw != 1) and (n_pw != len(val)):
                    print(
                        f'WARNING: # possible worlds was {n_pw} but now '
                        f'is {len(val)}. Something went wrong.'
                    )
                return len(val)

            # Otherwise, return the existing n_pw
            return n_pw

        n_pw: int = 1

        if overwrite:
            # Overwrite existing features with new ones:
            self.features.update(additional_features)

            for key, val in additional_features.items():
                n_pw = update_possible_worlds(n_pw, key, val)

            updated = True

        else:
            # Only update with new keys, do not overwrite existing keys:
            updated = False
            for key, val in additional_features.items():
                if key not in self.features:
                    # write
                    self.features[key] = val

                    n_pw = update_possible_worlds(n_pw, key, val)

                    updated = True

        return updated, n_pw


    def get_geometry(self) -> BaseGeometry | None:
        """
        Convert coordinates into a robust Shapely geometry object.

        Wraps `GeoTools.list_of_lists_to_geometry` to handle errors gracefully.

        Returns:
            BaseGeometry | None: A valid Shapely geometry, or None if the
            coordinates are malformed or invalid.
        """
        if not self.coordinates:
            return None

        try:
            return GeoTools.list_of_lists_to_geometry(self.coordinates)

        except ValueError as e:
            print(f"WARNING: Asset geometry invalid. {e}")
            return None


    def get_centroid(self) -> list[list[float | None]]:
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
            [[-122.4192, 37.77495]]

            >>> empty_asset = Asset(asset_id='empty', coordinates=[])
            Coordinates input is empty for empty; defaulting to an empty list.
            >>> empty_asset.get_centroid()
            [[None, None]]
        """
        geometry = self.get_geometry()

        if geometry:
            centroid = geometry.centroid
            return [[centroid.x, centroid.y]]

        return [[None, None]]

    def hash_asset(self) -> str:
        """
        Generate a unique hash for this asset based on its coordinates and features.

        This hash can be used to quickly identify duplicate assets by comparing
        geometry and attribute data. Both the coordinates and features are
        serialized as strings and then hashed using MD5.

        Returns:
            str: A hexadecimal string representing the MD5 hash of the asset.

        Example:
            >>> asset1 = Asset(
            ...     asset_id='123',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4180, 37.7755]],
            ...     features={'roof_type': 'gable'}
            ... )
            >>> asset2 = Asset(
            ...     asset_id='124',
            ...     coordinates=[[-122.4194, 37.7749], [-122.4180, 37.7755]],
            ...     features={'roof_type': 'flat'}
            ... )
            >>> hash1 = asset1.hash_asset()
            >>> print(hash1)
            1e629fc184329ea688c648e7663b439d
            >>> hash2 = asset2.hash_asset()
            >>> print(hash2)
            ac0d104a411f92bffff8cc1398257dd8
            >>> hash1 != hash2
            True
        """
        coord_str = str(self.coordinates)
        feat_str = str(self.features)
        return hashlib.md5((coord_str + feat_str).encode()).hexdigest()

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
        if not isinstance(features_to_remove, Iterable) or not all(
            isinstance(k, str) for k in features_to_remove
        ):
            raise TypeError('features_to_remove must be an iterable of strings.')

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
        print('Coordinates: ', self.coordinates)
        features_json = json.dumps(self.features, indent=2)
        print(f'Features: {features_json}')


class AssetInventory:
    """
    A class representing a collection of Assets managed as an inventory.

    This class provides methods to add, manipulate, join, write and query
    a collection of :`class:Asset` objects.

    To import the :class:`AssetInventory` class, use:

    .. code-block:: python

        from brails.types.asset_inventory import AssetInventory
    """

    def __init__(self) -> None:
        """Initialize AssetInventory with an empty inventory dictionary."""
        self.inventory: dict = {}
        self.housing_unit_inventory: HousingUnitInventory | None = None
        self.n_pw = 1
        self.vector_features: list[str] = ['HousingUnits']

    def add_asset(self, asset_id: str | int, asset: Asset) -> bool:
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
            raise TypeError('Expected an instance of Asset.')

        if asset_id in self.inventory:
            print(f'Asset with id {asset_id} already exists. Asset was not added')
            return False

        self.inventory[asset_id] = asset
        return True

    def add_asset_coordinates(
        self,
        asset_id: str | int,
        coordinates: list[list[float]] | list[list[list[float]]]
    ) -> bool:
        """
        Add an ``Asset`` to the inventory by adding its coordinate information.

        Args:
            asset_id(str or int):
                The unique identifier for the asset.
            coordinates (list[list[float]] | list[list[list[float]]]):
                A nested list of floats representing the geometry.
                Accepts Points, LineStrings, Polygons, and Multi-geometries.

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
            print(
                f'Asset with id {asset_id} already exists. Coordinates were '
                'not added'
            )
            return False

        # Create asset and add using id as the key:
        asset = Asset(asset_id, coordinates)
        self.inventory[asset_id] = asset

        return True

    def add_asset_features(
        self,
        asset_id: str | int,
        new_features: dict[str, Any],
        *,
        overwrite: bool = True,
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
            print(
                f'No existing Asset with id {asset_id} found. Asset '
                'features not added.'
            )
            return False

        status, n_pw = asset.add_features(
            new_features,
            vector_features=self.vector_features,
            overwrite=overwrite
        )
        if n_pw == 1:
            # The new feature is deterministic.
            # This doesn't change the inventory's n_pw, so do nothing.
            pass
        elif self.n_pw == 1:
            # The inventory was deterministic (n_pw=1), but this new
            # feature is probabilistic. Set the inventory's n_pw.
            self.n_pw = n_pw
        elif self.n_pw == n_pw:
            # The new feature's n_pw matches the inventory's n_pw.
            # This is the consistent case, so do nothing.
            pass
        else:
            # This is a mismatch. The inventory had one n_pw (e.g., 5)
            # and the new feature has a different one (e.g., 3).
            print(
                f'WARNING: # possible worlds was {self.n_pw} but is now '
                f'{n_pw}. Something went wrong.'
            )
            # Set to the new value as per the original logic
            self.n_pw = n_pw
        return status

    def add_model_predictions(
        self, predictions: dict[Any, Any], feature_key: str
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
            raise ValueError(
                "None of the keys in 'predictions' exist in 'self.inventory'."
            )

        # Validate feature_key input:
        if not isinstance(feature_key, str):
            raise TypeError("Expected 'feature_key' to be a string.")

        # Update inventory items with corresponding predictions:
        for asset_id in self.get_asset_ids():
            if asset_id in predictions:
                self.add_asset_features(
                    asset_id, {feature_key: predictions.get(asset_id)}
                )

    def change_feature_names(self, feature_name_mapping: dict[str, str]) -> None:
        """
        Rename feature names in ``AssetInventory`` via user-specified mapping.

        Args:
            feature_name_mapping (dict):
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
            raise TypeError("The 'feature_name_mapping' must be a dictionary.")

        # Validate that all keys and values are strings:
        for original_name, new_name in feature_name_mapping.items():
            if not isinstance(original_name, str) or not isinstance(new_name, str):
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
                    asset.features[new_name] = asset.features.pop(original_name)

    def combine(
        self,
        other_inventory: AssetInventory,
        asset_id_map: dict[str | int, str | int] | None = None,
    ) -> dict[str | int, str | int]:
        """
        Merge another AssetInventory into this one, handling conflicts.

        This method merges all unique assets from `other_inventory` into
        the current inventory (`self`).

        - **Asset De-duplication:** It automatically skips any incoming
          asset that is an exact duplicate of an existing one (based on
          a hash of its geometry and features).
        - **Asset ID Conflicts:** It resolves all asset ID collisions.
          You can provide an `asset_id_map` to suggest new IDs. If a
          suggested ID (or an original ID) already exists, a new
          sequential numeric ID will be assigned.
        - **Housing Unit Merging:** If both inventories have linked housing unit
          data, this method validates both, then merges the incoming
          `HousingUnitInventory`, re-indexes all new housing unit IDs to
          prevent conflicts, and updates the `HousingUnits` feature
          on all newly-added assets.

        Args:
            other_inventory (AssetInventory):
                The secondary inventory to merge into this one.
            asset_id_map (dict, optional):
                A dictionary mapping {original_asset_id: suggested_asset_id}.
                Use this to rename asset IDs from `other_inventory` during
                the merge.

        Returns:
            dict[str | int, str | int]:
                A `final_id_map` dictionary, which acts as a "receipt"
                mapping {original_asset_id: final_asset_id}. This is
                critical for data traceability, as it reports the
                actual ID assigned after resolving all conflicts.

        Raises:
            ValueError, LookupError, TypeError:
                If either `self` or `other_inventory` has an invalid
                housing unit state (e.g., orphan IDs, invalid data types)
                before the merge.
        """
        # --- 1. Asset Merging ---

        # Build a set of existing hashes for O(1) duplicate checking
        existing_hashes = {
            asset.hash_asset(): key for key, asset in self.inventory.items()
        }
        next_id = self._get_next_numeric_id()
        final_id_map = {}

        for orig_key, asset in other_inventory.inventory.items():
            asset_hash = asset.hash_asset()

            # Skip any asset that is a perfect duplicate
            if asset_hash in existing_hashes:
                continue

            # Determine the suggested ID, using the map or the original key
            suggested_id = (
                asset_id_map.get(orig_key, orig_key) if asset_id_map else orig_key
            )
            final_id = suggested_id

            # Ensure uniqueness.
            while final_id in self.inventory:
                final_id = next_id
                next_id += 1

            # Add the asset to the inventory and record its final ID
            self.add_asset(final_id, asset)
            final_id_map[orig_key] = final_id
            existing_hashes[asset_hash] = (
                final_id  # Add hash to prevent duplicates from within other_inventory
            )

        # --- 2. Housing Unit Inventory Merging ---
        hu_id_remap = {}

        # Case 1: We have no inventory, but the incoming one does.
        if (
            self.housing_unit_inventory is None
            and other_inventory.housing_unit_inventory is not None
        ):
            # First, validate the incoming inventory to prevent merging bad data
            print('Validating incoming housing unit inventory...')
            other_inventory.validate_housing_unit_assignments()

            # Use deepcopy to avoid two AssetInventory objects pointing to
            # the same mutable HousingUnitInventory object.
            self.housing_unit_inventory = deepcopy(other_inventory.housing_unit_inventory)
            # No ID remapping is needed; the new assets' links are already correct.

        # Case 2: Both inventories have housing unit data.
        elif (
                self.housing_unit_inventory is not None
                and other_inventory.housing_unit_inventory is not None
        ):
            # Validate both inventories before attempting to merge.
            print('Validating current housing unit inventory...')
            self.validate_housing_unit_assignments()
            print('Validating incoming housing unit inventory...')
            other_inventory.validate_housing_unit_assignments()

            # Merge the incoming inventory into our own. This returns a
            # map of {old_hu_id: new_hu_id} for all merged housing units.
            print('Merging housing unit inventories... Re-indexing incoming data.')
            hu_id_remap = self.housing_unit_inventory.merge_inventory(
                other_inventory.housing_unit_inventory
            )

        # Case 3 (both are None) -> do nothing.
        # Case 4 (self has one, other is None) -> do nothing.

        # --- 3. Asset Housing-Unit Link Fixing ---

        # If we remapped housing unit IDs (Case 2), we must update the
        # 'HousingUnits' feature on all newly-added assets.
        if hu_id_remap:
            print('Updating housing unit links for merged assets...')

            # We only need to check the assets we *just added*.
            for final_id in final_id_map.values():
                asset = self.inventory[final_id]
                old_hu_id_list = asset.features.get('HousingUnits')

                if old_hu_id_list:
                    # Rebuild the list using the re-map.
                    remapped_hu_id_list = [
                        hu_id_remap[old_hu_id] for old_hu_id in old_hu_id_list
                    ]

                    self.add_asset_features(
                        final_id, {'HousingUnits': remapped_hu_id_list}, overwrite=True
                    )

        return final_id_map

    def convert_polygons_to_centroids(self) -> None:
        """
        Convert geometries in the inventory to their centroid points.

        Iterates through the asset inventory and replaces the coordinates of
        each geometry with the coordinates of its centroid using the
        logic defined in `Asset.get_centroid()`.

        Notes:
            - Points are left unchanged.
            - Polygons, LineStrings, and Multi-geometries are converted to their centroid.
            - Invalid or malformed geometries will be replaced with ``[[None, None]]``
              and a warning will be printed by the Asset class.

        Modifies:
            self.inventory (dict):
                Updates the ``coordinates`` field of each asset in-place.

        Example:
            >>> inventory = AssetInventory()
            >>> # 1. Polygon
            >>> inventory.add_asset_coordinates(
            ...     'asset1',
            ...     [
            ...         [-122.4194, 37.7749],
            ...         [-122.4194, 37.7849],
            ...         [-122.4094, 37.7849],
            ...         [-122.4094, 37.7749],
            ...         [-122.4194, 37.7749]
            ...     ]
            ... )
            >>> # 2. LineString
            >>> inventory.add_asset_coordinates(
            ...     'asset2',
            ...     [
            ...         [-122.4194, 37.7749],
            ...         [-122.4094, 37.7849]
            ...     ]
            ... )
            >>> # 3. MultiPolygon
            >>> inventory.add_asset_coordinates(
            ...     'asset3',
            ...     [
            ...         [
            ...             [-122.4194, 37.7749], [-122.4194, 37.7849],
            ...             [-122.4094, 37.7849], [-122.4094, 37.7749],
            ...             [-122.4194, 37.7749]
            ...         ],
            ...         [
            ...             [-122.3000, 37.8000], [-122.3000, 37.8100],
            ...             [-122.2900, 37.8100], [-122.2900, 37.8000],
            ...             [-122.3000, 37.8000]
            ...         ]
            ...     ]
            ... )
            >>> inventory.convert_polygons_to_centroids()
            >>> inventory.get_asset_coordinates('asset1')
            [[-122.4144, 37.7799]]
            >>> inventory.get_asset_coordinates('asset2')
            [[-122.4144, 37.7799]]
        """
        for asset in self.inventory.values():
            asset.coordinates = asset.get_centroid()

    def get_asset_coordinates(self, asset_id: str | int) -> tuple[bool, list]:
        """
        Get the coordinates of a particular asset.

        Args:
            asset_id (str or int):
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
        self, asset_id: str | int
    ) -> tuple[bool, dict[str, Any]]:
        """
        Get features of a particular asset.

        Args:
            asset_id (str or int):
                The unique identifier for the asset.

        Returns:
            tuple[bool, dict]:
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

    def get_all_asset_features(self) -> set[str]:
        """
        Retrieves a set of unique feature keys present across all assets.

        Iterates through every asset and collects the keys from their 'features'
        dictionaries. This operation handles deduplication automatically.

        Returns:
            set[str]: A collection of unique feature names found in the inventory.
        """
        return {
            feature
            for asset in self.inventory.values()
            for feature in asset.features
        }

    def get_asset_ids(self) -> list[str | int]:
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

    def get_assets_intersecting_polygon(self, bpoly: BaseGeometry):
        """
        Get assets with geometries intersecting a bounding polygon.

        This method performs a spatial intersection check between each asset's
        geometry in the inventory and a provided bounding polygon (or
        multipolygon). Assets that intersect the polygon are identified and
        retained. All non-intersecting assets are removed from the inventory.

        Args:
            bpoly (shapely.geometry.base.BaseGeometry):
                The bounding polygon or multipolygon used to determine spatial
                intersections.

        Raises:
            TypeError:
                If ``bpoly`` is not a ``Polygon`` or ``MultiPolygon``.

        Example:
            >>> from shapely.geometry import Polygon
            >>> inventory = AssetInventory()
            >>> # A LineString in Dallas, TX (will intersect the Dallas bpoly):
            >>> _ = inventory.add_asset_coordinates(
            ...     'bridge_A',
            ...     [[-96.8003, 32.7767], [-96.7998, 32.7770]]
            ... )
            >>> # A Polygon in Los Angeles, CA (will NOT intersect the bpoly):
            >>> _ = inventory.add_asset_coordinates(
            ...     'tower_B',
            ...     [
            ...         [-118.2450, 34.0537],
            ...         [-118.2450, 34.0540],
            ...         [-118.2445, 34.0540],
            ...         [-118.2445, 34.0537],
            ...         [-118.2450, 34.0537],
            ...     ]
            ... )
            >>> # A bounding polygon roughly around downtown Dallas:
            >>> bpoly = Polygon([
            ...     (-96.81, 32.77),
            ...     (-96.81, 32.78),
            ...     (-96.79, 32.78),
            ...     (-96.79, 32.77),
            ...     (-96.81, 32.77)
            ... ])
            >>>
            >>> inventory.get_assets_intersecting_polygon(bpoly)
            >>> 'bridge_A' in inventory.inventory
            True
            >>> 'tower_B' in inventory.inventory
            False
        """
        # Validate bounding polygon type:
        if bpoly.geom_type not in ['Polygon', 'MultiPolygon']:
            raise TypeError(
                f'Invalid bounding polygon type: {bpoly.geom_type}. '
                "Expected 'Polygon' or 'MultiPolygon'."
            )

        # Fix the bounding polygon if it is not well-formed:
        if not bpoly.is_valid:
            bpoly = bpoly.buffer(0)

        valid_geometries = {}
        keys_to_remove = set()

        for key, asset in self.inventory.items():
            geom = asset.get_geometry()

            if geom and not geom.is_empty and geom.is_valid:
                valid_geometries[key] = geom
            else:
                keys_to_remove.add(key)

        if valid_geometries:
            non_intersecting_keys = GeoTools.compare_geometries(
                bpoly, valid_geometries, 'intersects'
            )
            keys_to_remove.update(non_intersecting_keys)

        for key in keys_to_remove:
            self.remove_asset(key)

    def get_coordinates(
        self,
    ) -> tuple[list[list[list[float]] | list[list[list[float]]]], list[str | int]]:
        """
        Get geometry(coordinates) and keys of all assets in the inventory.

        Returns:
            tuple:
                A tuple containing:
                - A list of coordinates for each asset. Depending on the geometry
                  type, this will be:
                  - Depth 2: ``[[lon, lat], ...]`` (Point, LineString, Polygon)
                  - Depth 3: ``[[[lon, lat], ...], ...]`` (MultiLineString, MultiPolygon)
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
        seed: int | float | str | bytes | bytearray | None = None,
    ) -> AssetInventory:
        """
        Generate a smaller asset inventory with a random selection of assets.

        This method randomly selects ``nsamples`` assets from the existing
        inventory and returns a new :class:`AssetInventory` instance containing
        only these sampled assets. The randomness can be controlled using an
        optional ``seed`` for reproducibility.

        Args:
            nsamples (int):
                The number of assets to randomly sample from the inventory.
                Must be a positive integer not exceeding the total number of
                assets.
            seed (int or float or str or bytes or bytearray or None, optional):
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
            raise ValueError(
                'Number of samples cannot exceed the number of '
                'assets in the inventory'
            )

        if seed is not None:
            random.seed(seed)

        random_keys = random.sample(list(self.inventory.keys()), nsamples)

        result = AssetInventory()
        for key in random_keys:
            result.add_asset(key, self.inventory[key])

        return result

    def get_extent(self, buffer: str | list[float] = 'default') -> box:
        """
        Calculate the geographical extent of the inventory.

        Args:
            buffer (str or list[float]): A string or a list of 4 floats.

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
        error_msg = (
            'Invalid buffer input. Valid options for the buffer input'
            "are 'default', 'none', or a list of 4 integers."
        )

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

        # Determine the geographical extent of the inventory
        valid_geoms = [
            asset.get_geometry()
            for asset in self.inventory.values()
        ]
        valid_geoms = [g for g in valid_geoms if g and not g.is_empty]

        if not valid_geoms:
            return box(0, 0, 0, 0)

        min_x, min_y, max_x, max_y = GeometryCollection(valid_geoms).bounds

        # Return a Shapely polygon of the bounding box:
        return box(
            min_x - buffer_levels[0],
            min_y - buffer_levels[1],
            max_x + buffer_levels[2],
            max_y + buffer_levels[3],
        )

    def get_geojson(self) -> dict[str, Any]:
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
            brails_version = version('BRAILS')
        except PackageNotFoundError:
            brails_version = 'NA'

        geojson = {
            'type': 'FeatureCollection',
            'generated': str(datetime.now(timezone.utc)),
            'brails_version': brails_version,
            'crs': {
                'type': 'name',
                'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'},
            },
            'features': [],
        }

        # Initialize list to track assets with invalid geometries
        failed_ids = []

        for key, asset in self.inventory.items():
            geometry_obj = asset.get_geometry()

            # Only store valid geometries
            if geometry_obj and not geometry_obj.is_empty:
                properties = asset.features | {'id': key}

                geojson['features'].append(
                    {
                        'type': 'Feature',
                        'properties': properties,
                        'geometry': mapping(geometry_obj),
                    }
                )
            else:
                # Track failures
                failed_ids.append(str(key))

        # Report Failures
        if failed_ids:
            print(f"\nWarning: {len(failed_ids)} assets were skipped due to invalid or unknown geometry.")
            print(f"Skipped IDs: {', '.join(failed_ids)}")

        return geojson

    def join(
        self,
        inventory_to_join: AssetInventory,
        method: str = 'GetPointsInPolygons',
    ) -> None:
        """
        Merge with another AssetInventory using specified spatial join method.

        Args:
            inventory_to_join (AssetInventory):
                The inventory to be joined with the current one.
            method (str):
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
            raise TypeError(
                'Inventory input specified for join needs to be an AssetInventory'
            )

        # Ensure method is a valid string:
        if not isinstance(method, str):
            raise TypeError('Join method should be a valid string')

        # Perform the spatial join using the specified method:
        self = SpatialJoinMethods.execute(method, self, inventory_to_join)

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
        print('Inventory stored in: ', self.inventory.__class__.__name__)
        for key, asset in self.inventory.items():
            print('Key: ', key, 'Asset:')
            asset.print_info()

    def read_from_geojson(  # noqa: C901
        self,
        file_path: str,
        asset_type: str = 'building',
        id_column: str | None = None,
    ) -> bool:
        """
        Reads a GeoJSON file and imports assets into the asset inventory.

        This method loads a GeoJSON FeatureCollection, validates its
        structure, checks geometries, and converts each asset_data into a
        BRAILS `Asset` object.

        It also supports mapping asset IDs from the GeoJSON file's properties
        (using `id_column`) or auto-generating new numeric IDs.

        Args:
            file_path (str):
                Path to the GeoJSON file to be read. Must represent a valid
                GeoJSON FeatureCollection.
            asset_type (str, optional):
                The 'type' assigned to all imported assets.
                Defaults to `'building'`.
            id_column (str, optional):
                The name of the asset_data-level or properties key containing
                the unique asset ID. If `None` (default), IDs are
                auto-generated sequentially.

        Returns:
            bool:
                ``True`` if the GeoJSON file is successfully read and assets
                are added to the inventory.

        Raises:
            FileNotFoundError:
                If the specified `file_path` does not exist or is not a file.
            json.JSONDecodeError:
                If the file is not valid JSON.
            ValueError:
                If the file is invalid JSON, not a valid GeoJSON
                FeatureCollection, or contains no features.

        Example:
            Example GeoJSON file (`seattle_buildings.geojson`):

            .. code-block:: json

                {
                  "type": "FeatureCollection",
                  "features": [
                    {
                      "type": "Feature",
                      "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                          [
                            [-122.3355, 47.6080],
                            [-122.3350, 47.6080],
                            [-122.3350, 47.6085],
                            [-122.3355, 47.6085],
                            [-122.3355, 47.6080]
                          ]
                        ]
                      },
                      "properties": {
                        "name": "Building A",
                        "height_m": 18.7
                      }
                    },
                    {
                      "type": "Feature",
                      "geometry": {
                        "type": "Point",
                        "coordinates": [-122.3321, 47.6062]
                      },
                      "properties": {
                        "name": "Building B",
                        "height_m": 12.4
                      }
                    }
                  ]
                }

            Example usage:

                >>> inv = AssetInventory()
                >>> inv.read_from_geojson(
                ...     'seattle_buildings.geojson',
                ...     asset_type='building',
                ...     id_column='name'
                ... )
                True
                >>> inv.print_info()
                AssetInventory
                Inventory stored in:  dict
                Key:  Building A Asset:
                     Coordinates:  [[-122.3355, 47.608], [-122.335, 47.608],
                 [-122.335, 47.6085], [-122.3355, 47.6085],
                 [-122.3355, 47.608]]
                     Features:  {'name': 'Building A', 'height_m': 18.7,
                 'type': 'building'}
                Key:  Building B Asset:
                     Coordinates:  [[-122.3321, 47.6062]]
                     Features:  {'name': 'Building B', 'height_m': 12.4,
                 'type': 'building'}

        Note:
            All geometries are expected to follow the GeoJSON standard
            ``[longitude, latitude]`` order and use the WGS-84 geographic
            coordinate reference system (EPSG:4326).
        """
        # Path checks:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f'File not found: {file_path}')

        # JSON parsing:
        try:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON format in {file_path}.') from e

        # GeoJSON structure validation:
        if not isinstance(data, dict) or data.get('type') != 'FeatureCollection':
            raise ValueError('Input file is not a valid GeoJSON FeatureCollection.')

        features = data.get('features', [])
        if not isinstance(features, list) or not features:
            raise ValueError('GeoJSON FeatureCollection contains no valid features.')

        id_counter = 1
        for i, asset_data in enumerate(features):
            # Validate asset_data structure
            if not isinstance(asset_data, dict):
                print(
                    f'Warning: Skipping invalid asset data at index {i} '
                    f'(not a dictionary)'
                )
                continue

            if asset_data.get('type') != 'Feature':
                print(
                    f'Warning: Skipping asset data at index {i} '
                    f"(type is not 'Feature')"
                )
                continue

            # Parse Geometry
            try:
                geometry = GeoTools.parse_geojson_geometry(
                    asset_data.get('geometry')
                )
            except (TypeError, ValueError, NotImplementedError, IndexError) as e:
                # Catch-all for any geometry parsing error
                print(
                    f'Warning: Skipping asset data at index {i} '
                    f'(invalid geometry structure: {e})'
                )
                continue

            if not geometry:
                print(
                    f'Warning: Skipping asset data at index {i} '
                    f'(unsupported or empty geometry)'
                )
                continue

            # Get properties and add asset_type
            properties = asset_data.get('properties', {})
            if not isinstance(properties, dict):
                print(
                    f'Warning: Asset data at index {i} has invalid properties '
                    f'(not a dictionary), using empty properties'
                )
                properties = {}

            properties['type'] = asset_type

            # Determine asset ID
            # Use id_column if provided, otherwise default to "id"
            id_key = id_column if id_column else 'id'

            if id_key in asset_data:
                asset_id = asset_data[id_key]
            elif id_key in properties:
                asset_id = properties[id_key]
            else:
                # Use auto-generated ID
                asset_id = id_counter
                id_counter += 1

            # Make sure numeric IDs are represented as ints to avoid duplicates
            if isinstance(asset_id, str) and asset_id.isdigit():
                asset_id = int(asset_id)

            # Make sure the asset_id is hashable
            if not isinstance(asset_id, Hashable):
                print(
                    f'Warning: Skipping asset data at index {i}. '
                    f"The asset ID '{asset_id}' is not a hashable type "
                    '(e.g., it might be a list or dict).'
                )
                continue

            # Create and add the asset
            asset = Asset(asset_id, geometry, properties)
            success = self.add_asset(asset_id, asset)
            if not success:
                print(
                    f'Warning: Asset with ID {asset_id} already exists, '
                    f'skipping asset data at index {i}'
                )

        return True

    def remove_asset(self, asset_id: str | int) -> bool:
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
        if not isinstance(features_to_remove, Iterable) or not all(
            isinstance(f, str) for f in features_to_remove
        ):
            raise TypeError('features_to_remove must be an iterable of strings.')

        results = [
            asset.remove_features(features_to_remove)
            for asset in self.inventory.values()
        ]
        return any(results)

    def remove_nonmatching_assets(self, image_set, verbose=False) -> None:
        """
        Remove assets that do not have corresponding entries in image set.

        This method compares asset keys in the inventory with those in the
        provided ImageSet and removes any asset whose key does not exist in
        the ImageSet.

        Args:
            image_set(ImageSet):
                The image set containing valid image keys.
            verbose(bool, optional):
                If ``True``, prints a summary of removed keys. Default is
                ``False``.

        Modifies:
            self.inventory(dict):
                Removes nonmatching asset entries directly from the inventory.
                The object is updated in-place.

        Example:
            >>> inventory = AssetInventory()
            >>> _ = inventory.add_asset(
            ...     asset_id='house_A',
            ...     asset=Asset(
            ...         'house_A',
            ...         [[-118.5123, 34.0451], [-118.5120, 34.0454],
            ...          [-118.5117, 34.0452], [-118.5121, 34.0449],
            ...          [-118.5123, 34.0451]],
            ...         features={'type': 'residential', 'floors': 2}
            ...     )
            ... )
            >>> _ = inventory.add_asset(
            ...     asset_id='warehouse_B',
            ...     asset=Asset(
            ...         'warehouse_B',
            ...         [[-118.5105, 34.0467], [-118.5102, 34.0471],
            ...          [-118.5099, 34.0468], [-118.5103, 34.0464],
            ...          [-118.5105, 34.0467]],
            ...         features={'type': 'commercial', 'floors': 1}
            ...     )
            ... )
            >>> inventory.print_info()
            AssetInventory
            Inventory stored in:  dict
            Key:  house_A Asset:
                     Coordinates:  [[-118.5123, 34.0451], [-118.512, 34.0454],
            [-118.5117, 34.0452], [-118.5121, 34.0449], [-118.5123, 34.0451]]
                     Features:  {'type': 'residential', 'floors': 2}
            Key:  warehouse_B Asset:
                     Coordinates:  [[-118.5105, 34.0467], [-118.5102, 34.0471],
            [-118.5099, 34.0468], [-118.5103, 34.0464], [-118.5105, 34.0467]]
                     Features:  {'type': 'commercial', 'floors': 1}
            >>> img_set = ImageSet()
            >>> img1 = Image('bldg1.jpg')
            >>> img2 = Image('bldg2.jpg')
            >>> _ = img_set.add_image('house_A', img1)
            >>> _ = img_set.add_image('house_B', img2)
            >>> inventory.remove_nonmatching_assets(img_set, verbose=True)
            Removed 1 nonmatching assets: ['warehouse_B']
        """
        # Collect keys from both sets
        footprint_keys = set(self.inventory.keys())
        imageset_keys = set(image_set.images.keys())

        # Identify keys missing in image set
        missing_keys = footprint_keys - imageset_keys

        # Remove unmatched assets (in-place modification)
        for key in missing_keys:
            self.remove_asset(key)

        # Optionally, print a summary of removed assets
        if verbose:
            print(
                f'Removed {len(missing_keys)} nonmatching assets: '
                f'{sorted(missing_keys)}'
            )

    def write_to_geojson(self, output_file: str = '') -> dict:
        """
        Write inventory to a GeoJSON file and return the GeoJSON dictionary.

        This method generates a GeoJSON representation of the asset inventory,
        writes it to the specified file path (if provided), and returns the
        GeoJSON object.

        Args:
            output_file(str, optional):
                Path to the output GeoJSON file. If empty, no file is written.

        Returns:
            dict:
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
            Wrote 1 asset to {../output.geojson}
        """
        geojson = self.get_geojson()

        # Ensure each feature has an 'id' field in the top-level feature object
        # This is the variation of GeoJSON that is used in R2D:
        for index, feature in enumerate(geojson['features']):
            feature_id = feature['properties'].pop('id', str(index))
            feature['id'] = str(feature_id)

        # If a file name is provided, write the created GeoJSON dictionary into
        # a GeoJSON file:
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(clean_floats(geojson), f, indent=2)
                num_elements = len(geojson['features'])
                element = 'asset' if num_elements == 1 else 'assets'
                print(
                    f'Wrote {num_elements} {element} to '
                    f'{os.path.abspath(output_file)}'
                )
        return geojson

    def read_from_csv(
        self,
        file_path: str,
        keep_existing: bool,
        str_type: str = 'building',
        id_column: str | None = None,
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
                  ``True`` if assets were added, ``False`` otherwise.
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

        if keep_existing:
            if len(self.inventory) == 0:
                print(
                    'No existing inventory found. Reading in the new '
                    'inventory from the file.'
                )
                id_counter = 1
            else:
                # we don't want a duplicate the id
                id_counter = max(self.inventory.keys()) + 1
        else:
            self.inventory = {}
            id_counter = 1

        # Attempt to open the file
        try:
            with open(file_path) as csvfile:
                csv_reader = csv.dictReader(csvfile)
                rows = list(csv_reader)
        except FileNotFoundError:
            raise Exception(f'The file {file_path} does not exist.')

        # Check if latitude/longitude exist
        lat = ['latitude', 'lat']
        lon = ['longitude', 'lon', 'long']
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
            if 'type' in bldg_features.keys():
                if bldg_features['type'] not in ['building', 'bridge']:
                    raise Exception(
                        f"The csv file {file_path} cannot have a column named 'type'"
                    )
            else:
                bldg_features['type'] = str_type

            # is the id provided by user?
            if id_column is None:
                # if not we assin the id
                id = id_counter
            else:
                if id_column not in bldg_features.keys():
                    raise Exception(
                        f"The key '{id_column}' not found in {file_path}"
                    )
                id = bldg_features[id_column]

            asset = Asset(id, coordinates, bldg_features)
            self.add_asset(id, asset)
            id_counter += 1

        return True

    def add_asset_features_from_csv(
        self, file_path: str, id_column: str | None
    ) -> bool:
        """
        Read inventory data from a CSV file and add it to the inventory.

        Args:
            file_path(str):
                  The path to the CSV file
            id_column(str):
                  The name of column that contains id values. If ``None``, new
                  indices will be assigned

        Returns:
            bool: ``True`` if assets were added, ``False`` otherwise.
        """
        try:  # Attempt to open the file
            with open(file_path) as csvfile:
                csv_reader = csv.dictReader(csvfile)
                rows = list(csv_reader)
        except FileNotFoundError:
            raise Exception(f'The file {csvfile} does not exist.')

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
                    f"The key '{id_column}' not found in {file_path}"
                )
            id = bldg_features[id_column]

            self.add_asset_features(id, bldg_features)

        return True

    def get_dataframe(self) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Convert the asset inventory into two structured DataFrames and count.

        This method processes the internal asset inventory and returns:

        - A ``DataFrame`` containing the features of each asset, with support
          for multiple possible worlds (realizations).
        - A ``DataFrame`` containing centroid coordinates (longitude and
          latitude) for spatial operations.
        - The total number of assets in the inventory.

        The method flattens feature lists into separate columns if multiple
        possible worlds exist. It also derives centroid coordinates from the
        geometry for each asset.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, int]:

                - Asset feature ``DataFrame`` (indexed by asset ID).
                - Asset geometry ``DataFrame`` with 'Lat' and 'Lon' columns.
                - Total number of assets (int).

        Raises:
            ValueError:
                If a feature list's length does not match the expected number
                of possible worlds.
        """
        n_possible_worlds = self.get_n_pw()

        # Lists to hold data for DataFrame construction
        ids = []
        lats = []
        lons = []
        feature_rows = []

        # Single pass through the inventory to collect all data
        for asset_id, asset in self.inventory.items():
            ids.append(asset_id)

            # Geometry: Reuse the robust logic in Asset.get_centroid()
            centroid_data = asset.get_centroid()[0]
            lons.append(centroid_data[0])
            lats.append(centroid_data[1])

            # Features: Collect for processing below
            features = asset.features.copy()
            features['index'] = asset_id
            feature_rows.append(features)

        nbldg = len(ids)

        # Process Features considering Possible Worlds
        if n_possible_worlds == 1:
            bldg_properties_df = pd.DataFrame(feature_rows)
        else:
            # Identify columns that are lists (vector columns) in ANY asset
            # These are the ones with multiple worlds
            vector_columns = set()
            for entry in feature_rows:
                vector_columns.update(
                    key for key, value in entry.items() if isinstance(value, list)
                )

            flat_data = []
            for entry in feature_rows:
                # Base row with scalar values
                row = {
                    key: value
                    for key, value in entry.items()
                    if key not in vector_columns
                }

                # Expand vector columns
                for col in vector_columns:
                    val = entry.get(col)

                    # Safety Check: Handle mixed types.
                    # If this asset has a scalar for a column that is a vector
                    # elsewhere, we replicate that scalar across all worlds.
                    if not isinstance(val, list):
                        for i in range(n_possible_worlds):
                            row[f'{col}_{i + 1}'] = val
                        continue

                    # Validate length
                    if len(val) != n_possible_worlds:
                        raise ValueError(
                            f"Asset {entry['index']}: The specified # of "
                            f"possible worlds is {n_possible_worlds} but "
                            f"'{col}' contains {len(val)} realizations."
                        )

                    # Flatten
                    for i in range(n_possible_worlds):
                        row[f'{col}_{i + 1}'] = val[i]

                flat_data.append(row)

            bldg_properties_df = pd.DataFrame(flat_data)

        # Cleanup Feature DataFrame
        if 'type' in bldg_properties_df.columns:
            bldg_properties_df = bldg_properties_df.drop(columns=['type'])

        bldg_properties_df = bldg_properties_df.set_index('index')

        # --- Construct Geometry DataFrame ---
        bldg_geometries_df = pd.DataFrame({
            'Lat': lats,
            'Lon': lons,
            'index': ids
        })
        bldg_geometries_df = bldg_geometries_df.set_index('index')

        return bldg_properties_df, bldg_geometries_df, nbldg

    def get_world_realization(self, id: int = 0) -> AssetInventory:
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
                'Cannot retrieve different realizations as the inventory '
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
        self, id: int, world_realization: AssetInventory
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

        if self.n_pw < id:
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

                # initialize
                if isinstance(original_value, list):
                    new_value = original_value
                else:
                    new_value = [original_value] * self.n_pw

                # update
                new_value[id] = val

                # if identical, shrink it
                if len(set(new_value)) == 1:
                    new_value = new_value[0]

                # overwrite existing ones.
                self.add_asset_features(i, {key: new_value}, overwrite=True)

    def get_n_pw(self) -> int:  # move to Asset
        """
        Get the number of possible worlds (realizations) in the inventory.

        Returns:
            int:
                The number of possible worlds stored in the inventory.
        """
        return self.n_pw

    def get_multi_keys(self) -> tuple[list[str], list[str]]:  # move to Asset
        """
        Identify features that contain multiple realizations across assets.

        Iterates through all assets and returns two lists:

        - Keys associated with multi-valued features(i.e., lists).
        - All unique feature keys present in the inventory.

        Returns:
            tuple[list[str], list[str]]:
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

    def _get_next_numeric_id(self) -> int:
        """
        Compute the next available numeric asset ID in the inventory.

        Returns:
            int:
                The next available numeric ID (max numeric key + 1).
                Returns 0 if the inventory contains no numeric keys.

        Notes:
            - Non-numeric keys (e.g., 'A101') are ignored.
            - This function is typically used to generate sequential
              numeric identifiers for new assets.
        """
        numeric_ids = [int(k) for k in self.inventory if str(k).isdigit()]
        return (max(numeric_ids) + 1) if numeric_ids else 0

    def set_housing_unit_inventory(
        self,
        hu_inventory: HousingUnitInventory,
        hu_assignment: dict[str | int, list] | None = None,
        *,
        validate: bool = True,
    ) -> None:
        """
        Set the housing unit inventory and optionally assign housing units to assets.

        This method links a HousingUnitInventory object to the AssetInventory.
        It can also be used to assign housing unit ID lists to individual assets
        using the `hu_assignment` dictionary.

        This enables two primary workflows:
        1.  Assigning new housing units: Pass both `hu_inventory` and
            `hu_assignment`. The function will link the inventory and add
            the 'HousingUnits' feature to each asset in the assignment dict.
        2.  Linking a loaded inventory: After loading an AssetInventory
            (e.g., from GeoJSON) that already has 'HousingUnits' features,
            pass only the `hu_inventory` object. The function will link it
            and can optionally validate the existing assignments.

        Args:
            hu_inventory (HousingUnitInventory):
                The `HousingUnitInventory` object to link to this
                `AssetInventory`.
            hu_assignment (dict[str | int, list], optional):
                A dictionary mapping `asset_id` to a list of housing unit IDs.
                If provided, this list will be added as the 'HousingUnits'
                feature for each corresponding asset, overwriting any
                existing 'HousingUnits' feature. Defaults to `None`.
            validate (bool, optional):
                If `True`, run `validate_housing_unit_assignments()` after
                linking to check for mismatches. Defaults to `True`.

        Raises:
            TypeError: If `hu_inventory` is not an instance of
                       `HousingUnitInventory` or if `hu_assignment` is
                       provided and is not a `dict`.
        """
        if not isinstance(hu_inventory, HousingUnitInventory):
            raise TypeError(
                f'hu_inventory must be an instance of HousingUnitInventory, '
                f'not {type(hu_inventory)}.'
            )

        self.housing_unit_inventory = hu_inventory

        if hu_assignment is not None:
            if not isinstance(hu_assignment, dict):
                raise TypeError(
                    f'hu_assignment must be a dictionary or None, '
                    f'not {type(hu_assignment)}.'
                )

            print(f'Assigning housing units to {len(hu_assignment)} assets...')
            assets_not_found = 0
            for asset_id, hu_list in hu_assignment.items():
                success = self.add_asset_features(
                    asset_id, {'HousingUnits': hu_list}, overwrite=True
                )
                if not success:
                    assets_not_found += 1

            if assets_not_found > 0:
                print(
                    f'Warning: {assets_not_found} asset(s) listed in '
                    f'hu_assignment were not found in the inventory.'
                )

        if validate:
            print('Validating housing unit assignments...')
            self.validate_housing_unit_assignments()
            print('Validation successful.')

    def validate_housing_unit_assignments(self) -> bool:
        """
        Check the integrity of housing unit assignments.

        This function performs two main checks:
        1.  If any assets have a 'HousingUnits' feature, it checks that
            `self.housing_unit_inventory` is not `None`.
        2.  If `self.housing_unit_inventory` is set, it checks that every
            housing unit ID in every asset's 'HousingUnits' list exists in the
            `self.housing_unit_inventory`.

        Returns:
            bool:
                `True` if all assignments are valid or if there are
                no assignments.

        Raises:
            TypeError:
                If an asset's 'HousingUnits' feature is not a `list`.
            LookupError:
                If assets have 'HousingUnits' assignments but
                `self.housing_unit_inventory` is `None`.
            ValueError:
                If any housing unit IDs assigned to assets are not found
                in the `self.housing_unit_inventory`.
        """
        all_asset_hu_ids = set()
        found_assignments = False
        assets_with_bad_data = []

        # First, collect all housing unit IDs from assets
        for asset_id, asset in self.inventory.items():
            hu_data = asset.features.get('HousingUnits')
            if hu_data is not None:
                found_assignments = True
                if not isinstance(hu_data, list):
                    assets_with_bad_data.append(asset_id)
                else:
                    all_asset_hu_ids.update(hu_data)

        # Check for non-list 'HousingUnits' features
        if assets_with_bad_data:
            raise TypeError(
                f"The 'HousingUnits' feature must be a list. Found invalid "
                f'data in assets: {assets_with_bad_data}'
            )

        # If no assets have assignments, we are in a valid state.
        if not found_assignments:
            return True

        # If we have assignments, we MUST have an inventory linked.
        if self.housing_unit_inventory is None:
            raise LookupError(
                'Assets have housing unit assignments, but no '
                'HousingUnitInventory is linked. Call '
                'set_housing_unit_inventory() first.'
            )

        # Check for orphan IDs
        valid_hu_ids = set(self.housing_unit_inventory.get_housing_unit_ids())
        missing_ids = all_asset_hu_ids - valid_hu_ids

        if missing_ids:
            sample = list(missing_ids)[:10]
            raise ValueError(
                f'Found {len(missing_ids)} orphan housing unit IDs assigned to '
                f'assets that are not in the HousingUnitInventory. '
                f'Example orphans: {sample}'
            )

        # All checks passed
        return True

    def remove_housing_unit_inventory(self, *, clear_assignments: bool = True) -> None:
        """
        Remove the linked housing unit inventory and clear all assignments.

        This method removes the `HousingUnitInventory` object from the
        `AssetInventory` by setting `self.housing_unit_inventory` to `None`.

        By default, it also removes the 'HousingUnits' feature from all
        assets in the inventory, as the assignments are specific to the
        inventory being removed. This ensures the `AssetInventory`
        remains in a valid state.

        Args:
            clear_assignments (bool, optional):
                If `True` (the default), this method will iterate through
                all assets and remove the 'HousingUnits' feature from them.
                If set to `False`, only the main `housing_unit_inventory`
                attribute will be removed, which will likely result
                in an invalid state that fails validation.
        """
        self.housing_unit_inventory = None
        print('HousingUnitInventory removed.')

        if clear_assignments:
            print("Removing 'HousingUnits' feature from all assets...")
            # Use the existing remove_features method
            self.remove_features(['HousingUnits'])
            print('Asset assignments cleared.')
        else:
            print(
                "Warning: Asset 'HousingUnits' features were not cleared. "
                'The inventory is likely in an invalid state.'
            )
