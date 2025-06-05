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
#
# Last updated:
# 06-02-2025

"""
This module defines abstract SpatialJoinMethods class.

.. autosummary::

    SpatialJoinMethods
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from brails.utils.input_validator import InputValidator
from brails.utils.inventory_validator import InventoryValidator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory


class SpatialJoinMethods(ABC):
    """Abstract base class for spatial join methods."""

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses with their class name."""
        super().__init_subclass__(**kwargs)
        SpatialJoinMethods._registry[cls.__name__] = cls

    @classmethod
    def join_inventories(cls,
                         inventory1: AssetInventory,
                         inventory2: AssetInventory) -> AssetInventory:
        """
        Perform a spatial join between two AssetInventory instances.

        This method validates the inputs and delegates the join logic to
        the concrete subclass implementation.

        Args:
            inventory1 (AssetInventory):
                The primary asset inventory.
            inventory2 (AssetInventory):
                The secondary asset inventory to join with.

        Returns:
            AssetInventory:
                A new AssetInventory instance resulting from the spatial join.

        Raises:
            TypeError: If either input is not an instance of AssetInventory.
        """
        # Input validation in the abstract class:
        if not (InventoryValidator.is_inventory(inventory1) and
                InventoryValidator.is_inventory(inventory2)):
            raise TypeError('Both inputs must be instances of AssetInventory.')

        # Call the concrete class implementation:
        instance = cls()
        return instance._join_implementation(inventory1, inventory2)

    @abstractmethod
    def _join_implementation(self,
                             inventory1,
                             inventory2):
        """Concrete classes must implement this join logic."""

    @staticmethod
    def execute(method_name: str,
                inventory1: AssetInventory,
                inventory2: AssetInventory) -> AssetInventory:
        """
        Run a spatial join using the given method name.

        Args:
            method_name (str):
                The name of the concrete join method class.
            inventory1 (AssetInventory):
                The primary asset inventory.
            inventory2 (AssetInventory):
                The secondary asset inventory to join with.

        Returns:
            AssetInventory:
                Result of the spatial join.

        Raises:
            ValueError:
                If the method is not found.
        """
        method_class = SpatialJoinMethods._registry.get(method_name)
        if method_class is None:
            raise ValueError(f"Join method '{method_name}' not found.")

        return method_class.join_inventories(inventory1, inventory2)

    def _merge_inventory_features(self,
                                  receiving_inventory: AssetInventory,
                                  merging_inventory: AssetInventory,
                                  matched_items: dict[int | str, int | str]
                                  ) -> AssetInventory:
        """
        Merge features from merging_inventory into receiving_inventory.

        Args:
            receiving_inventory (AssetInventory):
                Target inventory to receive features.
            merging_inventory (AssetInventory):
                Source inventory to extract features from.
            matched_items (dict[int | str, int | str]):
                Mapping from receiving ID to merging ID.

        Returns:
            AssetInventory:
                Updated receiving_inventory with merged features.
        """
        inventory_data = merging_inventory.inventory

        for receiving_id, merging_id in matched_items.items():

            # Extract the features from the merging item:
            features = inventory_data[merging_id].features

            # Write the extracted features to the receiving inventory:
            receiving_inventory.add_asset_features(receiving_id, features)

        return receiving_inventory

    def _get_point_indices(self,
                           inventory: AssetInventory) -> list[str | int]:
        """
        Retrieve the list of keys for assets with point geometry.

        Args:
            inventory (AssetInventory):
                The asset inventory to process.

        Returns:
            list[str | int]:
                A list of keys of assets with point geometry.

        Raises:
            TypeError:
                If the provided inventory is not an instance of AssetInventory.
        """
        InventoryValidator.validate_inventory(inventory)

        return [key for key, asset in inventory.inventory.items()
                if InputValidator.is_point(asset.coordinates)]

    def _get_linestring_indices(self,
                                inventory: AssetInventory) -> list[str | int]:
        """
        Retrieve the list of keys for assets with linestring geometry.

        Args:
            inventory (AssetInventory):
                The asset inventory to process.

        Returns:
            list[str | int]:
                A list of keys of assets with linestring geometry.

        Raises:
            TypeError:
                If the provided inventory is not an instance of AssetInventory.
        """
        InventoryValidator.validate_inventory(inventory)

        return [key for key, asset in inventory.inventory.items()
                if InputValidator.is_linestring(asset.coordinates)]

    def _get_polygon_indices(self,
                             inventory: AssetInventory) -> list[str | int]:
        """
        Retrieve the list of keys for assets with polygon geometry.

        Args:
            inventory (AssetInventory):
                The asset inventory to process.

        Returns:
            list[str | int]:
                A list of keys of assets with polygon geometry.

        Raises:
            TypeError:
                If the provided inventory is not an instance of AssetInventory.
        """
        InventoryValidator.validate_inventory(inventory)

        return [key for key, asset in inventory.inventory.items()
                if InputValidator.is_polygon(asset.coordinates)]
