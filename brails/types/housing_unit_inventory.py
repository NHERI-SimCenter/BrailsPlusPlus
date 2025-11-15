# Copyright (c) 2025 The Regents of the University of California
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
# Adam Zsarnoczay

"""
This module defines classes associated with housing unit inventories.

.. autosummary::

    HousingUnitInventory
    HousingUnit
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # For Python <3.8, use the backport
    from importlib_metadata import PackageNotFoundError, version

import jsonschema
from jsonschema import ValidationError, validate

from brails.utils.clean_floats import clean_floats


class HousingUnit:
    """A housing unit with features and attributes.

    To import the :class:`HousingUnit` class, use:

    .. code-block:: python

        from brails.types.housing_unit_inventory import HousingUnit


    Attributes:
        features (dict[str, Any]):
            A dictionary of features (attributes) for the housing unit.
    """

    def __init__(
        self,
        features: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a HousingUnit with features.

        Args:
            features (dict[str, Any], optional): A dictionary of features.
                Defaults to an empty dict.

        Raises:
            TypeError: If ``features`` is provided and is not a dictionary.
        """
        if features is not None and not isinstance(features, dict):
            msg = 'features must be a dict when provided.'
            raise TypeError(msg)
        self.features = features if features is not None else {}

    def add_features(
        self, additional_features: Dict[str, Any], *, overwrite: bool = True
    ) -> bool:
        """
        Update the existing features in the housing unit.

        Args:
            additional_features (dict[str, any]): New features to merge into
                the housing unit's features.
            overwrite (bool, optional): Whether to overwrite existing features.
                Defaults to True.

        Returns:
            bool: True if features were updated, False otherwise.

        Raises:
            TypeError: If ``additional_features`` is not a dictionary.
        """
        if not isinstance(additional_features, dict):
            msg = 'additional_features must be a dict.'
            raise TypeError(msg)

        if overwrite:
            self.features.update(additional_features)
            return True

        updated = False
        for key, value in additional_features.items():
            if key not in self.features:
                self.features[key] = value
                updated = True

        return updated

    def remove_features(self, feature_list: List[str]) -> None:
        """
        Remove specified features from the housing unit.

        Args:
            feature_list (List[str]): List of features to be removed

        Raises:
            TypeError: If ``feature_list`` is not a list of strings.
        """
        if not isinstance(feature_list, list) or not all(
            isinstance(item, str) for item in feature_list
        ):
            msg = 'feature_list must be a list of strings.'
            raise TypeError(msg)

        for key in feature_list:
            if key in self.features:
                del self.features[key]

    def print_info(self) -> None:
        """Print the features of the housing unit."""
        print(f'\t Features: {json.dumps(self.features, indent=2)}')


class HousingUnitInventory:
    """
    A class representing a collection of housing units managed as an inventory.

    This class provides methods to add, manipulate, write and query
    a collection of :class:`HousingUnit` objects.

    To import the :class:`HousingUnitInventory` class, use:

    .. code-block:: python

        from brails.types.housing_unit_inventory import HousingUnitInventory
    """

    def __init__(self) -> None:
        """Initialize HousingUnitInventory with an empty inventory dictionary."""
        self.inventory: Dict = {}

    def add_housing_unit(
        self,
        housing_unit_id: str,
        housing_unit: HousingUnit,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Add a housing unit to the inventory.

        Args:
            housing_unit_id (str):
                The unique identifier for the housing unit.
            housing_unit (HousingUnit):
                The housing unit to be added.
            overwrite (bool):
                Replace existing housing unit if it exists. Defaults to False.

        Raises:
            TypeError: If ``housing_unit`` is not an instance of :class:`HousingUnit`.

        Examples:
            >>> housing_unit = HousingUnit(
            ...     features={'income': 50000, 'size': 3}
            ... )
            >>> inventory = HousingUnitInventory()
            >>> inventory.add_housing_unit("1", housing_unit)
        """
        if not isinstance(housing_unit, HousingUnit):
            msg = 'Expected an instance of HousingUnit.'
            raise TypeError(msg)

        if housing_unit_id in self.inventory and not overwrite:
            print(
                f'The inventory already has a housing unit with id {housing_unit_id}.'
            )
            return

        self.inventory[housing_unit_id] = housing_unit

    def change_feature_names(self, feature_name_mapping: Dict[str, str]) -> None:
        """
        Rename features in a HousingUnitInventory using user-specified mapping.

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
            raise TypeError("Expected 'feature_name_mapping' to be a dictionary.")

        # Validate that all keys and values in the mapping are strings:
        for old_name, new_name in feature_name_mapping.items():
            if not isinstance(old_name, str) or not isinstance(new_name, str):
                raise TypeError(
                    "All keys and values in 'feature_name_mapping' must be "
                    f'strings. Invalid pair: ({old_name}, {new_name})'
                )

        # Apply the feature name changes to each housing unit in the inventory:
        for housing_unit in self.inventory.values():
            for old_name, new_name in feature_name_mapping.items():
                if old_name in housing_unit.features:
                    if new_name in housing_unit.features:
                        raise NameError(
                            f'New feature name {new_name} already exists.'
                        )

                    # Move the feature to the new name and remove the old one:
                    housing_unit.features[new_name] = housing_unit.features.pop(
                        old_name
                    )

    def get_housing_unit_ids(self) -> list[str]:
        """
        Retrieve the IDs of all housing units in the inventory.

        Returns:
            list[str]:
                A list of housing unit IDs.
        """
        return list(self.inventory.keys())

    def print_info(self) -> None:
        """
        Print summary information about the HousingUnitInventory.

        This method outputs the name of the class, the type of data structure
        used to store the inventory, and basic information about each housing unit
        in the inventory, including its key and features.
        """
        print(self.__class__.__name__)
        print('Inventory stored in: ', self.inventory.__class__.__name__)
        for key, housing_unit in self.inventory.items():
            print('Key: ', key, 'Housing unit:')
            housing_unit.print_info()

    def remove_housing_unit(self, housing_unit_id: str) -> None:
        """
        Remove a housing unit from the inventory.

        Args:
            housing_unit_id (str):
                The unique identifier for the housing unit.

        """
        if housing_unit_id in self.inventory:
            del self.inventory[housing_unit_id]

    def remove_features(self, feature_list: List[str]) -> None:
        """
        Remove specified features from all housing units in the inventory.

        Args:
            feature_list (list[str]):
                List of feature names to remove from all housing units.

        Raises:
            TypeError: If ``feature_list`` is not a list of strings.
        """
        # Validate input type: must be a list of strings
        if not isinstance(feature_list, list) or not all(
            isinstance(item, str) for item in feature_list
        ):
            msg = 'feature_list must be a list of strings.'
            raise TypeError(msg)

        for housing_unit in self.inventory.values():
            housing_unit.remove_features(feature_list)

    def to_json(self, output_file: str = '') -> dict[str, Any]:
        """
        Generate JSON representation and optionally write to file.

        This method generates a JSON representation of the housing unit
        inventory, writes it to the specified file path (if provided), and
        returns the JSON object.

        Args:
            output_file (str, optional):
                Path to the output JSON file. If empty, no file is written.

        Returns:
            dict[str, Any]:
                A dictionary containing the JSON representation of the
                inventory.
        """
        try:
            brails_version = version('BRAILS')
        except PackageNotFoundError:
            brails_version = 'NA'

        json_data = {
            'type': 'HousingUnitInventory',
            'generated': str(datetime.now(timezone.utc)),
            'brails_version': brails_version,
            'housing_units': {},
        }

        for housing_unit_id, housing_unit in self.inventory.items():
            json_data['housing_units'][str(housing_unit_id)] = housing_unit.features

        # Write the created JSON dictionary into a JSON file:
        if output_file:
            with Path(output_file).open('w', encoding='utf-8') as file_out:
                json.dump(clean_floats(json_data), file_out, indent=2)

        return json_data

    def read_from_json(self, json_data: Union[str, Dict[str, Any]]) -> None:
        """
        Read inventory data from a JSON file, string, or dictionary and add it to the inventory.

        Args:
            json_data (Union[str, Dict[str, Any]]):
                  Either a path to a JSON file, a JSON string, or a dictionary object
        """
        # Determine input type (dict, file path, or JSON string)
        data = None
        if isinstance(json_data, dict):
            # If json_data is already a dictionary, use it directly
            data = json_data
        else:
            try:
                # First, try to treat it as a file path
                json_data_file = Path(json_data)
                if json_data_file.exists():
                    with json_data_file.open(encoding='utf-8') as jsonfile:
                        data = json.load(jsonfile)
                else:
                    # If not a file path, treat it as a JSON string
                    try:
                        data = json.loads(json_data)
                    except json.JSONDecodeError as e:
                        msg = 'The input is neither a valid file path nor a valid JSON string.'
                        raise json.JSONDecodeError(msg) from e
            except Exception as e:
                msg = f'Error processing input: {e!s}'
                raise ValueError(msg) from e

        # Load and validate against JSON schema
        schema_path = (
            Path(__file__).resolve().parent / 'housing_unit_inventory_schema.json'
        )

        try:
            with schema_path.open(encoding='utf-8') as schema_file:
                schema = json.load(schema_file)

            # Validate against schema
            validate(instance=data, schema=schema)

        except FileNotFoundError as e:
            msg = f'Schema file not found at {schema_path}'
            raise FileNotFoundError(msg) from e

        except ValidationError as e:
            msg = f'Invalid JSON data: {e.message}'
            raise ValueError(msg) from e

        # Extract housing units data after validation
        housing_unit_data = data.get('housing_units')

        # Load data after successful validation
        for housing_unit_id_raw, features in housing_unit_data.items():
            housing_unit_id = (
                int(housing_unit_id_raw)
                if isinstance(housing_unit_id_raw, str)
                and housing_unit_id_raw.isdigit()
                else housing_unit_id_raw
            )

            # Create and add each housing unit
            self.add_housing_unit(housing_unit_id, HousingUnit(features))

    def _get_next_numeric_id(self) -> int:
        """
        Compute the next available numeric housing unit ID in the inventory.

        Returns:
            int:
                The next available numeric ID (max numeric key + 1).
                Returns 0 if the inventory contains no numeric keys.

        Notes:
            - Non-numeric keys (e.g., 'HU-A101') are ignored.
            - This function is typically used to generate sequential
              numeric identifiers for new housing units.
        """
        numeric_ids = [int(k) for k in self.inventory if str(k).isdigit()]
        return (max(numeric_ids) + 1) if numeric_ids else 0

    def merge_inventory(
        self, other_inventory: HousingUnitInventory
    ) -> Dict[Union[str, int], Union[str, int]]:
        """
        Merge another housing unit inventory into this one, resolving ID conflicts.

        This method iterates through the `other_inventory`. It attempts
        to add each housing unit using its original ID. If that ID already
        exists in the current inventory, it generates a new unique
        numeric ID for that housing unit.

        Args:
            other_inventory (HousingUnitInventory):
                The inventory to merge into this one.

        Returns:
            Dict[Union[str, int], Union[str, int]]:
                A dictionary mapping {old_housing_unit_id: new_housing_unit_id}.
                This map tracks all ID changes.

        Raises:
            TypeError:
                If `other_inventory` is not an instance of
                ``HousingUnitInventory``.
        """
        if not isinstance(other_inventory, HousingUnitInventory):
            raise TypeError(
                'Can only merge with another HousingUnitInventory instance.'
            )

        # Get the next safe starting ID for use *if* we find conflicts
        next_id = self._get_next_numeric_id()
        hu_id_remap = {}

        for old_id, housing_unit in other_inventory.inventory.items():
            new_id = old_id  # Try to use the original ID first

            if new_id in self.inventory:
                new_id = next_id
                next_id += 1

            # Since we've guaranteed new_id is unique, we can set
            # overwrite=False.
            self.add_housing_unit(new_id, housing_unit, overwrite=False)

            # Store the mapping, even if the ID didn't change
            hu_id_remap[old_id] = new_id

        return hu_id_remap
