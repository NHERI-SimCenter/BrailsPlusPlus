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
This module defines classes associated with household inventories.

.. autosummary::

    HouseholdInventory
    Household
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path

try:
    # Python 3.8+
    from importlib.metadata import version
except ImportError:
    # For Python <3.8, use the backport
    from importlib_metadata import version

import jsonschema
from jsonschema import validate, ValidationError

# TODO: refactor clean_floats
# This function is a copy of the function `clean_floats` in
# `asset_inventory.py`. It would be better to have it in a separate file in
# utils and import it from there.

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


class Household:
    """A household with features and attributes.

    To import the :class:`Household` class, use:

    .. code-block:: python

        from brails.types.household_inventory import Household


    Attributes:
        features (dict[str, Any]):
            A dictionary of features (attributes) for the household.
    """

    def __init__(
        self,
        features: Dict[str, Any] = None,
    ) -> None:
        """
        Initialize a Household with a household ID and features.

        Args:
            features (dict[str, Any], optional): A dictionary of features.
                Defaults to an empty dict.
        """
        self.features = features if features is not None else {}

    def add_features(
            self,
            additional_features: Dict[str, Any],
            overwrite: bool = True
    ) -> bool:
        """
        Update the existing features in the household.

        Args:
            additional_features (dict[str, any]): New features to merge into
                the household's features.
            overwrite (bool, optional): Whether to overwrite existing features.
                Defaults to True.

        Returns:
            bool: True if features were updated, False otherwise.
        """

        if overwrite:
            self.features.update(additional_features)
            return True

        updated = False
        for key, value in additional_features.items():
            if key not in self.features:
                self.features[key] = value
                updated = True

        return updated

    def remove_features(self, feature_list: List[str]):
        """
        Remove specified features from the household.

        Args:
            feature_list (List[str]): List of features to be removed
        """

        for key in feature_list:
            if key in self.features:
                del self.features[key]

    def print_info(self) -> None:
        """Print the features of the household."""
        print(f"\t Features: {json.dumps(self.features, indent=2)}")


class HouseholdInventory:
    """
    A class representing a collection of Households managed as an inventory.

    This class provides methods to add, manipulate, write and query
    a collection of :class:`Household` objects.

    To import the :class:`HouseholdInventory` class, use:

    .. code-block:: python

        from brails.types.household_inventory import HouseholdInventory
    """

    def __init__(self) -> None:
        """Initialize HouseholdInventory with an empty inventory dictionary."""
        self.inventory: Dict = {}

    def add_household(self, household_id: str, household: Household,
                      overwrite: bool = False):
        """
        Add a Household to the inventory.

        Args:
            household_id (str):
                The unique identifier for the household.
            household (Household):
                The household to be added.
            overwrite (bool):
                Replace existing household if it exists. Defaults to False.

        Raises:
            TypeError: If ``household`` is not an instance of :class:`Household`.

        Examples:
            >>> household = Household(
            ...     features={'income': 50000, 'size': 3}
            ... )
            >>> inventory = HouseholdInventory()
            >>> inventory.add_household("1", household)
        """
        if not isinstance(household, Household):
            msg = "Expected an instance of Household."
            raise TypeError(msg)

        if ~overwrite and household_id in self.inventory:
            print(f'The inventory already has a household with id {household_id}.')
            return

        self.inventory[household_id] = household


    def change_feature_names(
            self,
            feature_name_mapping: Dict[str, str]
    ) -> None:
        """
        Rename features in a HouseholdInventory using user-specified mapping.

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
                "Expected 'feature_name_mapping' to be a dictionary."
            )

        # Validate that all keys and values in the mapping are strings:
        for old_name, new_name in feature_name_mapping.items():
            if not isinstance(old_name, str) or not isinstance(new_name, str):
                raise TypeError(
                    "All keys and values in 'feature_name_mapping' must be "
                    f"strings. Invalid pair: ({old_name}, {new_name})"
                )

        # Apply the feature name changes to each household in the inventory:
        for household in self.inventory.values():
            for old_name, new_name in feature_name_mapping.items():
                if old_name in household.features:
                    if new_name in household.features:
                        raise NameError(f"New feature name {new_name} already exists.")

                    # Move the feature to the new name and remove the old one:
                    household.features[new_name] = household.features.pop(old_name)


    def get_household_ids(self) -> list[str]:
        """
        Retrieve the IDs of all households in the inventory.

        Returns:
            list[str]:
                A list of household IDs.
        """
        return list(self.inventory.keys())


    def print_info(self):
        """
        Print summary information about the HouseholdInventory.

        This method outputs the name of the class, the type of data structure
        used to store the inventory, and basic information about each household
        in the inventory, including its key and features.
        """
        print(self.__class__.__name__)
        print("Inventory stored in: ", self.inventory.__class__.__name__)
        for key, household in self.inventory.items():
            print("Key: ", key, "Household:")
            household.print_info()


    def remove_household(self, household_id: str):
        """
        Remove a Household from the inventory.

        Args:
            household_id (str):
                The unique identifier for the household.

        """
        if household_id in self.inventory:
            del self.inventory[household_id]


    def remove_features(self, feature_list: List[str]):
        """
        Remove specified features from all households in the inventory.

        Args:
            feature_list (list[str]):
                List of feature names to remove from all households.

        """
        for _, household in self.inventory.items():
            household.remove_features(feature_list)


    def to_json(self, output_file: str = "") -> dict[str, Any]:
        """
        Generate JSON representation and optionally write to file.

        This method generates a JSON representation of the household inventory,
        writes it to the specified file path (if provided), and returns the
        JSON object.

        Args:
            output_file (str, optional):
                Path to the output JSON file. If empty, no file is written.

        Returns:
            dict[str, Any]:
                A dictionary containing the JSON representation of the 
                inventory.
        """
        try:
            brails_version = version("BRAILS")
        except Exception:
            brails_version = "NA"

        json_data = {
            "type": "HouseholdInventory",
            "generated": str(datetime.now()),
            "brails_version": brails_version,
            "households": {}
        }

        for household_id, household in self.inventory.items():
            json_data["households"][str(household_id)] = household.features

        # Write the created JSON dictionary into a JSON file:
        if output_file:
            with open(output_file, "w", encoding="utf-8") as file_out:
                json.dump(clean_floats(json_data), file_out, indent=2)

        return json_data

    def read_from_json(
        self,
        json_data: Union[str, Dict[str, Any]]
    ):
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
                # First try to treat it as a file path
                if os.path.exists(json_data):
                    with open(json_data, mode="r", encoding="utf-8") as jsonfile:
                        data = json.load(jsonfile)
                else:
                    # If not a file path, treat it as a JSON string
                    try:
                        data = json.loads(json_data)
                    except json.JSONDecodeError:
                        msg = f"The input is neither a valid file path nor a valid JSON string."
                        raise Exception(msg)
            except Exception as e:
                msg = f"Error processing input: {str(e)}"
                raise Exception(msg)

        # Load and validate against JSON schema
        schema_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "household_inventory_schema.json"
        )
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as schema_file:
                schema = json.load(schema_file)
            
            # Validate against schema
            validate(instance=data, schema=schema)
            
        except FileNotFoundError:
            msg = f"Schema file not found at {schema_path}"
            raise Exception(msg)
        except ValidationError as e:
            msg = f"Invalid JSON data: {e.message}"
            raise ValueError(msg)

        # Extract households data after validation
        households_data = data.get("households", {})

        # Load data after successful validation
        for household_id, features in households_data.items():

            if isinstance(household_id, str) and household_id.isdigit():
                household_id = int(household_id)

            # Create and add each household
            self.add_household(household_id, Household(features))


    def _get_next_numeric_id(self) -> int:
        """
        Compute the next available numeric household ID in the inventory.

        Returns:
            int:
                The next available numeric ID (max numeric key + 1).
                Returns 0 if the inventory contains no numeric keys.

        Notes:
            - Non-numeric keys (e.g., 'HH-A101') are ignored.
            - This function is typically used to generate sequential
              numeric identifiers for new households.
        """
        numeric_ids = [int(k) for k in self.inventory.keys() if str(k).isdigit()]
        return (max(numeric_ids) + 1) if numeric_ids else 0


    def merge_inventory(
        self,
        other_inventory: 'HouseholdInventory'
    ) -> Dict[Union[str, int], Union[str, int]]:
        """
        Merge another household inventory into this one, resolving ID
        conflicts.

        This method iterates through the `other_inventory`. It attempts
        to add each household using its original ID. If that ID already
        exists in the current inventory, it generates a new unique
        numeric ID for that household.

        Args:
            other_inventory (HouseholdInventory):
                The inventory to merge into this one.

        Returns:
            Dict[Union[str, int], Union[str, int]]:
                A dictionary mapping {old_household_id: new_household_id}.
                This map tracks all ID changes.

        Raises:
            TypeError:
                If `other_inventory` is not an instance of
                ``HouseholdInventory``.
        """
        if not isinstance(other_inventory, HouseholdInventory):
            raise TypeError(
                "Can only merge with another HouseholdInventory instance.")

        # Get the next safe starting ID for use *if* we find conflicts
        next_id = self._get_next_numeric_id()
        hh_id_remap = {}

        for old_id, household in other_inventory.inventory.items():
            new_id = old_id  # Try to use the original ID first

            if new_id in self.inventory:
                new_id = next_id
                next_id += 1

            # Since we've guaranteed new_id is unique, we can set
            # overwrite=False.
            self.add_household(new_id, household, overwrite=False)

            # Store the mapping, even if the ID didn't change
            hh_id_remap[old_id] = new_id

        return hh_id_remap
