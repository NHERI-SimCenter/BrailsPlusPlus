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
        household_id (str):
            Unique identifier for the household.
        features (dict[str, Any]):
            A dictionary of features (attributes) for the household.
    """

    def __init__(
        self,
        household_id: str,
        features: Dict[str, Any] = None,
    ) -> None:
        """
        Initialize a Household with a household ID and features.

        Args:
            household_id (str): The unique identifier for the household.
            features (dict[str, Any], optional): A dictionary of features.
                Defaults to an empty dict.
        """
        self.household_id = household_id
        self.features = features if features is not None else {}

    def add_features(
            self,
            additional_features: Dict[str, Any],
            overwrite: bool = True
    ) -> Tuple[bool, int]:
        """
        Update the existing features in the household.

        Args:
            additional_features (dict[str, any]): New features to merge into
                the household's features.
            overwrite (bool, optional): Whether to overwrite existing features.
                Defaults to True.

        Returns:
            tuple[bool, int]: A tuple containing:
                - bool: True if features were updated, False otherwise.
                - int: Number of possible worlds (for compatibility).
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
        Remove specified features from the household.

        Args:
            feature_list (List[str]): List of features to be removed

        Return:
            bool: True if features are removed
        """
        for key in list(feature_list):
            self.features.pop(key, None)

        return True

    def print_info(self) -> None:
        """Print the features of the household."""
        print(f"\t Household ID: {self.household_id}")
        features_json = json.dumps(self.features, indent=2)
        print(f"\t Features: {features_json}")


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
        self.n_pw = 1

    def add_household(self, household_id: str, household: Household) -> bool:
        """
        Add a Household to the inventory.

        Args:
            household_id (str):
                The unique identifier for the household.
            household (Household):
                The household to be added.

        Returns:
            bool: ``True`` if the household was added successfully, ``False``
            otherwise.

        Raises:
            TypeError: If ``household`` is not an instance of :class:`Household`.

        Examples:
            >>> household = Household(
            ...     household_id="1",
            ...     features={'income': 50000, 'size': 3}
            ... )
            >>> inventory = HouseholdInventory()
            >>> success = inventory.add_household("1", household)
            >>> print(success)
            True

            >>> # Adding the same household_id again will fail
            >>> success = inventory.add_household("1", household)
            >>> print(success)
            False
        """
        if not isinstance(household, Household):
            raise TypeError("Expected an instance of Household.")

        if household_id in self.inventory:
            print(f'Household with id {household_id} already exists. '
                  f'Household was not added')
            return False

        self.inventory[household_id] = household
        return True

    def add_household_features(
        self,
        household_id: str,
        new_features: Dict[str, Any],
        overwrite: bool = True
    ) -> bool:
        """
        Add new household features to the Household with the specified ID.

        Args:
            household_id (str):
                The unique identifier for the household.
            new_features (dict):
                A dictionary of features to add to the household.
            overwrite (bool):
                Whether to overwrite existing features with the
                same keys. Defaults to True.

        Returns:
            bool: True if features were successfully added, False if the 
                household does not exist or the operation fails.
        """
        household = self.inventory.get(household_id, None)
        if household is None:
            print(f'No existing Household with id {household_id} found. '
                  f'Household features not added.')
            return False

        status, n_pw = household.add_features(new_features, overwrite)
        if n_pw == 1:
            pass
        elif (n_pw == self.n_pw) or (self.n_pw == 1):
            self.n_pw = n_pw
        else:
            print(f'WARNING: # possible worlds was {self.n_pw} but is now '
                  f'{n_pw}. Something went wrong.')
            self.n_pw = n_pw
        return status

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
                    # Move the feature to the new name and remove the old one:
                    household.features[new_name] = household.features.pop(old_name)

    def get_household_features(
            self,
            household_id: str
    ) -> tuple[bool, dict[str, Any]]:
        """
        Get features of a particular household.

        Args:
            household_id (str):
                The unique identifier for the household.

        Returns:
            tuple[bool, dict]:
                A tuple where the first element is a boolean
                indicating whether the household was found, and the second 
                element is a dictionary containing the household's features if 
                the household is present. Returns an empty dictionary if the 
                household is not found.
        """
        household = self.inventory.get(household_id, None)
        if household is None:
            return False, {}

        return True, household.features

    def get_household_ids(self) -> list[str]:
        """
        Retrieve the IDs of all households in the inventory.

        Returns:
            list[str]:
                A list of household IDs.
        """
        return list(self.inventory.keys())


    def print_info(self) -> None:
        """
        Print summary information about the HouseholdInventory.

        This method outputs the name of the class, the type of data structure
        used to store the inventory, and basic information about each household
        in the inventory, including its key and features.

        Returns:
            None
        """
        print(self.__class__.__name__)
        print("Inventory stored in: ", self.inventory.__class__.__name__)
        for key, household in self.inventory.items():
            print("Key: ", key, "Household:")
            household.print_info()

    def remove_household(self, household_id: str) -> bool:
        """
        Remove a Household from the inventory.

        Args:
            household_id (str):
                The unique identifier for the household.

        Returns:
            bool:
                True if household was removed, False otherwise.
        """
        if household_id in self.inventory:
            del self.inventory[household_id]
            return True
        return False

    def remove_features(self, feature_list: List[str]) -> bool:
        """
        Remove specified features from all households in the inventory.

        Args:
            feature_list (list[str]):
                List of feature names to remove from all households.

        Returns:
            bool:
                True if features were removed, False otherwise.
        """
        for _, household in self.inventory.items():
            household.remove_features(feature_list)

        return True

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
        json_data: Union[str, Dict[str, Any]],
        keep_existing: bool
    ) -> bool:
        """
        Read inventory data from a JSON file, string, or dictionary and add it to the inventory.

        Args:
            json_data (Union[str, Dict[str, Any]]):
                  Either a path to a JSON file, a JSON string, or a dictionary object
            keep_existing (bool):
                  If False, the inventory will be initialized

        Returns:
            bool:
                  True if households were added, False otherwise.
        """
        if not keep_existing:
            self.inventory = {}

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
                        raise Exception(f"The input is neither a valid file path nor a valid JSON string.")
            except Exception as e:
                raise Exception(f"Error processing input: {str(e)}")

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
            raise Exception(f"Schema file not found at {schema_path}")
        except ValidationError as e:
            raise ValueError(f"Invalid JSON data: {e.message}")

        # Extract households data after validation
        households_data = data.get("households", {})

        # Load data after successful validation
        for household_id, features in households_data.items():
            # Create and add household
            household = Household(household_id, features)
            self.add_household(household_id, household)

        return True
