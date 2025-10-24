# -*- coding: utf-8 -*-
#
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
# Sang-ri Yi
#
# Last updated:
# 11-12-2024

import os
import sys
import copy

from brails.types.asset_inventory import AssetInventory
from brails.inferers.inference_engine import InferenceEngine

import importlib.util
import logging

# Configure logging:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserInferer(InferenceEngine):
    """
    Executes the inference model provided by the user

    Attributes:
        n_pw (int):
                The number of possible worlds (i.e. samples or realizations)
        seed (int):
                For reproducibility

    Methods:


    """

    def __init__(
        self,
        input_inventory: AssetInventory,
        user_path,
        overwrite=True,
    ):
        self.input_inventory = input_inventory
        self.overwrite = overwrite
        self.user_path = user_path

    def infer(self) -> AssetInventory:
        #
        # read the user-defined function named "user_inferer"
        #

        input_inventory = self.input_inventory
        module_name = os.path.splitext(os.path.basename(self.user_path))[0]
        spec = importlib.util.spec_from_file_location(
            module_name, self.user_path)
        user_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = user_module
        spec.loader.exec_module(user_module)
        if not hasattr(user_module, "user_inferer"):
            msg = f"Function 'user_inferer' should exist in {self.user_path}."
            raise Exception(msg)

        prop = {}
        n_pw = input_inventory.get_n_pw()
        for nw in range(n_pw):
            this_inventory = input_inventory.get_world_realization(nw)

            #
            # convert to json
            #

            inventory_json = self.to_json(this_inventory)

            #
            # call user defined module and merge in to a json file
            #

            prop_world_i = user_module.user_inferer(inventory_json)

            if nw == 0:
                prop = prop_world_i
            else:
                prop = self.merge_two_json(
                    prop, prop_world_i, shrink=(nw == n_pw - 1))

        #
        # Update the inventory
        #

        output_inventory = copy.deepcopy(input_inventory)
        count = 0
        for index, feature in prop.items():
            updated = output_inventory.add_asset_features(
                index, feature, overwrite=self.overwrite
            )
            count += updated

        if len(prop) == 0:
            logger.warning("Nothing happened to the inventory.")
        elif count == 0:
            logger.warning(
                "Nothing happened to the inventory. Did you want to turn on the overwriting?"
            )
        elif count < len(prop):
            print(f"{count/len(prop)*100} % of assets are updated")
        else:
            print("All assets are updated")

        return output_inventory

    def merge_two_json(self, A, B, shrink=False):
        if A == {}:
            return B

        if B == {}:
            return A

        C = {}

        for key in A:
            # Initialize the merged values for the current key
            merged_values = {}

            # Loop through each feature in A and B
            for feature in A[key]:
                # Get the value from A, ensure it's a list
                value_a = A[key][feature]
                if not isinstance(value_a, list):
                    value_a = [value_a]

                # Get the value from B, ensure it's a list
                value_b = B[key].get(feature, [])
                if not isinstance(value_b, list):
                    value_b = [value_b]

                # Merge the two lists
                if not shrink:
                    merged_values[feature] = value_a + value_b
                else:
                    val = value_a + value_b
                    if isinstance(val, list) and (len(set(val)) == 1):
                        merged_values[feature] = val[0]
                    else:
                        merged_values[feature] = val

            # Assign the merged values for the current key to C
            C[key] = merged_values

        return C

    def to_json(self, this_inventory):
        inventory_json = {}
        for key, asset in this_inventory.inventory.items():
            if len(asset.coordinates) == 1:
                geometry = {"type": "Point", "coordinates": [
                    asset.coordinates[0][:]]}
            else:
                geometry = {"type": "Polygon",
                            "coordinates": asset.coordinates}

            feature = {
                "type": "Feature",
                "properties": asset.features,
                "geometry": geometry,
            }
            if "type" in asset.features:
                feature["type"] = asset.features["type"]

            inventory_json[key] = feature

        return inventory_json
