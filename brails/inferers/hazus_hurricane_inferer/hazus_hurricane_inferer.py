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

import time
import copy
import sys

import numpy as np

from brails.types.asset_inventory import AssetInventory
from brails.inferers.inferenceEngine import InferenceEngine
from brails.inferers.hazus_hurricane_inferer.auto_HU_NJ import auto_populate


# To be replaced with old brails ++ codes
import warnings
import logging

# Configure logging:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

class HazusHurricaneInferer(InferenceEngine):
    """
    Make inference based on Hazus 6 rulesets

    Attributes:

    Methods:


    """

    def __init__(
        self,
        input_inventory: AssetInventory,
        n_possible_worlds=1,
        include_features=["ReplacementCost", "StructureType"],
        seed=1,
        overwirte_existing=True,
        clean_features=False,

        planArea_key = "PlanArea",
        numberOfStories_key = "NumberOfStories",
        occupancyClass_key = "OccupancyClass",
        buildingMaterial_key = "BuildingMaterial",
        designLevel_H_key = "DesignLevel_H",
        yearBuilt_key = "YearBuilt",
        roofSystem_key = "RoofFrameType",
        roofShape_key = "RoofShape",
        roofSlope_key = "RoofSlope",
        avgJanTemp_key = "AvgJanTemp",
        meanRoofHt_key = "MeanRoofHt",
        windowArea_key = "WindowArea",
        numberOfUnits_key = "NumberOfUnits",
        garage_key = "Garage",
        floodZone_key = "FloodZone",
        windZone_key = "WindZone",
        designWindSpeed_key = "DesignWindSpeed",
        LULC_key = "LULC",
        z0_key = "Z0",
        sheathingThickness_key = "SheathingThickness"
    ):
        """
        Make inference based on Auto population script developed for NJ

        """

        self.input_inventory = input_inventory
        self.n_possible_worlds = n_possible_worlds
        self.include_features = include_features
        self.seed = seed
        self.overwirte_existing = overwirte_existing

        self.name_mapping = {
            planArea_key: "PlanArea",
            numberOfStories_key: "NumberOfStories",
            occupancyClass_key: "OccupancyClass",
            buildingMaterial_key: "BuildingMaterial",
            designLevel_H_key: "DesignLevel_H",
            yearBuilt_key: "YearBuilt",
            roofSystem_key: "RoofFrameType",
            roofShape_key: "RoofShape",
            roofSlope_key: "RoofSlope",
            avgJanTemp_key: "AvgJanTemp",
            meanRoofHt_key: "MeanRoofHt",
            windowArea_key: "WindowArea",
            numberOfUnits_key: "NumberOfUnits",
            garage_key: "Garage",
            floodZone_key: "FloodZone",
            windZone_key: "WindZone",
            designWindSpeed_key: "DesignWindSpeed",
            LULC_key: "LULC",
            z0_key: "Z0",
            sheathingThickness_key: "SheathingThickness"
        }

        self.clean_features = clean_features

    def infer(self) -> AssetInventory:

        input_inventory = self.input_inventory
        n_possible_worlds = self.n_possible_worlds

        #
        # Determine n_pw
        #

        elapseStart = time.time()
        existing_worlds = input_inventory.get_n_pw()

        if existing_worlds is None:
            msg = "ERROR: All assets should have same number of possible worlds to run the inference."
            raise Exception(msg)

        if existing_worlds == 1:
            n_pw = n_possible_worlds  # if zero, it will give the most likely value
        else:
            if n_possible_worlds == 0:
                pass
            elif (
                (n_possible_worlds == 1)
                or (n_possible_worlds == 1)
                or (n_possible_worlds == existing_worlds)
            ):
                logger.warning(
                    f"Existing {existing_worlds} worlds detacted. {existing_worlds} samples will generated per feature"
                )
                n_pw = 1  # n_pw per exisitng pw
            else:
                msg = f"ERROR: the number of possible worlds {n_possible_worlds} should be the same as the existing possible worlds. Choose {existing_worlds} or 0 (to get only the most likely value) to run the inference."
                raise Exception(msg)

        #
        # TODO Update keynames in inventory
        #


        #
        # TODO utilize n possible worlds
        #

        #
        # set seed
        #

        np.random.seed(self.seed)

        #
        # run the first world
        #

        input_inventory_subset = input_inventory.get_world_realization(0)
        input_inventory_json = self.to_json(input_inventory_subset)
        essential_features = self.infer_building_one_by_one(input_inventory_json,n_pw)

        #
        # loop over the second ~ n_pw worlds if needed
        #

        for nw in range(1, existing_worlds):
            # get inventory realization
            inventory_realization = input_inventory.get_world_realization(nw)
            input_inventory_json = self.to_json(inventory_realization)
            essential_features_tmp = self.infer_building_one_by_one(input_inventory_json,n_pw)

            essential_features = self.merge_two_json(
                essential_features_tmp, essential_features, shrink=(nw == existing_worlds - 1)
            )

        #
        # update features
        #

        if self.clean_features:
            # create a fresh inventory
            output_inventory = AssetInventory()
            for index, feature in essential_features.items():
                output_inventory.add_asset_coordinates(index, input_inventory.get_asset_coordinates(index)[1])
                output_inventory.add_asset_features(index, feature)
                updated = True

        else:
            output_inventory = copy.deepcopy(input_inventory)
            for index, feature in essential_features.items():
                output_inventory.add_asset_features(index, feature, overwrite=True)
                updated = True

        #
        # Return the valuee
        #

        if not updated:
            logger.warning("Nothing happened to the inventory.")

        elapseEnd = (time.time() - elapseStart) / 60
        print("Done inference. It took {:.2f} mins".format(elapseEnd))

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

    def infer_building_one_by_one(self, inventory_json,n_pw):

        # TODO: utilize n_pw

        new_features = {}

        for key, bldg in inventory_json.items():

            # TODO user chooce wheather to use default or not
            try:
                essential_features = auto_populate(bldg)
            except ValueError as e:
                msg = f"Failed in building {key} \n"
                msg += str(e)  # Convert exception to string
                logger.error(msg)
                sys.exit(-1)

            new_features[key] = essential_features

        return new_features

    def to_json(self, this_inventory):
        inventory_json = {}
        for key, asset in this_inventory.inventory.items():
            if len(asset.coordinates) == 1:
                geometry = {"type": "Point", "coordinates": [asset.coordinates[0][:]]}
            else:
                geometry = {"type": "Polygon", "coordinates": asset.coordinates}

            feature = {
                "type": "Feature",
                "properties": asset.features,
                "geometry": geometry,
            }
            if "type" in asset.features:
                feature["type"] = asset.features["type"]

            inventory_json[key] = feature

        return inventory_json