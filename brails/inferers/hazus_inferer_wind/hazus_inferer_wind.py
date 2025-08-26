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
import os
import json
import pandas as pd
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError

import numpy as np

from brails.types.asset_inventory import AssetInventory
from brails.inferers.inference_engine import InferenceEngine
from brails.inferers.hazus_inferer_wind.auto_HU_NJ import auto_populate

from brails.utils import GeoTools

# To be replaced with old brails ++ codes
import warnings
import logging

# Configure logging:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)


class HazusInfererWind(InferenceEngine):
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
        planArea_key="PlanArea",
        numberOfStories_key="NumberOfStories",
        occupancyClass_key="OccupancyClass",
        BuildingType_key="BuildingType",
        designLevel_H_key="DesignLevel_H",
        yearBuilt_key="YearBuilt",
        roofSystem_key="RoofSystem",
        roofShape_key="RoofShape",
        roofSlope_key="RoofSlope",
        avgJanTemp_key="AvgJanTemp",
        Height_key="Height",
        windowArea_key="WindowArea",
        numberOfUnits_key="NumberOfUnits",
        hasGarage_key="HasGarage",
        floodZone_key="FloodZone",
        windZone_key="WindZone",
        designWindSpeed_key="DesignWindSpeed",
        LULC_key="LULC",
        z0_key="Z0",
        sheathingThickness_key="SheathingThickness",
        use_default=True
    ):
        """
        Make inference based on Auto population script developed for NJ

        """

        self.input_inventory = input_inventory
        self.n_possible_worlds = n_possible_worlds
        self.include_features = include_features
        self.seed = seed
        self.overwirte_existing = overwirte_existing
        self.use_default = use_default

        self.name_mapping = {
            planArea_key: "PlanArea",
            numberOfStories_key: "NumberOfStories",
            occupancyClass_key: "OccupancyClass",
            BuildingType_key: "BuildingType",
            designLevel_H_key: "DesignLevel_H",
            yearBuilt_key: "YearBuilt",
            roofSystem_key: "RoofSystem",
            roofShape_key: "RoofShape",
            roofSlope_key: "RoofSlope",
            avgJanTemp_key: "AvgJanTemp",
            Height_key: "Height",
            windowArea_key: "WindowArea",
            numberOfUnits_key: "NumberOfUnits",
            hasGarage_key: "HasGarage",
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

        if n_possible_worlds == 0:
            msg = f"ERROR: most likely attributes option is not supported for the wind inferer. Please select n_possible_worlds>0"
            logger.error(msg)
            sys.exit(-1)

        #
        # Determine existing_worlds and n_pw
        #
        # existing_worlds : count the number of possible worlds in the current inventory
        # only if existing_worlds==1, n_pw possible worlds will be generated. otherwise n_pw is not used

        elapseStart = time.time()
        existing_worlds = input_inventory.get_n_pw()

        if existing_worlds is None:
            msg = "ERROR: All assets should have same number of possible worlds to run the inference."
            raise Exception(msg)

        if existing_worlds == 1:
            n_pw = n_possible_worlds  # if zero, it will give the most likely value
            logger.warning(
                f"The existing inventory does not contain multiple possible worlds. {n_pw} worlds will be generated for new features"
            )

        else:
            if (
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
        # Update keynames in inventory
        #

        input_inventory.change_feature_names(self.name_mapping)

        #
        # set seed
        #

        np.random.seed(self.seed)

        #
        # run the first world
        #

        input_inventory_subset = input_inventory.get_world_realization(0)
        input_inventory_json = self.to_json(input_inventory_subset)
        essential_features = self.infer_building_one_by_one(
            input_inventory_json, n_pw)

        #
        # loop over the second ~ n_pw worlds if needed
        #

        for nw in range(1, existing_worlds):
            # get inventory realization
            inventory_realization = input_inventory.get_world_realization(nw)
            input_inventory_json = self.to_json(inventory_realization)
            essential_features_tmp = self.infer_building_one_by_one(
                input_inventory_json, n_pw)

            essential_features = self.merge_two_json(
                essential_features_tmp, essential_features, shrink=(
                    nw == existing_worlds - 1)
            )

        #
        # update features
        #

        if self.clean_features:
            # create a fresh inventory
            output_inventory = AssetInventory()
            for index, feature in essential_features.items():
                output_inventory.add_asset_coordinates(
                    index, input_inventory.get_asset_coordinates(index)[1])
                output_inventory.add_asset_features(index, feature)
                updated = True

        else:
            output_inventory = copy.deepcopy(input_inventory)
            for index, feature in essential_features.items():
                output_inventory.add_asset_features(
                    index, feature, overwrite=True)
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

    def infer_building_one_by_one(self, inventory_json, n_pw):

        for pw in range(n_pw):
            new_features_tmp = {}
            for key, bldg in inventory_json.items():
                try:
                    essential_features = auto_populate(bldg, self.use_default)
                except ValueError as e:
                    msg = f"Failed in building {key} \n"
                    msg += str(e)  # Convert exception to string
                    logger.error(msg)
                    sys.exit(-1)

                new_features_tmp[key] = essential_features

            if pw == 0:
                # first possible world
                new_features = new_features_tmp
            else:
                new_features = self.merge_two_json(
                    new_features_tmp, new_features, shrink=(pw == n_pw - 1)
                )

        return new_features

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

    def validate(self, input_inventory, silence=False):

        init = time.time()

        # load the schema assuming it is called "input_schema.json" and it is
        # stored next to the mapping script
        current_file_path = os.path.dirname(__file__)
        schema_path = os.path.join(current_file_path, "input_schema.json")

        with Path(schema_path).open(encoding="utf-8") as f:
            input_schema = json.load(f)

        def check_building(gi):
            # gi is a dictionary of {feature:value} pairs

            # make sure missing data is properly represented as null in the JSON
            for key, item in gi.items():
                if pd.isna(item):
                    gi[key] = None

            # validate the provided features against the required inputs
            validate(instance=gi, schema=input_schema)

        invalid_id = []
        error_record = {}
        bldg_count = 0
        for id in input_inventory.get_asset_ids():
            try:
                check_building(input_inventory.get_asset_features(id)[1])
            except ValidationError as e:
                invalid_id += [id]
                error_record[id] = e.message
            bldg_count += 1

        print(f'Done validation. It took {round(time.time() - init, 2)} sec.')

        if not silence:
            if len(invalid_id) > 0:
                logger.warning(
                    f"The inventory has {len(invalid_id)} assets ({round(len(invalid_id) / bldg_count * 100, 2)}%) that are identified as invalid. This means the information on the specific combinations of input features are not found in the Hazus DL library. You can use <correct> method to quickly project those features to a valid feature combination. See documentation."
                )
            else:
                print("Good to go")

        return invalid_id, error_record

    def correct(self, input_inventory, invalid_id=None, weights={}):

        init = time.time()

        # re-running validation, in case the user changed some inputs
        if invalid_id == None:
            invalid_id, error_record = self.validate(
                input_inventory, silence=True)

        n_assets = len(input_inventory.inventory)
        n_invalid = len(invalid_id)

        output_inventory = copy.deepcopy(input_inventory)

        # record the invalid building properties
        target_bldg = {}
        target_loc = {}
        for count_invalid, id in enumerate(invalid_id):
            target_bldg[count_invalid] = copy.deepcopy(
                output_inventory.get_asset_features(id)[1])
            target_loc[count_invalid] = np.mean(
                output_inventory.get_asset_coordinates(id)[1], axis=0)
            # clean up the features so that it will not get a highscore
            output_inventory.inventory[id].remove_features(
                target_bldg[count_invalid].keys())

        # compute the score of each of the valid assets
        score = np.zeros([n_assets, n_invalid])
        for count_assets, bldg_id in enumerate(output_inventory.get_asset_ids(), start=0):
            # for this building,

            bldg_features = output_inventory.get_asset_features(bldg_id)[1]
            for count_invalid in range(n_invalid):

                # compute the score
                for target_feature, target_value in target_bldg[count_invalid].items():
                    if target_feature in bldg_features:
                        score[count_assets, count_invalid] += (
                            bldg_features[target_feature] == target_value) * weights.get(target_feature, 1)

        # index_of_max_score = np.argmax(score, axis = 0)

        # for count_invalid, id in enumerate(invalid_id):
        #     output_inventory.add_asset_features(id, output_inventory.get_asset_features(index_of_max_score[count_invalid])[1])

        max_score = np.max(score, axis=0)

        for count_invalid, id in enumerate(invalid_id):

            tie_indices = np.where(
                max_score[count_invalid] == score[:, count_invalid])[0]

            if len(tie_indices) == 1:
                # only one candidate
                selected_neighbor = tie_indices[0]
            else:
                # select the closest building
                target_point = target_loc[count_invalid]
                my_distance = []
                for t_idx in tie_indices:
                    candi_point = np.mean(
                        output_inventory.get_asset_coordinates(t_idx)[1], axis=0)
                    my_distance += [GeoTools.haversine_dist(
                        target_point.tolist(), candi_point.tolist())]
                selected_neighbor = tie_indices[np.argmin(my_distance)]

            output_inventory.add_asset_features(
                id, output_inventory.get_asset_features(selected_neighbor)[1])

        if len(invalid_id) == 0:
            print(
                f"Nothing happened. All good to go."
            )
        else:
            print(
                f"{n_invalid} Assets ({round(n_invalid / n_assets * 100, 2)}%) are corrected. Now good to go."
            )
        print(f'Done correction. It took {round(time.time() - init, 2)} sec.')

        return output_inventory
