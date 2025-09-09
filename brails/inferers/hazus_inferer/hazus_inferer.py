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
# 02-11-2025

import time
import copy

import numpy as np
import logging
import pandas as pd

from brails.types.asset_inventory import AssetInventory
from brails.inferers.inference_engine import InferenceEngine
from brails.inferers.hazus_inferer.hazus_rulesets import (
    get_hazus_occ_type_mapping,
    get_hazus_state_region_mapping,
    get_hazus_height_classes,
    get_hazus_year_classes,
)
from brails.inferers.hazus_inferer.hazus_rulesets import (
    get_hazus_region_to_garage,
    get_hazus_income_to_const_class,
    get_hazus_height_classes_RES1,
    get_hazus_base_replacement_cost,
)
from itertools import product

# sy - note this may not be the most accurate package but it's fast
import reverse_geocode

# To be replaced with old brails ++ codes
import warnings

# Configure logging:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)


class HazusInferer(InferenceEngine):
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
        yearBuilt_key="YearBuilt",
        occupancyClass_key="OccupancyClass",
        numberOfStories_key="NumberOfStories",
        income_key="Income",
        planArea_key="PlanArea",
        splitLevel_key="SplitLevel",
        garageType_key="GarageType",  # optional
        constructionClass_key="ConstructionClass",  # optional
        structureType_key="StructureType",
        replacementCost_key="ReplacementCost",
    ):
        """
        Make inference based on Hazus 6 rulesets

        """

        self.input_inventory = input_inventory
        self.n_possible_worlds = n_possible_worlds
        self.include_features = include_features
        self.seed = seed
        self.overwirte_existing = overwirte_existing
        self.yearBuilt_key = yearBuilt_key
        self.occupancyClass_key = occupancyClass_key
        self.numberOfStories_key = numberOfStories_key
        self.income_key = income_key
        self.planArea_key = planArea_key
        self.splitLevel_key = splitLevel_key
        self.garageType_key = garageType_key
        self.constructionClass_key = constructionClass_key
        self.structureType_key = structureType_key
        self.replacementCost_key = replacementCost_key

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
        # set some variables : TODO: move to the constructer
        #

        # self.n_pw = n_possible_worlds
        # self.seed = seed
        # self.overwirte_existing = overwirte_existing
        self.skip_buildings_if_needed = False
        output_inventory = copy.deepcopy(input_inventory)

        #
        # set seed
        #

        np.random.seed(self.seed)

        #
        # run the first world
        #

        input_inventory_subset = input_inventory.get_world_realization(0)

        if ("StructureType" in self.include_features) or (
            self.structureType_key in self.include_features
        ):
            occ_runable = self.check_keys(
                needed_keys=[
                    self.yearBuilt_key,
                    self.occupancyClass_key,
                    self.numberOfStories_key,
                ],
                target_key=self.structureType_key,
                inventory=input_inventory_subset,
            )
            if occ_runable:
                occ_prop, inventory_realization_df = self.get_str_from_occ(
                    input_inventory_subset,
                    self.yearBuilt_key,
                    self.occupancyClass_key,
                    self.numberOfStories_key,
                    n_pw,
                    self.structureType_key,
                )

        if ("ReplacementCost" in self.include_features) or (
            self.replacementCost_key in self.include_features
        ):
            repl_cost_runable = self.check_keys(
                needed_keys=[
                    self.income_key,
                    self.occupancyClass_key,
                    self.numberOfStories_key,
                    self.planArea_key,
                    self.splitLevel_key,
                ],
                optional_needed_keys=[
                    self.garageType_key, self.constructionClass_key],
                target_key=self.replacementCost_key,
                inventory=input_inventory_subset,
            )
            if repl_cost_runable:
                repl_cost_prop, inventory_realization_df = self.get_replacement_cost(
                    input_inventory_subset,
                    self.income_key,
                    self.occupancyClass_key,
                    self.numberOfStories_key,
                    self.planArea_key,
                    self.splitLevel_key,
                    self.garageType_key,
                    self.constructionClass_key,
                    n_pw,
                    self.replacementCost_key,
                )

        #
        # loop over the second ~ n_pw worlds if needed
        #
        # TODO: Note that there may be inefficiency. Even if you have a probablistic inventory, if year, occ, nstory is non-probablistic, you really don't need to run it 10 times.
        #

        for nw in range(1, existing_worlds):
            # get inventory realization
            inventory_realization = input_inventory.get_world_realization(nw)

            if ("StructureType" in self.include_features) or (
                self.structureType_key in self.include_features
            ):
                # occupancy type
                if occ_runable:
                    occ_prop_tmp, inventory_realization_df = self.get_str_from_occ(
                        inventory_realization,
                        self.yearBuilt_key,
                        self.occupancyClass_key,
                        self.numberOfStories_key,
                        n_pw,
                        self.structureType_key,
                    )
                    occ_prop = self.merge_two_json(
                        occ_prop_tmp, occ_prop, shrink=(
                            nw == existing_worlds - 1)
                    )

            if ("ReplacementCost" in self.include_features) or (
                self.replacementCost_key in self.include_features
            ):
                if repl_cost_runable:
                    repl_cost_prop_tmp, inventory_realization_df = (
                        self.get_replacement_cost(
                            inventory_realization,
                            self.income_key,
                            self.occupancyClass_key,
                            self.numberOfStories_key,
                            self.planArea_key,
                            self.splitLevel_key,
                            self.garageType_key,
                            self.constructionClass_key,
                            n_pw,
                            self.replacementCost_key,
                        )
                    )
                    repl_cost_prop = self.merge_two_json(
                        repl_cost_prop_tmp,
                        repl_cost_prop,
                        shrink=(nw == existing_worlds - 1),
                    )

        #
        # update features
        #

        updated = False

        if (
            ("StructureType" in self.include_features)
            or (self.structureType_key in self.include_features)
        ) and occ_runable:
            for index, feature in occ_prop.items():
                output_inventory.add_asset_features(
                    index, feature, overwrite=True)
                updated = True

        if (
            ("ReplacementCost" in self.include_features)
            or (self.replacementCost_key in self.include_features)
        ) and repl_cost_runable:
            for index, feature in repl_cost_prop.items():
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

    def check_keys(self, needed_keys, target_key, inventory, optional_needed_keys=[]):
        #
        # Convert inventory to df
        #

        bldg_properties_df, bldg_geometries_df, nbldg = inventory.get_dataframe()
        bldg_properties_df = bldg_properties_df.replace(
            "NA", np.nan, inplace=False
        )  # missing
        bldg_properties_df = bldg_properties_df.replace(
            "", np.nan, inplace=False
        )  # missing
        provided_keys = bldg_properties_df.columns

        #
        # check if the dataframe column has the keys
        #

        # Check if needed_keys is a subset of provided_keys
        if set(needed_keys).issubset(set(provided_keys)):
            pass
        else:
            # Find elements in needed_keys that are not in provided_keys
            not_in_provided_keys = [
                item for item in needed_keys if item not in provided_keys
            ]
            logger.warning(
                f"The keys needed to estimate {target_key} is not there: {not_in_provided_keys}"
            )
            logger.warning(f"Skipping hazus inference of {target_key}.")
            return False

        #
        # Check if input is missing
        #

        ready = True

        for key in needed_keys + optional_needed_keys:
            if (key in optional_needed_keys) and (key not in provided_keys):
                # don't care
                continue

            my_col = bldg_properties_df[key]

            if bldg_properties_df[key].isnull().any():  # if null found
                missing_values_index = my_col[my_col.isnull()].index.tolist()

                if self.skip_buildings_if_needed:
                    pass
                    # TODO: remove dataframe rows that have missing data and ready is true, return dataframe

                else:
                    if len(missing_values_index) > 10:
                        print(
                            f"The feature {key} is missing in many buildings including: ",
                            missing_values_index[0:10],
                        )
                    else:
                        print(
                            f"The feature {key} is missing in following buildings: ",
                            missing_values_index,
                        )
                    ready = False

        if not ready:
            print(
                f"Skipping hazus inference of {target_key}. If you still want to perform the inference, run imputer first and come back."
            )
            return False

        #
        # warning message for overwritting
        #

        if target_key in provided_keys:
            avail_percentage = (
                100
                - sum(bldg_properties_df[key].isnull())
                / len(bldg_properties_df[key])
                * 100
            )

            if self.overwirte_existing and (avail_percentage < 100):
                logger.warning(
                    f"the feature {target_key} available for {avail_percentage} % of inventories. They will be overwritten, unless otherwise specified."
                )
            elif (not self.overwirte_existing) and (avail_percentage < 100):
                logger.warning(
                    f"the feature {target_key} available for {avail_percentage} % of inventories. The feature will be inferred only for the missing inventories, unless otherwise specified."
                )
            elif (not self.overwirte_existing) and (avail_percentage == 100):
                logger.warning(
                    f"the feature {target_key} is already complete. If you still want to perform the inference, please turn on the option to overwrite"
                )

        #
        # Count the existing worlds
        #

        # if set(needed_keys) & set(inventory.get_multi_keys()):
        #     # "There is at least one overlapping element."
        #     existing_worlds = inventory.get_n_pw
        # else:
        #     existing_worlds = 1

        return True

    def get_str_from_occ(
        self,
        input_inventory,
        yearBuilt_key,
        occupancyClass_key,
        numberOfStories_key,
        n_pw,
        structureType_key,
    ):
        #
        # convert inventory to df
        #

        bldg_properties_df, bldg_geometries_df, nbldg = input_inventory.get_dataframe()

        #
        # get hazus rulesets
        #

        states_to_region = get_hazus_state_region_mapping()
        height_classes = get_hazus_height_classes()
        year_classes = get_hazus_year_classes()
        type_lists, type_weights = get_hazus_occ_type_mapping()

        #
        # Add "State" and "region" columns
        #

        geo_locs = reverse_geocode.search(
            [(row[0], row[1])
             for i, row in enumerate(bldg_geometries_df.values)]
        )
        states_list = [bldg["state"] for bldg in geo_locs]
        region_list = [states_to_region[state]["RegionGroup"]
                       for state in states_list]
        bldg_properties_df["State"] = states_list
        bldg_properties_df["Region"] = region_list

        #
        # Add "BuildingRise" column
        #

        bldg_properties_df["BuildingRise"] = ""
        for height_class, story_list in height_classes.items():
            in_class_index = bldg_properties_df[numberOfStories_key].isin(
                story_list)
            if sum(in_class_index) > 0:
                bldg_properties_df.loc[in_class_index,
                                       "BuildingRise"] = height_class

        #
        # Add "YearClass" column
        #

        bldg_properties_df["YearClass"] = ""
        for year_class, year_list in year_classes.items():
            in_class_index = bldg_properties_df[yearBuilt_key].isin(year_list)
            if sum(in_class_index) > 0:
                bldg_properties_df.loc[in_class_index,
                                       "YearClass"] = year_class

        #
        # Clean occupancy class and add as a new column
        #

        bldg_properties_df[f"{occupancyClass_key}_clean"] = bldg_properties_df[
            occupancyClass_key
        ].apply(self.modulate_occ)
        occupancyClass_key = f"{occupancyClass_key}_clean"

        #
        # Get all cases of interest
        #

        region_list = list(set(bldg_properties_df["Region"]))
        occ_list = list(set(bldg_properties_df[occupancyClass_key]))
        height_list = height_classes.keys()
        state_list = list(set(bldg_properties_df["State"]))
        classes_in_inventory = list(
            product(region_list, occ_list, height_list))

        #
        # Run inference
        #

        new_prop = {}

        for (
            region,
            occ,
            height,
        ) in (
            classes_in_inventory
        ):  # for all regions that appear at least once in inventory
            subset_inventory = bldg_properties_df[
                (bldg_properties_df["Region"] == region)
                & (bldg_properties_df[occupancyClass_key] == occ)
                & (bldg_properties_df["BuildingRise"] == height)
            ]
            nbldg_subset = len(subset_inventory)

            if nbldg_subset == 0:
                # no instance found
                continue

            if region == "West Coast":
                # year built is considered only in west coast

                for year_class in year_classes:
                    subset_inventory2 = subset_inventory[
                        (subset_inventory["YearClass"] == year_class)
                    ]  # inventory with specific region, occ, height, year
                    nbldg_subset2 = len(subset_inventory2)

                    if nbldg_subset2 == 0:
                        # no instance found
                        continue

                    if occ == "RES1":
                        # print(f"{occ} {year_class} {nbldg_subset2}")

                        for state in state_list:
                            subset_inventory3 = subset_inventory2[
                                (subset_inventory2["State"] == state)
                            ]  # inventory with specific region, occ, height, year, state
                            nbldg_subset3 = len(subset_inventory2)

                            weights = (
                                np.array(
                                    type_weights[region][occ][year_class][state])
                                / 100.0
                            )
                            structure_types = np.array(
                                type_lists[region]["RES1"])
                            weights, structure_types = self.modulate_weights(
                                weights,
                                structure_types,
                                region,
                                occ,
                                year_class,
                                height,
                            )

                            if len(weights) == 0:
                                logger.warning(
                                    f"HAZUS does not provide structural type information for {region}-{occ}-{height}-{year_class}-{state}. {structureType_key} will be missing in id={subset_inventory2.index.tolist()}"
                                )

                            new_prop = self.add_features_to_asset(
                                new_prop,
                                structureType_key,
                                structure_types,
                                weights,
                                n_pw,
                                nbldg_subset3,
                                global_asset_indices=subset_inventory3.index,
                            )

                    else:
                        weights = (
                            np.array(type_weights[region]
                                     [occ][height][year_class])
                            / 100.0
                        )
                        structure_types = np.array(type_lists[region][height])
                        weights, structure_types = self.modulate_weights(
                            weights, structure_types, region, occ, year_class, height
                        )

                        if len(weights) == 0:
                            logger.warning(
                                f" HAZUS does not provide structural type information for {region}-{occ}-{height}-{year_class}. {structureType_key} will be missing in id={subset_inventory2.index.tolist()}"
                            )

                        new_prop = self.add_features_to_asset(
                            new_prop,
                            structureType_key,
                            structure_types,
                            weights,
                            n_pw,
                            nbldg_subset2,
                            global_asset_indices=subset_inventory2.index,
                        )

            elif (region == "Mid-West") or (region == "East Coast"):
                if occ == "RES1":
                    # print(f"{occ} {year_class} {nbldg_subset2}")

                    for state in state_list:
                        subset_inventory3 = subset_inventory[
                            (subset_inventory["State"] == state)
                        ]  # inventory with specific region, occ, height, year, state
                        nbldg_subset3 = len(subset_inventory)

                        weights = np.array(
                            type_weights[region][occ][state]) / 100.0
                        structure_types = np.array(type_lists[region]["RES1"])
                        weights, structure_types = self.modulate_weights(
                            weights, structure_types, region, occ, year_class, height
                        )

                        if len(weights) == 0:
                            logger.warning(
                                f"HAZUS does not provide structural type information for {region}-{occ}-{height}-{year_class}-{state}. {structureType_key} will be missing in id={subset_inventory2.index.tolist()}"
                            )

                        new_prop = self.add_features_to_asset(
                            new_prop,
                            structureType_key,
                            structure_types,
                            weights,
                            n_pw,
                            nbldg_subset3,
                            global_asset_indices=subset_inventory3.index,
                        )
                else:
                    # define prob and categories
                    weights = np.array(
                        type_weights[region][occ][height]) / 100.0
                    structure_types = np.array(type_lists[region][height])
                    weights, structure_types = self.modulate_weights(
                        weights, structure_types, region, occ, year_class, height
                    )

                    if len(weights) == 0:
                        logger.warning(
                            f"HAZUS does not provide structural type information for {region}-{occ}-{height}. {structureType_key} will be missing in id={subset_inventory.index.tolist()}"
                        )

                    # add assets
                    new_prop = self.add_features_to_asset(
                        new_prop,
                        structureType_key,
                        structure_types,
                        weights,
                        n_pw,
                        nbldg_subset,
                        global_asset_indices=subset_inventory.index,
                    )

        return new_prop, bldg_properties_df

    # Clean up occ classes
    def modulate_occ(self, s):
        # if s="RES1A", return s="RES1"
        if not s[-1].isdigit():  # Check if the last character is not a digit
            return s[:-1]  # Remove the last character
        return s  # Return the string unchanged if it ends with a number

    def flatten_array(self, arr):
        if len(arr) == 1:
            return arr[0]  # Flatten to string
        else:
            return arr  # Leave the array as is or process differently

    def add_features_to_asset(
        self,
        new_prop,
        structureType_key,
        structure_types,
        weights,
        n_pw,
        n_bldg_subset,
        global_asset_indices,
    ):
        if len(weights) == 0:
            for count, index in enumerate(global_asset_indices):
                # new_prop[index] = {structureType_key: "NOT IN HAZUS" }
                # does not exist in ruleset
                new_prop[index] = {structureType_key: "NA"}
            return new_prop

        if n_pw == 0:
            # most likely struct
            struct_pick = [structure_types[np.argmax(weights)]] * n_bldg_subset
        else:
            # sample nbldg x n_pw
            struct_pick = np.random.choice(
                structure_types, size=[n_bldg_subset, n_pw], replace=True, p=weights
            ).tolist()

        for count, index in enumerate(global_asset_indices):
            # shrinks to a scalar value if same.
            val_vec = struct_pick[count]
            if not isinstance(val_vec, list):
                val = val_vec[0]
            elif len(set(val_vec)) == 1:
                val = val_vec[0]
            else:
                val = val_vec

            new_prop[index] = {
                structureType_key: val
            }  # if #elem in list is 1, convert it to integer

            # new_prop[index] = {structureType_key: self.flatten_array(struct_pick[count])} # if #elem in list is 1, convert it to integer

        return new_prop

    def modulate_weights(
        self, weights, structure_types, region, occ, year_class, height
    ):
        pass

        return weights, structure_types

    def get_replacement_cost(
        self,
        input_inventory,
        income_key,
        occupancyClass_key,
        numberOfStories_key,
        planArea_key,
        splitLevel_key,
        garageType_key,
        constructionClass_key,
        n_pw,
        replacementCost_key,
    ):
        #
        # convert inventory to df
        #

        bldg_properties_df, bldg_geometries_df, nbldg = input_inventory.get_dataframe()

        #
        # get hazus rulesets
        #

        states_to_region = get_hazus_state_region_mapping()
        # each to garage (random), construction class(random), height class(deterministic) columns and fianl replacement cost(random)
        garage_type_list, census_to_garage_weight = get_hazus_region_to_garage()
        (
            income_ratio_thres,
            income_group_list,
            const_class_list,
            income_to_const_class_weight,
            state_average_income,
        ) = get_hazus_income_to_const_class()
        res1_height_classes = get_hazus_height_classes_RES1()
        replacement_cost_per_ft2, garage_cost_per_residence = (
            get_hazus_base_replacement_cost()
        )

        #
        # From 'Lat' and 'Long' to "State" and "Region" columns
        #

        geo_locs = reverse_geocode.search(
            [(row[0], row[1])
             for i, row in enumerate(bldg_geometries_df.values)]
        )
        states_list = [bldg["state"] for bldg in geo_locs]
        region_list = [states_to_region[state]["CensusRegion"]
                       for state in states_list]
        bldg_properties_df["State"] = states_list
        bldg_properties_df["Region"] = region_list

        #
        # From 'nstory' to "BuildingRise" (Only used for RES1)
        #

        bldg_properties_df["BuildingRise"] = ""
        for height_class, story_list in res1_height_classes.items():
            in_class_index = bldg_properties_df[numberOfStories_key].isin(
                story_list)
            if sum(in_class_index) > 0:
                bldg_properties_df.loc[in_class_index,
                                       "BuildingRise"] = height_class

        # overwrite if split level
        bldg_properties_df.loc[
            bldg_properties_df[splitLevel_key] == "Yes", "BuildingRise"
        ] = "Split level"

        #
        # From "State" and 'income' to 'income group' # this can be more efficient
        #

        state_average_income_list = [
            state_average_income[state] for state in states_list
        ]
        bldg_properties_df["StateAverageIncome"] = state_average_income_list
        # income_ratio_list = [bldg[income_key]/state_average_income[bldg["State"]] for bldg in geo_locs]
        bldg_properties_df["IncomeRatio"] = (
            bldg_properties_df[income_key] /
            bldg_properties_df["StateAverageIncome"]
        )
        bldg_properties_df["IncomeGroup"] = pd.cut(
            bldg_properties_df["IncomeRatio"],
            bins=income_ratio_thres,
            labels=income_group_list,
        )

        #
        # From "Region" to 'Garage' (random sampling) - always do 1 pw at a time
        #

        if garageType_key in bldg_properties_df.columns:
            print(
                f"{garageType_key} info found in the inventory. Skipping the inference of Garage Type."
            )
            garage_df = pd.DataFrame(
                {i: bldg_properties_df[garageType_key] for i in range(n_pw)}
            )

        else:
            print(
                f"{garageType_key} info not found in the inventory. Making inference using Hazus 6."
            )

            # bldg_properties_df["garage_type"] = ""
            garage_df = pd.DataFrame(
                np.nan, index=bldg_properties_df.index, columns=range(n_pw)
            )
            region_list = list(set(bldg_properties_df["Region"]))

            # for npp in range(n_pw):
            for (
                region
            ) in region_list:  # for all regions that appear at least once in inventory
                subset_inventory = bldg_properties_df[
                    bldg_properties_df["Region"] == region
                ]
                nbldg_subset = len(subset_inventory)

                weights = np.array(census_to_garage_weight[region]) / 100.0
                garage_type = np.array(garage_type_list)

                if n_pw == 0:
                    # most likely struct
                    garage_pick = [
                        garage_type[np.argmax(weights)]] * nbldg_subset
                else:
                    # sample nbldg x n_pw
                    garage_pick = np.random.choice(
                        garage_type, size=[nbldg_subset, n_pw], replace=True, p=weights
                    ).tolist()

                # garage_pick = pd.DataFrame(garage_pick)

                # garage_df.loc[subset_inventory.index, range(n_pw)] = garage_pick.astype('str')
                garage_df.loc[subset_inventory.index,
                              range(max(1, n_pw))] = garage_pick

        #
        # From 'Income Group' to 'Construction Class' (random sampling)
        #

        # bldg_properties_df["const_class"] = ""

        if constructionClass_key in bldg_properties_df.columns:
            print(
                f"{constructionClass_key} info found in the inventory. Skipping the inference of Garage Type."
            )
            const_class_df = pd.DataFrame(
                {i: bldg_properties_df[constructionClass_key]
                    for i in range(n_pw)}
            )

        else:
            print(
                f"{constructionClass_key} info not found in the inventory. Making inference using Hazus 6."
            )

            const_class_df = pd.DataFrame(
                np.nan, index=bldg_properties_df.index, columns=range(n_pw)
            )
            income_group_list = list(set(bldg_properties_df["IncomeGroup"]))

            for income_group in (
                income_group_list
            ):  # for all regions that appear at least once in inventory
                subset_inventory = bldg_properties_df[
                    bldg_properties_df["IncomeGroup"] == income_group
                ]
                nbldg_subset = len(subset_inventory)

                weights = np.array(
                    income_to_const_class_weight[income_group]) / 100.0
                const_class = np.array(const_class_list)

                if n_pw == 0:
                    # most likely struct
                    const_class_pick = [
                        const_class[np.argmax(weights)]] * nbldg_subset

                    const_class_df.loc[subset_inventory.index,
                                       0] = const_class_pick

                else:
                    # sample nbldg x n_pw
                    const_class_pick = np.random.choice(
                        const_class, size=[nbldg_subset, n_pw], replace=True, p=weights
                    ).tolist()

                    # const_class_pick = pd.DataFrame(const_class_pick)

                    # const_class_df.loc[subset_inventory.index, range(n_pw)] = const_class_pick.astype('str')
                    const_class_df.loc[subset_inventory.index, range(n_pw)] = (
                        const_class_pick
                    )

        #
        # From 'occtype' and 'height Class' to 'basecost'
        #

        new_prop = {}
        # for i,row in enumerate(bldg_properties_df.values):
        # print(garage_df)

        for i, occ, fparea, height, region in zip(
            bldg_properties_df.index,
            bldg_properties_df[occupancyClass_key],
            bldg_properties_df[planArea_key],
            bldg_properties_df["BuildingRise"],
            bldg_properties_df["Region"],
        ):
            # occ = row[occupancyClass_key]
            # fparea = row[planArea_key]
            if occ == "RES1":
                base_cost = np.zeros((max(1, n_pw),))
                garage_cost = np.zeros(
                    max(1, n_pw),
                )
                for npp in range(max(1, n_pw)):
                    # const = row['const_class']
                    # garage = row['garage_type']
                    const = const_class_df.loc[i, npp]
                    garage = garage_df.loc[i, npp]
                    # height = row['height']
                    base_cost[npp] = (
                        replacement_cost_per_ft2[occ][const][height]["finished"]
                        * fparea
                    )  # TODO-ADAM: assuming finished
                    # print(garage_cost_per_residence[const])
                    garage_cost[npp] = garage_cost_per_residence[const][garage]
                if n_pw > 1:
                    final_cost = (base_cost + garage_cost).tolist()
                else:
                    final_cost = float((base_cost + garage_cost)[0])

            elif occ == "RES2":
                # region = row["Region"]
                final_cost = replacement_cost_per_ft2[occ][region] * fparea
            else:
                final_cost = replacement_cost_per_ft2[occ] * fparea

            # row['replace_cost'] = base_cost + garage_cost
            new_prop[i] = {replacementCost_key: final_cost}

        #
        # Below is potentially faster?
        #

        # bldg_properties_df["base_rep_cost_per_ft2"] = ""
        # occtype_group_list = list(set(bldg_properties_df['occtype']))

        # for occ in occtype_group_list: # for all regions that appear at least once in inventory

        #     subset_inventory =  bldg_properties_df[bldg_properties_df[occupancyClass_key]==occ]
        #     nbldg_subset = len(subset_inventory)

        #     if not (occ=="RES1"):
        #         cost = np.array(replacement_cost_per_ft2[occ])/100.

        #     else:
        #         const_class_list = list(set(subset_inventory['const_class']))
        #         height_class_list = list(set(subset_inventory["BuildingRise"]))
        #         classes_in_inventory = list(product(const_class_list,height_class_list))

        #         for const, height in classes_in_inventory: # for all regions that appear at least once in inventory

        #             subset_inventory2 = subset_inventory[(subset_inventory['const_class']==const) & (subset_inventory["BuildingRise"]==height)]
        #             nbldg_subset2 = len(subset_inventory2)

        #             cost = np.array(replacement_cost_per_ft2[occ][occ])/100.

        return new_prop, bldg_properties_df
