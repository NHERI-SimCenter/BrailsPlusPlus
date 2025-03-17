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
import sys
from copy import deepcopy

import numpy as np
import logging
import pandas as pd
from brails.utils import Importer
from brails.types.asset_inventory import AssetInventory
from brails.inferers.inferenceEngine import InferenceEngine

# Configure logging:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HazusInfererEarthquake(InferenceEngine):
    """
    To provide variables needed for Hazus damage and loss assessment

    Attributes:

    Methods:

    """

    def __init__(
        self,
        input_inventory: AssetInventory,
        n_possible_worlds=1,
        seed=1,
        overwirte_existing=True,
        yearBuilt_key="YearBuilt",
        occupancyClass_key="OccupancyClass",
        numberOfStories_key="NumberOfStories",
        income_key="Income",
        planArea_key="PlanArea",
        splitLevel_key="SplitLevel",
        buildingRise_key="BuildingRise",
        garageType_key="GarageType",  # optional
        constructionClass_key="ConstructionClass",  # optional
        foundationType_key="FoundationType",  # optional
        designLevel_key="DesignLevel",  # optional
        structureType_key="StructureType",  # optional
        replacementCost_key="ReplacementCost",  # optional
        clean_features=False,
    ):
        """
        To provide variables needed for Hazus damage and loss assessment

        """

        self.input_inventory = input_inventory
        self.n_possible_worlds = n_possible_worlds
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
        self.foundationType_key = foundationType_key
        self.structureType_key = structureType_key
        self.replacementCost_key = replacementCost_key
        self.buildingRise_key = buildingRise_key
        self.designLevel_key = designLevel_key
        self.clean_features = clean_features

    def infer(self) -> AssetInventory:
        elapseStart = time.time()
        np.random.seed(self.seed)

        self.skip_buildings_if_needed = False

        #
        # Names of final variables
        #

        target_keys_json = {
            self.structureType_key: "StructureType",  # Damage
            self.buildingRise_key: "BuildingRise",  # Damage
            self.designLevel_key: "DesignLevel",  # Damage
            self.foundationType_key: "FoundationType",  # Damage
            self.replacementCost_key: "ReplacementCost",  # Loss
            self.occupancyClass_key: "OccupancyClass",  # Loss
        }

        #
        # What we need at the end
        #

        target_keys = target_keys_json.keys()

        #
        # Keys that needs to be inferred
        #

        dummy, provided_keys = self.input_inventory.get_multi_keys()
        keys_to_infer = set(target_keys).difference(set(provided_keys))

        #
        # Step 1. "Occupancy type" cannot be inferred
        #

        print(
            f">> Step1 : Checking if OccupancyClass ({self.occupancyClass_key}) exist."
        )
        if self.occupancyClass_key in keys_to_infer:
            # occupancyClass_key cannot be inferred
            # sy - TODO; find a better way of doing it..
            bldg_properties_df, bldg_geometries_df, nbldg = (
                self.input_inventory.get_dataframe()
            )
            bldg_properties_df = bldg_properties_df.replace(
                "NA", np.nan, inplace=False
            )  # missing
            bldg_properties_df = bldg_properties_df.replace(
                "", np.nan, inplace=False
            )  # missing

            self.check_keys(
                needed_keys=[self.occupancyClass_key], inventory_df=bldg_properties_df
            )

        #
        # Step 2. "StructureType" and "ReplacementCost" can be inferred from Hazus mapping (Hazus inventory manual 6)
        #

        print(
            f">> Step2-1 : Checking if StructureType ({self.structureType_key}) and ReplacementCost ({self.replacementCost_key}) exist"
        )

        keys_to_infer_from_haz_inferer = set(
            [self.structureType_key, self.replacementCost_key]
        ).intersection(set(keys_to_infer))

        if keys_to_infer_from_haz_inferer:  # non empty
            print(f">> Step2-2 : Inferring {keys_to_infer_from_haz_inferer}")

            importer = Importer()
            simcenter_inferer_class = importer.get_class("SimCenterInferer")
            inferer = simcenter_inferer_class(
                input_inventory=self.input_inventory,
                n_possible_worlds=self.n_possible_worlds,
                seed=self.seed,
                yearBuilt_key=self.yearBuilt_key,
                occupancyClass_key=self.occupancyClass_key,
                numberOfStories_key=self.numberOfStories_key,
                income_key=self.income_key,
                planArea_key=self.planArea_key,
                splitLevel_key=self.splitLevel_key,
                structureType_key=self.structureType_key,
                replacementCost_key=self.replacementCost_key,
                include_features=list(keys_to_infer_from_haz_inferer),
            )

            self.input_inventory = inferer.infer()

        #
        # Step 3. "BuildingRise", "DesignLevel", "FoundationType", Simple deterministic mapping
        #

        print(
            f">> Step3-1 : Checking if BuildingRise ({self.buildingRise_key}), DesignLevel ({self.designLevel_key}) and FoundationType ({self.foundationType_key}) exist"
        )

        keys_to_infer_DL = set(
            [self.buildingRise_key, self.designLevel_key, self.foundationType_key]
        ).intersection(set(keys_to_infer))

        if keys_to_infer_DL:
            n_pw, existing_worlds = self.compute_n_possible_worlds(
                self.n_possible_worlds
            )

            print(f">> Step3-2 : Inferring {keys_to_infer_DL}")

            inventory_realization = self.input_inventory.get_world_realization(0)
            bldg_properties_df, bldg_geometries_df, nbldg = (
                inventory_realization.get_dataframe()
            )
            bldg_properties_df = bldg_properties_df.replace(
                "NA", np.nan, inplace=False
            )  # missing
            bldg_properties_df = bldg_properties_df.replace(
                "", np.nan, inplace=False
            )  # missing

            #
            # check if you have enough
            #

            if self.buildingRise_key in keys_to_infer_DL:
                heightclass_runable = self.check_keys(
                    needed_keys=[self.numberOfStories_key],
                    target_key=self.buildingRise_key,
                    inventory_df=bldg_properties_df,
                )
            else:
                heightclass_runable = False

            if self.designLevel_key in keys_to_infer_DL:
                designlevel_runable = self.check_keys(
                    needed_keys=[self.yearBuilt_key, self.structureType_key],
                    target_key=self.designLevel_key,
                    inventory_df=bldg_properties_df,
                )
            else:
                designlevel_runable = False

            if (not designlevel_runable) and self.designLevel_key in keys_to_infer_DL:
                keys_to_infer_DL.remove(self.designLevel_key)

            if (not heightclass_runable) and self.buildingRise_key in keys_to_infer_DL:
                keys_to_infer_DL.remove(self.buildingRise_key)

            #
            # generation
            #

            # inventory_realization = self.input_inventory.get_world_realization(0)
            new_prop, inventory_realization_df = self.get_era_height_foundation_class(
                inventory_realization,
                self.structureType_key,
                self.yearBuilt_key,
                self.numberOfStories_key,
                self.buildingRise_key,
                self.designLevel_key,
                self.foundationType_key,
                keys_to_infer_DL,
            )

            for nw in range(1, existing_worlds):
                # get inventory realization
                inventory_realization = self.input_inventory.get_world_realization(nw)
                new_prop_tmp, inventory_realization_df = (
                    self.get_era_height_foundation_class(
                        inventory_realization,
                        self.structureType_key,
                        self.yearBuilt_key,
                        self.numberOfStories_key,
                        self.buildingRise_key,
                        self.designLevel_key,
                        self.foundationType_key,
                        keys_to_infer_DL,
                    )
                )
                new_prop = self.merge_two_json(
                    new_prop_tmp, new_prop, shrink=(nw == existing_worlds - 1)
                )

            #
            # Save file
            #
            output_inventory = deepcopy(self.input_inventory)

            for index, feature in new_prop.items():
                output_inventory.add_asset_features(index, feature, overwrite=True)
                # updated = True

        #
        # Change names
        #
        print(">> Step4 : Changing feature names to what R2D (pelicun) can recognize")
        output_inventory.change_feature_names(target_keys_json)

        #
        # Clean_up_unnessesary
        #

        if self.clean_features:
            essential_keys = target_keys_json.values()
            keys_to_remove = set(provided_keys).difference(set(essential_keys))
            output_inventory.remove_features(keys_to_remove)

        elapseEnd = (time.time() - elapseStart) / 60
        print("Done inference. It took {:.2f} mins".format(elapseEnd))
        sys.stdout.flush()  # Forces immediate output to the console

        #
        # Final assessment
        #

        dummy, provided_keys = output_inventory.get_multi_keys()
        missing_keys = set(target_keys_json.values()).difference(set(provided_keys))
        if missing_keys:  # nonempty
            logger.warning(
                f"Unable to create inventory that is readily runnable in R2D (pelicun) due to missing {', '.join(missing_keys)}"
            )

        return output_inventory

    def compute_n_possible_worlds(self, n_possible_worlds):
        existing_worlds = self.input_inventory.get_n_pw()

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

        return n_pw, existing_worlds

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

    def check_keys(
        self, needed_keys, inventory_df, target_key=None, optional_needed_keys=[]
    ):
        provided_keys = inventory_df.columns

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

            if target_key is None:
                logger.warning(
                    f"{not_in_provided_keys} is not something that can be inferred. Unable to create inventory that is readily runnable in R2D (pelicun) due to missing {', '.join(not_in_provided_keys)}. Use either scrapers, image processer, or user inferer to fill in this values."
                )

            else:
                logger.warning(
                    f"The keys needed to estimate {target_key} is not there: {', '.join(not_in_provided_keys)}"
                )
                logger.warning(f"Skipping hazus inference of {target_key}.")
                logger.warning(
                    "Unable to create readily runnable inventory due to missing values. Run imputer to fill in missing values."
                )

            return False

        #
        # Check if input is missing
        #

        ready = True

        for key in needed_keys + optional_needed_keys:
            if (key in optional_needed_keys) and (key not in provided_keys):
                # don't care
                continue

            my_col = inventory_df[key]

            if inventory_df[key].isnull().any():  # if null found
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
            if target_key is None:
                logger.warning(
                    "Unable to create readily runnable inventory due to missing values. Run imputer to fill in missing values."
                )
            else:
                logger.warning(
                    f"Skipping hazus inference of {target_key}. If you still want to perform the inference, run imputer first."
                )
                logger.warning(
                    "Unable to create readily runnable inventory due to missing values. Run imputer to fill in missing values."
                )
            return False

        #
        # warning message for overwritting
        #

        if target_key in provided_keys:
            avail_percentage = (
                100 - sum(inventory_df[key].isnull()) / len(inventory_df[key]) * 100
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

    def get_era_height_foundation_class(
        self,
        input_inventory,
        structureType_key,
        yearBuilt_key,
        numberOfStories_key,
        buildingRise_key,
        designLevel_key,
        foundationType_key,
        keys_to_infer,
    ):
        #
        # convert inventory to df
        #

        bldg_properties_df, bldg_geometries_df, nbldg = input_inventory.get_dataframe()

        #
        # get hazus DL rulesets
        #

        height_classes = {"L": [1, 2, 3], "M": [4, 5, 6], "H": list(range(7, 200))}

        # design_level_W1 = {
        #         "MC": list(range(1,1975)),
        #         "HC": list(range(1975,2030))
        # }

        design_level = {
            "LC": list(range(1, 1940)),
            "MC": list(range(1940, 1975)),
            "HC": list(range(1975, 2030)),
        }

        # height_classes = get_hazus_height_classes()
        # year_classes = get_hazus_year_classes()

        #
        # Add  HeightClass column
        #

        if buildingRise_key in keys_to_infer:
            bldg_properties_df[buildingRise_key] = ""
            for height_class, story_list in height_classes.items():
                in_class_index = bldg_properties_df[numberOfStories_key].isin(
                    story_list
                )
                if sum(in_class_index) > 0:
                    bldg_properties_df.loc[in_class_index, buildingRise_key] = (
                        height_class
                    )
        else:
            bldg_properties_df[buildingRise_key] = ""

        #
        # Add DesignLevel column
        #

        if designLevel_key in keys_to_infer:
            bldg_properties_df[designLevel_key] = ""
            for design_class, year_list in design_level.items():
                in_class_index = bldg_properties_df[yearBuilt_key].isin(year_list)
                if sum(in_class_index) > 0:
                    bldg_properties_df.loc[in_class_index, designLevel_key] = (
                        design_class
                    )
        else:
            bldg_properties_df[designLevel_key] = ""

        design_level = bldg_properties_df[designLevel_key]
        structural_type = bldg_properties_df[structureType_key]

        # Boolean indexing with local variables for faster access
        mask = (design_level == "LC") & (structural_type == "W1")
        # Update in place
        bldg_properties_df.loc[mask, designLevel_key] = "MC"

        #
        # Foundation Column
        #

        if foundationType_key in keys_to_infer:
            bldg_properties_df[foundationType_key] = "S"
        else:
            bldg_properties_df[foundationType_key] = ""

        #
        # This is a bit slow. Hope the below one is faster although it is a lot longer.
        #

        # for i, heightclass, designlevel in zip(bldg_properties_df.index, bldg_properties_df[buildingRise_key], bldg_properties_df[designLevel_key]):
        #    new_prop[index] = {buildingRise_key: heightclass, designLevel_key: designlevel, foundationType_key: 'S'} # if #elem in list is 1, convert it to integer

        # Convert keys_to_infer to a set once for comparison
        keys_to_infer_set = set(keys_to_infer)

        # Mapping of key combinations to the corresponding DataFrame columns
        key_combinations = {
            frozenset([buildingRise_key]): pd.DataFrame(
                {
                    buildingRise_key: bldg_properties_df[buildingRise_key],
                }
            ),
            frozenset([designLevel_key]): pd.DataFrame(
                {
                    designLevel_key: bldg_properties_df[designLevel_key],
                }
            ),
            frozenset([foundationType_key]): pd.DataFrame(
                {
                    foundationType_key: bldg_properties_df[foundationType_key],
                }
            ),
            frozenset([buildingRise_key, designLevel_key]): pd.DataFrame(
                {
                    buildingRise_key: bldg_properties_df[buildingRise_key],
                    designLevel_key: bldg_properties_df[designLevel_key],
                }
            ),
            frozenset([designLevel_key, foundationType_key]): pd.DataFrame(
                {
                    designLevel_key: bldg_properties_df[designLevel_key],
                    foundationType_key: bldg_properties_df[foundationType_key],
                }
            ),
            frozenset([buildingRise_key, foundationType_key]): pd.DataFrame(
                {
                    buildingRise_key: bldg_properties_df[buildingRise_key],
                    foundationType_key: bldg_properties_df[foundationType_key],
                }
            ),
            frozenset(
                [buildingRise_key, designLevel_key, foundationType_key]
            ): pd.DataFrame(
                {
                    buildingRise_key: bldg_properties_df[buildingRise_key],
                    designLevel_key: bldg_properties_df[designLevel_key],
                    foundationType_key: bldg_properties_df[foundationType_key],
                }
            ),
        }

        # Check if keys_to_infer matches any combination and assign the result
        for key_combination, value in key_combinations.items():
            if keys_to_infer_set == key_combination:
                if isinstance(value, pd.DataFrame):
                    new_prop = value.to_dict(orient="index")
                else:
                    new_prop = value.to_dict()

        return new_prop, bldg_properties_df
