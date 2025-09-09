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
from brails.inferers.inference_engine import InferenceEngine
import reverse_geocode


# To be replaced with old brails ++ codes
import warnings
import logging

# Configure logging:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)


class HazusInfererFlood(InferenceEngine):
    """
    Make inference based on Hazus 6 rulesets

    Attributes:

    Methods:


    """

    def __init__(
        self,
        input_inventory: AssetInventory,
        n_possible_worlds=1,
        include_features=None,
        seed=1,
        overwirte_existing=True,
        clean_features=False,
        city_key="City",
        yearBuilt_key="YearBuilt",
        floodZone_key="FloodZone",
        firstFloorElevation_key="FirstFloorElevation",
        splitLevel_key="SplitLevel",
        basement_key="Basement",
        occupancyClass_key="OccupancyClass",
        postFIRM_key="PostFIRM",
        floodType_key="FloodType",
        basementType_key="BasementType",
        numberOfStories_key="NumberOfStories",
        PostFIRM_year_by_city=None
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
            city_key: "City",
            yearBuilt_key: "YearBuilt",
            floodZone_key: "FloodZone",
            firstFloorElevation_key: "FirstFloorElevation",
            splitLevel_key: "SplitLevel",
            basement_key: "Basement",
            occupancyClass_key: "OccupancyClass",
            postFIRM_key: "PostFIRM",
            floodType_key: "FloodType",
            basementType_key: "BasementType",
            numberOfStories_key: "NumberOfStories",
        }

        self.clean_features = clean_features

        if PostFIRM_year_by_city == None:
            # we provide default for NJ
            self.PostFIRM_year_by_city = {
                'Absecon': 1976,
                'Atlantic': 1971,
                'Brigantine': 1971,
                'Buena': 1983,
                'Buena Vista': 1979,
                'Corbin City': 1981,
                'Egg Harbor City': 1982,
                'Egg Harbor': 1983,
                'Estell Manor': 1978,
                'Folsom': 1982,
                'Galloway': 1983,
                'Hamilton': 1977,
                'Hammonton': 1982,
                'Linwood': 1983,
                'Longport': 1974,
                'Margate City': 1974,
                'Mullica': 1982,
                'Northfield': 1979,
                'Pleasantville': 1983,
                'Port Republic': 1983,
                'Somers Point': 1982,
                'Ventnor City': 1971,
                'Weymouth': 1979
            }
        else:
            self.PostFIRM_year_by_city = PostFIRM_year_by_city

    def infer(self) -> AssetInventory:

        input_inventory = self.input_inventory
        n_possible_worlds = self.n_possible_worlds

        if n_possible_worlds == 0:
            msg = f"ERROR: most likely attributes option is not supported for the flood inferer. Please select n_possible_worlds>0"
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
                f"The existing inventory does not contain multiple possible worlds. {n_pw} world(s) will be generated for new features"
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

            # Infer building one by one
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
                    essential_features = self.auto_populate(bldg)
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

    def auto_populate(self, inventory):

        BIM = inventory["properties"]

        available_features = BIM.keys()

        #
        # Infer city
        #

        if "City" in BIM:
            city = BIM["City"]
        else:
            geo_coord = inventory["geometry"]["coordinates"][0]
            geo_locs = reverse_geocode.search([[geo_coord[1], geo_coord[0]]])
            city = geo_locs[0]["city"]

        # to improve competability between the provided value and dictionary list

        #
        # Infer postFIRM
        #
        city = city.lower().replace(' ', '').replace('city', '')

        self.PostFIRM_year_by_city = {key.lower().replace(' ', '').replace(
            'city', ''): value for key, value in self.PostFIRM_year_by_city.items()}
        if not (city in self.PostFIRM_year_by_city):
            logger.warning(
                f"PostFIRM information not provided. Setting conservative condition PostFIRM = False"
            )
            # print("Warining")
            # print(city)
            # sys.exit(1)
            PostFIRM = False
        elif self.is_ready_to_infer(available_features, ['YearBuilt'], 'PostFIRM'):
            PostFIRM_year = self.PostFIRM_year_by_city[city]
            PostFIRM = BIM['YearBuilt'] > PostFIRM_year

        # sys.exit(-1)
        #
        # Infer postFIRM
        #

        # TODO: obtain splitlevel
        # Basement Type

        # For the foundation type, let's follow the ruleset from NSI (e.g. I) instead of NJDEP (e.g.3501)
        # C = 3505 = Crawl, B = 3504 = Basement, S = 3507 = Slab, P = 3502 = Pier, I = 3501 = Pile, F = 3506 = Fill, W = 3503 = Solid Wall

        if "BasementType" in BIM:
            basement_type = BIM["BasementType"]

        # elif self.is_ready_to_infer(available_features,['SplitLevel','FoundationType'],'BasementType'):
        #     if BIM['SplitLevel']=='Yes' and (BIM['FoundationType'] == 'B'):
        #     #if BIM['SplitLevel'] and (BIM['FoundationType'] == 3504):
        #         basement_type = 'spt' # Split-Level Basement
        #     elif BIM['FoundationType'] in ['I', 'P', 'W', 'C', 'F', 'S']:
        #     #elif BIM['FoundationType'] in [3501, 3502, 3503, 3505, 3506, 3507]:
        #         basement_type = 'bn' # No Basement
        #     elif (BIM['SplitLevel']=='No') and (BIM['FoundationType'] == 'B'):
        #     #elif (not BIM['SplitLevel']) and (BIM['FoundationType'] == 3504):
        #         basement_type = 'bw' # Basement
        #     else:
        #         logger.warning(
        #             f"FoundationType {BIM['FoundationType']} not recognized. Assuming conservative condition with a basement (bw)"
        #         )
        #         basement_type = 'bw' # Default

        elif self.is_ready_to_infer(available_features, ['SplitLevel', 'Basement'], 'BasementType'):
            if BIM['SplitLevel'] == 'Yes' and (BIM['Basement'] == 'Yes'):
                basement_type = 'spt'  # Split-Level Basement
            elif BIM['Basement'] == 'No':
                basement_type = 'bn'  # No Basement
            elif (BIM['SplitLevel'] == 'No') and (BIM['Basement'] == 'Yes'):
                basement_type = 'bw'  # Basement
            else:
                logger.warning(
                    f"FoundationType {BIM['FoundationType']} not recognized. Assuming conservative condition with a basement (bw)"
                )
                basement_type = 'bw'  # Default

        #
        # Flood Type
        #

        if "FloodType" in BIM:
            flood_type = BIM["FloodType"]

        elif self.is_ready_to_infer(available_features, ['FloodZone'], 'FloodType'):
            if BIM['FloodZone'] == 'AO':
                flood_type = 'raz'  # Riverine/A-Zone
            elif BIM['FloodZone'] in ['A', 'AE', 'AH']:
                flood_type = 'caz'  # Costal-Zone A
            elif BIM['FloodZone'].startswith('V'):
                flood_type = 'cvz'  # Costal-Zone V
            else:
                logger.warning(
                    f"FloodZone {BIM['FloodZone']} not recognized. Setting default FloodType Costal-Zone A (caz)"
                )
                flood_type = 'caz'  # Default

        #
        # Infer first floor elevation
        #
        if "EffectiveFirstFloorElevation" in BIM:
            FFE = BIM["EffectiveFirstFloorElevation"]

        elif self.is_ready_to_infer(available_features, ['FirstFloorElevation'], 'EffectiveFirstFloorElevation'):
            if flood_type in ['raz', 'caz']:
                FFE = BIM['FirstFloorElevation']
            else:
                FFE = BIM['FirstFloorElevation'] - 1.0

        self.is_ready_to_infer(available_features=available_features, needed_features=[
                               'OccupancyClass', 'SplitLevel', 'NumberOfStories'], inferred_feature="Hazus flood properties")

        essential_features = dict(
            EffectiveFirstFloorElevation=FFE,
            OccupancyClass=BIM['OccupancyClass'],
            FloodType=flood_type,
            SplitLevel=BIM['SplitLevel'],
            NumberOfStories=BIM['NumberOfStories'],
            BasementType=basement_type,
        )

        return essential_features

    def to_json(self, this_inventory):

        this_inventory.convert_polygons_to_centroids()
        # inventory_json = this_inventory.get_geojson()

        inventory_json = {}
        for key, asset in this_inventory.inventory.items():
            geometry = {"type": "Point", "coordinates": [
                asset.coordinates[0][:]]}
            # if len(asset.coordinates) == 1:
            #    geometry = {"type": "Point", "coordinates": [asset.coordinates[0][:]]}
            # else:
            #    geometry = {"type": "Polygon", "coordinates": asset.coordinates}

            feature = {
                "type": "Feature",
                "properties": asset.features,
                "geometry": geometry,
            }
            if "type" in asset.features:
                feature["type"] = asset.features["type"]

            inventory_json[key] = feature

        return inventory_json

    def is_ready_to_infer(self, available_features, needed_features, inferred_feature):

        missing_keys = set(needed_features).difference(set(available_features))

        if missing_keys:
            msg = f"You need {missing_keys} to infer '{inferred_feature}'. You only have {available_features}."
            raise ValueError(msg)

        return True
