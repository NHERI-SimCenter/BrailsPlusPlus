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
This module assigns households to housing units using the pyncoda package

"""

import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path

from brails.types.asset_inventory import AssetInventory
from brails.types.household_inventory import Household, HouseholdInventory
from brails.utils import Importer
from pyncoda.ncoda_07i_process_communities import process_community_workflow
from pyncoda import ncoda_00h_bldg_archetype_structure as bldg_arch

# Define the absolute paths to the data files
SCRIPT_DIR = Path(__file__).resolve().parent
FIPS_LOOKUP_PATH = SCRIPT_DIR / 'supporting_data' / 'fips_lookup.json'
STATE_ABBREV_PATH = SCRIPT_DIR / 'supporting_data' / 'state_abbreviations.json'

def get_county_and_state_names(census_tracts: list)-> Tuple[Dict[str, Any], str]:
    """
    Identify the counties and state covered by an asset inventory.

    This function uses the 'TRACT_GEOID' column in the input GeoDataFrame to
    extract the 5-digit county FIPS codes of every county that has at least one
    asset in it. It then maps these FIPS codes to human-readable names using
    the fips_lookup.json and state_abbreviations.json files.

    Args:
        census_tracts(list): A list of 11-digit FIPS codes identifying the
            census tracts covered by the asset inventory.

    Returns:
        counties(dict): A dictionary providing the FIPS code and the name of
            each county.
        state_name(str): The full name of the state that contains the counties.
    """

    # --- Load the lookup tables from the JSON files ---
    with open(FIPS_LOOKUP_PATH, 'r', encoding='utf-8') as f:
        fips_names = json.load(f)
    with open(STATE_ABBREV_PATH, 'r', encoding='utf-8') as f:
        state_abbreviations = json.load(f)

    # --- Extract unique 5-digit county FIPS codes from the census tract list ---
    unique_county_fips = sorted(list({tract_id[:5] for tract_id in census_tracts}))

    # --- Build the output dictionary ---
    counties_dict = {'counties': {}}
    county_counter = 1

    for fips_code in unique_county_fips:
        state_fips = fips_code[:2]

        # Look up the county and state names from the fips_names dictionary
        county_name = fips_names.get(fips_code, "Unknown County")
        state_name = fips_names.get(state_fips, "Unknown State")

        # Look up the state abbreviation
        state_abbr = state_abbreviations.get(state_name, "")

        # Format the final name string
        formatted_name = f"{county_name}, {state_abbr}" if state_abbr else county_name

        # Add the entry to the dictionary
        counties_dict['counties'][county_counter] = {
            'FIPS Code': fips_code,
            'Name': formatted_name
        }
        county_counter += 1

    return counties_dict, state_name

def validate_key_features_dict(key_features: Dict[str, Any]):
    """
    Validate the key_features dictionary.

    Check that the required keys are present and have the correct data type.

    Args:
        key_features(dict): This dictionary helps keep inventory feature naming
            flexible by allowing users to specify the labels they use for
            important features and their length unit. We expect the following
            keys in the dictionary: occupancy_col, plan_area_col,
            story_count_col, length_unit. For each key, we expect a string
            value. Currently, only 'ft' and 'm' are supported as length units.

    Raises:
        ValueError: If any of the required keys are missing, have an incorrect
            data type, or the provided length unit is not supported.
    """
    required_keys = {
        'occupancy_col': str,
        'plan_area_col': str,
        'story_count_col': str,
        'length_unit': str
    }

    missing_keys = []
    type_errors = []

    for key, expected_type in required_keys.items():
        if key not in key_features:
            missing_keys.append(key)
            continue  # No need to check type if key is missing

        value = key_features[key]
        if not isinstance(value, expected_type):
            type_errors.append(
                f"Key '{key}': expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

        if key == 'length_unit' and value not in ['ft', 'm']:
            type_errors.append(
                f"Key '{key}': expected 'ft' or 'm', got '{value}'"
            )

    # Consolidate errors and raise a single, informative message
    if missing_keys or type_errors:
        error_msg = "Invalid 'key_features' argument:\n"
        if missing_keys:
            error_msg += f"- Missing required keys: {', '.join(missing_keys)}\n"
        if type_errors:
            error_msg += f"- Type errors: \n  " + "\n  ".join(type_errors)
        raise ValueError(error_msg)

def prepare_building_inventory(
        inventory: AssetInventory,
        key_features: Dict[str, Any]
)-> gpd.GeoDataFrame:
    """
    Prepare building inventory data for pyncoda.

    Load the building inventory data from an AssetInventory object and filter
    the data to include only residential buildings. Then, check for and remove
    buildings with missing required columns. Finally, convert the plan area
    to square feet if necessary and return a GeoDataFrame with the desired
    columns.

    Args:
        inventory(AssetInventory): The AssetInventory object containing
            building data.
        key_features(dict): A dictionary containing the names of the columns
            containing important attributes of each building. The expected keys
            are documented in the validate_key_features_dict function.

    Returns:
        lean_buildings_gdf(GeoDataFrame): Building inventory data with only
            residential buildings and the desired columns for pyncoda.
    """
    validate_key_features_dict(key_features)

    buildings_geojson = inventory.get_geojson()
    buildings_gdf = gpd.GeoDataFrame.from_features(
        buildings_geojson["features"],
        crs="EPSG:4326"
    )
    buildings_gdf.set_index('id', inplace=True)

    print(f"Initial number of buildings loaded: {len(buildings_gdf)}")
    print("-" * 30)

    # Pyncoda expects ids in a building_id column
    buildings_gdf = buildings_gdf.rename(columns={'id': 'building_id'})

    occupancy_col = key_features['occupancy_col']
    plan_area_col = key_features['plan_area_col']
    story_count_col = key_features['story_count_col']
    length_unit = key_features['length_unit']

    # Keep only the residential buildings
    buildings_gdf = buildings_gdf.loc[buildings_gdf[occupancy_col].str.startswith('RES')]

    print(f"Number of residential buildings in the inventory: {len(buildings_gdf)}")

    # Report and drop buildings with missing required columns
    required_cols = [occupancy_col, plan_area_col, story_count_col, 'geometry']
    missing_cols_mask = buildings_gdf[required_cols].isnull().any(axis=1)
    dropped_for_cols = buildings_gdf[missing_cols_mask]['building_id'].tolist()

    if dropped_for_cols:
        print(f"\nAssets dropped due to missing required columns ('{occupancy_col}', '{plan_area_col}', '{story_count_col}', 'geometry'): {len(dropped_for_cols)}")
        print(f"  IDs: {dropped_for_cols[:5]}{'...' if len(dropped_for_cols) > 5 else ''}")

    # Keep only the rows that are NOT in the missing_cols_mask
    buildings_gdf = buildings_gdf[~missing_cols_mask]

    print("-" * 30)
    print(f"Number of buildings remaining after filtering: {len(buildings_gdf)}")

    # Ensure columns have the correct data type for calculations
    buildings_gdf['building_id'] = buildings_gdf['building_id'].astype(int)
    buildings_gdf[plan_area_col] = buildings_gdf[plan_area_col].astype(float)
    buildings_gdf[story_count_col] = buildings_gdf[story_count_col].astype(float)

    # Perform the unit conversion on the plan area column
    if length_unit == "m":
        buildings_gdf[plan_area_col] *= 10.7639

    # Select and rename the columns to match the desired output properties.
    lean_buildings_gdf = buildings_gdf[[
        'building_id',
        'geometry'
    ]].copy()

    lean_buildings_gdf['occtype'] = buildings_gdf[occupancy_col]
    lean_buildings_gdf['gross_area_in_sqft'] = (
        buildings_gdf[plan_area_col] * buildings_gdf[story_count_col])

    lean_buildings_gdf.set_index('building_id', inplace=True)

    return lean_buildings_gdf


def create_household_inventory(
        assigned_households: pd.DataFrame
)->HouseholdInventory:
    """
    Create a HouseholdInventory object from a DataFrame of household assignments.

    Process the raw pyncoda output into a HouseholdInventory object by filtering
    the columns we need and converting the labels to SimCenter standard and
    mapping the concise digits to human-readable values.

    Args:
        assigned_households(DataFrame): A DataFrame containing the raw pyncoda
            output of the household assignment.

    Returns:
        household_inventory(HouseholdInventory): A HouseholdInventory object
            containing the household assignments.
    """
    # keep only the columns we need from the raw pyncoda output
    cols_to_keep = [
        'blockid',
        'numprec',
        'ownershp',
        'race',
        'hispan',
        'family',
        'vacancy',
        'gqtype',
        'incomegroup',
        'randincome',
        'poverty',
        'building_id'
    ]
    households = assigned_households[cols_to_keep].copy()

    # we don't need to save building id to households
    del households['building_id']

    # rename columns to use standard SimCenter names
    households.rename(columns={
        'blockid': 'BLOCK_GEOID',
        'numprec': 'NumberOfPersons',
        'ownershp': 'Ownership',
        'race': 'Race',
        'hispan': 'Hispanic',
        'family': 'Family',
        'vacancy': 'VacancyStatus',
        'gqtype': 'GroupQuartersType',
        'incomegroup': 'IncomeGroup',
        'randincome': 'IncomeSample',
        'poverty': 'Poverty'
    }, inplace=True)

    # Map digits to actual values for clarity

    # No need to map Hispanic, Family, and Poverty, just use the raw boolean values
    # No need to map IncomeSample, just use the raw float value

    households['Ownership'] = households['Ownership'].map({
        1: 'Owner occupied',
        2: 'Renter occupied'
    })

    households['Race'] = households['Race'].map({
        1: 'White',
        2: 'Black',
        3: 'American Indian',
        4: 'Asian',
        5: 'Pacific Islander',
        6: 'Some Other Race',
        7: 'Two or More Races'
    })

    households['VacancyStatus'] = households['VacancyStatus'].map({
        0: 'Occupied',
        1: 'For Rent',
        2: 'Rented, not occupied',
        3: 'For sale only',
        4: 'Sold, not occupied',
        5: 'For seasonal, recreational, or occasional use',
        6: 'For migrant workers',
        7: 'Other vacant'
    })

    households['GroupQuartersType'] = households['GroupQuartersType'].map({
        0: 'Not a group quarters building',
        1: 'Correctional facilities for adults',
        2: 'Juvenile facilities',
        3: 'Nursing facilities/Skilled-nursing facilities',
        4: 'Other institutional facilities',
        5: 'College/University student housing',
        6: 'Military quarters',
        7: 'Other noninstitutional facilities'
    })

    households['IncomeGroup'] = households['IncomeGroup'].map({
        0: np.nan,
        1: 'Less than $10,000',
        2: '$10,000 to $14,999',
        3: '$15,000 to $19,999',
        4: '$20,000 to $24,999',
        5: '$25,000 to $29,999',
        6: '$30,000 to $34,999',
        7: '$35,000 to $39,999',
        8: '$40,000 to $44,999',
        9: '$45,000 to $49,999',
        10: '$50,000 to $59,999',
        11: '$60,000 to $74,999',
        12: '$75,000 to $99,999',
        13: '$100,000 to $124,999',
        14: '$125,000 to $149,999',
        15: '$150,000 to $199,999',
        16: '$200,000 or more',
    })

    # create the household inventory and add the households
    household_inventory = HouseholdInventory()

    for household_id, household_features in households.iterrows():
        household_inventory.add_household(
            household_id,
            Household(household_features.dropna().to_dict()))

    return household_inventory

def assign_households_to_buildings(
    building_inventory: AssetInventory,
    key_features: Dict[str, Any],
    vintage: str,
    output_folder: str
):
    """
    Assign synthetic households to buildings in an AssetInventory.

    This function prepares building inventory data and runs the household 
    assignment process using the pyncoda workflow.

    Args:
        building_inventory (AssetInventory): The AssetInventory object containing
            building data.
        key_features (dict): A dictionary containing the names of the columns
            containing important attributes of each building. The expected keys
            are documented in the validate_key_features_dict function.
        vintage (str): The reference vintage for the household information.
        output_folder (str): Path to directory where temporary and output files 
            will be stored.

    Raises:
        FileNotFoundError: If required files are not found.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Prepare Building Inventory Input
    # Create a temporary GeoJSON file for the pyncoda-optimized inventory
    temp_geojson_path = os.path.join(
        output_folder,
        "temp_building_inventory.geojson"
    )

    lean_building_gdf = prepare_building_inventory(
        building_inventory,
        key_features
    )
    lean_building_gdf.to_file(temp_geojson_path, driver='GeoJSON')

    # Step 2: Prepare the Census tract, State, and County information

    # Create a lean inventory that only has the buildings that were kept
    filtered_inventory = AssetInventory()
    filtered_inventory.inventory = {
        k: v for k,v in building_inventory.inventory.items() if k in lean_building_gdf.index
    }

    # Run scraper to identify the census tracts covered by the buildings
    importer = Importer()
    census_scraper = importer.get_class('CensusScraper')()
    census_tract_dict = census_scraper.get_census_tracts(filtered_inventory)
    census_tracts = list(census_tract_dict.keys())

    # Get county list and state name
    counties, state_name = get_county_and_state_names(census_tracts)

    # Step 2: Run Household Assignment
    # Set up community info
    if vintage not in ['2010', '2020']:
        raise ValueError(f"The provided {vintage} vintage is not valid")

    community_info = {
        'BRAILS Community': {
            'community_name': 'BRAILS Community',
            'focalplace_name': 'BRAILS Community Focal Place',
            'STATE': state_name.upper(),
            'years': [vintage],
            'counties': counties['counties'],
            'building_inventory': {
                'filename': temp_geojson_path,
                'note': 'Community for BRAILS Building Inventory',
                'archetype_var': 'occtype',
                'bldg_uniqueid': 'building_id',
                'building_area_var': 'gross_area_in_sqft',
                'building_area_cutoff': 300,
                'use_incore': False,
                'id': 'filtered NSI',
                'residential_archetypes': bldg_arch.HAZUS_residential_archetypes
            }
        }
    }

    # Create the workflow
    workflow = process_community_workflow(
        community_info,
        seed=9877,
        version='2.1.0',
        version_text='v2-1-0',
        basevintage=vintage,
        outputfolder=output_folder,
        census_tracts=census_tracts,
        #import_hui_path='housing_unit_inventory_filtered.csv',
        export_hui_path=os.path.join(output_folder, 'housing_unit_inventory_filtered.csv'),
        force_hua_rerun=True
    )

    # Process communities and return results
    assigned_households = workflow.process_communities()

    # remove unassigned households and convert data types
    assigned_households = assigned_households.loc[assigned_households['building_id'] != 'missing building id'].copy()
    assigned_households['building_id'] = assigned_households['building_id'].astype(float).astype(int)
    assigned_households['gqtype'] = assigned_households['gqtype'].astype(int)

    household_inventory = create_household_inventory(assigned_households)

    assigned_households['household_id'] = assigned_households.index.copy()
    households_by_bldg_id = assigned_households.groupby('building_id').agg(
        household_ids = ('household_id', lambda x: x.tolist())
    )
    for building_id, household_ids in households_by_bldg_id.iterrows():
        building_inventory.add_asset_features(
            asset_id = building_id,
            new_features = dict(Households = household_ids['household_ids'])
        )

    building_inventory.household_inventory = household_inventory