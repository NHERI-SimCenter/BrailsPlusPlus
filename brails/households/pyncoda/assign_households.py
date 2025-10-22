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
from typing import Dict, List, Any

import pandas as pd
import geopandas as gpd
import json
from pathlib import Path

from brails.types.asset_inventory import AssetInventory
from brails.utils import Importer
from pyncoda.ncoda_07i_process_communities import process_community_workflow
from pyncoda import ncoda_00h_bldg_archetype_structure as bldg_arch

# Define the absolute paths to the data files
SCRIPT_DIR = Path(__file__).resolve().parent
FIPS_LOOKUP_PATH = SCRIPT_DIR / 'supporting_data' / 'fips_lookup.json'
STATE_ABBREV_PATH = SCRIPT_DIR / 'supporting_data' / 'state_abbreviations.json'

def get_county_and_state_names(census_tracts: list):
    """
    Identify the counties and state covered by an asset inventory.

    This function uses the 'TRACT_GEOID' column in the input GeoDataFrame to
    extract the 5-digit county FIPS codes of every county that has at least one
    asset in it. It then maps these FIPS codes to human-readable names using
    the fips_lookup.json and state_abbreviations.json files.

    Parameters
    ----------
    census_tracts : list
        A list of 11-digit FIPS codes identifying the census tracts covered by
        the asset inventory.

    Returns
    -------
    counties : dict
        A dictionary providing the FIPS code and the name of each county.
    state_name: str
        The full name of the state that contains the counties.
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

def assign_households_to_buildings(
    inventory: AssetInventory,
    occupancy_col: str,
    plan_area_col: str,
    length_unit: str,
    vintage: str,
    output_folder: str
) -> Dict[str, Any]:
    """
    Assign synthetic households to buildings in an AssetInventory.

    This function prepares building inventory data and runs the household 
    assignment process using the pyncoda workflow.

    Args:
        inventory (AssetInventory): The AssetInventory object containing 
            building data.
        occupancy_col (str): The name of the feature containing occupancy type 
            (e.g., "OccupancyClass").
        plan_area_col (str): The name of the feature for the building's plan area
            (e.g., "PlanArea").
        length_unit (str): The length unit for plan area. Must be "ft" or "m".
        vintage (str): The reference vintage for the household information.
        output_folder (str): Path to directory where temporary and output files 
            will be stored.

    Returns:
        Dict[str, Any]: Result from the pyncoda workflow process.

    Raises:
        ValueError: If length_unit is not "ft" or "m".
        FileNotFoundError: If required files are not found.
    """
    # Validate length_unit parameter
    if length_unit not in ["ft", "m"]:
        raise ValueError("length_unit must be either 'ft' or 'm'")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Prepare Building Inventory Input
    # Create temporary GeoJSON file with filtered properties
    temp_geojson_path = os.path.join(output_folder, "temp_building_inventory.geojson")

    geojson_data = inventory.get_geojson()
    asset_gdf = gpd.GeoDataFrame.from_features(
        geojson_data["features"],
        crs="EPSG:4326"
    )
    asset_gdf.set_index('id', inplace=True)

    print(f"Initial number of assets loaded: {len(asset_gdf)}")
    print("-" * 30)

    # Pyncoda expects ids in a building_id column
    asset_gdf = asset_gdf.rename(columns={'id': 'building_id'})

    # Report and drop assets with missing required columns
    required_cols = [occupancy_col, plan_area_col, 'geometry']
    missing_cols_mask = asset_gdf[required_cols].isnull().any(axis=1)
    dropped_for_cols = asset_gdf[missing_cols_mask]['building_id'].tolist()

    if dropped_for_cols:
        print(f"\nAssets dropped due to missing required columns ('{occupancy_col}', '{plan_area_col}', etc.): {len(dropped_for_cols)}")
        print(f"  IDs: {dropped_for_cols[:5]}{'...' if len(dropped_for_cols) > 5 else ''}")

    # Keep only the rows that are NOT in the missing_cols_mask
    asset_gdf = asset_gdf[~missing_cols_mask]

    print("-" * 30)
    print(f"Number of assets remaining after filtering: {len(asset_gdf)}")

    # Ensure columns have the correct data type for calculations
    asset_gdf['building_id'] = asset_gdf['building_id'].astype(int)
    asset_gdf[plan_area_col] = asset_gdf[plan_area_col].astype(float)

    # Perform the unit conversion on the plan area column
    if length_unit == "m":
        asset_gdf['plan_area_in_sqft'] = asset_gdf[plan_area_col] * 10.7639
    else:
        asset_gdf['plan_area_in_sqft'] = asset_gdf[plan_area_col]

    # Select and rename the columns to match the desired output properties.
    lean_asset_gdf = asset_gdf[[
        'building_id',
        'geometry'
    ]].copy() # Use .copy() to avoid SettingWithCopyWarning

    lean_asset_gdf['occtype'] = asset_gdf[occupancy_col]
    lean_asset_gdf['plan_area_in_sqft'] = asset_gdf['plan_area_in_sqft']

    lean_asset_gdf.set_index('building_id', inplace=True)
    lean_asset_gdf.to_file(temp_geojson_path, driver='GeoJSON')

    # Step 2: Prepare the Census tract, State, and County information

    # Create a lean inventory that only has the assets that were kept
    filtered_inventory = AssetInventory()
    filtered_inventory.inventory = {
        k: v for k,v in inventory.inventory.items() if k in lean_asset_gdf.index
    }

    # Run scraper to identify the census tracts covered by the assets
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
                'building_area_var': 'plan_area_in_sqft',
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
        force_hua_rerun=True
    )

    # Process communities and return results
    hua_hui_gdf_dict = workflow.process_communities()

    return hua_hui_gdf_dict