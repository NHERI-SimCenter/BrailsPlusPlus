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
This module assigns housing units to residential buildings in an AssetInventory.

It leverages the pyncoda package to generate detailed housing unit demographics
based on census data. The main entry point is the
`assign_housing_units_to_buildings` function, which orchestrates the entire
workflow from data preparation to final assignment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyncoda import ncoda_00h_bldg_archetype_structure as bldg_arch
from pyncoda.ncoda_07i_process_communities import process_community_workflow

from brails.types.asset_inventory import AssetInventory
from brails.types.housing_unit_inventory import HousingUnit, HousingUnitInventory
from brails.utils import Importer

# Define the absolute paths to the data files
SCRIPT_DIR = Path(__file__).resolve().parent
FIPS_LOOKUP_PATH = SCRIPT_DIR / 'supporting_data' / 'fips_lookup.json'
STATE_ABBREV_PATH = SCRIPT_DIR / 'supporting_data' / 'state_abbreviations.json'


def get_county_and_state_names(census_tracts: list) -> Tuple[Dict[str, Any], str]:
    """
    Identify the counties and state covered by an asset inventory.

    This function takes a list of census tract identifiers and extracts the
    5-digit county FIPS codes for every county that has at least one asset in it.
    It then maps these FIPS codes to human-readable names using the
    fips_lookup.json and state_abbreviations.json files.

    Note:
        This function assumes that all input tracts belong to the same state.

    Args:
        census_tracts (list[str]): A list of 11-digit FIPS codes (as strings)
            identifying the census tracts covered by the asset inventory.

    Returns:
        counties (dict): A dictionary providing the FIPS code and the name of
            each county.
        state_name (str): The full name of the state that contains the counties.
    """
    if not isinstance(census_tracts, list):
        raise TypeError('census_tracts must be a list of 11-digit strings.')
    if len(census_tracts) == 0:
        raise ValueError(
            'census_tracts list is empty; provide at least one tract id.'
        )
    invalid_items = [
        x
        for x in census_tracts
        if not isinstance(x, str) or len(x) != 11 or not x.isdigit()  # noqa: PLR2004
    ]
    if invalid_items:
        raise ValueError(
            'All items in census_tracts must be 11-digit strings; invalid items: '
            + ', '.join(map(str, invalid_items))
        )

    with FIPS_LOOKUP_PATH.open(encoding='utf-8') as f:
        fips_names = json.load(f)
    with STATE_ABBREV_PATH.open(encoding='utf-8') as f:
        state_abbreviations = json.load(f)

    unique_county_fips = sorted({tract_id[:5] for tract_id in census_tracts})

    counties_dict = {'counties': {}}
    county_counter = 1

    state_name = 'Unknown State'

    for county_fips in unique_county_fips:
        state_fips = county_fips[:2]

        county_name = fips_names.get(county_fips, 'Unknown County')
        state_name = fips_names.get(state_fips, 'Unknown State')

        state_abbr = state_abbreviations.get(state_name, '')

        formatted_name = (
            f'{county_name}, {state_abbr}' if state_abbr else county_name
        )

        counties_dict['counties'][county_counter] = {
            'FIPS Code': county_fips,
            'Name': formatted_name,
        }
        county_counter += 1

    return counties_dict, state_name


def validate_key_features_dict(key_features: Dict[str, Any]) -> None:
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
        'length_unit': str,
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
                f'got {type(value).__name__}'
            )

        if key == 'length_unit' and value not in ['ft', 'm']:
            type_errors.append(f"Key '{key}': expected 'ft' or 'm', got '{value}'")

    if missing_keys or type_errors:
        error_msg = "Invalid 'key_features' argument:\n"
        if missing_keys:
            error_msg += f'- Missing required keys: {", ".join(missing_keys)}\n'
        if type_errors:
            error_msg += '- Type errors: \n  ' + '\n  '.join(type_errors)
        raise ValueError(error_msg)


def prepare_building_inventory(
    inventory: AssetInventory, key_features: Dict[str, Any]
) -> gpd.GeoDataFrame:
    """
    Prepare building inventory data for pyncoda.

    Load the building inventory data from an AssetInventory object and filter
    the data to include only residential buildings. Then, check for and remove
    buildings with missing required columns. Finally, convert the plan area
    to square feet if necessary and return a GeoDataFrame with the desired
    columns.

    Note:
        All input building geometries are converted to their centroids to
        provide point representations for downstream processing.

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
    if not isinstance(inventory, AssetInventory):
        raise TypeError('inventory must be an instance of AssetInventory')

    # Helper to produce an empty GeoDataFrame with expected structure
    def _empty_buildings_gdf() -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame(
            {
                'building_id': pd.Series(dtype='int64'),
                'geometry': gpd.GeoSeries(dtype='geometry'),
                'occtype': pd.Series(dtype='object'),
                'gross_area_in_sqft': pd.Series(dtype='float64'),
            },
            geometry='geometry',
            crs='EPSG:4326',
        )
        return gdf.set_index('building_id', drop=True)

    # Empty inventory short-circuit
    if not inventory.inventory:
        print(
            'No buildings found in the provided inventory. Returning empty GeoDataFrame.'
        )
        return _empty_buildings_gdf()

    buildings_geojson = inventory.get_geojson()
    features = buildings_geojson.get('features', [])
    if not features:
        print('Inventory geojson has no features. Returning empty GeoDataFrame.')
        return _empty_buildings_gdf()

    buildings_gdf = gpd.GeoDataFrame.from_features(features, crs='EPSG:4326')

    # Pyncoda expects ids in a building_id column
    buildings_gdf = buildings_gdf.rename(columns={'id': 'building_id'})

    print(f'Initial number of buildings loaded: {len(buildings_gdf)}')
    print('-' * 30)

    occupancy_col = key_features['occupancy_col']
    plan_area_col = key_features['plan_area_col']
    story_count_col = key_features['story_count_col']
    length_unit = key_features['length_unit']

    required_cols = [occupancy_col, plan_area_col, story_count_col, 'geometry']
    missing_cols_mask = buildings_gdf[required_cols].isna().any(axis=1)
    dropped_for_cols = buildings_gdf[missing_cols_mask]['building_id'].tolist()

    if len(buildings_gdf) > 0 and missing_cols_mask.all():
        print(
            'All buildings are missing one or more required columns '
            f"('{occupancy_col}', '{plan_area_col}', '{story_count_col}', 'geometry'). "
            'Dropping all buildings and returning empty GeoDataFrame.'
        )
        return _empty_buildings_gdf()

    if dropped_for_cols:
        cols_to_show = 5
        print(
            f'\nAssets dropped due to missing required columns '
            f"('{occupancy_col}', '{plan_area_col}', '{story_count_col}', "
            f"'geometry'): {len(dropped_for_cols)}"
        )
        print(
            f'  IDs: {dropped_for_cols[:cols_to_show]}'
            f'{"..." if len(dropped_for_cols) > cols_to_show else ""}'
        )

    buildings_gdf = buildings_gdf[~missing_cols_mask]

    print(
        f'Number of buildings with all required features present: {len(buildings_gdf)}'
    )

    buildings_gdf = buildings_gdf.loc[
        buildings_gdf[occupancy_col].str.startswith('RES')
    ]

    if buildings_gdf.empty:
        print(
            'No residential buildings found in the inventory. Returning empty GeoDataFrame.'
        )
        return _empty_buildings_gdf()

    print('-' * 30)
    print(f'Number of buildings remaining after filtering: {len(buildings_gdf)}')

    buildings_gdf['building_id'] = buildings_gdf['building_id'].astype(int)
    buildings_gdf[plan_area_col] = buildings_gdf[plan_area_col].astype(float)
    buildings_gdf[story_count_col] = buildings_gdf[story_count_col].astype(float)

    if length_unit == 'm':
        buildings_gdf[plan_area_col] *= 10.7639

    # replace all geometries with their centroids
    buildings_gdf['geometry'] = buildings_gdf['geometry'].centroid

    # Select and rename the columns to match the desired output properties.
    lean_buildings_gdf = buildings_gdf[['building_id', 'geometry']].copy()

    lean_buildings_gdf['occtype'] = buildings_gdf[occupancy_col]
    lean_buildings_gdf['gross_area_in_sqft'] = (
        buildings_gdf[plan_area_col] * buildings_gdf[story_count_col]
    )

    return lean_buildings_gdf.set_index('building_id')


def _validate_pyncoda_output(output_df: pd.DataFrame) -> None:
    """
    Validate that the pyncoda output DataFrame contains all required columns.

    Args:
        output_df (pd.DataFrame): The pyncoda output to validate.

    Raises:
        ValueError: If any of the required columns are missing.
    """
    required_columns = [
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
        'building_id',
    ]

    if not isinstance(output_df, pd.DataFrame):
        raise TypeError('The pyncoda output must be provided as a DataFrame')

    missing = [col for col in required_columns if col not in output_df.columns]
    if missing:
        raise ValueError(
            'Assigned housing units DataFrame is missing required columns: '
            + ', '.join(missing)
        )


def create_housing_unit_inventory(
    assigned_housing_units: pd.DataFrame,
) -> HousingUnitInventory:
    """
    Create a HousingUnitInventory from a DataFrame of housing unit assignments.

    Process the raw pyncoda output into a HousingUnitInventory object by filtering
    the columns we need and converting the labels to SimCenter standard and
    mapping the concise digits to human-readable values.

    Args:
        assigned_housing_units(DataFrame): A DataFrame containing the raw pyncoda
            output of the housing unit assignment.

    Returns:
        housing_unit_inventory(HousingUnitInventory): A HousingUnitInventory object
            containing the housing unit assignments.
    """
    _validate_pyncoda_output(assigned_housing_units)

    if assigned_housing_units.empty:
        return HousingUnitInventory()

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
        'building_id',
    ]
    housing_units = assigned_housing_units[cols_to_keep].copy()

    # we don't need to save building id to housing units
    del housing_units['building_id']

    # rename columns to use standard SimCenter names
    housing_units = housing_units.rename(
        columns={
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
            'poverty': 'Poverty',
        }
    )

    # No need to map Hispanic, Family, and Poverty, just use the raw boolean values
    # No need to map IncomeSample, just use the raw float value

    housing_units['Ownership'] = housing_units['Ownership'].map(
        {1: 'Owner occupied', 2: 'Renter occupied'}
    )

    housing_units['Race'] = housing_units['Race'].map(
        {
            1: 'White',
            2: 'Black',
            3: 'American Indian',
            4: 'Asian',
            5: 'Pacific Islander',
            6: 'Some Other Race',
            7: 'Two or More Races',
        }
    )

    housing_units['VacancyStatus'] = housing_units['VacancyStatus'].map(
        {
            0: 'Occupied',
            1: 'For Rent',
            2: 'Rented, not occupied',
            3: 'For sale only',
            4: 'Sold, not occupied',
            5: 'For seasonal, recreational, or occasional use',
            6: 'For migrant workers',
            7: 'Other vacant',
        }
    )

    housing_units['GroupQuartersType'] = housing_units['GroupQuartersType'].map(
        {
            0: 'Not a group quarters building',
            1: 'Correctional facilities for adults',
            2: 'Juvenile facilities',
            3: 'Nursing facilities/Skilled-nursing facilities',
            4: 'Other institutional facilities',
            5: 'College/University student housing',
            6: 'Military quarters',
            7: 'Other noninstitutional facilities',
        }
    )

    housing_units['IncomeGroup'] = housing_units['IncomeGroup'].map(
        {
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
        }
    )

    # create the housing unit inventory and add the housing units
    housing_unit_inventory = HousingUnitInventory()

    for housing_unit_id, housing_unit_features in housing_units.iterrows():
        housing_unit_inventory.add_housing_unit(
            housing_unit_id, HousingUnit(housing_unit_features.dropna().to_dict())
        )

    return housing_unit_inventory


def assign_housing_units_to_buildings(
    building_inventory: AssetInventory,
    key_features: Dict[str, Any],
    vintage: str,
    output_folder: str,
) -> None:
    """
    Assign synthetic housing units to buildings in an AssetInventory.

    This function prepares building inventory data and runs the housing unit
    assignment process using the pyncoda workflow.

    Args:
        building_inventory (AssetInventory): The AssetInventory object containing
            building data.
        key_features (dict): A dictionary containing the names of the columns
            containing important attributes of each building. The expected keys
            are documented in the validate_key_features_dict function.
        vintage (str): The reference vintage for the housing unit information.
        output_folder (str): Path to directory where temporary and output files
            will be stored.

    Returns:
        None: The function modifies the `building_inventory` object in-place.

    Raises:
        FileNotFoundError: If required files are not found.
    """
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_geojson_path = output_dir / 'temp_building_inventory.geojson'

    lean_building_gdf = prepare_building_inventory(building_inventory, key_features)

    if lean_building_gdf.empty:
        print(
            'No valid buildings were found to process. Exiting assignment workflow.'
        )
        return

    lean_building_gdf.to_file(temp_geojson_path, driver='GeoJSON')

    # Step 2: Prepare the Census tract, State, and County information
    filtered_inventory = AssetInventory()
    filtered_inventory.read_from_geojson(temp_geojson_path)

    # filtered_inventory = AssetInventory()
    # for k, v in building_inventory.inventory.items():
    #    if k in lean_building_gdf.index:
    #        filtered_inventory.add_asset(k, v)

    importer = Importer()
    census_scraper = importer.get_class('CensusScraper')()
    census_tract_dict = census_scraper.get_census_tracts(filtered_inventory)

    if not census_tract_dict:
        raise ValueError(
            'No census tracts were found for the filtered inventory; '
            'housing units cannot be assigned without census tract information.'
        )

    census_tracts = list(census_tract_dict.keys())
    counties, state_name = get_county_and_state_names(census_tracts)

    if vintage not in ['2010', '2020']:
        raise ValueError(f'The provided {vintage} vintage is not valid')

    try:
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
                    'residential_archetypes': bldg_arch.HAZUS_residential_archetypes,
                },
            }
        }

        workflow = process_community_workflow(
            community_info,
            seed=9877,
            version='2.1.0',
            version_text='v2-1-0',
            basevintage=vintage,
            outputfolder=output_folder,
            census_tracts=census_tracts,
            # import_hui_path='housing_unit_inventory_filtered.csv',
            export_hui_path=str(output_dir / 'housing_unit_inventory_filtered.csv'),
            force_hua_rerun=True,
        )

        assigned_housing_units = workflow.process_communities()

    except (OSError, ValueError, TypeError, KeyError, FileNotFoundError) as e:
        print(f'An error occurred while running pyncoda: {e}')
        return

    # 5) Validate pyncoda output columns
    _validate_pyncoda_output(assigned_housing_units)

    # remove unassigned housing units and convert data types
    assigned_housing_units = assigned_housing_units.loc[
        assigned_housing_units['building_id'] != 'missing building id'
    ].copy()

    # After filtering, if no rows remain, exit
    if assigned_housing_units.empty:
        print(
            'No housing units were assigned to buildings. Exiting assignment workflow.'
        )
        return

    assigned_housing_units['building_id'] = (
        assigned_housing_units['building_id'].astype(float).astype(int)
    )
    assigned_housing_units['gqtype'] = assigned_housing_units['gqtype'].astype(int)

    housing_unit_inventory = create_housing_unit_inventory(assigned_housing_units)

    assigned_housing_units['housing_unit_id'] = assigned_housing_units.index.copy()
    housing_units_by_bldg_id = assigned_housing_units.groupby('building_id').agg(
        housing_unit_ids=('housing_unit_id', lambda x: x.tolist())
    )

    building_inventory.set_housing_unit_inventory(
        hu_inventory=housing_unit_inventory,
        hu_assignment=housing_units_by_bldg_id['housing_unit_ids'].to_dict(),
        validate=True,
    )
