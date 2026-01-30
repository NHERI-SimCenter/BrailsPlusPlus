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
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyncoda import ncoda_00h_bldg_archetype_structure as bldg_arch
from pyncoda.ncoda_07i_process_communities import process_community_workflow

from brails.scrapers import CensusTractScraper
from brails.types import AssetInventory, HousingUnit, HousingUnitInventory

# Define the absolute paths to the data files
SCRIPT_DIR = Path(__file__).resolve().parent
FIPS_LOOKUP_PATH = SCRIPT_DIR / 'supporting_data' / 'fips_lookup.json'
STATE_ABBREV_PATH = SCRIPT_DIR / 'supporting_data' / 'state_abbreviations.json'


class PyncodaHousingUnitAllocator:
    """
    Class-based allocator that orchestrates the housing unit assignment workflow.

    This class manages the configuration and execution of the pyncoda-based
    housing unit allocation process. It validates input parameters upon
    initialization and provides a clean interface for assigning synthetic
    housing units to an AssetInventory.

    Attributes:
        inventory (AssetInventory): The bound asset inventory to which housing
            units will be assigned. The same instance is modified in-place by
            the allocator's workflow.
        vintage (str): The census data vintage to use ('2010' or '2020').
        seed (int): Random seed for pyncoda's probabilistic generation.
        key_features (dict): Mapping from required keys to inventory column names.
            It must contain the following keys:
            - 'occupancy_col' (str): Column name for occupancy type (e.g., 'OccupancyClass').
            - 'plan_area_col' (str): Column name for building footprint area.
            - 'story_count_col' (str): Column name for number of stories.
            - 'length_unit' (str): Unit of measurement for length ('ft' or 'm').
        work_dir (Path): The effective working directory used by pyncoda. It is
            created as a subdirectory named 'pyncoda_working_dir' under the
            user-provided base `work_dir` (or under the current working directory
            if no base is provided). All temporary and output files are written
            inside this subdirectory.
    """

    # Working subdirectory name for pyncoda temporary/output files
    PYNCODA_SUBDIR = 'pyncoda_working_dir'

    def __init__(
        self,
        inventory: AssetInventory,
        vintage: str = '2020',
        seed: int = 9877,
        key_features: dict[str, Any] | None = None,
        work_dir: str | Path | None = None,
        clean_work_dir: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the allocator with an inventory and configuration settings.

        Args:
            inventory (AssetInventory): The `AssetInventory` to which housing units
                will be assigned. The same instance is stored on the allocator and
                modified in-place during the allocation workflow.
            vintage (str, optional): Census data vintage to use ('2010' or '2020').
                Defaults to '2020'.
            seed (int, optional): Random seed for pyncoda's probabilistic
                generation. Defaults to 9877.
            key_features (dict, optional): Mapping from required keys to inventory
                column names. Required keys:
                - 'occupancy_col' (str): Column name for occupancy type (e.g., 'OccupancyClass').
                - 'plan_area_col' (str): Column name for building footprint area.
                - 'story_count_col' (str): Column name for number of stories.
                - 'length_unit' (str): Unit of measurement for length ('ft' or 'm').
                Defaults to None.
            work_dir (str or Path, optional): Base directory under which a
                subdirectory will be created to store temporary and output files.
                If not provided, uses current working directory. The final working
                directory will be `(work_dir or cwd) / PYNCODA_SUBDIR`. Defaults
                to None.
            clean_work_dir (bool, optional): Initialization behavior flag. When
                True and the working subdirectory already exists, it is removed
                before being recreated to ensure a clean state. Defaults to True.
        """
        if not isinstance(inventory, AssetInventory):
            raise TypeError('inventory must be an instance of AssetInventory')
        if vintage not in ['2010', '2020']:
            raise ValueError(f'The provided {vintage} vintage is not valid')

        self.inventory = inventory
        self.vintage = vintage
        self.seed = seed
        self.key_features = key_features or {}
        # Resolve working directory as (provided base or cwd) joined with subdir
        base_dir = Path(work_dir) if work_dir is not None else Path.cwd()
        # Validate that base_dir, if it exists, is a directory
        if base_dir.exists() and not base_dir.is_dir():
            raise NotADirectoryError(
                f'The provided work_dir is not a directory: {base_dir}'
            )
        self.work_dir = base_dir / self.PYNCODA_SUBDIR

        # validate key features if provided (some static wrappers may not need it)
        if self.key_features:
            self._validate_key_features(self.key_features)

        # initialize working directory
        self._initialize_working_directory(clean_work_dir)

    # --------------------- Private helpers (static where appropriate) ---------------------
    def _initialize_working_directory(self, clean: bool) -> None:  # noqa: FBT001
        """Ensure the pyncoda working directory exists, optionally cleaning it.

        If `clean` is True and the directory already exists, it will be removed
        before being recreated. The final directory is available at
        `self.work_dir`.
        """
        work_dir_path = Path(self.work_dir)
        if clean and work_dir_path.exists():
            print(
                f'Existing pyncoda working directory found. Removing and recreating: {work_dir_path}'
            )
            shutil.rmtree(work_dir_path)
        work_dir_path.mkdir(parents=True, exist_ok=True)

    def _get_county_and_state_names(
        self,
        census_tracts: list[str],
    ) -> tuple[dict[str, Any], str]:
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

    def _validate_key_features(self, key_features: dict[str, Any]) -> None:
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

        missing_keys: list[str] = []
        type_errors: list[str] = []

        for key, expected_type in required_keys.items():
            if key not in key_features:
                missing_keys.append(key)
                continue  # No need to check type if key is missing

            value = key_features[key]
            if not isinstance(value, expected_type):
                type_errors.append(
                    f"Key '{key}': expected {expected_type.__name__}, got {type(value).__name__}"
                )

            if key == 'length_unit' and value not in ['ft', 'm']:
                type_errors.append(
                    f"Key '{key}': expected 'ft' or 'm', got '{value}'"
                )

        if missing_keys or type_errors:
            error_msg = "Invalid 'key_features' argument:\n"
            if missing_keys:
                error_msg += f'- Missing required keys: {", ".join(missing_keys)}\n'
            if type_errors:
                error_msg += '- Type errors: \n  ' + '\n  '.join(type_errors)
            raise ValueError(error_msg)

    def _prepare_inventory(self) -> gpd.GeoDataFrame:
        """
        Prepare building inventory data for pyncoda.

        Load the building inventory data from the bound AssetInventory object and
        filter the data to include only residential buildings. Then, check for and
        remove buildings with missing required columns. Finally, convert the plan
        area to square feet if necessary and return a GeoDataFrame with the desired
        columns.

        Note:
            All input building geometries are converted to their centroids to
            provide point representations for downstream processing.

        Returns:
            lean_buildings_gdf(GeoDataFrame): Building inventory data with only
                residential buildings and the desired columns for pyncoda.
        """
        key_features = self.key_features

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
        if not self.inventory.inventory:
            print(
                'No buildings found in the provided inventory. Returning empty GeoDataFrame.'
            )
            return _empty_buildings_gdf()

        buildings_geojson = self.inventory.get_geojson()
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
                f"\nAssets dropped due to missing required columns ('{occupancy_col}', '{plan_area_col}', '{story_count_col}', 'geometry'): {len(dropped_for_cols)}"
            )
            print(
                f'  IDs: {dropped_for_cols[:cols_to_show]}{"..." if len(dropped_for_cols) > cols_to_show else ""}'
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

    def _validate_pyncoda_output(self, output_df: pd.DataFrame) -> None:
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

    def _create_housing_unit_inventory(
        self,
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
        self._validate_pyncoda_output(assigned_housing_units)
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
                housing_unit_id,
                HousingUnit(housing_unit_features.dropna().to_dict()),
            )

        return housing_unit_inventory

    # --------------------- Public API ---------------------
    def allocate(self) -> None:
        """
        Assign probabilistic housing units to buildings in the bound AssetInventory.

        This function prepares building inventory data and runs the housing unit
        assignment process using the pyncoda workflow.

        Returns:
            None: The function modifies the bound `AssetInventory` object in-place.

        Raises:
            FileNotFoundError: If required files are not found.
        """
        output_dir = Path(self.work_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_geojson_path = output_dir / 'temp_building_inventory.geojson'

        lean_building_gdf = self._prepare_inventory()
        if lean_building_gdf.empty:
            print(
                'No valid buildings were found to process. Exiting assignment workflow.'
            )
            return
        lean_building_gdf.to_file(temp_geojson_path, driver='GeoJSON')

        # Prepare the Census tract, State, and County information
        filtered_inventory = AssetInventory()
        filtered_inventory.read_from_geojson(temp_geojson_path)

        census_tract_dict = CensusTractScraper().get_census_tracts(
            filtered_inventory
        )
        if not census_tract_dict:
            raise ValueError(
                'No census tracts were found for the filtered inventory; housing units cannot be assigned without census tract information.'
            )
        census_tracts = list(census_tract_dict.keys())
        counties, state_name = self._get_county_and_state_names(census_tracts)

        try:
            community_info = {
                'BRAILS Community': {
                    'community_name': 'BRAILS Community',
                    'focalplace_name': 'BRAILS Community Focal Place',
                    'STATE': state_name.upper(),
                    'years': [self.vintage],
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
                seed=self.seed,
                version='2.1.0',
                version_text='v2-1-0',
                basevintage=self.vintage,
                outputfolder=str(output_dir),
                census_tracts=census_tracts,
                export_hui_path=str(
                    output_dir / 'housing_unit_inventory_filtered.csv'
                ),
                force_hua_rerun=True,
            )
            assigned_housing_units = workflow.process_communities()
        except (OSError, ValueError, TypeError, KeyError, FileNotFoundError) as e:
            print(f'An error occurred while running pyncoda: {e}')
            return

        # Validate pyncoda output columns
        self._validate_pyncoda_output(assigned_housing_units)

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
        assigned_housing_units['gqtype'] = assigned_housing_units['gqtype'].astype(
            int
        )

        housing_unit_inventory = self._create_housing_unit_inventory(
            assigned_housing_units
        )

        assigned_housing_units['housing_unit_id'] = (
            assigned_housing_units.index.copy()
        )
        housing_units_by_bldg_id = assigned_housing_units.groupby('building_id').agg(
            housing_unit_ids=('housing_unit_id', list)
        )

        self.inventory.set_housing_unit_inventory(
            hu_inventory=housing_unit_inventory,
            hu_assignment=housing_units_by_bldg_id['housing_unit_ids'].to_dict(),
            validate=True,
        )


class PyncodaHousingUnitSummarizer:
    """
    Summarizes housing unit information at the building level.

    Attributes:
        inventory (AssetInventory): The inventory containing buildings and housing units.
    """

    def __init__(self, inventory: AssetInventory) -> None:
        """
        Initialize the summarizer.

        Args:
            inventory (AssetInventory): The asset inventory to summarize.
        """
        self.inventory = inventory

    def summarize(  # noqa: C901
        self,
        overwrite: bool = True,  # noqa: FBT001, FBT002
        selected_features: list[str] | None = None,
    ) -> None:
        """
        Aggregate housing unit statistics and attach them to building assets.

        Args:
            overwrite (bool): If True, overwrite existing building features.
            selected_features (list[str] | None): List of features to calculate.
                If None, all available statistics are calculated.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        all_features = [
            'Population',
            'OwnerOccupiedUnits',
            'RenterOccupiedUnits',
            'VacantUnits',
            'GroupQuarters',
            'MeanIncome',
            'MedianIncome',
            'Race',
            'Hispanic',
            'Family',
            'Poverty',
        ]

        # Validate selected_features
        if selected_features is not None:
            invalid_features = set(selected_features) - set(all_features)
            if invalid_features:
                raise ValueError(
                    f'Invalid features requested: {invalid_features}. '
                    f'Available features: {all_features}'
                )
            features_to_calc = selected_features
        else:
            features_to_calc = all_features

        # Check if housing unit inventory exists
        if not hasattr(self.inventory, 'housing_unit_inventory') or (
            self.inventory.housing_unit_inventory is None
        ):
            raise ValueError(
                'AssetInventory has no HousingUnitInventory attached. '
                'Cannot summarize.'
            )

        hu_inv_lookup = self.inventory.housing_unit_inventory.inventory

        for asset_id, asset in self.inventory.inventory.items():
            # Retrieve linked housing units
            hu_ids = asset.features.get('HousingUnits', [])
            units = []
            for uid in hu_ids:
                if uid not in hu_inv_lookup:
                    raise ValueError(
                        f'HousingUnit ID {uid} linked to Asset {asset_id} '
                        'not found in HousingUnitInventory.'
                    )
                units.append(hu_inv_lookup[uid])

            # Prepare stats dictionary
            stats = {}

            # --- Counts ---
            # Population (Sum of NumberOfPersons)
            stats['Population'] = int(
                sum(u.features.get('NumberOfPersons', 0) for u in units)
            )

            # Occupancy Counts
            stats['OwnerOccupiedUnits'] = int(
                sum(
                    1
                    for u in units
                    if u.features.get('Ownership') == 'Owner occupied'
                )
            )
            stats['RenterOccupiedUnits'] = int(
                sum(
                    1
                    for u in units
                    if u.features.get('Ownership') == 'Renter occupied'
                )
            )
            stats['VacantUnits'] = int(
                sum(
                    1
                    for u in units
                    if u.features.get('VacancyStatus', '') != 'Occupied'
                )
            )
            stats['GroupQuarters'] = int(
                sum(
                    1
                    for u in units
                    if u.features.get('GroupQuartersType', '')
                    != 'Not a group quarters building'
                )
            )

            # --- Demographics (Households Only) ---
            # Households are Occupied AND Not Group Quarters
            households = [
                u
                for u in units
                if u.features.get('VacancyStatus') == 'Occupied'
                and u.features.get('GroupQuartersType')
                == 'Not a group quarters building'
            ]

            # Income
            if not households:
                stats['MeanIncome'] = 'N/A'
                stats['MedianIncome'] = 'N/A'
                stats['Race'] = 'N/A'
                stats['Hispanic'] = 'N/A'
                stats['Family'] = 'N/A'
                stats['Poverty'] = 'N/A'
            else:
                # Mean/Median Income
                incomes = [
                    u.features.get('IncomeSample')
                    for u in households
                    if u.features.get('IncomeSample') is not None
                ]
                stats['MeanIncome'] = float(np.mean(incomes))
                stats['MedianIncome'] = float(np.median(incomes))

                # Race
                # Logic: If all consistent -> use value. If mixed -> "Two or More Races"
                races = {u.features.get('Race') for u in households}
                if len(races) == 1:
                    stats['Race'] = races.pop()
                elif len(races) > 1:
                    stats['Race'] = 'Two or More Races'

                # Boolean Maps (Hispanic, Family, Poverty) -> "Mixed" if not uniform
                for key in ['Hispanic', 'Family', 'Poverty']:
                    vals = {u.features.get(key) for u in households}
                    if len(vals) == 1:
                        val = vals.pop()
                        stats[key] = 'Yes' if val else 'No'
                    elif len(vals) > 1:
                        stats[key] = 'Mixed'

            # --- Assignment ---
            # Filter stats to only include selected features
            final_stats = {k: v for k, v in stats.items() if k in features_to_calc}

            if final_stats:
                self.inventory.add_asset_features(
                    asset_id, final_stats, overwrite=overwrite
                )
