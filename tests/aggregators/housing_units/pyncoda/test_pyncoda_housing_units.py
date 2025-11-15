"""
Test suite for the pyncoda_housing_units module.

This suite includes:
- Unit tests for individual helper functions.
- Mocked integration tests for the main workflow, covering various scenarios
  and failure modes.
- A "live" integration test that runs the actual pyncoda process on a
  controlled dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal as pdt_assert_frame_equal
from shapely.geometry import Point

from brails.aggregators.housing_units.pyncoda import pyncoda_housing_units as ah
from brails.types.asset_inventory import Asset, AssetInventory
from brails.types.housing_unit_inventory import HousingUnitInventory

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def valid_key_features() -> dict[str, str]:
    """Return a valid baseline key_features dictionary."""
    return {
        'occupancy_col': 'Occupancy',
        'plan_area_col': 'PlanArea',
        'story_count_col': 'Stories',
        'length_unit': 'ft',
    }


# -----------------------------
# Tests: validate_key_features_dict
# -----------------------------


def test_validate_key_features_dict_accepts_valid_input(
    valid_key_features: dict[str, str],
) -> None:
    """It should not raise when called with a valid dictionary."""
    # Arrange
    data = valid_key_features

    # Act / Assert (no exception)
    ah.validate_key_features_dict(data)


@pytest.mark.parametrize(
    'broken_data',
    [
        pytest.param(  # Missing key
            {'occupancy_col': 'Occ', 'plan_area_col': 'Area', 'length_unit': 'ft'},
            id='missing_story_count_col',
        ),
        pytest.param(  # Wrong type for a key
            {
                'occupancy_col': 123,
                'plan_area_col': 'Area',
                'story_count_col': 'Stories',
                'length_unit': 'ft',
            },
            id='wrong_type_for_occupancy_col',
        ),
        pytest.param(  # Unsupported length unit
            {
                'occupancy_col': 'Occ',
                'plan_area_col': 'Area',
                'story_count_col': 'Stories',
                'length_unit': 'cm',
            },
            id='unsupported_length_unit',
        ),
    ],
)
def test_validate_key_features_dict_raises_for_single_issue(
    broken_data: dict[str, Any],
) -> None:
    """It should raise ValueError for each single, isolated validation issue.

    Uses parameterization to cover different invalid dictionaries.
    """
    # Act / Assert
    with pytest.raises(ValueError, match="Invalid 'key_features' argument"):
        ah.validate_key_features_dict(broken_data)


def test_validate_key_features_dict_reports_all_errors(
    valid_key_features: dict[str, str],
) -> None:
    """It should report multiple errors in a single, consolidated message."""
    # Arrange: introduce multiple issues
    data = dict(valid_key_features)
    data.pop('story_count_col')  # missing key
    data['plan_area_col'] = 999  # wrong type
    data['length_unit'] = 'cm'  # unsupported unit

    # Act / Assert
    with pytest.raises(
        ValueError, match="Invalid 'key_features' argument"
    ) as exc_info:
        ah.validate_key_features_dict(data)

    msg = str(exc_info.value)

    # Assert message contains all discrete problems (format-sensitive but resilient)
    assert "Invalid 'key_features' argument:" in msg
    assert 'Missing required keys' in msg
    assert 'story_count_col' in msg
    assert 'Type errors:' in msg
    assert "Key 'plan_area_col': expected str, got int" in msg
    assert "Key 'length_unit': expected 'ft' or 'm', got 'cm'" in msg


# -----------------------------
# Tests: get_county_and_state_names
# -----------------------------


def test_get_county_and_state_names_multiple_counties_same_state() -> None:
    """It returns correct counties dict and state name for tracts across counties in one state."""
    # Arrange: Two counties within California (06)
    tracts = [
        '06001432100',  # Alameda
        '06001000000',  # Alameda
        '06013000000',  # Contra Costa
    ]

    # Act
    counties_dict, state_name = ah.get_county_and_state_names(tracts)

    # Assert
    assert isinstance(counties_dict, dict)
    assert set(counties_dict.keys()) == {'counties'}
    counties = counties_dict['counties']
    assert isinstance(counties, dict)
    # Should have two entries (keys numbered starting at 1)
    assert sorted(counties.keys()) == [1, 2]
    # Sorted by county FIPS -> Alameda (06001) then Contra Costa (06013)
    assert counties[1]['FIPS Code'] == '06001'
    assert counties[1]['Name'] == 'Alameda County, CA'
    assert counties[2]['FIPS Code'] == '06013'
    assert counties[2]['Name'] == 'Contra Costa County, CA'
    assert state_name == 'California'


@pytest.mark.parametrize(
    ('bad_input', 'expected_exception', 'expected_match'),
    [
        pytest.param(
            None, TypeError, 'must be a list of 11-digit strings', id='none'
        ),
        pytest.param(
            {},
            TypeError,
            'must be a list of 11-digit strings',
            id='dict_instead_of_list',
        ),
        pytest.param(
            [12345678901],
            ValueError,
            'All items in census_tracts must be 11-digit strings',
            id='non_string_item',
        ),
        pytest.param(
            ['060013'],
            ValueError,
            'All items in census_tracts must be 11-digit strings',
            id='malformed_fips_length',
        ),
        pytest.param(
            ['06A01000000'],
            ValueError,
            'All items in census_tracts must be 11-digit strings',
            id='non_digit_string',
        ),
    ],
)
def test_get_county_and_state_names_invalid_inputs(
    bad_input: Any,
    expected_exception: type[Exception],
    expected_match: str,
) -> None:
    """It should raise appropriate exceptions for invalid inputs."""
    with pytest.raises(expected_exception, match=expected_match):
        ah.get_county_and_state_names(bad_input)  # type: ignore[arg-type]


def test_get_county_and_state_names_empty_list_returns_empty() -> None:
    """It should raise ValueError for an empty input list (stricter handling)."""
    # Act / Assert
    with pytest.raises(ValueError, match='list is empty'):
        ah.get_county_and_state_names([])


@pytest.mark.parametrize(
    ('tracts', 'expected_full_names', 'expected_state'),
    [
        pytest.param(
            ['99999000000'],  # Unknown state ("99"), unknown county
            ['Unknown County'],
            'Unknown State',
            id='unknown_state_and_county',
        ),
        pytest.param(
            ['06098000000'],  # Known state (06 -> California), unknown county
            ['Unknown County, CA'],
            'California',
            id='known_state_unknown_county',
        ),
    ],
)
def test_get_county_and_state_names_unknown_placeholders(
    tracts: list[str],
    expected_full_names: list[str],
    expected_state: str,
) -> None:
    """It should gracefully handle unknown FIPS by using "Unknown" placeholders."""
    # Act
    counties_dict, state_name = ah.get_county_and_state_names(list(tracts))

    # Assert: Direct comparison using parameterized expected names
    names = [data['Name'] for _, data in sorted(counties_dict['counties'].items())]
    assert names == list(expected_full_names)
    assert state_name == expected_state


# -----------------------------
# Tests: prepare_building_inventory
# -----------------------------


@pytest.fixture
def empty_inventory() -> AssetInventory:
    """Return an empty AssetInventory."""
    return AssetInventory()


@pytest.fixture
def non_residential_inventory() -> AssetInventory:
    """Inventory containing only non-residential buildings (no 'RES*')."""
    inv = AssetInventory()
    inv.add_asset(
        1,
        Asset(
            1,
            coordinates=[[-122.0, 37.0]],
            features={'Occupancy': 'COM1', 'PlanArea': 100.0, 'Stories': 1},
        ),
    )
    inv.add_asset(
        2,
        Asset(
            2,
            coordinates=[[-122.1, 37.1]],
            features={'Occupancy': 'IND1', 'PlanArea': 200.0, 'Stories': 2},
        ),
    )
    return inv


@pytest.fixture
def all_missing_data_inventory() -> AssetInventory:
    """Inventory where all residential buildings are missing a required value (PlanArea)."""
    inv = AssetInventory()
    inv.add_asset(
        1,
        Asset(
            1,
            coordinates=[[-122.0, 37.0]],
            features={'Occupancy': 'RES1', 'PlanArea': None, 'Stories': 1},
        ),
    )
    inv.add_asset(
        2,
        Asset(
            2,
            coordinates=[[-122.1, 37.1]],
            features={'Occupancy': 'RES2', 'PlanArea': None, 'Stories': 2},
        ),
    )
    return inv


@pytest.fixture
def comprehensive_inventory_meters() -> AssetInventory:
    """Mixed inventory: residential valid/invalid and non-residential; areas in meters for conversion test."""
    inv = AssetInventory()
    # Valid residential
    inv.add_asset(
        10,
        Asset(
            10,
            coordinates=[[-122.0, 37.0]],
            features={'Occupancy': 'RES1', 'PlanArea': 10.0, 'Stories': 2},
        ),
    )
    # Residential missing PlanArea (will be dropped)
    inv.add_asset(
        11,
        Asset(
            11,
            coordinates=[[-122.05, 37.05]],
            features={'Occupancy': 'RES2', 'PlanArea': None, 'Stories': 1},
        ),
    )
    # Non-residential
    inv.add_asset(
        12,
        Asset(
            12,
            coordinates=[[-122.1, 37.1]],
            features={'Occupancy': 'COM1', 'PlanArea': 50.0, 'Stories': 1},
        ),
    )
    # Another valid residential
    inv.add_asset(
        13,
        Asset(
            13,
            coordinates=[[-122.2, 37.2]],
            features={'Occupancy': 'RES3', 'PlanArea': 5.0, 'Stories': 3},
        ),
    )
    return inv


@pytest.mark.parametrize(
    ('bad_inventory', 'expected_exception'),
    [
        pytest.param(None, TypeError, id='inventory_wrong_type'),
    ],
)
def test_prepare_building_inventory_input_validation_invalid_type(
    bad_inventory: Any,
    expected_exception: type[Exception],
    valid_key_features: dict[str, str],
) -> None:
    """It should raise TypeError when inventory is not an AssetInventory."""
    with pytest.raises(
        expected_exception, match='must be an instance of AssetInventory'
    ):
        ah.prepare_building_inventory(bad_inventory, valid_key_features)  # type: ignore[arg-type]


def test_prepare_building_inventory_empty_inventory_returns_empty(
    empty_inventory: AssetInventory, valid_key_features: dict[str, str]
) -> None:
    """It should return an empty GeoDataFrame for an empty AssetInventory."""
    gdf = ah.prepare_building_inventory(empty_inventory, valid_key_features)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.empty
    for col in ('occtype', 'gross_area_in_sqft'):
        assert col in gdf.columns


def test_prepare_building_inventory_no_residential_returns_empty(
    non_residential_inventory: AssetInventory, valid_key_features: dict[str, str]
) -> None:
    """It should return an empty GeoDataFrame when no residential buildings present."""
    gdf = ah.prepare_building_inventory(
        non_residential_inventory, valid_key_features
    )
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.empty


def test_prepare_building_inventory_all_dropped_warning(
    all_missing_data_inventory: AssetInventory,
    valid_key_features: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """It should return empty GeoDataFrame and print warning when all residential rows are invalid."""
    gdf = ah.prepare_building_inventory(
        all_missing_data_inventory, valid_key_features
    )
    captured = capsys.readouterr()
    assert gdf.empty
    assert 'All buildings are missing one or more required columns' in captured.out


def test_prepare_building_inventory_core_logic_meters_conversion(
    comprehensive_inventory_meters: AssetInventory,
    valid_key_features: dict[str, str],
    mocker: MockerFixture,
) -> None:
    """Core logic: drops non-res/invalid, computes area, applies 'm'→'ft' conversion; validator is mocked."""
    mocker.patch.object(ah, 'validate_key_features_dict', autospec=True)

    key_features_m = dict(valid_key_features)
    key_features_m['length_unit'] = 'm'

    gdf = ah.prepare_building_inventory(
        comprehensive_inventory_meters, key_features_m
    )

    # Expect only assets 10 and 13 (both residential and with valid PlanArea/Stories)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert not gdf.empty
    ids = sorted(map(int, gdf.index.tolist()))
    assert ids == [10, 13]

    assert set(gdf.loc[ids, 'occtype'].tolist()) == {'RES1', 'RES3'}

    # Check area calculation with conversion (10 m^2 * 10.7639 * 2 stories)
    area_10 = gdf.loc[10, 'gross_area_in_sqft']
    assert area_10 == pytest.approx(10.0 * 10.7639 * 2.0, rel=1e-6)

    # Check second valid row (5 m^2 * 10.7639 * 3)
    area_13 = gdf.loc[13, 'gross_area_in_sqft']
    assert area_13 == pytest.approx(5.0 * 10.7639 * 3.0, rel=1e-6)


def test_prepare_building_inventory_integration_validator_error(
    comprehensive_inventory_meters: AssetInventory,
    valid_key_features: dict[str, str],
) -> None:
    """Integration: with invalid key_features, the ValueError from validator must propagate."""
    bad_features = dict(valid_key_features)
    bad_features.pop('story_count_col')  # make it invalid

    with pytest.raises(ValueError, match="Invalid 'key_features' argument"):
        ah.prepare_building_inventory(comprehensive_inventory_meters, bad_features)


# -----------------------------
# Tests: _validate_pyncoda_output and create_housing_unit_inventory
# -----------------------------


@pytest.fixture
def pyncoda_required_columns() -> list[str]:
    """Return the list of columns required by `_validate_pyncoda_output`."""
    return [
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


@pytest.fixture
def df_valid_pyncoda_small(pyncoda_required_columns: list[str]) -> pd.DataFrame:
    """Small valid pyncoda-like DataFrame for success-path tests.

    Two rows designed to exercise categorical mappings:
    - ownershp: 1 -> Owner occupied; 2 -> Renter occupied
    - race: 4 -> Asian; 2 -> Black
    - vacancy: 0 -> Occupied; 5 -> For seasonal, recreational, or occasional use
    - gqtype: 0 -> Not a group quarters building; 7 -> Other noninstitutional facilities
    - incomegroup: 11 -> "$60,000 to $74,999"; 0 -> NaN (should drop from features)
    """
    data = {
        'blockid': ['00000000001', '00000000002'],
        'numprec': [3, 1],
        'ownershp': [1, 2],
        'race': [4, 2],
        'hispan': [True, False],
        'family': [False, True],
        'vacancy': [0, 5],
        'gqtype': [0, 7],
        'incomegroup': [11, 0],
        'randincome': [65000.0, 12000.5],
        'poverty': [False, True],
        'building_id': [10, 13],
    }
    return pd.DataFrame(data, columns=pyncoda_required_columns)


@pytest.fixture
def df_missing_columns(df_valid_pyncoda_small: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the valid DF missing a required column (vacancy)."""
    return df_valid_pyncoda_small.drop(columns=['vacancy'])  # missing required


@pytest.fixture
def df_empty_with_required_columns(
    pyncoda_required_columns: list[str],
) -> pd.DataFrame:
    """Empty DataFrame that still has all required columns."""
    return pd.DataFrame(columns=pyncoda_required_columns)


# --- Tests: _validate_pyncoda_output ---


def test_validate_pyncoda_output_accepts_valid(
    df_valid_pyncoda_small: pd.DataFrame,
) -> None:
    """It should not raise when all required columns are present."""
    ah._validate_pyncoda_output(df_valid_pyncoda_small)


def test_validate_pyncoda_output_raises_for_missing_columns(
    df_missing_columns: pd.DataFrame,
) -> None:
    """It should raise ValueError when any required column is missing."""
    with pytest.raises(ValueError, match='missing required columns'):
        ah._validate_pyncoda_output(df_missing_columns)


# --- Tests: create_housing_unit_inventory ---


@pytest.mark.parametrize(
    ('bad_input', 'expected_exception'),
    [
        pytest.param(None, TypeError, id='invalid_type'),
    ],
)
def test_create_housing_unit_inventory_input_validation_invalid_type(
    bad_input: Any, expected_exception: type[Exception]
) -> None:
    """It should raise a TypeError for invalid input types (per spec)."""
    with pytest.raises(expected_exception, match='must be provided as a DataFrame'):
        ah.create_housing_unit_inventory(bad_input)  # type: ignore[arg-type]


def test_create_housing_unit_inventory_missing_columns_raises(
    df_missing_columns: pd.DataFrame,
) -> None:
    """It should raise ValueError when input DataFrame lacks required columns."""
    with pytest.raises(ValueError, match='missing required columns'):
        ah.create_housing_unit_inventory(df_missing_columns)


def test_create_housing_unit_inventory_empty_df_returns_empty_inventory(
    df_empty_with_required_columns: pd.DataFrame,
) -> None:
    """It should return an empty HousingUnitInventory for an empty DataFrame."""
    inv = ah.create_housing_unit_inventory(df_empty_with_required_columns)

    assert isinstance(inv, HousingUnitInventory)
    assert inv.inventory == {}


def test_create_housing_unit_inventory_success_path_and_mappings(
    df_valid_pyncoda_small: pd.DataFrame,
) -> None:
    """Success path: verify renaming, mappings, and that building_id is dropped."""
    # Arrange: copy for immutability check
    original = df_valid_pyncoda_small.copy(deep=True)

    # Act
    inv = ah.create_housing_unit_inventory(df_valid_pyncoda_small)

    # Assert: type and count
    assert isinstance(inv, HousingUnitInventory)
    ids = inv.get_housing_unit_ids()
    assert len(ids) == 2

    # Row 0 expectations
    h0 = inv.inventory[ids[0]].features
    assert h0['BLOCK_GEOID'] == '00000000001'
    assert h0['NumberOfPersons'] == 3
    assert h0['Ownership'] == 'Owner occupied'
    assert h0['Race'] == 'Asian'
    assert h0['Hispanic'] is True
    assert h0['Family'] is False
    assert h0['VacancyStatus'] == 'Occupied'
    assert h0['GroupQuartersType'] == 'Not a group quarters building'
    assert h0['IncomeGroup'] == '$60,000 to $74,999'
    assert h0['IncomeSample'] == 65000.0
    assert h0['Poverty'] is False
    assert 'building_id' not in h0

    # Row 1 expectations
    h1 = inv.inventory[ids[1]].features
    assert h1['BLOCK_GEOID'] == '00000000002'
    assert h1['NumberOfPersons'] == 1
    assert h1['Ownership'] == 'Renter occupied'
    assert h1['Race'] == 'Black'
    assert h1['Hispanic'] is False
    assert h1['Family'] is True
    assert h1['VacancyStatus'] == 'For seasonal, recreational, or occasional use'
    assert h1['GroupQuartersType'] == 'Other noninstitutional facilities'
    assert 'IncomeGroup' not in h1  # incomegroup=0 -> NaN -> dropped by dropna
    assert h1['IncomeSample'] == 12000.5
    assert h1['Poverty'] is True
    assert 'building_id' not in h1

    # Immutability: ensure the input DataFrame was not modified
    pdt_assert_frame_equal(original, df_valid_pyncoda_small)


# -----------------------------
# Part 2: Integration tests for main workflow (mocked pyncoda)
# -----------------------------


@pytest.fixture
def lean_buildings_gdf_one() -> gpd.GeoDataFrame:
    """Return a minimal non-empty GeoDataFrame compatible with workflow.

    Columns required by the workflow: index named by building_id, geometry,
    occtype, gross_area_in_sqft. CRS WGS84.
    """
    gdf = gpd.GeoDataFrame(
        {
            'building_id': [1],
            'geometry': [Point(-122.0, 37.0)],
            'occtype': ['RES1'],
            'gross_area_in_sqft': [100.0],
        },
        geometry='geometry',
        crs='EPSG:4326',
    )
    return gdf.set_index('building_id')


@pytest.fixture
def simple_inventory() -> AssetInventory:
    """A simple AssetInventory with one residential-like asset id=1."""
    inv = AssetInventory()
    # Geometry and features do not matter since we mock preparation and scraping
    inv.add_asset(
        1,
        Asset(
            1,
            coordinates=[[-122.0, 37.0]],
            features={'Occupancy': 'RES1', 'PlanArea': 100.0, 'Stories': 1},
        ),
    )
    return inv


def _mock_happy_path_dependencies(
    mocker: MockerFixture,
    lean_gdf: gpd.GeoDataFrame,
    *,
    census_tracts: list[str] | None = None,
) -> None:
    """Patch dependencies to reach specific checkpoints in the workflow.

    - prepare_building_inventory -> returns provided lean_gdf
    - GeoDataFrame.to_file -> no-op
    - AssetInventory.read_from_geojson -> no-op (new behavior reloads filtered inventory)
    - Importer().get_class('CensusScraper')() -> object with get_census_tracts()
    - get_county_and_state_names -> returns a fixed counties/state tuple
    """
    mocker.patch.object(ah, 'prepare_building_inventory', return_value=lean_gdf)

    # Avoid actual file IO
    mocker.patch('geopandas.GeoDataFrame.to_file', autospec=True)
    mocker.patch(
        'brails.aggregators.housing_units.pyncoda.pyncoda_housing_units.AssetInventory.read_from_geojson',
        autospec=True,
    )

    # Fake census scraper
    fake_scraper = mocker.Mock()
    fake_scraper.get_census_tracts.return_value = (
        {'06001432100': {}}
        if census_tracts is None
        else {t: {} for t in census_tracts}
    )
    fake_importer = mocker.Mock()
    fake_importer.get_class.return_value = lambda: fake_scraper
    mocker.patch.object(ah, 'Importer', return_value=fake_importer)

    # Counties/state
    mocker.patch.object(
        ah,
        'get_county_and_state_names',
        return_value=(
            {'counties': {1: {'FIPS Code': '06001', 'Name': 'Alameda County, CA'}}},
            'California',
        ),
    )


def test_assign_housing_units_invalid_vintage(
    simple_inventory: AssetInventory,
    valid_key_features: dict[str, str],
    lean_buildings_gdf_one: gpd.GeoDataFrame,
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """It should raise ValueError when an unsupported vintage is provided."""
    _mock_happy_path_dependencies(mocker, lean_buildings_gdf_one)

    output_dir = tmp_path / 'test_output'
    output_dir.mkdir()

    with pytest.raises(ValueError, match='vintage is not valid') as exc:
        ah.assign_housing_units_to_buildings(
            building_inventory=simple_inventory,
            key_features=valid_key_features,
            vintage='2015',  # invalid per spec
            output_folder=output_dir,
        )
    assert 'vintage is not valid' in str(exc.value)


def test_assign_housing_units_census_scraper_failure(
    simple_inventory: AssetInventory,
    valid_key_features: dict[str, str],
    lean_buildings_gdf_one: gpd.GeoDataFrame,
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """It should raise ValueError when CensusScraper returns no tracts."""
    # Base mocks
    mocker.patch.object(
        ah, 'prepare_building_inventory', return_value=lean_buildings_gdf_one
    )
    mocker.patch('geopandas.GeoDataFrame.to_file', autospec=True)

    # Importer -> CensusScraper that returns empty dict
    fake_scraper = mocker.Mock()
    fake_scraper.get_census_tracts.return_value = {}
    fake_importer = mocker.Mock()
    fake_importer.get_class.return_value = lambda: fake_scraper
    mocker.patch.object(ah, 'Importer', return_value=fake_importer)

    # New behavior: filtered inventory is reloaded from file; avoid actual IO
    mocker.patch(
        'brails.aggregators.housing_units.pyncoda.pyncoda_housing_units.AssetInventory.read_from_geojson',
        autospec=True,
    )

    output_dir = tmp_path / 'test_output'
    output_dir.mkdir()

    with pytest.raises(ValueError, match='No census tracts were found'):
        ah.assign_housing_units_to_buildings(
            building_inventory=simple_inventory,
            key_features=valid_key_features,
            vintage='2020',
            output_folder=output_dir,
        )


def test_assign_housing_units_pyncoda_process_failure(
    simple_inventory: AssetInventory,
    valid_key_features: dict[str, str],
    lean_buildings_gdf_one: gpd.GeoDataFrame,
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """It should catch pyncoda exceptions and return None without crashing."""
    _mock_happy_path_dependencies(mocker, lean_buildings_gdf_one)

    # Raise during workflow construction
    mocker.patch.object(
        ah, 'process_community_workflow', side_effect=ValueError('boom')
    )

    output_dir = tmp_path / 'test_output'
    output_dir.mkdir()

    result = ah.assign_housing_units_to_buildings(
        building_inventory=simple_inventory,
        key_features=valid_key_features,
        vintage='2020',
        output_folder=output_dir,
    )
    assert result is None


def test_assign_housing_units_pyncoda_empty_result(
    simple_inventory: AssetInventory,
    valid_key_features: dict[str, str],
    lean_buildings_gdf_one: gpd.GeoDataFrame,
    mocker: MockerFixture,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """It should print a message and return None when no housing units are assigned."""
    _mock_happy_path_dependencies(mocker, lean_buildings_gdf_one)

    # Mock a workflow object with process_communities returning a DF that becomes empty after filtering
    required_cols = [
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
    df = pd.DataFrame(
        {
            'blockid': ['00000000001'],
            'numprec': [1],
            'ownershp': [1],
            'race': [1],
            'hispan': [False],
            'family': [False],
            'vacancy': [0],
            'gqtype': [0],
            'incomegroup': [1],
            'randincome': [100.0],
            'poverty': [False],
            'building_id': ['missing building id'],
        },
        columns=required_cols,
    )

    fake_workflow = mocker.Mock()
    fake_workflow.process_communities.return_value = df
    mocker.patch.object(ah, 'process_community_workflow', return_value=fake_workflow)

    output_dir = tmp_path / 'test_output'
    output_dir.mkdir()

    result = ah.assign_housing_units_to_buildings(
        building_inventory=simple_inventory,
        key_features=valid_key_features,
        vintage='2020',
        output_folder=output_dir,
    )

    captured = capsys.readouterr()
    assert 'No housing units were assigned to buildings' in captured.out
    assert result is None


# -----------------------------
# Part 2: New geometry-handling integration test (mocked pyncoda)
# -----------------------------


def _square_ring_coords(
    center_lon: float, center_lat: float, half_size: float = 0.01
) -> list[list[float]]:
    """Return coordinates for a square linear ring around a center point.

    The ring is closed (first point repeated at the end) and suitable for
    constructing a Polygon in AssetInventory via Asset(coordinates=[ring]).
    """
    minx, miny = center_lon - half_size, center_lat - half_size
    maxx, maxy = center_lon + half_size, center_lat + half_size
    ring = [
        [minx, miny],
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny],
        [minx, miny],
    ]
    return ring  # noqa: RET504


@pytest.fixture
def point_only_inventory() -> AssetInventory:
    """AssetInventory with only Point geometries (two nearby points)."""
    inv = AssetInventory()
    inv.add_asset_coordinates('p1', [[-120.000, 35.000]])
    inv.add_asset_coordinates('p2', [[-120.002, 35.001]])
    return inv


@pytest.fixture
def polygon_only_inventory() -> AssetInventory:
    """AssetInventory with only Polygon geometries (two small squares)."""
    inv = AssetInventory()
    coords1 = _square_ring_coords(-120.005, 35.002, half_size=0.01)
    coords2 = _square_ring_coords(-119.995, 34.998, half_size=0.01)
    inv.add_asset('g1', Asset('g1', coordinates=[coords1], features={}))
    inv.add_asset('g2', Asset('g2', coordinates=[coords2], features={}))
    return inv


@pytest.fixture
def mixed_geometry_inventory() -> AssetInventory:
    """AssetInventory mixing a Point and a Polygon geometry."""
    inv = AssetInventory()
    # One point
    inv.add_asset_coordinates('m_p', [[-120.001, 35.0005]])
    # One polygon
    coords = _square_ring_coords(-120.005, 35.002, half_size=0.01)
    inv.add_asset('m_g', Asset('m_g', coordinates=[coords], features={}))
    return inv


@pytest.mark.parametrize(
    'inventory_fixture_name',
    [
        'point_only_inventory',
        'polygon_only_inventory',
        'mixed_geometry_inventory',
    ],
)
def test_assign_housing_units_handles_various_geometries(
    request: Any,
    inventory_fixture_name: str,
    valid_key_features: dict[str, str],
    lean_buildings_gdf_one: gpd.GeoDataFrame,
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """Workflow should complete and link a HousingUnitInventory for any geometry mix.

    The pyncoda process is mocked to return a valid, canned assigned-housing-units
    DataFrame loaded from tests/fixtures/pyncoda_output/loving_county_assigned_housing_units.csv.
    We assert that the workflow links a HousingUnitInventory to the provided
    building inventory regardless of original geometry types.
    """
    # Arrange: get the parameterized input inventory
    building_inventory: AssetInventory = request.getfixturevalue(
        inventory_fixture_name
    )

    # Patch dependencies up to and including census tract resolution
    _mock_happy_path_dependencies(mocker, lean_buildings_gdf_one)

    # Load the canned pyncoda output and mock the workflow to return it
    fixture_csv = (
        Path(__file__).parents[3]
        / 'fixtures'
        / 'pyncoda_output'
        / 'loving_county_assigned_housing_units.csv'
    )
    df_ok = pd.read_csv(fixture_csv, dtype={'blockid': str})

    fake_workflow = mocker.Mock()
    fake_workflow.process_communities.return_value = df_ok
    mocker.patch.object(ah, 'process_community_workflow', return_value=fake_workflow)

    # Act: run the workflow
    result = ah.assign_housing_units_to_buildings(
        building_inventory=building_inventory,
        key_features=valid_key_features,
        vintage='2020',
        output_folder=str(tmp_path),
    )

    # Assert: function returns None and links a HousingUnitInventory
    assert result is None

    assert hasattr(building_inventory, 'housing_unit_inventory')
    assert isinstance(
        building_inventory.housing_unit_inventory, HousingUnitInventory
    )
    # Should have at least one housing unit created by the mocked DataFrame
    assert len(building_inventory.housing_unit_inventory.get_housing_unit_ids()) > 0


# -----------------------------
# Part 2: End-to-end mocked pyncoda test
# -----------------------------


def test_assign_housing_units_end_to_end_mocked(  # noqa: C901
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """End-to-end workflow using canned pyncoda output (mocked process).

    This test exercises the full pipeline from a valid AssetInventory to a
    linked HousingUnitInventory, while mocking only the external pyncoda
    processing step to return a canned DataFrame loaded from the fixture file
    at tests/fixtures/pyncoda_output/loving_county_assigned_housing_units.csv.

    IMPORTANT: Use the Loving County input inventory so building_ids match the
    canned pyncoda output fixture.

    Assertions follow Part 2 of the test plan:
    1) Process Completion
    2) State Change (HousingUnitInventory linked)
    3) Housing Unit Linking (0 < linked <= residential assets)
    4) Housing Unit Count sanity bounds
    5) Data Transformation Spot Check on one non-vacant, non-GQ housing unit
    """
    # Arrange: load AssetInventory from the Loving County fixture
    fixture_geojson = (
        Path(__file__).parents[3]
        / 'fixtures'
        / 'live_test'
        / 'loving_county_inventory.geojson'
    )
    inv = AssetInventory()
    inv.read_from_geojson(fixture_geojson)

    # Key features specific to the Loving County fixture
    key_features = {
        'occupancy_col': 'occtype',
        'plan_area_col': 'footprintArea',
        'story_count_col': 'num_story',
        'length_unit': 'ft',
    }

    # Avoid actual file IO in prepare -> to_file and the filtered-inventory reload
    mocker.patch('geopandas.GeoDataFrame.to_file', autospec=True)
    mocker.patch(
        'brails.aggregators.housing_units.pyncoda.pyncoda_housing_units.AssetInventory.read_from_geojson',
        autospec=True,
    )

    # Mock census dependencies to provide non-empty tract and county/state data
    fake_scraper = mocker.Mock()
    fake_scraper.get_census_tracts.return_value = {'48301000000': {}}
    fake_importer = mocker.Mock()
    fake_importer.get_class.return_value = lambda: fake_scraper
    mocker.patch.object(ah, 'Importer', return_value=fake_importer)

    mocker.patch.object(
        ah,
        'get_county_and_state_names',
        return_value=(
            {'counties': {1: {'FIPS Code': '48301', 'Name': 'Loving County, TX'}}},
            'TEXAS',
        ),
    )

    # Load canned pyncoda output and mock the workflow to return it
    fixture_csv = (
        Path(__file__).parents[3]
        / 'fixtures'
        / 'pyncoda_output'
        / 'loving_county_assigned_housing_units.csv'
    )
    df_assigned = pd.read_csv(fixture_csv, dtype={'blockid': str})

    fake_workflow = mocker.Mock()
    fake_workflow.process_communities.return_value = df_assigned
    mocker.patch.object(ah, 'process_community_workflow', return_value=fake_workflow)

    # Act: run the workflow; the function returns None on success
    try:
        result = ah.assign_housing_units_to_buildings(
            building_inventory=inv,
            key_features=key_features,
            vintage='2020',
            output_folder=str(tmp_path),
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f'assign_housing_units_to_buildings raised unexpectedly: {exc}')

    # Assert 1: Process Completion
    assert result is None

    # Assert 2: State Change — HousingUnitInventory linked
    assert hasattr(inv, 'housing_unit_inventory')
    assert isinstance(inv.housing_unit_inventory, HousingUnitInventory)

    # Assert 3: Housing Unit Linking — count assets with HousingUnits feature
    asset_ids = inv.get_asset_ids()
    linked_assets = 0
    for aid in asset_ids:
        found, features = inv.get_asset_features(aid)
        assert found is True
        if isinstance(features.get('HousingUnits'), list):
            linked_assets += 1
    assert linked_assets > 0

    # Upper bound: cannot exceed the number of residential-like assets in input
    res_like = 0
    for aid in asset_ids:
        _, features = inv.get_asset_features(aid)
        occtype = features.get('occtype') or features.get('Occupancy')
        if isinstance(occtype, str) and occtype.startswith('RES'):
            res_like += 1
    if res_like > 0:
        assert linked_assets <= res_like
    else:
        assert linked_assets <= len(asset_ids)

    # Assert 4: Housing Unit Count — sanity bounds (>0 and below a generous cap)
    total_housing_units = len(inv.housing_unit_inventory.get_housing_unit_ids())
    assert total_housing_units > 0
    # Set a generous upper cap so the test remains robust to fixture updates
    assert total_housing_units <= 10000

    # Assert 5: Data Transformation Spot Check — human-readable mappings
    readable_ownership = {'Owner occupied', 'Renter occupied'}
    readable_race = {
        'White',
        'Black',
        'American Indian',
        'Asian',
        'Pacific Islander',
        'Some Other Race',
        'Two or More Races',
    }
    readable_vacancy = {
        'Occupied',
        'For Rent',
        'Rented, not occupied',
        'For sale only',
        'Sold, not occupied',
        'For seasonal, recreational, or occasional use',
        'For migrant workers',
        'Other vacant',
    }
    readable_gq = {
        'Not a group quarters building',
        'Correctional facilities for adults',
        'Juvenile facilities',
        'Nursing facilities/Skilled-nursing facilities',
        'Other institutional facilities',
        'College/University student housing',
        'Military quarters',
        'Other noninstitutional facilities',
    }

    candidate_features = None
    for hid in inv.housing_unit_inventory.get_housing_unit_ids():
        feats = inv.housing_unit_inventory.inventory[hid].features
        if (
            feats.get('VacancyStatus') == 'Occupied'
            and feats.get('GroupQuartersType') == 'Not a group quarters building'
        ):
            candidate_features = feats
            break

    # Fallback: any housing unit for basic presence checks
    if (
        candidate_features is None
        and inv.housing_unit_inventory.get_housing_unit_ids()
    ):
        candidate_features = inv.housing_unit_inventory.inventory[
            inv.housing_unit_inventory.get_housing_unit_ids()[0]
        ].features

    assert candidate_features is not None

    if 'Ownership' in candidate_features:
        assert candidate_features['Ownership'] in readable_ownership
    if 'Race' in candidate_features:
        assert candidate_features['Race'] in readable_race
    if 'VacancyStatus' in candidate_features:
        assert candidate_features['VacancyStatus'] in readable_vacancy
    if 'GroupQuartersType' in candidate_features:
        assert candidate_features['GroupQuartersType'] in readable_gq


# -----------------------------
# Part 3: Live pyncoda test
# -----------------------------


@pytest.mark.live
def test_assign_housing_units_live_pyncoda(tmp_path: Path) -> None:  # noqa: C901
    """Live end-to-end test running the real pyncoda workflow on Loving County data.

    This test uses the provided GeoJSON fixture and runs assign_housing_units_to_buildings
    without mocking. It performs robust assertions equivalent to the mocked
    end-to-end test: process completion, state change, housing unit linking, total
    housing unit count sanity bounds, and data transformation spot check.
    """
    # Arrange: load AssetInventory from the live fixture
    fixture_path = (
        Path(__file__).parents[3]
        / 'fixtures'
        / 'live_test'
        / 'loving_county_inventory.geojson'
    )

    inv = AssetInventory()
    inv.read_from_geojson(fixture_path)

    # Key features per plan
    key_features = {
        'occupancy_col': 'occtype',
        'plan_area_col': 'footprintArea',
        'story_count_col': 'num_story',
        'length_unit': 'ft',
    }

    # Act: run the real workflow (expect ~3 minutes)
    try:
        result = ah.assign_housing_units_to_buildings(
            building_inventory=inv,
            key_features=key_features,
            vintage='2020',
            output_folder=str(tmp_path),
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f'assign_housing_units_to_buildings raised unexpectedly: {exc}')

    # Assert 1: Process Completion (function returns None by design)
    assert result is None

    # Assert 2: State Change — a HousingUnitInventory is linked
    assert hasattr(inv, 'housing_unit_inventory')
    assert isinstance(inv.housing_unit_inventory, HousingUnitInventory)

    # Assert 3: Housing Unit Linking — count assets with HousingUnits feature
    asset_ids = inv.get_asset_ids()
    linked_assets = 0
    for aid in asset_ids:
        found, features = inv.get_asset_features(aid)
        assert found is True
        if isinstance(features.get('HousingUnits'), list):
            linked_assets += 1
    assert linked_assets > 0

    # Determine the number of residential buildings in the input (upper bound)
    res_like = 0
    for aid in asset_ids:
        _, features = inv.get_asset_features(aid)
        occtype = features.get('occtype')
        if isinstance(occtype, str) and occtype.startswith('RES'):
            res_like += 1
    if res_like > 0:
        assert linked_assets <= res_like
    else:
        # If occtype not present in features, at least ensure we didn't link more than total assets
        assert linked_assets <= len(asset_ids)

    # Assert 4: Housing Unit Count — sanity bounds (>0 and reasonably small for Loving County)
    total_housing_units = len(inv.housing_unit_inventory.get_housing_unit_ids())
    assert total_housing_units > 0
    assert total_housing_units <= 2000  # generous upper bound to remain robust

    # Assert 5: Data Transformation Spot Check — verify human-readable mappings
    readable_ownership = {'Owner occupied', 'Renter occupied'}
    readable_race = {
        'White',
        'Black',
        'American Indian',
        'Asian',
        'Pacific Islander',
        'Some Other Race',
        'Two or More Races',
    }
    readable_vacancy = {
        'Occupied',
        'For Rent',
        'Rented, not occupied',
        'For sale only',
        'Sold, not occupied',
        'For seasonal, recreational, or occasional use',
        'For migrant workers',
        'Other vacant',
    }
    readable_gq = {
        'Not a group quarters building',
        'Correctional facilities for adults',
        'Juvenile facilities',
        'Nursing facilities/Skilled-nursing facilities',
        'Other institutional facilities',
        'College/University student housing',
        'Military quarters',
        'Other noninstitutional facilities',
    }

    # Find the first housing unit that is not Vacant and not Group Quarters
    candidate_features = None
    for hid in inv.housing_unit_inventory.get_housing_unit_ids():
        feats = inv.housing_unit_inventory.inventory[hid].features
        if (
            feats.get('VacancyStatus') == 'Occupied'
            and feats.get('GroupQuartersType') == 'Not a group quarters building'
        ):
            candidate_features = feats
            break

    # If none found (edge case for tiny datasets), fall back to any housing unit for basic checks
    if (
        candidate_features is None
        and inv.housing_unit_inventory.get_housing_unit_ids()
    ):
        candidate_features = inv.housing_unit_inventory.inventory[
            inv.housing_unit_inventory.get_housing_unit_ids()[0]
        ].features

    assert candidate_features is not None

    if 'Ownership' in candidate_features:
        assert candidate_features['Ownership'] in readable_ownership
    if 'Race' in candidate_features:
        assert candidate_features['Race'] in readable_race
    # Vacancy mapped
    if 'VacancyStatus' in candidate_features:
        assert candidate_features['VacancyStatus'] in readable_vacancy
    # Group quarters mapped
    if 'GroupQuartersType' in candidate_features:
        assert candidate_features['GroupQuartersType'] in readable_gq
