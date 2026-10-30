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

"""Tests for the NSI SimCenter conventions, the raw data path and its file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from brails.scrapers.nsi_parser.nsi_conventions import (
    NSI_RAW_SCHEMA,
    NSI_TO_SIMCENTER,
    to_simcenter,
)
from brails.scrapers.nsi_parser.nsi_parser import NSI_Parser
from tests.scrapers.conftest import Region, canned_response
from tests.scrapers.data.nsi_fixture_generator import FIXTURE_PATH, fetch_sample

if TYPE_CHECKING:
    from collections.abc import Callable


def make_record(fd_id: int, lon: float, lat: float, **overrides: object) -> dict:
    """
    Build a raw NSI 2026-style property record.

    Args:
        fd_id (int):
            The NSI structure identifier.
        lon (float):
            Longitude of the point.
        lat (float):
            Latitude of the point.
        **overrides (object):
            Raw NSI fields to override or add.

    Returns:
        dict:
            The property record.
    """
    record = {
        'fd_id': fd_id,
        'x': lon,
        'y': lat,
        'occtype': 'RES1-2SWB',
        'bldgtype': 'W',
        'num_story': 2,
        'med_yr_blt': 1955,
        'sqft': 1800.0,
        'val_struct': 500000.0,
        'found_type': 'S',
        'found_ht': 1.5,
        'ground_elv': 30.0,
        'firmzone': 'X',
        'bldheight': 24.0,
        'source': 'P',
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------
# the conversion function
# ---------------------------------------------------------------------


def test_to_simcenter_renames_and_unpacks() -> None:
    """SimCenter names, decoded occupancy, and no fabricated types."""
    converted = to_simcenter(make_record(11, 0.0005, 0.0005))

    assert converted['OccupancyClass'] == 'RES1'
    assert converted['Basement'] is True
    assert converted['SplitLevel'] is False
    assert converted['NumberOfStories'] == 2
    assert converted['YearBuiltTractMedian'] == 1955
    # A tract median must not reach the inventory as an observed year.
    assert 'YearBuilt' not in converted
    assert converted['PlanArea'] == 1800.0
    assert converted['ReplacementCost'] == 500000.0
    # The general class passes through; nothing invents a specific type.
    assert converted['BuildingType'] == 'W'
    assert 'StructureType' not in converted
    # The NSI foundation type is kept verbatim under FoundationType.
    assert converted['FoundationType'] == 'S'
    # Unmapped fields pass through with their NSI names.
    assert converted['bldheight'] == 24.0
    assert converted['fd_id'] == 11


def test_to_simcenter_does_not_modify_the_input() -> None:
    """The conversion returns a new dictionary."""
    record = make_record(11, 0.0005, 0.0005)

    to_simcenter(record)

    assert record == make_record(11, 0.0005, 0.0005)


@pytest.mark.parametrize(
    ('occtype', 'occupancy', 'split_level', 'basement'),
    [
        ('RES1-SLNB', 'RES1', True, False),
        ('RES1-1SWB', 'RES1', False, True),
        ('RES1', 'RES1', None, None),
        ('RES3B', 'RES3B', None, None),
        ('COM4', 'COM4', None, None),
    ],
)
def test_occtype_unpacking(
    occtype: str,
    occupancy: str,
    split_level: bool | None,  # noqa: FBT001
    basement: bool | None,  # noqa: FBT001
) -> None:
    """The packed occupancy code decodes into separate features."""
    converted = to_simcenter(make_record(11, 0.0005, 0.0005, occtype=occtype))

    assert converted['OccupancyClass'] == occupancy
    assert converted.get('SplitLevel') is split_level
    assert converted.get('Basement') is basement


def test_manufactured_housing_code_maps_to_mh() -> None:
    """The NSI H code becomes the Hazus MH general class."""
    converted = to_simcenter(
        make_record(11, 0.0005, 0.0005, bldgtype='H', occtype='RES2')
    )

    assert converted['BuildingType'] == 'MH'


BBOX = (-0.01, -0.01, 0.01, 0.01)


# ---------------------------------------------------------------------
# the download itself
# ---------------------------------------------------------------------


def test_the_download_keys_records_by_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Records sharing a coordinate survive the download itself."""
    canned_response(
        monkeypatch,
        [
            make_record(11, 0.0005, 0.0005),
            make_record(22, 0.0005, 0.0005, occtype='COM1'),
            make_record(33, 0.0015, 0.0015, occtype='COM4'),
        ],
    )

    records = NSI_Parser._get_nsi_data(BBOX)

    assert sorted(records) == [11, 22, 33]
    geometry, properties = records[11]
    assert (geometry.x, geometry.y) == (0.0005, 0.0005)
    assert properties['occtype'] == 'RES1-2SWB'
    assert records[22][0].equals(geometry)


def test_repeated_record_ids_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two records under one id would silently lose one of them."""
    canned_response(
        monkeypatch,
        [make_record(11, 0.0005, 0.0005), make_record(11, 0.0015, 0.0015)],
    )

    with pytest.raises(ValueError, match='records would be lost'):
        NSI_Parser._get_nsi_data(BBOX)


def test_a_record_without_an_id_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed record must not be reported as a failed download."""
    record = make_record(11, 0.0005, 0.0005)
    del record['fd_id']
    canned_response(monkeypatch, [record])

    with pytest.raises(KeyError):
        NSI_Parser._get_nsi_data(BBOX)


# ---------------------------------------------------------------------
# the raw data path
# ---------------------------------------------------------------------


def test_get_raw_data_returns_untouched_nsi_fields(canned_fetch: Callable[..., None]) -> None:
    """By default the raw path keeps NSI names and values as served."""
    canned_fetch([make_record(11, 0.0005, 0.0005)])

    inventory = NSI_Parser().get_raw_data(Region())

    _, features = inventory.get_asset_features(11)
    assert features['occtype'] == 'RES1-2SWB'
    assert features['bldgtype'] == 'W'
    assert features['med_yr_blt'] == 1955
    assert features['type'] == 'building'
    assert 'OccupancyClass' not in features
    assert 'StructureType' not in features


def test_get_raw_data_applies_simcenter_format(canned_fetch: Callable[..., None]) -> None:
    """With the flag on, each record goes through the conversion."""
    canned_fetch(
        [
            make_record(11, 0.0005, 0.0005),
            make_record(22, 0.0015, 0.0015, occtype='COM4', bldgtype='S'),
        ]
    )

    inventory = NSI_Parser().get_raw_data(Region(), simcenter_format=True)

    _, features = inventory.get_asset_features(11)
    assert features['OccupancyClass'] == 'RES1'
    assert features['Basement'] is True
    assert features['BuildingType'] == 'W'
    assert features['type'] == 'building'
    assert 'occtype' not in features
    _, features = inventory.get_asset_features(22)
    assert features['OccupancyClass'] == 'COM4'
    assert features['BuildingType'] == 'S'
    assert 'Basement' not in features


def test_records_sharing_a_coordinate_all_survive(
    canned_fetch: Callable[..., None],
) -> None:
    """NSI serves one record per occupancy of a mixed-use structure."""
    canned_fetch(
        [
            make_record(11, 0.0005, 0.0005),
            make_record(22, 0.0005, 0.0005, occtype='COM1', bldgtype='S'),
            make_record(33, 0.0005, 0.0005, occtype='COM4', sqft=900.0),
        ]
    )

    inventory = NSI_Parser().get_raw_data(Region())

    assert sorted(inventory.get_asset_ids()) == [11, 22, 33]
    assert inventory.get_asset_coordinates(11)[1] == [[0.0005, 0.0005]]
    assert inventory.get_asset_coordinates(33)[1] == [[0.0005, 0.0005]]
    assert inventory.get_asset_features(33)[1]['sqft'] == 900.0


# ---------------------------------------------------------------------
# the raw-pull file
# ---------------------------------------------------------------------


def test_the_raw_file_holds_nsi_names_in_fd_id_order(
    canned_fetch: Callable[..., None], tmp_path: Path
) -> None:
    """The file is the raw pull even when the inventory is converted."""
    canned_fetch([make_record(22, 0.0015, 0.0015), make_record(11, 0.0005, 0.0005)])
    path = tmp_path / 'pull.parquet'

    NSI_Parser().get_raw_data(Region(), simcenter_format=True, raw_parquet_path=path)
    table = pq.read_table(path)

    assert table.column_names == NSI_RAW_SCHEMA.names
    assert table['fd_id'].to_pylist() == [11, 22]
    assert table['occtype'].to_pylist() == ['RES1-2SWB', 'RES1-2SWB']


def test_only_records_inside_the_region_are_written(
    canned_fetch: Callable[..., None], tmp_path: Path
) -> None:
    """The file holds what the inventory holds."""
    canned_fetch([make_record(11, 0.0005, 0.0005), make_record(22, 0.5, 0.5)])
    path = tmp_path / 'pull.parquet'

    pulled = NSI_Parser().get_raw_data(Region(), raw_parquet_path=path)

    assert pq.read_table(path)['fd_id'].to_pylist() == [11]
    assert pulled.get_asset_ids() == [11]


def test_no_raw_path_writes_nothing(
    canned_fetch: Callable[..., None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``raw_parquet_path``, no file is written."""
    canned_fetch([make_record(11, 0.0005, 0.0005)])
    monkeypatch.chdir(tmp_path)

    NSI_Parser().get_raw_data(Region())

    assert list(tmp_path.iterdir()) == []


def test_a_served_null_is_kept_as_none(canned_fetch: Callable[..., None]) -> None:
    """The parser keeps a null NSI serves; the file reader omits it."""
    canned_fetch([make_record(11, 0.0005, 0.0005, firmzone=None)])

    inventory = NSI_Parser().get_raw_data(Region())

    assert inventory.get_asset_features(11)[1]['firmzone'] is None


# ---------------------------------------------------------------------
# the stored NSI sample
# ---------------------------------------------------------------------


def load_fixture_records() -> list[dict]:
    """
    Read the stored NSI API sample.

    Returns:
        list[dict]:
            The raw properties of each sampled point.
    """
    sample = json.loads(Path(FIXTURE_PATH).read_text())
    return [feature['properties'] for feature in sample['features']]


def test_fixture_carries_every_mapped_field() -> None:
    """Each stored record provides all the fields the conversion renames."""
    records = load_fixture_records()

    assert len(records) > 10
    for record in records:
        assert set(NSI_TO_SIMCENTER) <= set(record)


def test_to_simcenter_handles_the_real_records() -> None:
    """The conversion decodes every stored record without inventing values."""
    for record in load_fixture_records():
        converted = to_simcenter(dict(record))

        assert '-' not in converted['OccupancyClass']
        assert converted['BuildingType'] in {'W', 'S', 'C', 'M', 'MH'}
        assert 'StructureType' not in converted
        for key in ('SplitLevel', 'Basement'):
            if key in converted:
                assert isinstance(converted[key], bool)


@pytest.mark.live
def test_live_nsi_schema_matches_the_fixture() -> None:
    """The live API still serves the field set the fixture was cut from.

    This is the schema-drift alarm. When it fails, rerun
    ``tests/scrapers/data/nsi_fixture_generator.py`` to refresh the
    stored sample and review the conventions module's field mapping.
    """
    live = fetch_sample()

    live_fields = set(live['features'][0]['properties'])
    fixture_fields = {
        field for record in load_fixture_records() for field in record
    }
    assert live_fields == fixture_fields
