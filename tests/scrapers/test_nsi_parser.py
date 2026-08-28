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

"""Tests for the NSI SimCenter conventions and the raw data path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from shapely.geometry import Point, box

from brails.scrapers.nsi_parser.nsi_conventions import (
    NSI_TO_SIMCENTER,
    to_simcenter,
)
from brails.scrapers.nsi_parser.nsi_parser import NSI_Parser
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


class Region:
    """A minimal stand-in for RegionBoundary."""

    @staticmethod
    def get_boundary() -> tuple:
        """Return a bounding polygon covering the canned points."""
        return box(-0.01, -0.01, 0.01, 0.01), None, None


@pytest.fixture
def canned_fetch(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Patch the NSI download with a canned point set."""

    def install(records: list[dict]) -> None:
        data = {
            Point(record['x'], record['y']): dict(record)
            for record in records
        }
        monkeypatch.setattr(
            NSI_Parser,
            '_get_nsi_data',
            staticmethod(lambda _bbox: dict(data)),
        )

    return install


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
