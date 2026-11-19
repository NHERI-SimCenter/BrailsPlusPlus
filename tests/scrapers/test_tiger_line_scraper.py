# Copyright (c) 2026 The Regents of the University of California
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
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.

"""Tests for TIGERLineScraper in brails.scrapers.us_census_scrapers."""

from __future__ import annotations

import socket

import geopandas as gpd
import pytest
from shapely.geometry import box

from brails.scrapers import TIGERLineScraper as ExportedScraper
from brails.scrapers.us_census_scrapers import tiger_line_scraper
from brails.scrapers.us_census_scrapers.tiger_line_scraper import (
    STATE_NAME_COLUMN,
    TIGERLineScraper,
)
from brails.types.geometry_region import GeometryRegion
from brails.utils import Importer
from brails.utils.tiger_files import DEFAULT_TIGER_VINTAGE

# Two states side by side, split at longitude -120, each with two block groups. A
# third state lies far to the east. The frames are projected, as a reader must
# project the region to match them.
STATES = gpd.GeoDataFrame(
    {'STATEFP': ['32', '06', '04'], 'NAME': ['Nevada', 'California', 'Arizona']},
    geometry=[
        box(-120, 38, -119, 39),
        box(-121, 38, -120, 39),
        box(-114, 32, -113, 33),
    ],
    crs='EPSG:4269',
).to_crs(3310)
BLOCK_GROUPS = {
    '06': gpd.GeoDataFrame(
        {'STATEFP': ['06', '06'], 'GEOID': ['060010000002', '060010000001']},
        geometry=[box(-120.5, 38, -120, 39), box(-121, 38, -120.5, 39)],
        crs='EPSG:4269',
    ).to_crs(3310),
    '32': gpd.GeoDataFrame(
        {'STATEFP': ['32', '32'], 'GEOID': ['320010000001', '320010000002']},
        geometry=[box(-120, 38, -119.5, 39), box(-119.5, 38, -119, 39)],
        crs='EPSG:4269',
    ).to_crs(3310),
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every connection and name lookup, so no test reaches a service."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = 'A test tried to open a network connection.'
        raise OSError(msg)

    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'getaddrinfo', refuse)


@pytest.fixture
def files_read(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """
    Serve the state and block-group frames above in place of read_tiger_file.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Replaces read_tiger_file in the scraper's module.

    Returns:
        list[tuple]:
            The layer, area and keyword arguments of each file read.
    """
    calls = []

    def read(layer: str, area: str, **options: object) -> gpd.GeoDataFrame:
        calls.append((layer, area, options))
        return STATES if layer == 'STATE' else BLOCK_GROUPS[area]

    monkeypatch.setattr(tiger_line_scraper, 'read_tiger_file', read)
    return calls


def test_region_over_two_states_reads_both_files(files_read: list[tuple]) -> None:
    """A region across a state line gets the block groups it meets in both states."""
    region = GeometryRegion(box(-120.4, 38.4, -119.6, 38.6))
    block_groups = TIGERLineScraper().get_block_groups(region)
    assert [call[:2] for call in files_read] == [
        ('STATE', 'us'),
        ('BG', '06'),
        ('BG', '32'),
    ]
    assert list(block_groups['GEOID']) == ['060010000002', '320010000001']
    assert list(block_groups[STATE_NAME_COLUMN]) == ['California', 'Nevada']
    assert block_groups.crs.to_epsg() == 3310


def test_region_in_one_state_reads_one_file(files_read: list[tuple]) -> None:
    """A region inside one state reads that state's file only, sorted by GEOID."""
    region = GeometryRegion(box(-120.9, 38.4, -120.1, 38.6))
    block_groups = TIGERLineScraper().get_block_groups(region)
    assert [call[:2] for call in files_read] == [('STATE', 'us'), ('BG', '06')]
    assert list(block_groups['GEOID']) == ['060010000001', '060010000002']


def test_release_and_download_settings_reach_the_reader(
    files_read: list[tuple],
) -> None:
    """Every read gets the vintage, timeout and retries, and the BG read a notice."""
    region = GeometryRegion(box(-120.9, 38.4, -120.1, 38.6))
    TIGERLineScraper(2020, timeout=5, retries=1).get_block_groups(region)
    settings = {'vintage': 2020, 'timeout': 5, 'retries': 1}
    assert files_read[0][2] == settings
    assert files_read[1][2] == {
        **settings,
        'notice': 'Reading the TIGER/Line 2020 block groups of California.',
    }


def test_neighbors_sharing_only_an_edge_are_left_out(
    files_read: list[tuple],
) -> None:
    """A region drawn on a state's edges brings in no neighbor across them."""
    region = GeometryRegion(box(-121, 38, -120, 39))
    block_groups = TIGERLineScraper().get_block_groups(region)
    assert [call[:2] for call in files_read] == [('STATE', 'us'), ('BG', '06')]
    assert list(block_groups['GEOID']) == ['060010000001', '060010000002']


@pytest.mark.usefixtures('files_read')
def test_region_in_no_state_is_refused() -> None:
    """A region that meets no state of the state file raises ValueError."""
    region = GeometryRegion(box(-150, 10, -149, 11))
    with pytest.raises(
        ValueError, match=f'no state of the TIGER/Line {DEFAULT_TIGER_VINTAGE}'
    ):
        TIGERLineScraper().get_block_groups(region)


def test_class_is_exported_and_found_by_the_importer() -> None:
    """brails.scrapers exports the class, and the Importer finds it by its name."""
    assert ExportedScraper is TIGERLineScraper
    assert Importer().get_class('TIGERLineScraper') is TIGERLineScraper
