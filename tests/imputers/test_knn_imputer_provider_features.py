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

"""Tests of how KnnImputer treats the features footprint scrapers add."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from brails.constants import PROVIDER_FEATURES
from brails.imputers.knn_imputer.knn_imputer import KnnImputer
from brails.scrapers.footprint_scraper import FootprintScraper

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from brails.types.asset_inventory import AssetInventory
    from brails.types.region_boundary import RegionBoundary

METERS_PER_DEGREE = math.pi * 6371008.8 / 180.0
LAT = 37.77


def square(index: int) -> list[list[float]]:
    """Return a closed 10 m square, the index-th of a row 25 m apart."""
    half_lat = 5.0 / METERS_PER_DEGREE
    half_lon = half_lat / math.cos(math.radians(LAT))
    lon = -122.45 + 5.0 * index * half_lon
    return [
        [lon - half_lon, LAT - half_lat],
        [lon + half_lon, LAT - half_lat],
        [lon + half_lon, LAT + half_lat],
        [lon - half_lon, LAT + half_lat],
        [lon - half_lon, LAT - half_lat],
    ]


class RowScraper(FootprintScraper):
    """A footprint scraper that returns a row of 40 squares as OSM footprints."""

    def __init__(self) -> None:
        """Use the default options."""
        super().__init__('RowScraper')

    def get_footprints(self, region: RegionBoundary | None = None) -> AssetInventory:
        """Build the inventory of the row."""
        del region
        footprints = [square(index) for index in range(40)]
        attributes: dict[str, list[Any]] = {
            'buildingheight': [3.0 + index % 7 for index in range(40)],
            'osm_id': [1000 + index for index in range(40)],
        }
        return self._create_asset_inventory(
            footprints, attributes, 'ft', provider='osm'
        )


def scraped_inventory() -> AssetInventory:
    """Return a scraped inventory with gaps to impute and provider features."""
    inventory = RowScraper().get_footprints()
    for key in inventory.get_asset_ids():
        features = inventory.inventory[key].features
        features['numstories'] = 'NA' if key % 4 == 0 else 1 + key % 3
        features['roofshape'] = 'NA' if key % 5 == 0 else ('gable', 'hip')[key % 2]
    return inventory


def test_provider_features_are_not_predictors_by_default(
    mocker: MockerFixture,
) -> None:
    """Provider ids and footprint_provider are left out unless asked for, and kept."""
    spy = mocker.spy(KnnImputer, 'category_in_df_to_indices')

    default_run = KnnImputer(scraped_inventory(), seed=1).impute()
    default_columns = set(spy.call_args_list[-1][0][1].columns)
    kept_run = KnnImputer(
        scraped_inventory(), seed=1, exclude_provider_ids=False
    ).impute()
    kept_columns = set(spy.call_args_list[-1][0][1].columns)

    assert not default_columns & set(PROVIDER_FEATURES)
    assert {'osm_id', 'footprint_provider'} <= kept_columns
    for run in (default_run, kept_run):
        features = run.inventory[0].features
        assert features['osm_id'] == 1000
        assert features['footprint_provider'] == 'osm'
        assert features['numstories'] != 'NA'
