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

"""
A synthetic study area for the compiler tests.

Two block groups of one tract make up the area. The western one holds
houses, apartments and shops that NSI reaches, and the eastern one mostly
footprints NSI does not reach. The census tables are empty and no
mobile-home park point is given. Positions are in meters in UTM zone 10N
near San Francisco and stored in EPSG:4326.
Footprint heights are in feet, the scraper's default unit.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from brails.compilers.hazus_eq_inventory_compiler import CompilerInputs
from brails.constants import (
    FOOTPRINT_PROVIDER_KEY,
    GEOGRAPHIC_CRS,
    OCCUPANCY_KEY,
    STORIES_KEY,
    STRUCTURE_TYPE_KEY,
    UNITS_KEY,
    YEAR_BUILT_KEY,
)
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import (
    HEIGHT_KEY,
    OVERTURE_PROVIDER,
    SOURCES_KEY,
)
from brails.types.asset_inventory import Asset, AssetInventory

METRIC_CRS = 'EPSG:32610'
X0, Y0 = 550_000.0, 4_180_000.0
_TO_LONLAT = Transformer.from_crs(METRIC_CRS, GEOGRAPHIC_CRS, always_xy=True)

SEED = 20261111
BG_WEST = '060750101001'
BG_EAST = '060750101002'
RELEASE = '2026-10-21.0'

OSM = 'dataset: OpenStreetMap, record_id: w1'
MIXED = 'dataset: Microsoft ML Buildings; property: /properties/height, dataset: OpenStreetMap'

# Footprints as id: (x, y, side, height in feet or None, sources). A
# footprint whose sources are None is an OpenStreetMap footprint without
# them; the others are Overture's.
FOOTPRINTS: dict[str, tuple[float, float, float, float | None, str | None]] = {
    'h01': (10, 10, 11, 13.0, OSM),
    'h02': (30, 10, 11, 26.0, OSM),
    'h03': (50, 10, 11, 13.0, OSM),
    'h04': (70, 10, 11, None, OSM),
    'h05': (10, 40, 11, None, OSM),
    'h06': (30, 40, 11, 13.0, MIXED),
    'h07': (50, 40, 11, None, OSM),
    'h08': (70, 40, 11, None, OSM),
    'w01': (10, 70, 11, None, OSM),
    'w02': (40, 70, 11, 13.0, OSM),
    'w03': (70, 70, 11, None, OSM),
    'p01': (10, 100, 25, 52.0, OSM),
    'c01': (50, 100, 20, 13.0, OSM),
    'c02': (90, 100, 30, None, OSM),
    'c03': (140, 100, 12, None, OSM),
    'c04': (160, 140, 20, 26.0, OSM),
    'c05': (120, 140, 20, None, OSM),
    'u01': (10, 170, 9, 13.0, OSM),
    'u02': (40, 170, 10, None, OSM),
    'u03': (70, 170, 12, 26.0, OSM),
    'u04': (100, 170, 14, None, OSM),
    'u05': (130, 170, 16, 13.0, None),
    'd01': (0, 210, 5, None, OSM),
    'r01': (220, 10, 11, None, OSM),
    'r02': (245, 10, 11, 13.0, OSM),
    'r03': (270, 10, 11, None, OSM),
    'r04': (295, 10, 11, None, OSM),
    'r05': (385, 185, 11, None, OSM),
    'l01': (360, 190, 11, None, OSM),
    'l02': (220, 60, 20, 52.0, OSM),
    'u06': (220, 100, 9, None, OSM),
    'u07': (250, 100, 10, 13.0, OSM),
    'u08': (280, 100, 11, None, OSM),
    'u09': (310, 100, 12, None, OSM),
    'u10': (340, 100, 13, 26.0, OSM),
    'u11': (220, 140, 14, None, OSM),
    'u12': (250, 140, 15, None, OSM),
    'u13': (280, 140, 9, None, OSM),
    'u14': (310, 140, 10, None, OSM),
    'u15': (340, 140, 12, None, OSM),
}

# NSI records as fd_id: (footprint, dx, dy, fields). The point lies at the
# footprint's lower left corner plus (dx, dy) meters, so (5, 5) is inside an
# eleven-meter house and a negative offset puts it outside, for the walk.
_HOUSE = {
    'occtype': 'RES1-1SNB',
    'resunits': 1,
    'num_story': 1,
    'sqft': 1400.0,
    'fullrep': 350000.0,
    'bldgtype': 'W',
    'med_yr_blt': 1955,
}
NSI_RECORDS: dict[int, tuple[str, float, float, dict[str, Any]]] = {
    1: ('h01', 5, 5, _HOUSE),
    2: ('h02', 5, 5, _HOUSE),
    3: ('h03', 5, 5, {**_HOUSE, 'num_story': 2}),
    4: ('h04', 5, 5, _HOUSE),
    5: ('h05', 5, 5, _HOUSE),
    6: ('h06', 5, 5, {**_HOUSE, 'num_story': 2}),
    7: ('h07', 5, 5, _HOUSE),
    8: ('h08', 5, 5, _HOUSE),
    9: ('w01', -6, 5, _HOUSE),
    10: ('w02', 16, 5, {k: v for k, v in _HOUSE.items() if k != 'num_story'}),
    11: ('w03', 5, 5, {k: v for k, v in _HOUSE.items() if k != 'num_story'}),
    12: (
        'p01',
        10,
        10,
        {
            'occtype': 'RES3C',
            'resunits': 8,
            'num_story': 3,
            'sqft': 9000.0,
            'fullrep': 2.0e6,
            'bldgtype': 'W',
        },
    ),
    13: (
        'c01',
        10,
        10,
        {
            'occtype': 'COM1',
            'num_story': 1,
            'sqft': 4000.0,
            'fullrep': 8.0e5,
            'bldgtype': 'C',
        },
    ),
    14: (
        'c02',
        10,
        10,
        {
            'occtype': 'EDU1',
            'num_story': 5,
            'sqft': 40000.0,
            'fullrep': 9.0e6,
            'bldgtype': 'C',
        },
    ),
    15: (
        'c03',
        6,
        6,
        {
            'occtype': 'RES2',
            'resunits': 1,
            'num_story': 3,
            'sqft': 1100.0,
            'fullrep': 9.0e4,
            'bldgtype': 'H',
        },
    ),
    16: (
        'c04',
        10,
        10,
        {
            'occtype': 'COM4',
            'num_story': 3,
            'sqft': 6000.0,
            'fullrep': 1.5e6,
            'bldgtype': 'S',
        },
    ),
    17: ('r01', 5, 5, _HOUSE),
    18: ('r02', 5, 5, _HOUSE),
    19: ('r03', 5, 5, _HOUSE),
    20: ('r04', 5, 5, _HOUSE),
    21: ('r05', 5, 5, _HOUSE),
    22: ('l01', 5, 5, {**_HOUSE, 'occtype': 'COM1', 'resunits': None}),
    28: ('c05', 10, 10, {'occtype': 'COM1', 'bldgtype': 'C'}),
    23: (
        'l02',
        10,
        10,
        {
            'occtype': 'COM1',
            'num_story': 2,
            'sqft': 5000.0,
            'fullrep': 1.0e6,
            'bldgtype': 'M',
        },
    ),
}

# Values known locally, by footprint.
LOCAL = {
    'l01': {OCCUPANCY_KEY: 'RES1', UNITS_KEY: 1, STORIES_KEY: 1},
    'l02': {STRUCTURE_TYPE_KEY: 'URM', YEAR_BUILT_KEY: 1990, STORIES_KEY: 2},
    'u07': {YEAR_BUILT_KEY: 1950},
    'c01': {UNITS_KEY: 4},
}


def lonlat(x: float, y: float) -> list[float]:
    """
    Return the longitude and latitude of a local metric position.

    Args:
        x (float):
            Meters east of the fixture origin.
        y (float):
            Meters north of it.

    Returns:
        list[float]:
            The longitude and latitude.
    """
    lon, lat = _TO_LONLAT.transform(X0 + x, Y0 + y)
    return [lon, lat]


def square(x: float, y: float, side: float) -> list[list[float]]:
    """
    Return a closed square ring with its lower left corner at a position.

    Args:
        x (float):
            Meters east of the fixture origin.
        y (float):
            Meters north of it.
        side (float):
            The side in meters.

    Returns:
        list[list[float]]:
            The ring's longitude and latitude pairs.
    """
    corners = [(x, y), (x + side, y), (x + side, y + side), (x, y + side), (x, y)]
    return [lonlat(cx, cy) for cx, cy in corners]


def build_inputs(*, reverse: bool = False) -> CompilerInputs:
    """
    Build the study area.

    Args:
        reverse (bool, optional):
            Whether to insert the footprints, the NSI points and the local
            assets in reverse order.

    Returns:
        CompilerInputs:
            The study area's data.
    """
    order = (lambda items: list(items)[::-1]) if reverse else list

    footprints = AssetInventory()
    for asset_id, (x, y, side, height, sources) in order(FOOTPRINTS.items()):
        features: dict[str, Any] = {'type': 'building'}
        if sources is None:
            features[FOOTPRINT_PROVIDER_KEY] = 'osm'
        else:
            features[FOOTPRINT_PROVIDER_KEY] = OVERTURE_PROVIDER
            features[SOURCES_KEY] = sources
        if height is not None:
            features[HEIGHT_KEY] = height
        footprints.add_asset(asset_id, Asset(asset_id, square(x, y, side), features))

    points = AssetInventory()
    for fd_id, (footprint, dx, dy, fields) in order(NSI_RECORDS.items()):
        x, y = FOOTPRINTS[footprint][:2]
        coordinate = lonlat(x + dx, y + dy)
        features = {k: v for k, v in fields.items() if v is not None}
        block_group = BG_WEST if x < 200 else BG_EAST
        features.update(
            fd_id=fd_id,
            cbfips=block_group + '000',
            x=coordinate[0],
            y=coordinate[1],
            type='building',
        )
        points.add_asset(fd_id, Asset(fd_id, [coordinate], features))

    local = AssetInventory()
    for asset_id, features in order(LOCAL.items()):
        x, y, side = FOOTPRINTS[asset_id][:3]
        local.add_asset(
            asset_id, Asset(asset_id, square(x, y, side), dict(features))
        )

    corners = {
        BG_WEST: ((-50, -50), (200, -50), (200, 250), (-50, 250)),
        BG_EAST: ((200, -50), (450, -50), (450, 250), (200, 250)),
    }
    block_groups = gpd.GeoDataFrame(
        {'GEOID': list(corners)},
        geometry=[
            shapely.Polygon([lonlat(*corner) for corner in ring])
            for ring in corners.values()
        ],
        crs=GEOGRAPHIC_CRS,
    )

    return CompilerInputs(
        footprints=footprints,
        nsi_points=points,
        block_groups=block_groups,
        b25024_block_groups=pd.DataFrame(),
        b25034_block_groups=pd.DataFrame(),
        b25034_tracts=pd.DataFrame(),
        b25127_tracts=pd.DataFrame(),
        mh_park_points=np.empty((0, 2)),
        footprint_release=RELEASE,
        local=local,
        study_area='06075',
    )
