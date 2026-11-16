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

"""Offline tests of how the footprint scrapers key their assets."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
import pytest
from shapely.geometry import Polygon

from brails.scrapers.footprint_scraper import FootprintScraper
from brails.scrapers.ms_footprint_scraper import ms_footprint_handler
from brails.scrapers.ms_footprint_scraper.ms_footprint_handler import (
    MS_FootprintScraper,
)
from brails.scrapers.osm_footprint_scraper import osm_footprint_scraper
from brails.scrapers.osm_footprint_scraper.osm_footprint_scraper import (
    OSM_FootprintScraper,
)
from brails.scrapers.overture_maps_scraper.overture_maps_footprint_scraper import (
    OvertureMapsFootprintScraper,
)
from brails.scrapers.usa_footprint_scraper import usa_footprint_scraper
from brails.scrapers.usa_footprint_scraper.usa_footprint_scraper import (
    USA_FootprintScraper,
)
from brails.types.asset_inventory import AssetInventory
from brails.types.region_boundary import RegionBoundary
from brails.utils.asset_id import assign_asset_ids

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytest_mock import MockerFixture

METERS_PER_DEGREE = math.pi * 6371008.8 / 180.0
SQFT_PER_M2 = 1.0 / 0.3048**2
LAT = 37.77
BOX = (-122.46, 37.76, -122.40, 37.78)


def meters_to_lon(meters: float) -> float:
    """Return the longitude span of a distance east at ``LAT``."""
    return meters / (METERS_PER_DEGREE * math.cos(math.radians(LAT)))


def rectangle(lon: float, width_m: float, height_m: float) -> list[list[float]]:
    """Return a closed rectangle centered on a longitude at ``LAT``."""
    half_lat = height_m / 2.0 / METERS_PER_DEGREE
    half_lon = meters_to_lon(width_m / 2.0)
    return [
        [lon - half_lon, LAT - half_lat],
        [lon + half_lon, LAT - half_lat],
        [lon + half_lon, LAT + half_lat],
        [lon - half_lon, LAT + half_lat],
        [lon - half_lon, LAT - half_lat],
    ]


def row_lon(index: float) -> float:
    """Return the longitude of the index-th footprint of the fixture row."""
    return -122.45 + meters_to_lon(index * 25.0)


# The fixture: twelve houses 25 m apart, a 4 m shed on the same centroid as
# house 3, a 5 by 6 m garage beside house 7, and a collinear footprint.
HOUSES = [rectangle(row_lon(index), 12.0, 10.0) for index in range(12)]
SHED = rectangle(row_lon(3), 4.0, 4.0)
GARAGE = rectangle(row_lon(7) + 0.0001, 5.0, 6.0)
FLAT = [[-122.44, LAT], [-122.4399, LAT], [-122.4398, LAT], [-122.44, LAT]]
FIXTURE = [*HOUSES, SHED, GARAGE, FLAT]
SHED_POSITION, GARAGE_POSITION, FLAT_POSITION = 12, 13, 14


class StubScraper(FootprintScraper):
    """A footprint scraper that returns the footprints passed to it."""

    def __init__(self, input_dict: dict[str, Any], footprints: list[Any]) -> None:
        """Store the footprints and the options."""
        self.length_unit = input_dict.get('length', 'ft')
        super().__init__('StubScraper', input_dict, self.length_unit)
        self.footprints_to_return = footprints

    def get_footprints(self, region: RegionBoundary | None = None) -> AssetInventory:
        """Build the inventory of the stored footprints."""
        del region
        count = len(self.footprints_to_return)
        attributes = {
            'buildingheight': [3.0 + index for index in range(count)],
            'osm_id': [1000 + index for index in range(count)],
        }
        return self._create_asset_inventory(
            list(self.footprints_to_return),
            attributes,
            self.length_unit,
            provider='osm',
        )


def keys_by_geometry(inventory: AssetInventory) -> dict[str, Any]:
    """Map each asset's coordinates to its key."""
    return {
        str(asset.coordinates): key for key, asset in inventory.inventory.items()
    }


def test_default_scheme_reproduces_positional_keys() -> None:
    """Without options the keys are positional and every footprint is kept."""
    scraper = StubScraper({}, FIXTURE)
    inventory = scraper.get_footprints()

    assert scraper.id_scheme == 'index'
    assert scraper.min_area == 0.0
    assert inventory.get_asset_ids() == list(range(len(FIXTURE)))
    for key, footprint in enumerate(FIXTURE):
        asset = inventory.inventory[key]
        assert asset.coordinates == footprint
        assert asset.features['buildingheight'] == 3.0 + key
        assert asset.features['osm_id'] == 1000 + key
        assert asset.features['footprint_provider'] == 'osm'
    assert scraper.asset_id_skipped == []
    assert scraper.asset_id_log == []


def test_explicit_floor_applies_under_the_default_scheme() -> None:
    """An explicit min_area drops small footprints and keeps keys contiguous."""
    scraper = StubScraper({'min_area': 450}, FIXTURE)
    inventory = scraper.get_footprints()

    assert inventory.get_asset_ids() == list(range(12))
    reasons = {
        entry['position']: entry['reason'] for entry in scraper.asset_id_skipped
    }
    assert reasons == {
        SHED_POSITION: 'below_min_area',
        GARAGE_POSITION: 'below_min_area',
        FLAT_POSITION: 'zero_area',
    }


def test_geohash_scheme_applies_the_floor_before_the_ids() -> None:
    """The shed under the default floor is dropped first, so its house keeps a bare geohash."""
    scraper = StubScraper({'id_scheme': 'geohash'}, FIXTURE)
    inventory = scraper.get_footprints()

    assert scraper.min_area == 450.0
    expected = [assign_asset_ids([house]).ids[0] for house in HOUSES]
    assert sorted(inventory.get_asset_ids()) == sorted(expected)
    assert assign_asset_ids([SHED]).ids[0] == expected[3]
    assert scraper.asset_id_log == []

    skipped = {entry['position']: entry for entry in scraper.asset_id_skipped}
    assert set(skipped) == {SHED_POSITION, GARAGE_POSITION, FLAT_POSITION}
    assert skipped[SHED_POSITION]['provider_id'] == 1000 + SHED_POSITION
    assert skipped[SHED_POSITION]['plan_area'] == pytest.approx(
        AssetInventory.centroid_and_plan_area(SHED)[2] * SQFT_PER_M2, rel=1e-9
    )
    for asset in inventory.inventory.values():
        assert asset.features['footprint_provider'] == 'osm'


def test_the_floor_is_measured_by_the_id_routine() -> None:
    """A square just under the floor is dropped although _polygon_area puts it above."""
    side_m = math.sqrt(449.6 / SQFT_PER_M2)
    square = rectangle(row_lon(20), side_m, side_m)
    scraper = StubScraper({'id_scheme': 'geohash'}, [*HOUSES, square])
    lons, lats = zip(*square)
    assert AssetInventory.centroid_and_plan_area(square)[2] * SQFT_PER_M2 < 450.0
    assert scraper._polygon_area(list(lats), list(lons), 'ft') > 450.0

    inventory = scraper.get_footprints()

    assert str(square) not in keys_by_geometry(inventory)
    assert [
        (entry['position'], entry['reason']) for entry in scraper.asset_id_skipped
    ] == [(12, 'below_min_area')]


def test_zero_floor_still_drops_footprints_without_area() -> None:
    """With min_area 0 the shed collides with its house and the flat ring is dropped."""
    scraper = StubScraper({'id_scheme': 'geohash', 'min_area': 0}, FIXTURE)
    inventory = scraper.get_footprints()

    keys = keys_by_geometry(inventory)
    cell = assign_asset_ids([HOUSES[3]]).ids[0]
    assert keys[str(HOUSES[3])] == f'{cell}-120'
    assert keys[str(SHED)] == f'{cell}-16'
    assert str(FLAT) not in keys
    assert [entry['reason'] for entry in scraper.asset_id_skipped] == ['zero_area']
    assert [entry['position'] for entry in scraper.asset_id_log] == [
        3,
        SHED_POSITION,
    ]


def test_floor_is_in_the_scrapers_plan_area_unit() -> None:
    """In meters the default floor is DEFAULT_MIN_AREA_SQFT in square meters."""
    scraper = StubScraper({'id_scheme': 'geohash', 'length': 'm'}, FIXTURE)
    inventory = scraper.get_footprints()

    assert scraper.min_area == pytest.approx(450.0 / SQFT_PER_M2, rel=1e-9)
    assert len(inventory.get_asset_ids()) == 12


@pytest.mark.parametrize('unit', ['ft', 'm', 'km', 'mi'])
def test_the_floor_keeps_the_same_footprints_in_every_unit(unit: str) -> None:
    """Squares just below and above the floor are kept alike in every unit."""
    floor_m2 = 450.0 / SQFT_PER_M2
    squares = [
        rectangle(row_lon(index), math.sqrt(area), math.sqrt(area))
        for index, area in enumerate((41.8062, 41.8070, 41.8100))
    ]
    assert [
        AssetInventory.centroid_and_plan_area(square)[2] < floor_m2
        for square in squares
    ] == [True, False, False]

    default = StubScraper({'id_scheme': 'geohash', 'length': unit}, squares)
    default.get_footprints()
    explicit = StubScraper(
        {
            'id_scheme': 'geohash',
            'length': unit,
            'min_area': default.min_area,
        },
        squares,
    )
    explicit.get_footprints()

    for scraper in (default, explicit):
        assert [entry['position'] for entry in scraper.asset_id_skipped] == [0]


def test_the_tiebreak_does_not_depend_on_the_length_unit() -> None:
    """Identical footprints rank by provider id, or by features free of units."""
    square = rectangle(row_lon(20), 10.0, 10.0)
    ranks = []
    for unit in ('ft', 'm'):
        scraper = StubScraper(
            {'id_scheme': 'geohash', 'length': unit}, [square, square]
        )
        inventory = scraper.get_footprints()
        ranks.append(
            {
                asset.features['osm_id']: key.rsplit('-', 1)[1]
                for key, asset in inventory.inventory.items()
            }
        )
    assert ranks == [{1000: '1', 1001: '2'}] * 2

    in_feet = {'type': 'building', 'buildingheight': 32.8, 'footprintArea': 1076}
    in_meters = {'type': 'building', 'buildingheight': 10.0, 'footprintArea': 100}
    assert FootprintScraper._footprint_tiebreak(in_feet, None) == (
        FootprintScraper._footprint_tiebreak(in_meters, None)
    )


def test_invalid_options_are_refused() -> None:
    """An unknown scheme or a negative floor raises at construction."""
    with pytest.raises(ValueError, match='id_scheme'):
        StubScraper({'id_scheme': 'uuid'}, FIXTURE)
    with pytest.raises(ValueError, match='min_area'):
        StubScraper({'min_area': -1}, FIXTURE)
    with pytest.raises(ValueError, match='min_area'):
        StubScraper({'min_area': True}, FIXTURE)


def test_usa_scraper_records_its_ids(mocker: MockerFixture) -> None:
    """The USA scraper requests UUID and keeps both provider ids."""
    records = [
        {
            'geometry': {'rings': [footprint]},
            'attributes': {
                'BUILD_ID': 500 + index,
                'HEIGHT': 6.0,
                'UUID': f'{{uuid-{index}}}',
            },
        }
        for index, footprint in enumerate([*HOUSES[:4], SHED])
    ]
    helper = mocker.MagicMock()
    helper.split_polygon_into_cells.side_effect = lambda bpoly: [bpoly]
    helper.download_attr_from_api.return_value = records
    mocker.patch.object(
        usa_footprint_scraper, 'ArcgisAPIServiceHelper', return_value=helper
    )
    region = RegionBoundary({'type': 'locationPolygon', 'data': BOX})

    scraper = USA_FootprintScraper({'id_scheme': 'geohash'})
    inventory = scraper.get_footprints(region)

    requested = helper.download_attr_from_api.call_args[0][1]
    assert 'UUID' in requested
    assert sorted(inventory.get_asset_ids()) == sorted(
        assign_asset_ids([house]).ids[0] for house in HOUSES[:4]
    )
    features = sorted(
        (asset.features['usa_build_id'], asset.features['usa_uuid'])
        for asset in inventory.inventory.values()
    )
    assert features == [(500 + index, f'{{uuid-{index}}}') for index in range(4)]
    assert {
        asset.features['footprint_provider']
        for asset in inventory.inventory.values()
    } == {'usa_structures'}
    assert [entry['provider_id'] for entry in scraper.asset_id_skipped] == [504]


def test_osm_scraper_records_its_ids(mocker: MockerFixture) -> None:
    """The OSM scraper keeps each way's id."""
    elements = []
    node_id = 1
    for way_index, footprint in enumerate(HOUSES[:3]):
        nodes = []
        for lon, lat in footprint[:-1]:
            elements.append({'type': 'node', 'id': node_id, 'lon': lon, 'lat': lat})
            nodes.append(node_id)
            node_id += 1
        elements.append(
            {
                'type': 'way',
                'id': 9000 + way_index,
                'nodes': [*nodes, nodes[0]],
                'tags': {'building': 'yes'},
            }
        )
    mocker.patch.object(
        osm_footprint_scraper,
        'safe_overpass_json',
        return_value={'elements': elements},
    )
    region = RegionBoundary({'type': 'locationPolygon', 'data': BOX})

    inventory = OSM_FootprintScraper({}).get_footprints(region)

    assert inventory.get_asset_ids() == [0, 1, 2]
    assert [inventory.inventory[key].features['osm_id'] for key in range(3)] == [
        9000,
        9001,
        9002,
    ]
    assert inventory.inventory[0].features['footprint_provider'] == 'osm'


def test_microsoft_scraper_records_its_provider(mocker: MockerFixture) -> None:
    """Microsoft footprints carry no id but name their provider."""
    mocker.patch.object(MS_FootprintScraper, '_bbox2quadkeys', return_value=[1])
    mocker.patch.object(
        MS_FootprintScraper,
        '_download_ms_tiles',
        return_value=(list(HOUSES[:3]), [5.0, 6.0, 7.0]),
    )
    assert ms_footprint_handler.FootprintScraper is FootprintScraper
    region = RegionBoundary({'type': 'locationPolygon', 'data': BOX})

    scraper = MS_FootprintScraper({'id_scheme': 'geohash'})
    inventory = scraper.get_footprints(region)

    assert sorted(inventory.get_asset_ids()) == sorted(
        assign_asset_ids([house]).ids[0] for house in HOUSES[:3]
    )
    assert {
        asset.features['footprint_provider']
        for asset in inventory.inventory.values()
    } == {'microsoft'}


def overture_frame(footprints: list[list[list[float]]]) -> pd.DataFrame:
    """Return Overture rows as the dataset read gives them."""
    return pd.DataFrame(
        {
            'id': [f'ov-{index}' for index in range(len(footprints))],
            'geometry': [Polygon(footprint).wkb for footprint in footprints],
            'sources': [None] * len(footprints),
            'names': [None] * len(footprints),
            'bbox': [None] * len(footprints),
            'height': [4.0] * len(footprints),
        }
    )


def mock_overture(
    mocker: MockerFixture,
    footprints: list[list[list[float]]],
    parts: Sequence[list[list[float]]] = (),
) -> type:
    """Return the Overture scraper serving buildings and parts, read offline."""
    mocker.patch.object(
        OvertureMapsFootprintScraper, 'resolve_release', return_value='2026-07-22.0'
    )

    def read(_release: str, overture_type: str, bbox: Any = None) -> pa.Table:
        del bbox
        if overture_type == 'building':
            return pa.Table.from_pandas(overture_frame(footprints))
        return pa.Table.from_pandas(overture_frame(list(parts)))

    mocker.patch.object(
        OvertureMapsFootprintScraper, '_read_table', side_effect=read
    )
    return OvertureMapsFootprintScraper


def test_overture_scraper_is_a_footprint_scraper(mocker: MockerFixture) -> None:
    """The Overture scraper derives from FootprintScraper and keeps its own state."""
    scraper = mock_overture(mocker, HOUSES[:5])({'length': 'ft'})
    assert isinstance(scraper, FootprintScraper)
    assert scraper.name == 'OvertureMapsFootprintScraper'
    assert scraper.overture_release == '2026-07-22.0'
    assert scraper.id_scheme == 'index'


def test_overture_default_scheme_keeps_positional_keys(
    mocker: MockerFixture,
) -> None:
    """Without options the Overture keys are positional and every footprint is kept."""
    region = RegionBoundary({'type': 'locationPolygon', 'data': BOX})
    scraper = mock_overture(mocker, [*HOUSES[:5], SHED])({'length': 'ft'})
    inventory = scraper.get_footprints(region)

    assert inventory.get_asset_ids() == list(range(6))
    assert [
        inventory.inventory[key].features['overture_id'] for key in range(6)
    ] == [f'ov-{index}' for index in range(6)]
    assert inventory.inventory[0].features['footprint_provider'] == 'overture'
    assert len(scraper.footprints) == len(scraper.centroids) == 6


def test_overture_geohash_scheme(mocker: MockerFixture) -> None:
    """Under geohash the shed is floored out and the houses keep bare geohashes."""
    region = RegionBoundary({'type': 'locationPolygon', 'data': BOX})
    scraper_class = mock_overture(mocker, [*HOUSES[:5], SHED])
    scraper = scraper_class({'length': 'ft', 'id_scheme': 'geohash'})
    inventory = scraper.get_footprints(region)

    assert sorted(inventory.get_asset_ids()) == sorted(
        assign_asset_ids([house]).ids[0] for house in HOUSES[:5]
    )
    assert [
        (entry['provider_id'], entry['reason']) for entry in scraper.asset_id_skipped
    ] == [('ov-5', 'below_min_area')]
    ids = assign_asset_ids(HOUSES[:5], tiebreak=[f'ov-{i}' for i in range(5)]).ids
    for index, key in enumerate(ids):
        assert inventory.inventory[key].features['overture_id'] == f'ov-{index}'


def test_overture_reads_buildings_only(mocker: MockerFixture) -> None:
    """The scraper requests the building type only, so a part never becomes an asset."""
    region = RegionBoundary({'type': 'locationPolygon', 'data': BOX})
    scraper_class = mock_overture(mocker, HOUSES[:2], parts=[HOUSES[5]])
    inventory = scraper_class({'length': 'ft'}).get_footprints(region)

    read = OvertureMapsFootprintScraper._read_table
    assert [call.args[1] for call in read.call_args_list] == ['building']
    assert set(keys_by_geometry(inventory)) == {str(house) for house in HOUSES[:2]}


def test_two_query_boxes_agree_outside_collision_sets(mocker: MockerFixture) -> None:
    """Footprints both boxes return keep their ids unless a box splits a collision set."""
    # A building and a smaller footprint inside it share a centroid 4 m west
    # of the east box's edge. The east box reaches the building but not the
    # smaller footprint.
    east_edge = row_lon(3.5)
    building = rectangle(east_edge - meters_to_lon(4.0), 16.0, 16.0)
    part = rectangle(east_edge - meters_to_lon(4.0), 7.0, 7.0)
    assert assign_asset_ids([building]).ids[0] == assign_asset_ids([part]).ids[0]
    scraper_class = mock_overture(mocker, [*HOUSES, building, part])

    pulls = {}
    log_sizes = {}
    for name, (west, east) in {
        'west': (-122.46, row_lon(7.5)),
        'east': (east_edge, -122.40),
    }.items():
        region = RegionBoundary(
            {'type': 'locationPolygon', 'data': (west, 37.76, east, 37.78)}
        )
        scraper = scraper_class({'length': 'ft', 'id_scheme': 'geohash'})
        pulls[name] = keys_by_geometry(scraper.get_footprints(region))
        log_sizes[name] = len(scraper.asset_id_log)

    shared_houses = [str(house) for house in HOUSES[4:8]]
    assert set(pulls['west']) & set(pulls['east']) == {*shared_houses, str(building)}
    for geometry in shared_houses:
        assert pulls['west'][geometry] == pulls['east'][geometry]

    cell = assign_asset_ids([building]).ids[0]
    assert pulls['west'][str(building)] == f'{cell}-256'
    assert pulls['east'][str(building)] == cell
    assert log_sizes == {'west': 2, 'east': 0}


def test_a_subclass_that_skips_the_base_init_keeps_positional_keys() -> None:
    """A subclass that does not call the base __init__ gets positional keys."""

    class LegacyScraper(FootprintScraper):
        def __init__(self) -> None:
            self.length_unit = 'ft'

        def get_footprints(
            self, region: RegionBoundary | None = None
        ) -> AssetInventory:
            del region
            return self._create_asset_inventory(
                list(FIXTURE), {'buildingheight': [1.0] * len(FIXTURE)}, 'ft'
            )

    inventory = LegacyScraper().get_footprints()
    assert inventory.get_asset_ids() == list(range(len(FIXTURE)))
    assert 'footprint_provider' not in inventory.inventory[0].features
