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

"""Tests for the merge of stacked Overture footprints."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
import pytest
import shapely
from shapely.geometry import Polygon

import brails.compilers
from brails.compilers.stacked_footprints import (
    PARTS_COLUMNS,
    RULE_COPY,
    RULE_STACKED,
    StackedFootprintMerge,
    StackedMergeResult,
)
from brails.inferers import StoriesFromHeightInferer
from brails.scrapers.overture_maps_scraper.overture_maps_footprint_scraper import (
    OvertureMapsFootprintScraper,
)
from brails.types.asset_inventory import Asset, AssetInventory
from brails.types.region_boundary import RegionBoundary
from brails.utils import Importer, UnitConverter
from brails.utils.asset_id import assign_asset_ids

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

METERS_PER_DEGREE = math.pi * 6371008.8 / 180.0
LAT = 37.77
LON = -122.42
REGION = (-122.43, 37.76, -122.40, 37.78)

MICROSOFT = 'dataset: Microsoft ML Buildings'
MICROSOFT_HEIGHT = 'property: /properties/height, dataset: Microsoft ML Buildings'
OSM = 'dataset: OpenStreetMap'
OSM_HEIGHT = 'property: /properties/height, dataset: OpenStreetMap, record_id: w1'

# Scales of the stories inferer, not the packaged calibration.
FIXTURE_SCALES = {
    'length_unit': 'm',
    'providers': {
        'overture': {
            'sources': {
                'Microsoft ML Buildings': {'scale': 3.92, 'max_stories': 2},
                'OpenStreetMap': {'scale': 4.0, 'max_stories': None},
            }
        }
    },
}


def box(x: float, y: float, width: float, depth: float) -> list[list[float]]:
    """
    Return a closed rectangle given in meters from (``LON``, ``LAT``).

    Args:
        x (float):
            The center's distance east, in meters.
        y (float):
            The center's distance north, in meters.
        width (float):
            The east-west side, in meters.
        depth (float):
            The north-south side, in meters.

    Returns:
        list[list[float]]:
            The ring as ``[lon, lat]`` pairs.
    """
    per_lon = METERS_PER_DEGREE * math.cos(math.radians(LAT))
    lon, lat = LON + x / per_lon, LAT + y / METERS_PER_DEGREE
    half_lon, half_lat = width / 2.0 / per_lon, depth / 2.0 / METERS_PER_DEGREE
    return [
        [lon - half_lon, lat - half_lat],
        [lon + half_lon, lat - half_lat],
        [lon + half_lon, lat + half_lat],
        [lon - half_lon, lat + half_lat],
        [lon - half_lon, lat - half_lat],
    ]


def overture(
    overture_id: str, ring: list[list[float]], **features: Any
) -> tuple[list[list[float]], dict[str, Any]]:
    """
    Return an Overture footprint as the scraper records it.

    Args:
        overture_id (str):
            The footprint's ``overture_id``.
        ring (list[list[float]]):
            Its coordinates.
        **features (Any):
            Its other features, such as ``height``.

    Returns:
        tuple[list[list[float]], dict[str, Any]]:
            The coordinates and the features.
    """
    return ring, {
        'type': 'building',
        'footprint_provider': 'overture',
        'overture_id': overture_id,
        **features,
    }


def inventory_of(
    *footprints: tuple[list[list[float]], dict[str, Any]],
) -> AssetInventory:
    """
    Key footprints by geohash with the Overture ids as tiebreak.

    Args:
        *footprints (tuple[list[list[float]], dict[str, Any]]):
            The coordinates and features of each footprint.

    Returns:
        AssetInventory:
            The keyed inventory.
    """
    keys = assign_asset_ids(
        [ring for ring, _ in footprints],
        tiebreak=[features['overture_id'] for _, features in footprints],
    ).ids
    inventory = AssetInventory()
    for key, (ring, features) in zip(keys, footprints):
        inventory.add_asset(key, Asset(key, ring, features))
    return inventory


def by_overture_id(result: StackedMergeResult) -> dict[str, tuple[str, Asset]]:
    """
    Index a merge's assets by ``overture_id``.

    Args:
        result (StackedMergeResult):
            The merge's result.

    Returns:
        dict[str, tuple[str, Asset]]:
            Each ``overture_id`` to the asset's key and the asset.
    """
    return {
        asset.features['overture_id']: (key, asset)
        for key, asset in result.inventory.inventory.items()
    }


PODIUM = box(0.0, 0.0, 40.0, 40.0)
# 95% inside the podium, so the union reaches past its east side.
TOWER = box(15.5, 0.0, 10.0, 10.0)


def test_a_tower_merges_into_its_podium() -> None:
    """A podium without height and the tower on it become one asset."""
    inventory = inventory_of(
        overture('podium', PODIUM, num_floors=2, sources=MICROSOFT),
        overture('tower', TOWER, height=40.0, num_floors=12, sources=OSM_HEIGHT),
    )
    result = StackedFootprintMerge(length_unit='m').merge(inventory)

    assert list(by_overture_id(result)) == ['podium']
    key, asset = by_overture_id(result)['podium']
    union = shapely.union_all([Polygon(PODIUM), Polygon(TOWER)])
    assert Polygon(asset.coordinates).equals(union)
    assert not Polygon(asset.coordinates).equals(Polygon(PODIUM))
    assert asset.features['height'] == 40.0
    assert asset.features['num_floors'] == 12
    assert asset.features['overture_part_ids'] == 'tower'

    assert list(result.parts.columns) == list(PARTS_COLUMNS)
    assert result.parts[
        ['asset_id', 'rule', 'head_overture_id', 'overture_id', 'is_head']
    ].to_dict('records') == [
        {
            'asset_id': key,
            'rule': RULE_STACKED,
            'head_overture_id': 'podium',
            'overture_id': overture_id,
            'is_head': overture_id == 'podium',
        }
        for overture_id in ('podium', 'tower')
    ]
    tower_row = result.parts.iloc[1]
    assert tower_row['height'] == 40.0
    assert tower_row['merged_height'] == 40.0
    assert tower_row['merged_num_floors'] == 12
    assert tower_row['plan_area_m2'] == pytest.approx(100.0, rel=1e-6)
    assert math.isnan(result.parts.iloc[0]['height'])


@pytest.mark.parametrize(
    ('part_height', 'merges'), [(13.5, True), (13.0, False), (12.0, False)]
)
def test_a_part_merges_when_taller_by_more_than_three_meters(
    part_height: float, *, merges: bool
) -> None:
    """A part 3.5 m taller than its container merges, one 3 m or 2 m taller not."""
    inventory = inventory_of(
        overture('container', PODIUM, height=10.0),
        overture('part', TOWER, height=part_height),
    )
    result = StackedFootprintMerge(length_unit='m').merge(inventory)
    assert ('part' not in by_overture_id(result)) is merges


@pytest.mark.parametrize('roof', ['container', 'part'])
def test_a_roof_outline_merges_whatever_the_heights(roof: str) -> None:
    """A roof-class container or member merges at equal heights."""
    inventory = inventory_of(
        overture(
            'container',
            PODIUM,
            height=10.0,
            **({'class': 'roof'} if roof == 'container' else {}),
        ),
        overture(
            'part',
            TOWER,
            height=10.0,
            **({'class': 'roof'} if roof == 'part' else {}),
        ),
    )
    result = StackedFootprintMerge(length_unit='m').merge(inventory)
    assert list(by_overture_id(result)) == ['container']


def test_a_nested_pair_without_signal_stays_two_assets() -> None:
    """Equal heights and no roof class leave a nested pair as it is."""
    inventory = inventory_of(
        overture('container', PODIUM, height=10.0, **{'class': 'apartments'}),
        overture('part', TOWER, height=10.0),
    )
    result = StackedFootprintMerge(length_unit='m').merge(inventory)
    assert result.inventory.get_asset_ids() == inventory.get_asset_ids()
    assert result.parts.empty


@pytest.mark.parametrize(('inside', 'merges'), [(0.85, False), (0.92, True)])
def test_a_member_must_lie_nine_tenths_inside(
    inside: float, *, merges: bool
) -> None:
    """A member 85% inside its container stays and one 92% inside merges."""
    # The member's west part, a share of its 10 m width, lies in the podium.
    member = box(20.0 - 10.0 * inside + 5.0, 0.0, 10.0, 10.0)
    inventory = inventory_of(
        overture('container', PODIUM),
        overture('member', member, height=30.0),
    )
    result = StackedFootprintMerge(length_unit='m').merge(inventory)
    assert ('member' not in by_overture_id(result)) is merges


def test_a_nest_merges_into_its_outermost_footprint() -> None:
    """A footprint in a container that lies in another merges into the outermost."""
    # The middle lies 92.5% inside the outer one, and the inner one only in
    # the middle, half of it outside the outer one.
    outer = box(0.0, 0.0, 40.0, 40.0)
    middle = box(11.5, 0.0, 20.0, 20.0)
    inner = box(20.0, 0.0, 3.0, 3.0)
    footprints = [
        overture('outer', outer),
        overture('middle', middle, height=10.0),
        overture('inner', inner, height=30.0),
    ]
    merge = StackedFootprintMerge(length_unit='m')
    result = merge.merge(inventory_of(*footprints))

    assert list(by_overture_id(result)) == ['outer']
    assert by_overture_id(result)['outer'][1].features['overture_part_ids'] == (
        'inner;middle'
    )
    assert set(result.parts['head_overture_id']) == {'outer'}

    # The same footprints in reverse order give the same result.
    reversed_result = merge.merge(inventory_of(*reversed(footprints)))
    assert reversed_result.inventory.get_asset_ids() == (
        result.inventory.get_asset_ids()
    )
    assert reversed_result.parts.equals(result.parts)


def test_copies_keep_one_drawing() -> None:
    """Copies keep the drawing with a height, else the larger one."""
    # Each pair overlaps at an intersection over union of about 0.8.
    with_height = box(0.0, 0.0, 10.0, 10.0)
    larger = box(1.0, 0.0, 10.0, 10.5)
    smaller = box(100.0, 0.0, 10.0, 10.0)
    largest = box(101.0, 0.0, 10.0, 10.5)
    footprints = (
        overture('with_height', with_height, height=6.0),
        overture('larger', larger),
        overture('smaller', smaller, height=6.0),
        overture('largest', largest, height=6.0),
    )
    result = StackedFootprintMerge(length_unit='m').merge(inventory_of(*footprints))
    assert sorted(by_overture_id(result)) == ['largest', 'with_height']
    assert set(result.parts['rule']) == {RULE_COPY}
    assert result.parts.groupby('head_overture_id')['overture_id'].apply(
        sorted
    ).to_dict() == {
        'with_height': ['larger', 'with_height'],
        'largest': ['largest', 'smaller'],
    }


def test_copies_inside_a_nest_merge_through_their_keeper() -> None:
    """Of two drawings of a tower on a podium only the kept one is a part."""
    # Copies at an intersection over union of 0.94, of which the larger is kept.
    kept = box(15.5, 0.3, 10.0, 10.4)
    inventory = inventory_of(
        overture('podium', PODIUM),
        overture('dropped', TOWER, height=50.0),
        overture('kept', kept, height=40.0),
    )
    result = StackedFootprintMerge(length_unit='m').merge(inventory)

    ((key, asset),) = result.inventory.inventory.items()
    assert asset.features['overture_id'] == 'podium'
    assert asset.features['overture_part_ids'] == 'kept'
    assert asset.features['height'] == 40.0
    assert result.parts[
        ['rule', 'overture_id', 'asset_id', 'merged_height']
    ].to_dict('records') == [
        {
            'rule': rule,
            'overture_id': overture_id,
            'asset_id': key,
            'merged_height': 40.0,
        }
        for rule, overture_id in (
            (RULE_COPY, 'kept'),
            (RULE_COPY, 'dropped'),
            (RULE_STACKED, 'podium'),
            (RULE_STACKED, 'kept'),
        )
    ]


def test_a_part_below_the_floor_lends_no_height(mocker: MockerFixture) -> None:
    """The scraper's floor removes a small part before the merge sees it."""
    shed = box(-10.0, 10.0, 5.0, 5.0)  # 25 m2, under DEFAULT_MIN_AREA_SQFT
    footprints = [PODIUM, TOWER, shed]
    mocker.patch.object(
        OvertureMapsFootprintScraper, 'resolve_release', return_value='2026-07-22.0'
    )
    frame = pd.DataFrame(
        {
            'id': ['podium', 'tower', 'shed'],
            'geometry': [Polygon(ring).wkb for ring in footprints],
            'sources': [None] * 3,
            'names': [None] * 3,
            'bbox': [None] * 3,
            'height': [float('nan'), 40.0, 60.0],
        }
    )
    mocker.patch.object(
        OvertureMapsFootprintScraper,
        '_read_table',
        side_effect=lambda _release, overture_type, **_: pa.Table.from_pandas(
            frame if overture_type == 'building' else frame.iloc[:0]
        ),
    )
    scraper = OvertureMapsFootprintScraper({'length': 'ft', 'id_scheme': 'geohash'})
    inventory = scraper.get_footprints(
        RegionBoundary({'type': 'locationPolygon', 'data': REGION})
    )
    assert [entry['provider_id'] for entry in scraper.asset_id_skipped] == ['shed']

    result = StackedFootprintMerge(length_unit='ft').merge(inventory)
    (asset,) = result.inventory.inventory.values()
    assert asset.features['height'] == UnitConverter.convert_length(40.0, 'm', 'ft')
    assert list(result.parts['overture_id']) == ['podium', 'tower']


def test_keys_are_the_ids_of_the_merged_set() -> None:
    """The merged set is re-keyed, and an unmerged footprint keeps its key."""
    house = box(100.0, 0.0, 12.0, 10.0)
    inventory = inventory_of(
        overture('podium', PODIUM),
        overture('tower', TOWER, height=40.0),
        overture('house', house, height=6.0),
    )
    result = StackedFootprintMerge(length_unit='m').merge(inventory)

    keys = list(result.inventory.inventory)
    assets = list(result.inventory.inventory.values())
    assert (
        keys
        == assign_asset_ids(
            [asset.coordinates for asset in assets],
            tiebreak=[asset.features['overture_id'] for asset in assets],
        ).ids
    )
    podium_key = assign_asset_ids([PODIUM]).ids[0]
    assert by_overture_id(result)['podium'][0] != podium_key, (
        'the fixture must move the centroid out of the podium cell'
    )
    assert by_overture_id(result)['house'][0] == assign_asset_ids([house]).ids[0]

    unmerged = inventory_of(overture('podium', PODIUM), overture('house', house))
    assert (
        StackedFootprintMerge().merge(unmerged).inventory.get_asset_ids()
        == unmerged.get_asset_ids()
    )


@pytest.mark.parametrize('part_height_m', [13.5, 12.5])
def test_the_length_unit_does_not_change_the_merges(part_height_m: float) -> None:
    """Heights in feet and in meters give the same merges."""
    merged = {}
    for unit in ('m', 'ft'):
        inventory = inventory_of(
            overture(
                'container',
                PODIUM,
                height=UnitConverter.convert_length(10.0, 'm', unit),
            ),
            overture(
                'part',
                TOWER,
                height=UnitConverter.convert_length(part_height_m, 'm', unit),
            ),
        )
        result = StackedFootprintMerge(length_unit=unit).merge(inventory)
        merged[unit] = sorted(by_overture_id(result))
    assert merged['m'] == merged['ft']
    assert merged['m'] == (
        ['container'] if part_height_m > 13.0 else ['container', 'part']
    )


def test_the_height_keeps_the_scale_of_its_source() -> None:
    """The stories inferer reads the merged height with the tallest part's scale."""
    # The tallest part names its dataset without a property, and a shorter
    # part names another dataset as the source of its own height.
    inventory = inventory_of(
        overture('head', PODIUM, sources=MICROSOFT),
        overture(
            'short', box(-15.0, 0.0, 6.0, 6.0), height=20.0, sources=MICROSOFT_HEIGHT
        ),
        overture('tall', TOWER, height=40.0, sources=OSM),
    )
    result = StackedFootprintMerge(length_unit='m').merge(inventory)
    (asset,) = result.inventory.inventory.values()
    assert asset.features['sources'] == f'{OSM}; {MICROSOFT}'
    assert asset.features['overture_part_ids'] == 'short;tall'

    inferer = StoriesFromHeightInferer(length_unit='m', scales=FIXTURE_SCALES)
    tall = inventory.inventory[assign_asset_ids([TOWER]).ids[0]].features
    assert inferer.stories_of(asset.features) == inferer.stories_of(tall) == 10


@pytest.mark.parametrize(
    ('case', 'message'),
    [
        ('provider', 'not an Overture footprint'),
        ('overture_id', 'not an Overture footprint'),
        ('integer_keys', 'not a string'),
        ('worlds', 'possible worlds'),
        ('no_area', 'not a polygon with area'),
    ],
)
def test_refusals(case: str, message: str) -> None:
    """The merge refuses what it cannot re-key or does not know how to read."""
    footprints = [
        overture('podium', PODIUM),
        overture('tower', TOWER, height=40.0),
    ]
    if case == 'provider':
        footprints[1][1]['footprint_provider'] = 'osm'
    if case == 'overture_id':
        del footprints[1][1]['overture_id']
    if case == 'no_area':
        west, south = PODIUM[0]
        footprints[1] = overture(
            'flat',
            [
                [west, south],
                [west + 1e-4, south],
                [west + 2e-4, south],
                [west, south],
            ],
        )
    inventory = AssetInventory()
    for index, (ring, features) in enumerate(footprints):
        key = index if case == 'integer_keys' else f'key{index}'
        inventory.add_asset(key, Asset(key, ring, features))
    if case == 'worlds':
        inventory.n_pw = 2
    with pytest.raises(ValueError, match=message):
        StackedFootprintMerge().merge(inventory)


def test_an_unknown_length_unit_is_refused() -> None:
    """A length unit the unit converter does not know raises."""
    with pytest.raises(ValueError, match='not a length unit'):
        StackedFootprintMerge(length_unit='kg')


def test_the_input_inventory_is_unchanged() -> None:
    """A merge leaves its input as it was."""
    inventory = inventory_of(
        overture('podium', PODIUM, sources=MICROSOFT),
        overture('tower', TOWER, height=40.0, sources=OSM_HEIGHT),
        overture('house', box(100.0, 0.0, 12.0, 10.0), height=6.0),
    )
    before = {
        key: (deepcopy(asset.coordinates), deepcopy(asset.features))
        for key, asset in inventory.inventory.items()
    }
    result = StackedFootprintMerge(length_unit='m').merge(inventory)
    for asset in result.inventory.inventory.values():
        asset.coordinates[0][0] = 0.0
        asset.features['height'] = -1.0
    assert {
        key: (asset.coordinates, asset.features)
        for key, asset in inventory.inventory.items()
    } == before


def test_discovered_and_not_exported() -> None:
    """The Importer finds the class, which the package does not export."""
    assert Importer().get_class('StackedFootprintMerge') is StackedFootprintMerge
    assert 'StackedFootprintMerge' not in brails.compilers.__all__
