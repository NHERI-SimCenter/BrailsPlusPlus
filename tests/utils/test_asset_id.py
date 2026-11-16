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


"""Tests for the footprint asset id routine."""

from __future__ import annotations

import ast
import math
import random
from pathlib import Path

import pytest
from shapely.geometry import MultiPolygon, Polygon

from brails.constants import EARTH_RADIUS_M
from brails.utils import asset_id
from brails.utils.asset_id import (
    GEOHASH_PRECISION,
    SUFFIX_SEPARATOR,
    assign_asset_ids,
    geohash,
)

# A 0.0002 by 0.0002 degree footprint in San Francisco and its pinned id. The
# id fixes the coordinate reference system, the centroid routine and the
# geohash precision together. A change to any of them changes it.
GOLDEN_RING = [
    [-122.4194, 37.7749],
    [-122.4192, 37.7749],
    [-122.4192, 37.7751],
    [-122.4194, 37.7751],
    [-122.4194, 37.7749],
]
GOLDEN_ID = '9q8yyk8zrd'

METERS_PER_DEGREE_LAT = math.pi * EARTH_RADIUS_M / 180.0


def rectangle(
    lon: float, lat: float, width_m: float, height_m: float
) -> list[list[float]]:
    """Return a closed counterclockwise rectangle centered on a point."""
    half_lat = height_m / 2.0 / METERS_PER_DEGREE_LAT
    half_lon = width_m / 2.0 / (METERS_PER_DEGREE_LAT * math.cos(math.radians(lat)))
    return [
        [lon - half_lon, lat - half_lat],
        [lon + half_lon, lat - half_lat],
        [lon + half_lon, lat + half_lat],
        [lon - half_lon, lat + half_lat],
        [lon - half_lon, lat - half_lat],
    ]


def footprint_row(count: int, lat: float = 37.77) -> list[list[list[float]]]:
    """Return a row of 12 m square footprints 20 m apart."""
    step = 20.0 / (METERS_PER_DEGREE_LAT * math.cos(math.radians(lat)))
    return [
        rectangle(-122.45 + index * step, lat, 12.0, 12.0) for index in range(count)
    ]


def test_geohash_matches_the_published_example() -> None:
    """The encoding reproduces the standard reference value."""
    assert geohash(10.40744, 57.64911, precision=11) == 'u4pruydqqvj'
    assert geohash(10.40744, 57.64911) == 'u4pruydqqv'
    assert GEOHASH_PRECISION == 10


def test_golden_id() -> None:
    """A fixed footprint keeps its id."""
    assert assign_asset_ids([GOLDEN_RING]).ids == [GOLDEN_ID]


def test_input_forms_give_the_same_id() -> None:
    """The list form, a GeoJSON mapping and a shapely geometry agree."""
    mapping = {'type': 'Polygon', 'coordinates': [GOLDEN_RING]}
    shapely_polygon = Polygon(GOLDEN_RING)
    ids = {
        assign_asset_ids([footprint]).ids[0]
        for footprint in (
            GOLDEN_RING,
            mapping,
            shapely_polygon,
            [tuple(point) for point in GOLDEN_RING],
        )
    }
    assert ids == {GOLDEN_ID}


def test_id_ignores_the_order_of_multipolygon_parts() -> None:
    """Reordering a multipolygon's parts leaves its id unchanged."""
    first = rectangle(-122.41, 37.78, 10.0, 8.0)
    second = rectangle(-122.4098, 37.7801, 6.0, 6.0)
    assert (
        assign_asset_ids([MultiPolygon([Polygon(second), Polygon(first)])]).ids
        == assign_asset_ids([[first, second]]).ids
    )


def test_a_hole_does_not_change_the_id() -> None:
    """A polygon with a hole gets the id of its exterior ring."""
    exterior = rectangle(-122.41, 37.78, 30.0, 30.0)
    hole = rectangle(-122.4100, 37.7801, 5.0, 5.0)
    with_hole = Polygon(exterior, [hole])
    assert assign_asset_ids([with_hole]).ids == assign_asset_ids([exterior]).ids


def test_projected_coordinates_raise() -> None:
    """Coordinates outside EPSG:4326 ranges are refused."""
    projected = [[551000.0, 4180000.0], [551010.0, 4180000.0], [551010.0, 4180010.0]]
    with pytest.raises(ValueError, match='EPSG:4326'):
        assign_asset_ids([projected])


def test_zero_area_rings_are_refused() -> None:
    """Collinear or repeated vertices have no area and no id."""
    collinear = [[-122.41, 37.78], [-122.4099, 37.78], [-122.4098, 37.78]]
    repeated = [[-122.41, 37.78], [-122.41, 37.78], [-122.41, 37.78]]
    for zero_area in (collinear, repeated):
        with pytest.raises(ValueError, match='encloses no area'):
            assign_asset_ids([GOLDEN_RING, zero_area])


def test_ids_are_unchanged_under_reordering() -> None:
    """Shuffling the input changes no footprint's id."""
    footprints = footprint_row(30)
    house = rectangle(-122.43, 37.76, 15.0, 15.0)
    shed = rectangle(-122.43, 37.76, 4.0, 4.0)
    footprints += [house, shed]
    expected = dict(zip(map(str, footprints), assign_asset_ids(footprints).ids))

    shuffled = list(footprints)
    random.Random(7).shuffle(shuffled)
    result = assign_asset_ids(shuffled)
    assert dict(zip(map(str, shuffled), result.ids)) == expected


def test_ids_outside_collision_sets_survive_subsetting() -> None:
    """Only a collision pair's ids depend on the subset, and only while both are in it."""
    footprints = footprint_row(20)
    house = rectangle(-122.43, 37.76, 15.0, 15.0)
    shed = rectangle(-122.43, 37.76, 4.0, 4.0)
    cell = assign_asset_ids([house]).ids[0]
    full = assign_asset_ids([*footprints, house, shed]).ids
    assert full[-2:] == [f'{cell}-225', f'{cell}-16']
    assert all(SUFFIX_SEPARATOR not in value for value in full[:-2])

    for subset, expected in (
        ([*footprints[3:11], house, shed], [*full[3:11], *full[-2:]]),
        ([*footprints[3:11], house], [*full[3:11], cell]),
        ([shed, *footprints[5:9]], [cell, *full[5:9]]),
    ):
        assert assign_asset_ids(subset).ids == expected


def test_a_collision_receives_the_area_suffix() -> None:
    """Footprints sharing a cell are suffixed, and each loses its suffix without its partner."""
    house = rectangle(-122.43, 37.76, 15.0, 15.0)
    shed = rectangle(-122.43, 37.76, 4.0, 4.0)
    cell = assign_asset_ids([house]).ids[0]
    assert assign_asset_ids([shed]).ids[0] == cell

    result = assign_asset_ids([house, shed])
    assert result.ids == [f'{cell}-225', f'{cell}-16']
    assert [entry['reason'] for entry in result.log] == ['collision', 'collision']
    assert [entry['position'] for entry in result.log] == [0, 1]

    assert assign_asset_ids([house]).ids == [cell]
    assert assign_asset_ids([shed]).ids == [cell]


def test_equal_areas_are_ranked_by_geometry_not_input_order() -> None:
    """An equal-area pair is ranked the same in any order and from any start vertex."""
    lat = 37.76
    square = rectangle(-122.43, lat, 10.0, 10.0)
    oblong = rectangle(-122.43, lat, 20.0, 5.0)
    assert assign_asset_ids([square]).ids == assign_asset_ids([oblong]).ids

    forward = assign_asset_ids([square, oblong])
    backward = assign_asset_ids([oblong, square])
    rotated_square = square[1:-1] + square[:2]
    rotated = assign_asset_ids([list(reversed(oblong)), rotated_square])

    assert forward.ids[0] == backward.ids[1] == rotated.ids[1]
    assert forward.ids[1] == backward.ids[0] == rotated.ids[0]
    assert sorted(forward.ids) == sorted(
        [f'{forward.log[0]["geohash"]}-100-1', f'{forward.log[0]["geohash"]}-100-2']
    )
    assert {entry['reason'] for entry in forward.log} == {'equal_area'}


def test_the_tiebreak_ranks_identical_geometry() -> None:
    """Identical geometry is ranked by the tiebreak, and logged as identical only without one."""
    square = rectangle(-122.43, 37.76, 10.0, 10.0)
    by_tiebreak = assign_asset_ids([square, square], tiebreak=['b', 'a'])
    assert [value.rsplit('-', 1)[1] for value in by_tiebreak.ids] == ['2', '1']
    assert {entry['reason'] for entry in by_tiebreak.log} == {'equal_area'}

    identical = assign_asset_ids([square, square])
    assert sorted(value.rsplit('-', 1)[1] for value in identical.ids) == ['1', '2']
    assert {entry['reason'] for entry in identical.log} == {'identical'}


def test_errors_name_the_input_at_fault() -> None:
    """An empty polygon and an invalid geohash coordinate say what is wrong."""
    with pytest.raises(ValueError, match='A Polygon footprint has no coordinates'):
        assign_asset_ids([{'type': 'Polygon', 'coordinates': []}])
    with pytest.raises(ValueError, match='A geohash needs'):
        geohash(200.0, 37.0)


def test_tiebreak_length_is_checked() -> None:
    """A tiebreak that does not match the geometries is refused."""
    with pytest.raises(ValueError, match='tiebreak'):
        assign_asset_ids([GOLDEN_RING], tiebreak=['a', 'b'])


def test_the_module_imports_the_standard_library_only() -> None:
    """The module imports only standard-library modules."""
    source = Path(asset_id.__file__).read_text(encoding='utf-8')
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or '').split('.')[0])
    assert modules <= {'__future__', 'collections', 'math', 'typing'}


def test_the_module_radius_is_the_shared_constant() -> None:
    """The module's Earth radius equals brails.constants.EARTH_RADIUS_M."""
    assert asset_id._EARTH_RADIUS_M == EARTH_RADIUS_M
