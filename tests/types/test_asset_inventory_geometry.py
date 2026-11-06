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


"""Tests for the footprint geometry static methods of AssetInventory."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from brails.types.asset_inventory import AssetInventory
from brails.utils.asset_id import EARTH_RADIUS_M

# A 0.0002 by 0.0002 degree footprint in San Francisco and its plan area.
GOLDEN_RING = [
    [-122.4194, 37.7749],
    [-122.4192, 37.7749],
    [-122.4192, 37.7751],
    [-122.4194, 37.7751],
    [-122.4194, 37.7749],
]
GOLDEN_AREA_M2 = 390.922

METERS_PER_DEGREE_LAT = math.pi * EARTH_RADIUS_M / 180.0


def rectangle(
    lon: float, lat: float, width_m: float, height_m: float
) -> list[list[float]]:
    """
    Return a closed counterclockwise rectangle centered on a point.

    Args:
        lon (float):
            Center longitude.
        lat (float):
            Center latitude.
        width_m (float):
            East-west side in meters.
        height_m (float):
            North-south side in meters.

    Returns:
        list[list[float]]:
            The ring in BRAILS coordinates.
    """
    half_lat = height_m / 2.0 / METERS_PER_DEGREE_LAT
    half_lon = width_m / 2.0 / (METERS_PER_DEGREE_LAT * math.cos(math.radians(lat)))
    return [
        [lon - half_lon, lat - half_lat],
        [lon + half_lon, lat - half_lat],
        [lon + half_lon, lat + half_lat],
        [lon - half_lon, lat + half_lat],
        [lon - half_lon, lat - half_lat],
    ]


class TestCentroidAndPlanArea:
    """The centroid and plan area of a footprint's exterior rings."""

    def test_golden_area(self) -> None:
        """A fixed footprint keeps its plan area."""
        _, _, area = AssetInventory.centroid_and_plan_area(GOLDEN_RING)
        assert area == pytest.approx(GOLDEN_AREA_M2, abs=1e-3)

    def test_centroid_agrees_with_shapely(self) -> None:
        """The planar centroid is the usual polygon centroid."""
        polygon = Polygon(
            [[-122.40, 37.78], [-122.3996, 37.78], [-122.3999, 37.7803]]
        )
        lon, lat, _ = AssetInventory.centroid_and_plan_area(
            list(polygon.exterior.coords)
        )
        assert lon == pytest.approx(polygon.centroid.x, abs=1e-12)
        assert lat == pytest.approx(polygon.centroid.y, abs=1e-12)

    def test_input_forms_agree(self) -> None:
        """The list form, a GeoJSON mapping and a Shapely geometry agree."""
        forms = (
            GOLDEN_RING,
            {'type': 'Polygon', 'coordinates': [GOLDEN_RING]},
            Polygon(GOLDEN_RING),
            [tuple(point) for point in GOLDEN_RING],
        )
        values = {AssetInventory.centroid_and_plan_area(form) for form in forms}
        assert len(values) == 1

    def test_vertex_order_orientation_and_closing_vertex(self) -> None:
        """Start vertex, orientation and closing vertex change neither value."""
        ring = [
            [-122.41, 37.78],
            [-122.4098, 37.7801],
            [-122.4097, 37.7804],
            [-122.4101, 37.7803],
        ]
        expected = AssetInventory.centroid_and_plan_area(ring)
        reversed_rotation = list(reversed(ring[1:] + ring[:1]))
        variants = [
            [*ring, ring[0]],
            ring[2:] + ring[:2],
            list(reversed(ring)),
            [*reversed_rotation, reversed_rotation[0]],
        ]
        for variant in variants:
            assert AssetInventory.centroid_and_plan_area(variant)[:2] == expected[:2]
            assert AssetInventory.centroid_and_plan_area(variant)[2] == (
                pytest.approx(expected[2], rel=1e-12)
            )

    def test_order_of_multipolygon_parts(self) -> None:
        """Reordering a multipolygon's parts changes neither value."""
        first = rectangle(-122.41, 37.78, 10.0, 8.0)
        second = rectangle(-122.4098, 37.7801, 6.0, 6.0)
        assert AssetInventory.centroid_and_plan_area(
            [first, second]
        ) == AssetInventory.centroid_and_plan_area([second, first])

    def test_a_hole_is_ignored(self) -> None:
        """A polygon with a hole has the centroid and area of its exterior ring."""
        exterior = rectangle(-122.41, 37.78, 30.0, 30.0)
        hole = rectangle(-122.4100, 37.7801, 5.0, 5.0)
        with_hole = Polygon(exterior, [hole])
        assert AssetInventory.centroid_and_plan_area(
            with_hole
        ) == AssetInventory.centroid_and_plan_area(exterior)
        assert with_hole.centroid.y != pytest.approx(
            AssetInventory.centroid_and_plan_area(exterior)[1], abs=1e-9
        )

    def test_errors_name_the_input_at_fault(self) -> None:
        """An empty polygon, projected coordinates and no area raise."""
        with pytest.raises(ValueError, match='A Polygon footprint has no coordinates'):
            AssetInventory.centroid_and_plan_area({'type': 'Polygon', 'coordinates': []})
        projected = [[551000.0, 4180000.0], [551010.0, 4180000.0], [551010.0, 4180010.0]]
        with pytest.raises(ValueError, match='EPSG:4326'):
            AssetInventory.centroid_and_plan_area(projected)
        collinear = [[-122.41, 37.78], [-122.4099, 37.78], [-122.4098, 37.78]]
        with pytest.raises(ValueError, match='no centroid'):
            AssetInventory.centroid_and_plan_area(collinear)


class TestHasArea:
    """Whether a footprint's exterior rings enclose any area."""

    def test_zero_area_rings(self) -> None:
        """Collinear or repeated vertices have no area; a square has some."""
        collinear = [[-122.41, 37.78], [-122.4099, 37.78], [-122.4098, 37.78]]
        repeated = [[-122.41, 37.78], [-122.41, 37.78], [-122.41, 37.78]]
        assert not AssetInventory.has_area(collinear)
        assert not AssetInventory.has_area(repeated)
        assert AssetInventory.has_area(GOLDEN_RING)

    def test_invalid_vertices_raise(self) -> None:
        """A vertex that is not a coordinate pair raises."""
        with pytest.raises(ValueError, match='not a coordinate pair'):
            AssetInventory.has_area([[-122.41, 37.78], [True, 37.78], [0.0, 0.0]])
