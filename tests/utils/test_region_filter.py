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


"""Tests for filtering an inventory to the assets inside a region."""

from __future__ import annotations

import io
import json
import sys
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from brails.constants import BLOCK_GROUP_KEY
from brails.scrapers.us_census_scrapers import census_tract_scraper
from brails.scrapers.us_census_scrapers.census_tract_scraper import (
    CensusTractScraper,
)
from brails.types.asset_inventory import Asset, AssetInventory
from brails.types.census_boundary import CensusBoundary
from brails.types.housing_unit_inventory import HousingUnitInventory
from brails.types.region_boundary import RegionBoundary
from brails.utils import region_filter
from brails.utils.region_filter import (
    filter_to_region,
    find_inside,
    read_boundary,
    reference_point,
)

if TYPE_CHECKING:
    from pathlib import Path

# The region of most tests: a square degree's hundredth around (-122.0, 38.0).
REGION = box(-122.01, 37.99, -121.99, 38.01)

# A point inside REGION, as GeoJSON text.
POINT = '{"type": "Point", "coordinates": [-122.0, 38.0]}'


pytestmark = pytest.mark.usefixtures('no_network')


def square(lon: float, lat: float, half: float = 0.0001) -> list[list[float]]:
    """
    Return a closed square ring centered on a point.

    Args:
        lon (float):
            Center longitude.
        lat (float):
            Center latitude.
        half (float, optional):
            Half the side in degrees. Defaults to 0.0001.

    Returns:
        list[list[float]]:
            The ring in BRAILS coordinates.
    """
    return [
        [lon - half, lat - half],
        [lon + half, lat - half],
        [lon + half, lat + half],
        [lon - half, lat + half],
        [lon - half, lat - half],
    ]


def inventory(
    records: dict[Any, tuple[list, dict[str, Any]]],
    n_pw: int = 1,
    vector_features: list[str] | None = None,
) -> AssetInventory:
    """
    Build an inventory from ids, coordinates and features.

    Args:
        records (dict[Any, tuple[list, dict[str, Any]]]):
            Asset id to its coordinates and features.
        n_pw (int, optional):
            The number of worlds. Defaults to 1.
        vector_features (list[str] | None, optional):
            The inventory's vector features, or ``None`` for the default.

    Returns:
        AssetInventory:
            The inventory.
    """
    inv = AssetInventory(vector_features=vector_features)
    for asset_id, (coordinates, features) in records.items():
        inv.add_asset(asset_id, Asset(asset_id, coordinates, features))
    inv.n_pw = n_pw
    return inv


def three_worlds() -> AssetInventory:
    """
    Return an inventory of three worlds, two assets inside the region.

    Returns:
        AssetInventory:
            Assets ``in_a`` and ``in_b`` inside :data:`REGION`, ``out``
            outside it, with list-valued, scalar and vector features.
    """
    return inventory(
        {
            'in_a': (
                square(-122.005, 38.0),
                {
                    'StructureType': ['W1', 'W2', 'C1'],
                    'PlanArea': 120.5,
                    'HousingUnits': [3, 4],
                    'name': 'Café',
                },
            ),
            'in_b': (
                [[-122.0, 38.005]],
                {'StructureType': ['S1', 'S1', 'S2'], 'PlanArea': None},
            ),
            'out': (
                square(-121.9, 38.0),
                {'StructureType': ['W1', 'W1', 'W1'], 'PlanArea': 80.0},
            ),
        },
        n_pw=3,
    )


def write(path: Path, document: dict[str, Any]) -> Path:
    """
    Write a JSON document as UTF-8 and return its path.

    Args:
        path (Path):
            The file.
        document (dict[str, Any]):
            The document.

    Returns:
        Path:
            ``path``.
    """
    path.write_text(json.dumps(document, ensure_ascii=False), encoding='utf-8')
    return path


class TestInside:
    """An asset is inside when its reference point lies strictly inside."""

    def test_points(self) -> None:
        """A point inside is kept, and points outside or on the edge are not."""
        inv = inventory(
            {
                'inside': ([[-122.0, 38.0]], {}),
                'outside': ([[-121.98, 38.0]], {}),
                'edge': ([[-122.01, 38.0]], {}),
                'corner': ([[-121.99, 38.01]], {}),
            }
        )
        assert filter_to_region(inv, REGION).get_asset_ids() == ['inside']

    def test_footprints_by_their_reference_point(self) -> None:
        """An outline across the boundary is kept or left out by its centroid."""
        inv = inventory(
            {
                # Two thirds inside, centroid inside.
                'mostly_in': (square(-121.9905, 38.0, half=0.0015), {}),
                # Two thirds outside, centroid outside, first vertex inside.
                'mostly_out': (square(-121.9895, 38.0, half=0.0015), {}),
                # An L whose arm reaches east past the boundary: the centroid
                # of its ring is inside, the center of its bounds outside.
                'l_shape': (
                    [
                        [-121.992, 37.999],
                        [-121.985, 37.999],
                        [-121.985, 37.9992],
                        [-121.991, 37.9992],
                        [-121.991, 38.001],
                        [-121.992, 38.001],
                        [-121.992, 37.999],
                    ],
                    {},
                ),
            }
        )
        assert sorted(filter_to_region(inv, REGION).inventory) == [
            'l_shape',
            'mostly_in',
        ]

    def test_multipolygon_footprint_and_region(self) -> None:
        """A footprint's parts weigh by area; a region's parts all count."""
        big_inside = square(-122.0, 38.0, half=0.001)
        small_outside = square(-121.98, 38.0, half=0.0001)
        inv = inventory(
            {
                'two_parts': ([big_inside, small_outside], {}),
                'east': ([[-121.95, 38.0]], {}),
                'far': ([[-121.0, 38.0]], {}),
            }
        )
        region = MultiPolygon([REGION, box(-121.96, 37.99, -121.94, 38.01)])
        assert sorted(filter_to_region(inv, region).inventory) == [
            'east',
            'two_parts',
        ]

    def test_the_tract_scraper_locates_by_the_same_point(self) -> None:
        """
        Off the shared edge, each half keeps the block group the scraper gives.

        On the edge the scraper gives the lowest GEOID and the filter neither.
        """
        assert census_tract_scraper.reference_point is reference_point
        west = box(-122.01, 37.99, -122.0, 38.01)
        east = box(-122.0, 37.99, -121.99, 38.01)
        frame = gpd.GeoDataFrame(
            {'GEOID': ['060010001001', '060010001002']},
            geometry=[west, east],
            crs='EPSG:4326',
        )
        records = {
            'w': (square(-122.005, 38.0), {}),
            'e': (square(-121.995, 38.0), {}),
            'across_w': (square(-122.0002, 38.0, half=0.0005), {}),
            'across_e': (square(-121.9998, 38.0, half=0.0005), {}),
            'point_e': ([[-121.993, 38.004]], {}),
            'on_edge': ([[-122.0, 38.003]], {}),
        }
        located, _ = CensusTractScraper().get_census_geographies(
            inventory(records), levels=('block_group',), polygons=frame
        )
        for geoid, half in (('060010001001', west), ('060010001002', east)):
            expected = {
                asset_id
                for asset_id in located.inventory
                if located.inventory[asset_id].features[BLOCK_GROUP_KEY] == geoid
                and asset_id != 'on_edge'
            }
            assert set(filter_to_region(inventory(records), half).inventory) == (
                expected
            )
        assert located.inventory['on_edge'].features[BLOCK_GROUP_KEY] == (
            '060010001001'
        )


class TestInventory:
    """An inventory keeps its worlds and vector features, as copies."""

    def test_worlds_and_vector_features(self) -> None:
        """The result has the source's n_pw and vector features and validates."""
        source = three_worlds()
        source.vector_features = ['HousingUnits', 'Extra']
        result = filter_to_region(source, REGION)
        assert sorted(result.inventory) == ['in_a', 'in_b']
        assert result.n_pw == 3
        assert result.vector_features == ['HousingUnits', 'Extra']
        assert result.vector_features is not source.vector_features
        result.validate_possible_worlds()
        assert (
            result.get_world_realization(2)
            .inventory['in_a']
            .features['StructureType']
            == 'C1'
        )

    def test_kept_assets_are_copies(self) -> None:
        """Changing the result leaves the source as it was."""
        source = three_worlds()
        before = source.get_geojson()['features']
        result = filter_to_region(source, REGION)
        asset = result.inventory['in_a']
        assert asset is not source.inventory['in_a']
        asset.features['StructureType'][0] = 'URM'
        asset.features['HousingUnits'].append(5)
        asset.features['YearBuilt'] = [1950, 1960, 1970]
        asset.coordinates[0][0] = -122.009
        result.remove_asset('in_b')
        assert source.get_geojson()['features'] == before
        assert sorted(source.inventory) == ['in_a', 'in_b', 'out']

    def test_empty_inventory(self) -> None:
        """An inventory without assets gives an empty one with its worlds."""
        result = filter_to_region(inventory({}, n_pw=4), REGION)
        assert result.inventory == {}
        assert result.n_pw == 4

    def test_housing_unit_inventory_refused(self) -> None:
        """An inventory holding housing units is refused, not divided."""
        source = three_worlds()
        source.housing_unit_inventory = HousingUnitInventory()
        with pytest.raises(ValueError, match=r'remove_housing_unit_inventory\(\)'):
            filter_to_region(source, REGION)
        with pytest.raises(ValueError, match='^The inventory holds a housing-unit'):
            filter_to_region(source, REGION)
        with pytest.raises(ValueError, match='^Inventory 2 of the list holds'):
            filter_to_region([three_worlds(), source], REGION)

    def test_asset_without_coordinates(self) -> None:
        """An asset without coordinates names itself in the error."""
        source = inventory({'bare': ([], {})})
        with pytest.raises(ValueError, match="'bare' has no coordinates"):
            filter_to_region(source, REGION)


class TestFile:
    """A GeoJSON file is streamed, its worlds kept, and written back equal."""

    @pytest.mark.parametrize('chunk', [1, 7, 64, 300, None])
    def test_file_equals_the_inventory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk: int | None
    ) -> None:
        """At any chunk size the file gives what the inventory gives."""
        if chunk is not None:
            monkeypatch.setattr(region_filter, 'READ_CHUNK_CHARS', chunk)
        source = three_worlds()
        path = tmp_path / 'county.geojson'
        source.write_to_geojson(str(path))
        from_file = filter_to_region(path, REGION)
        from_memory = filter_to_region(source, REGION)
        assert from_file.n_pw == 3
        assert sorted(from_file.inventory) == sorted(from_memory.inventory)
        for asset_id, asset in from_memory.inventory.items():
            assert from_file.inventory[asset_id].features == asset.features
            assert from_file.inventory[asset_id].coordinates == asset.coordinates
        from_file.validate_possible_worlds()

    def test_written_back_equal_to_the_source(self, tmp_path: Path) -> None:
        """write_to_geojson writes the kept features as the source holds them."""
        source_path = tmp_path / 'county.geojson'
        three_worlds().write_to_geojson(str(source_path))
        result = filter_to_region(str(source_path), REGION)
        assert all('type' not in a.features for a in result.inventory.values())
        out = tmp_path / 'place.geojson'
        result.write_to_geojson(str(out))
        written = json.loads(out.read_text(encoding='utf-8'))
        source = json.loads(source_path.read_text(encoding='utf-8'))
        kept = [f for f in source['features'] if f['id'] in {'in_a', 'in_b'}]
        assert sorted(written['features'], key=lambda f: f['id']) == kept

    def test_boundary_rule_on_the_file_route(self, tmp_path: Path) -> None:
        """A feature on an edge or a corner of the region is left out of a file."""
        source = inventory(
            {
                'inside': ([[-122.0, 38.0]], {}),
                'edge': ([[-122.01, 38.0]], {}),
                'corner': ([[-121.99, 38.01]], {}),
                'footprint_on_edge': (square(-122.0, 37.99), {}),
                'outside': ([[-121.98, 38.0]], {}),
            }
        )
        path = tmp_path / 'edges.geojson'
        source.write_to_geojson(str(path))
        assert list(filter_to_region(path, REGION).inventory) == ['inside']

    def test_integer_id_export_with_asset_type(self, tmp_path: Path) -> None:
        """A filtered county file is written with integer ids given asset_type."""
        path = tmp_path / 'county.geojson'
        three_worlds().write_to_geojson(str(path))
        result = filter_to_region(path, REGION)
        out = tmp_path / 'r2d.geojson'
        with pytest.raises(ValueError, match="no 'type' value.* Pass asset_type"):
            result.write_to_geojson(str(out), integer_ids=True)
        written = result.write_to_geojson(
            str(out), integer_ids=True, asset_type='building'
        )
        assert [f['id'] for f in written['features']] == [1, 2]
        assert {f['properties']['type'] for f in written['features']} == {'building'}
        assert all('type' not in a.features for a in result.inventory.values())

    def test_integer_id_round_trip(self, tmp_path: Path) -> None:
        """County file, filter, integer ids, filter, integer ids: equal features."""
        county = tmp_path / 'county.geojson'
        three_worlds().write_to_geojson(str(county))
        first = tmp_path / 'r2d_first.geojson'
        filter_to_region(county, REGION).write_to_geojson(
            str(first), integer_ids=True, asset_type='building'
        )
        again = filter_to_region(first, REGION)
        assert list(again.inventory) == ['in_a', 'in_b']
        assert all('unique_id' not in a.features for a in again.inventory.values())
        assert all(
            a.features['type'] == 'building' for a in again.inventory.values()
        )
        second = tmp_path / 'r2d_second.geojson'
        again.write_to_geojson(str(second), integer_ids=True)
        assert (
            json.loads(second.read_text(encoding='utf-8'))['features']
            == json.loads(first.read_text(encoding='utf-8'))['features']
        )

    def test_integer_id_file_of_several_types(self, tmp_path: Path) -> None:
        """Numbers repeat across types, and the unique_ids key the assets."""
        source = three_worlds()
        source.inventory['in_a'].features['type'] = 'building'
        source.inventory['in_b'].features['type'] = 'bridge'
        source.inventory['out'].features['type'] = 'building'
        path = tmp_path / 'r2d.geojson'
        written = source.write_to_geojson(str(path), integer_ids=True)
        assert sorted(f['id'] for f in written['features']) == [1, 1, 2]
        result = filter_to_region(path, REGION)
        assert {k: a.features['type'] for k, a in result.inventory.items()} == {
            'in_a': 'building',
            'in_b': 'bridge',
        }

    def test_digit_unique_ids_stay_strings(self, tmp_path: Path) -> None:
        """Digit unique_ids key the assets as strings, and the file is the same."""
        source = inventory(
            {
                5: ([[-122.0, 38.0]], {'type': 'building'}),
                12: ([[-122.001, 38.0]], {'type': 'building'}),
            }
        )
        path = tmp_path / 'r2d.geojson'
        first = source.write_to_geojson(str(path), integer_ids=True)
        result = filter_to_region(path, REGION)
        assert list(result.inventory) == ['5', '12']
        again = result.write_to_geojson(integer_ids=True)
        assert again['features'] == first['features']

    def test_default_form_unique_id_is_a_property(self, tmp_path: Path) -> None:
        """A unique_id beside a string id is kept as written, and written back."""
        source = inventory(
            {
                'a': ([[-122.0, 38.0]], {'unique_id': 'SRC-1'}),
                'b': ([[-122.001, 38.0]], {'unique_id': 'SRC-2'}),
            }
        )
        path = tmp_path / 'county.geojson'
        first = source.write_to_geojson(str(path))
        result = filter_to_region(path, REGION)
        assert list(result.inventory) == ['a', 'b']
        assert result.inventory['a'].features == {'unique_id': 'SRC-1'}
        again = result.write_to_geojson()
        assert again['features'] == first['features']

    def test_integer_id_with_other_unique_id_is_not_the_form(
        self, tmp_path: Path
    ) -> None:
        """An integer id beside a unique_id that is not a string is not re-keyed."""
        feature = {
            'type': 'Feature',
            'id': 1,
            'properties': {'unique_id': 7},
            'geometry': json.loads(POINT),
        }
        path = write(
            tmp_path / 'numbers.geojson',
            {'type': 'FeatureCollection', 'features': [feature]},
        )
        result = filter_to_region(path, REGION)
        assert list(result.inventory) == [1]
        assert result.inventory[1].features == {'unique_id': 7}

    def test_unique_id_in_some_features_refused(self, tmp_path: Path) -> None:
        """A file that mixes the two forms inside the region is refused."""
        features = [
            {
                'type': 'Feature',
                'id': 1,
                'properties': {'unique_id': 'a', 'type': 'building'},
                'geometry': json.loads(POINT),
            },
            {
                'type': 'Feature',
                'id': 'b',
                'properties': {},
                'geometry': {'type': 'Point', 'coordinates': [-122.001, 38.0]},
            },
        ]
        path = write(
            tmp_path / 'mixed.geojson',
            {'type': 'FeatureCollection', 'features': features},
        )
        with pytest.raises(
            ValueError, match='1 have the integer-id form .* 1 do not'
        ):
            filter_to_region(path, REGION)

    def test_empty_unique_id_refused(self, tmp_path: Path) -> None:
        """An empty unique_id names the feature."""
        feature = {
            'type': 'Feature',
            'id': 1,
            'properties': {'unique_id': ''},
            'geometry': json.loads(POINT),
        }
        path = write(
            tmp_path / 'empty_unique_id.geojson',
            {'type': 'FeatureCollection', 'features': [feature]},
        )
        with pytest.raises(ValueError, match='Feature 1 .* has an empty unique_id'):
            filter_to_region(path, REGION)

    def test_repeated_unique_id_refused_inside_only(self, tmp_path: Path) -> None:
        """A unique_id repeated inside the region is refused, outside it is not."""
        features = [
            {
                'type': 'Feature',
                'id': number,
                'properties': {'unique_id': 'a'},
                'geometry': {'type': 'Point', 'coordinates': [lon, 38.0]},
            }
            for number, lon in ((1, -122.0), (2, -121.9), (3, -122.001))
        ]
        path = tmp_path / 'r2d.geojson'
        write(path, {'type': 'FeatureCollection', 'features': features[:2]})
        assert list(filter_to_region(path, REGION).inventory) == ['a']
        write(path, {'type': 'FeatureCollection', 'features': features})
        with pytest.raises(ValueError, match='inside the region have the unique_id'):
            filter_to_region(path, REGION)

    def test_other_form_outside_the_region_not_checked(self, tmp_path: Path) -> None:
        """A feature of the default form outside the region is left out unchecked."""
        features = [
            {
                'type': 'Feature',
                'id': 1,
                'properties': {'unique_id': 'a'},
                'geometry': json.loads(POINT),
            },
            {
                'type': 'Feature',
                'id': 'b',
                'properties': {},
                'geometry': {'type': 'Point', 'coordinates': [-121.9, 38.0]},
            },
        ]
        path = write(
            tmp_path / 'mixed.geojson',
            {'type': 'FeatureCollection', 'features': features},
        )
        assert list(filter_to_region(path, REGION).inventory) == ['a']

    def test_polygon_read_by_its_exterior_ring(self, tmp_path: Path) -> None:
        """A kept polygon's holes are dropped."""
        shell = square(-122.0, 38.0, 0.001)
        feature = {
            'type': 'Feature',
            'id': 'p',
            'properties': {},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [shell, square(-122.0, 38.0, 0.0005)],
            },
        }
        path = write(
            tmp_path / 'holed.geojson',
            {'type': 'FeatureCollection', 'features': [feature]},
        )
        assert filter_to_region(path, REGION).inventory['p'].coordinates == shell

    def test_projected_file_refused_at_its_crs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A projected file is refused when its crs is read, before any feature."""
        ring = [
            [560000.0, 4200000.0],
            [560010.0, 4200000.0],
            [560010.0, 4200010.0],
            [560000.0, 4200000.0],
        ]
        feature = {
            'type': 'Feature',
            'id': 'p',
            'properties': {},
            'geometry': {'type': 'Polygon', 'coordinates': [ring]},
        }
        crs = {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::26910'}}
        path = write(
            tmp_path / 'utm.geojson',
            {'type': 'FeatureCollection', 'crs': crs, 'features': [feature]},
        )
        with pytest.raises(ValueError, match='Write the file in EPSG:4326'):
            filter_to_region(path, REGION)
        assert capsys.readouterr().out == ''
        path = write(
            tmp_path / 'utm_no_crs.geojson',
            {'type': 'FeatureCollection', 'features': [feature]},
        )
        with pytest.raises(
            ValueError, match='cannot read as longitude and latitude'
        ):
            filter_to_region(path, REGION)

    @pytest.mark.parametrize(
        'name',
        [
            'urn:ogc:def:crs:EPSG::4269',
            'EPSG:4269',
            'urn:ogc:def:crs:OGC:1.3:CRS83',
            'urn:ogc:def:crs:OGC::CRS83',
            'OGC:CRS83',
            'CRS83',
        ],
    )
    def test_nad83_accepted_with_a_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str
    ) -> None:
        """NAD83 coordinates are used as they are, and one line says so."""
        feature = {
            'type': 'Feature',
            'id': 'a',
            'properties': {},
            'geometry': json.loads(POINT),
        }
        path = write(
            tmp_path / 'nad83.geojson',
            {
                'type': 'FeatureCollection',
                'crs': {'type': 'name', 'properties': {'name': name}},
                'features': [feature],
            },
        )
        result = filter_to_region(path, REGION)
        assert result.inventory['a'].coordinates == [[-122.0, 38.0]]
        lines = capsys.readouterr().out.splitlines()
        assert lines == [
            f'{path} gives its coordinates in NAD83 ({name}). They are used as '
            'they are, as WGS84 longitudes and latitudes, from which they differ '
            'slightly.',
            'Kept 1 of 1 asset inside the region.',
        ]

    @pytest.mark.parametrize(
        'name',
        ['urn:ogc:def:crs:EPSG::26910', 'EPSG:3310', 'urn:ogc:def:crs:EPSG::3857'],
    )
    def test_projected_crs_still_refused(self, tmp_path: Path, name: str) -> None:
        """A projected system, NAD83's UTM zone included, is refused."""
        path = write(
            tmp_path / 'projected.geojson',
            {
                'type': 'FeatureCollection',
                'crs': {'type': 'name', 'properties': {'name': name}},
                'features': [],
            },
        )
        with pytest.raises(
            ValueError, match=r'only, named as one of .*CRS84.*EPSG::4269'
        ):
            filter_to_region(path, REGION)

    def test_null_crs_is_no_crs(self, tmp_path: Path) -> None:
        """A crs of null, which the 2008 specification allowed, is accepted."""
        feature = {
            'type': 'Feature',
            'id': 'a',
            'properties': {},
            'geometry': json.loads(POINT),
        }
        path = write(
            tmp_path / 'null_crs.geojson',
            {'type': 'FeatureCollection', 'crs': None, 'features': [feature]},
        )
        assert list(filter_to_region(path, REGION).inventory) == ['a']

    @pytest.mark.parametrize(
        'name',
        ['urn:ogc:def:crs:OGC:1.3:CRS84', 'OGC:CRS84', 'CRS84', 'EPSG:4326'],
    )
    def test_geographic_crs_names_accepted(self, tmp_path: Path, name: str) -> None:
        """A crs that names WGS84 longitude and latitude is accepted."""
        feature = {
            'type': 'Feature',
            'id': 'a',
            'properties': {},
            'geometry': json.loads(POINT),
        }
        path = write(
            tmp_path / 'wgs84.geojson',
            {
                'type': 'FeatureCollection',
                'crs': {'type': 'name', 'properties': {'name': name}},
                'features': [feature],
            },
        )
        assert list(filter_to_region(path, REGION).inventory) == ['a']

    def test_members_in_any_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Members before and after features, scalars and text, at every edge."""
        features = [
            {
                'type': 'Feature',
                'properties': {'name': 'Gare "Nord" éé\\', 'Stories': 12345},
                'geometry': {'type': 'Point', 'coordinates': [-122.0, 38.0]},
                'id': 'p1',
            },
            {
                'id': 'p2',
                'geometry': {'type': 'Point', 'coordinates': [-121.0, 38.0]},
                'properties': None,
                'type': 'Feature',
            },
        ]
        text = (
            '{"count": 1234567890, "features": '
            + json.dumps(features, ensure_ascii=False)
            + ', "flag": true, "nothing": null, "crs": {"type": "name", '
            '"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, '
            '"type": "FeatureCollection", "scale": -1.5e3}'
        )
        path = tmp_path / 'ordered.geojson'
        path.write_text(text, encoding='utf-8')
        for chunk in range(1, len(text) + 2):
            monkeypatch.setattr(region_filter, 'READ_CHUNK_CHARS', chunk)
            result = filter_to_region(path, REGION)
            assert list(result.inventory) == ['p1']
            assert result.inventory['p1'].features == features[0]['properties']
            assert result.n_pw == 1

    def test_byte_order_mark(self, tmp_path: Path) -> None:
        """A file that starts with a UTF-8 byte order mark is read."""
        path = tmp_path / 'bom.geojson'
        three_worlds().write_to_geojson(str(path))
        path.write_bytes(b'\xef\xbb\xbf' + path.read_bytes())
        assert sorted(filter_to_region(path, REGION).inventory) == ['in_a', 'in_b']

    def test_mixed_list_lengths_refused(self, tmp_path: Path) -> None:
        """Kept lists of different lengths name no number of worlds."""
        collection = three_worlds().write_to_geojson()
        del collection['n_worlds']
        collection['features'][1]['properties']['StructureType'] = ['S1', 'S2']
        path = write(tmp_path / 'mixed.geojson', collection)
        with pytest.raises(ValueError, match='lists of 2 and 3 values'):
            filter_to_region(path, REGION)

    def test_one_world_without_lists(self, tmp_path: Path) -> None:
        """A file of scalars, and a plain vector list, gives one world."""
        source = inventory(
            {'a': ([[-122.0, 38.0]], {'PlanArea': 1.0, 'HousingUnits': [1, 2, 3]})}
        )
        path = tmp_path / 'one.geojson'
        source.write_to_geojson(str(path))
        assert filter_to_region(path, REGION).n_pw == 1

    @pytest.mark.parametrize(
        ('text', 'message'),
        [
            ('[]', "expected one of '{'"),
            ('{"type": "FeatureCollection"}', 'no features member'),
            ('{"features": []}', 'no type member'),
            ('{"type": "Feature", "features": []}', "type 'Feature', not"),
            ('{"type": "FeatureCollection" "features": []}', "expected one of ',}'"),
            ('{"type": "FeatureCollection", "features": []} x', 'data follows'),
            ('{"type": "FeatureCollection", "features": [],}', 'comma ends the top'),
            (
                '{"type": "FeatureCollection", "features": ['
                f'{{"id": "a", "geometry": {POINT}}},]}}',
                'comma ends the feat',
            ),
            ('{"type": "FeatureCollection", 1: 2, "features": []}', 'not a string'),
            (
                '{"type": "FeatureCollection", "features": [], "features": []}',
                "'features' occurs twice",
            ),
            ('{"type": "FeatureCollection", "features": [{"id": tru', 'not valid'),
            (
                '{"type": "FeatureCollection", "crs": {"type": "name", '
                '"properties": {"name": "EPSG:26910"}}, "features": []}',
                'EPSG:26910',
            ),
            (
                '{"type": "FeatureCollection", "crs": {"type": "link", '
                '"properties": {"href": "x"}}, "features": []}',
                'gives the crs',
            ),
            (
                '{"type": "FeatureCollection", "crs": {"type": "name", '
                '"properties": {"name": "urn:ogc:def:crs:EPSG::3310"}}, '
                '"features": []}',
                'EPSG::3310',
            ),
            (
                '{"type": "FeatureCollection", "features": [{"type": "Feature", '
                f'"geometry": {POINT}}}]}}',
                'Feature 0 .* has no id',
            ),
            (
                '{"type": "FeatureCollection", "features": [{"id": [1], '
                f'"geometry": {POINT}}}]}}',
                r'has the id \[1\]',
            ),
            (
                '{"type": "FeatureCollection", "features": [{"id": "l", '
                f'"properties": [1], "geometry": {POINT}}}]}}',
                'not an object',
            ),
            (
                '{"type": "FeatureCollection", "features": [{"id": "e", '
                f'"properties": {{"YearBuilt": []}}, "geometry": {POINT}}}]}}',
                "empty list under 'YearBuilt'",
            ),
            (
                '{"type": "FeatureCollection", "features": [{"type": "Feature", '
                '"id": "far", "geometry": null}]}',
                r"'far' .* cannot read \(None\)\. Remove the feature",
            ),
            (
                '{"type": "FeatureCollection", "features": ['
                f'{{"id": "d", "geometry": {POINT}}}, '
                '{"id": "d", "geometry": {"type": "Point", '
                '"coordinates": [-122.001, 38.0]}}]}',
                "have the id 'd'",
            ),
        ],
    )
    def test_malformed_files_refused(
        self, tmp_path: Path, text: str, message: str
    ) -> None:
        """A file that is not a usable FeatureCollection raises ValueError."""
        path = tmp_path / 'bad.geojson'
        path.write_text(text, encoding='utf-8')
        with pytest.raises(ValueError, match=message):
            filter_to_region(path, REGION)

    def test_empty_collection(self, tmp_path: Path) -> None:
        """A collection without features gives an empty inventory."""
        path = write(
            tmp_path / 'empty.geojson', {'type': 'FeatureCollection', 'features': []}
        )
        result = filter_to_region(path, REGION)
        assert result.inventory == {}
        assert result.n_pw == 1


class TestRegion:
    """A region is a place name, a RegionBoundary or a polygon in degrees."""

    def test_place_name_is_looked_up_once(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A name goes to RegionBoundary once, and the name found is printed."""
        calls = []

        def get_boundary(
            self: RegionBoundary, *_args: object, **_kwargs: object
        ) -> tuple[Polygon, str, int]:
            calls.append((self.input_type, self.query_area))
            return REGION, 'Testville', 1

        monkeypatch.setattr(RegionBoundary, 'get_boundary', get_boundary)
        result = filter_to_region(
            three_worlds(), 'Testville, CA', place_source='osm'
        )
        assert calls == [('locationName', 'Testville, CA')]
        assert sorted(result.inventory) == ['in_a', 'in_b']
        assert capsys.readouterr().out == 'Kept 2 of 3 assets inside Testville.\n'

    def test_region_boundary_of_a_bounding_box(self) -> None:
        """A RegionBoundary of a bounding box needs no lookup."""
        region = RegionBoundary(
            {'type': 'locationPolygon', 'data': (-122.01, 37.99, -121.99, 38.01)}
        )
        assert sorted(filter_to_region(three_worlds(), region).inventory) == [
            'in_a',
            'in_b',
        ]

    @pytest.mark.parametrize('region', [Point(-122.0, 38.0), [-122.0, 38.0], None])
    def test_other_types_refused(self, region: object) -> None:
        """A point, a list or None is not a region."""
        with pytest.raises(TypeError, match='region must be'):
            filter_to_region(three_worlds(), region)

    def test_projected_polygon_refused(self) -> None:
        """A polygon in meters is not taken for degrees."""
        with pytest.raises(ValueError, match='not longitude and latitude'):
            filter_to_region(
                three_worlds(), box(560000.0, 4200000.0, 570000.0, 4210000.0)
            )

    def test_invalid_polygon_repaired(self) -> None:
        """An invalid region is used as shapely.make_valid repairs it."""
        # A hole that reaches past its shell. make_valid turns the part of the
        # hole outside the shell into a part of the region, which the invalid
        # polygon does not contain.
        hole = [
            (-121.995, 37.995),
            (-121.98, 37.995),
            (-121.98, 38.005),
            (-121.995, 38.005),
        ]
        invalid = Polygon(REGION.exterior.coords, [hole])
        assert not invalid.is_valid
        inv = inventory(
            {
                'shell': ([[-122.005, 38.0]], {}),
                'hole_in_shell': ([[-121.993, 38.0]], {}),
                'hole_outside_shell': ([[-121.985, 38.0]], {}),
            }
        )
        assert sorted(filter_to_region(inv, invalid).inventory) == [
            'hole_outside_shell',
            'shell',
        ]

    def test_polygon_holes_kept(self) -> None:
        """A polygon given directly keeps its holes."""
        hole = [
            (-122.002, 37.998),
            (-121.998, 37.998),
            (-121.998, 38.002),
            (-122.002, 38.002),
        ]
        holed = Polygon(REGION.exterior.coords, [hole])
        inv = inventory(
            {'ring': ([[-122.005, 38.0]], {}), 'enclave': ([[-122.0, 38.0]], {})}
        )
        assert filter_to_region(inv, holed).get_asset_ids() == ['ring']

    def test_missing_path_before_lookup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A path that is not a file is refused before the name is looked up."""
        calls = []
        monkeypatch.setattr(
            RegionBoundary, 'get_boundary', lambda *_args: calls.append(1)
        )
        with pytest.raises(FileNotFoundError, match='is not a file'):
            filter_to_region(
                tmp_path / 'missing.geojson', 'Testville, CA', place_source='osm'
            )
        assert calls == []

    def test_one_asset_in_the_singular(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A total of one is printed as one asset."""
        filter_to_region(inventory({'a': ([[-122.0, 38.0]], {})}), REGION)
        assert capsys.readouterr().out == 'Kept 1 of 1 asset inside the region.\n'

    def test_printed_name_without_its_accents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A console that cannot encode the name gets it without accents."""

        def get_boundary(
            *_args: object, **_kwargs: object
        ) -> tuple[Polygon, str, int]:
            return REGION, 'Cañada', 1

        monkeypatch.setattr(RegionBoundary, 'get_boundary', get_boundary)
        raw = io.BytesIO()
        console = io.TextIOWrapper(raw, encoding='ascii')
        monkeypatch.setattr(sys, 'stdout', console)
        filter_to_region(three_worlds(), 'Cañada, CA', place_source='osm')
        console.flush()
        assert raw.getvalue() == b'Kept 2 of 3 assets inside Canada.\n'


def two_counties(tmp_path: Path) -> tuple[Path, Path]:
    """
    Write two county-shaped files of three worlds that share the line -122.0.

    Args:
        tmp_path (Path):
            The folder.

    Returns:
        tuple[Path, Path]:
            The west file, with ``w_in`` inside :data:`REGION` and ``w_out``
            outside, and the east file, with ``e_in`` and ``e_out``.
    """
    west = inventory(
        {
            'w_in': (square(-122.005, 38.0), {'StructureType': ['W1', 'W2', 'C1']}),
            'w_out': (square(-122.05, 38.0), {'StructureType': ['W1', 'W1', 'W1']}),
        },
        n_pw=3,
    )
    east = inventory(
        {
            'e_in': ([[-121.995, 38.002]], {'StructureType': ['S1', 'S2', 'S2']}),
            'e_out': ([[-121.95, 38.0]], {'StructureType': ['C2', 'C2', 'C1']}),
        },
        n_pw=3,
    )
    west_path, east_path = tmp_path / 'west.geojson', tmp_path / 'east.geojson'
    west.write_to_geojson(str(west_path))
    east.write_to_geojson(str(east_path))
    return west_path, east_path


class TestSources:
    """A list of inventories and paths is filtered into one inventory."""

    def test_two_counties_by_one_boundary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A boundary across two files keeps the union of the two results."""
        west_path, east_path = two_counties(tmp_path)
        west = filter_to_region(west_path, REGION)
        east = filter_to_region(east_path, REGION)
        capsys.readouterr()
        both = filter_to_region([west_path, east_path], REGION)
        assert list(both.inventory) == ['w_in', 'e_in']
        assert both.n_pw == 3
        assert both.vector_features == west.vector_features
        for asset_id, asset in (west.inventory | east.inventory).items():
            assert both.inventory[asset_id].features == asset.features
            assert both.inventory[asset_id].coordinates == asset.coordinates
        both.validate_possible_worlds()
        assert capsys.readouterr().out == (
            f'Kept 1 of 2 assets of {west_path} inside the region.\n'
            f'Kept 1 of 2 assets of {east_path} inside the region.\n'
            'Kept 2 of 4 assets inside the region.\n'
        )

    def test_order_given(self, tmp_path: Path) -> None:
        """The sources are filtered in the order given."""
        west_path, east_path = two_counties(tmp_path)
        assert list(filter_to_region((east_path, west_path), REGION).inventory) == [
            'e_in',
            'w_in',
        ]

    def test_inventories_and_paths_together(self, tmp_path: Path) -> None:
        """An inventory and a path can be filtered together."""
        west_path, _ = two_counties(tmp_path)
        source = three_worlds()
        both = filter_to_region([source, west_path], REGION)
        assert list(both.inventory) == ['in_a', 'in_b', 'w_in']
        assert both.inventory['in_a'] is not source.inventory['in_a']

    def test_one_item_list_is_its_item(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A one-item list gives what its item gives, and prints the same."""
        west_path, _ = two_counties(tmp_path)
        capsys.readouterr()
        alone = filter_to_region(west_path, REGION)
        printed = capsys.readouterr().out
        listed = filter_to_region([west_path], REGION)
        assert capsys.readouterr().out == printed
        assert list(listed.inventory) == list(alone.inventory)
        assert listed.n_pw == alone.n_pw

    def test_boundary_looked_up_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A place name is looked up once for all sources."""
        calls = []

        def get_boundary(
            self: RegionBoundary, *_args: object, **_kwargs: object
        ) -> tuple[Polygon, str, int]:
            calls.append(self.query_area)
            return REGION, 'Testville', 1

        monkeypatch.setattr(RegionBoundary, 'get_boundary', get_boundary)
        filter_to_region(
            list(two_counties(tmp_path)), 'Testville, CA', place_source='osm'
        )
        assert calls == ['Testville, CA']

    def test_empty_list_refused(self) -> None:
        """An empty list names what to pass."""
        with pytest.raises(ValueError, match='list of sources is empty'):
            filter_to_region([], REGION)

    def test_other_source_refused(self) -> None:
        """A source that is neither an inventory nor a path is refused."""
        with pytest.raises(TypeError, match='Item 2 of the list is an int'):
            filter_to_region([three_worlds(), 7], REGION)

    def test_generator_is_not_a_list(self) -> None:
        """A generator is refused, with lists and tuples named."""
        sources = (source for source in [three_worlds()])
        with pytest.raises(TypeError, match='is a generator.* list or tuple'):
            filter_to_region(sources, REGION)

    def test_stated_worlds_without_kept_lists(self, tmp_path: Path) -> None:
        """A file's n_worlds sets its number of worlds when it keeps no lists."""
        west_path, _ = two_counties(tmp_path)
        five = inventory(
            {
                's': ([[-122.0, 38.004]], {'PlanArea': 10.0}),
                'far': ([[-121.0, 38.0]], {'StructureType': ['W1'] * 5}),
            },
            n_pw=5,
        )
        five_path = tmp_path / 'five.geojson'
        five.write_to_geojson(str(five_path))
        assert filter_to_region(five_path, REGION).n_pw == 5
        with pytest.raises(ValueError, match=r'5 in .*five.geojson, 3 in'):
            filter_to_region([five_path, west_path], REGION)

    def test_ids_of_one_string_form_refused(self) -> None:
        """Ids 7 and '7', in two sources or in one, would be written as one."""
        first = inventory({7: ([[-122.0, 38.0]], {})})
        second = inventory({'7': ([[-122.001, 38.0]], {})})
        with pytest.raises(ValueError, match=r"asset '7' .* both inventory 1"):
            filter_to_region([first, second], REGION)
        both = inventory({7: ([[-122.0, 38.0]], {}), '7': ([[-122.001, 38.0]], {})})
        with pytest.raises(ValueError, match="writes as the one id '7'"):
            filter_to_region(both, REGION)

    def test_every_path_checked_before_lookup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing second path is refused before the name is looked up."""
        calls = []
        monkeypatch.setattr(
            RegionBoundary, 'get_boundary', lambda *_args: calls.append(1)
        )
        west_path, _ = two_counties(tmp_path)
        with pytest.raises(FileNotFoundError, match='missing.geojson is not a file'):
            filter_to_region(
                [west_path, tmp_path / 'missing.geojson'],
                'Testville',
                place_source='osm',
            )
        assert calls == []

    def test_different_worlds_refused(self, tmp_path: Path) -> None:
        """Sources of different numbers of worlds name each source."""
        west_path, _ = two_counties(tmp_path)
        one_world = inventory({'x': ([[-122.0, 38.004]], {})})
        with pytest.raises(
            ValueError, match=r'3 in .*west.geojson, 1 in inventory 2'
        ):
            filter_to_region([west_path, one_world], REGION)

    def test_scalar_file_takes_the_others_worlds(self, tmp_path: Path) -> None:
        """A file without n_worlds and without kept lists takes the others' count."""
        west_path, _ = two_counties(tmp_path)
        collection = inventory(
            {'s': ([[-122.0, 38.004]], {'PlanArea': 10.0})}
        ).write_to_geojson()
        del collection['n_worlds']
        scalars = write(tmp_path / 'scalars.geojson', collection)
        both = filter_to_region([scalars, west_path], REGION)
        assert both.n_pw == 3
        both.validate_possible_worlds()

    def test_different_vector_features_refused(self) -> None:
        """Inventories of different vector features are refused."""
        first = three_worlds()
        second = three_worlds()
        second.inventory = {'x': second.inventory['in_b']}
        second.vector_features = ['HousingUnits', 'Extra']
        with pytest.raises(ValueError, match='same vector_features'):
            filter_to_region([first, second], REGION)

    def test_overlapping_sources_refused(self, tmp_path: Path) -> None:
        """An asset inside the region in two sources names both."""
        west_path, _ = two_counties(tmp_path)
        copy_path = tmp_path / 'west_again.geojson'
        copy_path.write_bytes(west_path.read_bytes())
        with pytest.raises(ValueError, match=r"'w_in' .* both .*west.geojson and"):
            filter_to_region([west_path, copy_path], REGION)


class TestStatedWorlds:
    """A file's n_worlds is checked and gives the source its number of worlds."""

    def test_count_without_kept_lists(self, tmp_path: Path) -> None:
        """A file of ten worlds whose kept assets hold no lists gives n_pw 10."""
        ten = inventory(
            {
                's': ([[-122.0, 38.004]], {'PlanArea': 10.0}),
                'far': ([[-121.0, 38.0]], {'StructureType': ['W1'] * 10}),
            },
            n_pw=10,
        )
        path = tmp_path / 'ten.geojson'
        ten.write_to_geojson(str(path))
        assert filter_to_region(path, REGION).n_pw == 10
        other = inventory(
            {'x': ([[-122.002, 38.0]], {'StructureType': ['W1'] * 10})}, n_pw=10
        )
        both = filter_to_region([path, other], REGION)
        assert both.n_pw == 10
        both.validate_possible_worlds()

    def test_kept_lists_of_another_length_refused(self, tmp_path: Path) -> None:
        """Kept lists of one length other than the stated count are refused."""
        collection = three_worlds().write_to_geojson()
        assert collection['n_worlds'] == 3
        for feature in collection['features'][:2]:
            feature['properties']['StructureType'] = ['S1', 'S2']
        path = write(tmp_path / 'short.geojson', collection)
        with pytest.raises(
            ValueError,
            match='hold lists of 2 values, but the file gives n_worlds as 3. Write',
        ):
            filter_to_region(path, REGION)

    def test_kept_lists_of_two_lengths_refused(self, tmp_path: Path) -> None:
        """Kept lists of two lengths are refused."""
        collection = three_worlds().write_to_geojson()
        collection['features'][1]['properties']['StructureType'] = ['S1', 'S2']
        path = write(tmp_path / 'mixed.geojson', collection)
        with pytest.raises(
            ValueError, match='lists of 2 and 3 values, so they share no number'
        ):
            filter_to_region(path, REGION)

    def test_one_world_beside_more_refused(self, tmp_path: Path) -> None:
        """A file of 1 world is refused beside a file of 3, as one world of a run."""
        west_path, _ = two_counties(tmp_path)
        one = tmp_path / 'one.geojson'
        inventory({'s': ([[-122.0, 38.004]], {'PlanArea': 10.0})}).write_to_geojson(
            str(one)
        )
        with pytest.raises(
            ValueError, match=r'1 in .*one.geojson. Pass .* A source of 1 world'
        ):
            filter_to_region([west_path, one], REGION)

    def test_list_outside_the_region_not_checked(self, tmp_path: Path) -> None:
        """Only the kept lists are checked against the stated count."""
        collection = three_worlds().write_to_geojson()
        collection['features'][2]['properties']['StructureType'] = ['W1']
        path = write(tmp_path / 'far_short.geojson', collection)
        assert filter_to_region(path, REGION).n_pw == 3

    def test_member_not_a_count_refused(self, tmp_path: Path) -> None:
        """A member that is not an integer of at least 1 is refused."""
        collection = three_worlds().write_to_geojson()
        collection['n_worlds'] = 0
        path = write(tmp_path / 'bad.geojson', collection)
        with pytest.raises(ValueError, match='gives n_worlds as'):
            filter_to_region(path, REGION)

    def test_member_after_features_checked(self, tmp_path: Path) -> None:
        """An n_worlds written after features is checked once they are read."""
        document = {
            'type': 'FeatureCollection',
            'features': [{'id': 's', 'geometry': json.loads(POINT)}],
            'n_worlds': 10,
        }
        assert (
            filter_to_region(write(tmp_path / 'ten.geojson', document), REGION).n_pw
            == 10
        )
        document['n_worlds'] = 0
        with pytest.raises(ValueError, match='gives n_worlds as 0'):
            filter_to_region(write(tmp_path / 'zero.geojson', document), REGION)


class TestSource:
    """A place name is looked up in census boundaries unless source is 'osm'."""

    @pytest.fixture
    def lookups(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
        """
        Replace both lookups by stand-ins that record their calls.

        Args:
            monkeypatch (pytest.MonkeyPatch):
                Replaces ``get_boundary`` of both classes.

        Returns:
            list[tuple[str, str]]:
                The class and the text of each lookup, in order.
        """
        calls: list[tuple[str, str]] = []

        def census(
            self: CensusBoundary, *_args: object, **_kwargs: object
        ) -> tuple[Polygon, str, str]:
            calls.append(('census', self.place))
            return REGION, 'Testville city, California', '0600001'

        def osm(
            self: RegionBoundary, *_args: object, **_kwargs: object
        ) -> tuple[Polygon, str, int]:
            calls.append(('osm', self.query_area))
            return REGION, 'Testville', 1

        monkeypatch.setattr(CensusBoundary, 'get_boundary', census)
        monkeypatch.setattr(RegionBoundary, 'get_boundary', osm)
        return calls

    @pytest.mark.parametrize('options', [{}, {'place_source': 'census'}])
    def test_name_goes_to_the_census(
        self,
        lookups: list,
        capsys: pytest.CaptureFixture[str],
        options: dict[str, str],
    ) -> None:
        """A name goes once to the census lookup, and the name found is printed."""
        result = filter_to_region(three_worlds(), 'Testville, CA', **options)
        assert lookups == [('census', 'Testville, CA')]
        assert sorted(result.inventory) == ['in_a', 'in_b']
        assert capsys.readouterr().out == (
            'Kept 2 of 3 assets inside Testville city, California.\n'
        )

    def test_osm_only_with_osm(self, lookups: list) -> None:
        """With place_source='osm' a name goes to RegionBoundary, not the census."""
        filter_to_region(three_worlds(), 'Testville, CA', place_source='osm')
        assert lookups == [('osm', 'Testville, CA')]

    def test_census_boundary_given(self, lookups: list) -> None:
        """A CensusBoundary given as the region is asked for its boundary."""
        filter_to_region(three_worlds(), CensusBoundary('Testville, CA'))
        assert lookups == [('census', 'Testville, CA')]

    def test_census_refusal_of_the_text_raised(self, lookups: list) -> None:
        """A text the census lookup refuses is not sent to OpenStreetMap."""
        with pytest.raises(ValueError, match="'Testville' names no state"):
            filter_to_region(three_worlds(), 'Testville')
        assert lookups == []

    def test_census_refusal_of_the_match_raised(
        self, lookups: list, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A name the census matches to nothing is not sent to OpenStreetMap."""

        def no_match(*_args: object, **_kwargs: object) -> None:
            lookups.append(('census', 'refused'))
            msg = 'No census place or county in California is named ...'
            raise ValueError(msg)

        monkeypatch.setattr(CensusBoundary, 'get_boundary', no_match)
        with pytest.raises(ValueError, match='No census place'):
            filter_to_region(three_worlds(), 'Testville, CA')
        assert lookups == [('census', 'refused')]

    @pytest.mark.parametrize('place_source', ['census', 'osm'])
    def test_source_beside_a_polygon_not_consulted(
        self, lookups: list, place_source: str
    ) -> None:
        """Either source beside a polygon is accepted and looks nothing up."""
        result = filter_to_region(three_worlds(), REGION, place_source=place_source)
        assert sorted(result.inventory) == ['in_a', 'in_b']
        assert lookups == []

    @pytest.mark.parametrize('place_source', ['census', 'osm'])
    def test_source_beside_a_region_boundary_not_consulted(
        self, lookups: list, place_source: str
    ) -> None:
        """Either source beside a RegionBoundary reads only the region's boundary."""
        data = (-122.01, 37.99, -121.99, 38.01)
        region = RegionBoundary({'type': 'locationPolygon', 'data': data})
        result = filter_to_region(three_worlds(), region, place_source=place_source)
        assert sorted(result.inventory) == ['in_a', 'in_b']
        assert lookups == [('osm', data)]

    @pytest.mark.parametrize('region', ['Testville, CA', REGION])
    @pytest.mark.parametrize('place_source', ['nominatim', None])
    def test_unknown_source_refused(
        self, lookups: list, region: object, place_source: object
    ) -> None:
        """A source other than 'census' or 'osm' is refused, whatever the region."""
        with pytest.raises(
            ValueError, match="place_source must be 'census' or 'osm'"
        ):
            filter_to_region(three_worlds(), region, place_source=place_source)
        assert lookups == []


def test_find_inside_strictly_and_in_order() -> None:
    """A point on the boundary is outside, and the ids keep the assets' order."""
    assets = {
        name: Asset(name, [[x, 38.0]], {})
        for name, x in (('c', -122.0), ('edge', -122.01), ('a', -122.005))
    }
    assert find_inside(assets, REGION) == ['c', 'a']


def test_read_boundary_keeps_the_polygonal_parts_of_a_repair() -> None:
    """An invalid polygon's repair keeps its polygonal parts, a polygon its name."""
    spiked = Polygon([(0, 0), (2, 0), (2, 2), (3, 3), (2, 2), (0, 2)])
    assert not spiked.is_valid
    boundary, name, identifier = read_boundary(spiked)
    assert isinstance(boundary, Polygon)
    assert boundary.equals(box(0, 0, 2, 2))
    assert (name, identifier) == ('the region', None)
    with pytest.raises(TypeError, match='not str'):
        read_boundary('Berkeley, CA')


def test_ring_that_encloses_no_area_keeps_no_asset() -> None:
    """A ring that doubles back keeps its repair, and no asset, on its line or off."""
    folded = Polygon([(-122.0, 38.0), (-121.99, 38.0), (-122.0, 38.0)])
    assert not folded.is_valid
    boundary, _, _ = read_boundary(folded)
    assert boundary.length > 0
    assert boundary.area == 0
    inventory = AssetInventory()
    inventory.add_asset_coordinates('A', [[-121.995, 38.0005]])
    inventory.add_asset_coordinates('B', [[-121.995, 38.0]])
    assert filter_to_region(inventory, folded).get_asset_ids() == []
