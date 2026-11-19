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
import socket
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
from brails.types.housing_unit_inventory import HousingUnitInventory
from brails.types.region_boundary import RegionBoundary
from brails.utils import region_filter
from brails.utils.region_filter import filter_to_region, reference_point

if TYPE_CHECKING:
    from pathlib import Path

# The region of most tests: a square degree's hundredth around (-122.0, 38.0).
REGION = box(-122.01, 37.99, -121.99, 38.01)

# A point inside REGION, as GeoJSON text.
POINT = '{"type": "Point", "coordinates": [-122.0, 38.0]}'


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every connection and name lookup, so no test reaches a service."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = 'A test tried to open a network connection.'
        raise OSError(msg)

    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'getaddrinfo', refuse)


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

    def test_integer_id_export_needs_a_type(self, tmp_path: Path) -> None:
        """A filtered county file takes integer ids once its assets have a type."""
        path = tmp_path / 'county.geojson'
        three_worlds().write_to_geojson(str(path))
        result = filter_to_region(path, REGION)
        out = tmp_path / 'r2d.geojson'
        with pytest.raises(ValueError, match="has no 'type' value"):
            result.write_to_geojson(str(out), integer_ids=True)
        for asset in result.inventory.values():
            asset.features['type'] = 'building'
        written = result.write_to_geojson(str(out), integer_ids=True)
        assert sorted(f['id'] for f in written['features']) == [1, 2]

    def test_integer_id_file_is_not_written_in_that_form_again(
        self, tmp_path: Path
    ) -> None:
        """A filtered integer-id file carries unique_id, which the export refuses."""
        source = three_worlds()
        for asset in source.inventory.values():
            asset.features['type'] = 'building'
        path = tmp_path / 'r2d.geojson'
        source.write_to_geojson(str(path), integer_ids=True)
        result = filter_to_region(path, REGION)
        with pytest.raises(ValueError, match="already carries a 'unique_id'"):
            result.write_to_geojson(
                str(tmp_path / 'again.geojson'), integer_ids=True
            )

    def test_integer_id_file_of_several_types_refused(self, tmp_path: Path) -> None:
        """Each type is numbered from 1, so kept features repeat an id."""
        source = three_worlds()
        source.inventory['in_a'].features['type'] = 'building'
        source.inventory['in_b'].features['type'] = 'bridge'
        source.inventory['out'].features['type'] = 'building'
        path = tmp_path / 'r2d.geojson'
        source.write_to_geojson(str(path), integer_ids=True)
        with pytest.raises(ValueError, match='several types numbers each type'):
            filter_to_region(path, REGION)

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

    def test_integer_ids(self, tmp_path: Path) -> None:
        """A file of integer ids is keyed by them and keeps unique_id and type."""
        source = three_worlds()
        for asset in source.inventory.values():
            asset.features['type'] = 'building'
        path = tmp_path / 'r2d.geojson'
        source.write_to_geojson(str(path), integer_ids=True)
        result = filter_to_region(path, REGION)
        assert sorted(result.inventory) == [1, 2]
        assert sorted(
            a.features['unique_id'] for a in result.inventory.values()
        ) == [
            'in_a',
            'in_b',
        ]
        assert all(
            a.features['type'] == 'building' for a in result.inventory.values()
        )

    def test_mixed_list_lengths_refused(self, tmp_path: Path) -> None:
        """Kept lists of different lengths name no number of worlds."""
        source = three_worlds()
        source.inventory['in_b'].features['StructureType'] = ['S1', 'S2']
        path = tmp_path / 'mixed.geojson'
        source.write_to_geojson(str(path))
        with pytest.raises(ValueError, match=r'lists of \[2, 3\] values'):
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
                '"properties": {"name": "urn:ogc:def:crs:EPSG::4269"}}, '
                '"features": []}',
                'EPSG::4269',
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
        result = filter_to_region(three_worlds(), 'Testville, CA')
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
            filter_to_region(tmp_path / 'missing.geojson', 'Testville, CA')
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
        filter_to_region(three_worlds(), 'Cañada, CA')
        console.flush()
        assert raw.getvalue() == b'Kept 2 of 3 assets inside Canada.\n'
