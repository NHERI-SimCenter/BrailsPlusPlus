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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.


"""Tests for the integer-id GeoJSON export and its inverse read."""

from __future__ import annotations

import copy
import csv
import datetime as dt
import json
from typing import TYPE_CHECKING, Any

import pytest

from brails.types import asset_inventory as asset_inventory_module
from brails.types.asset_inventory import Asset, AssetInventory

if TYPE_CHECKING:
    from pathlib import Path

# Geohash-style asset ids, including a suffixed collision and an all-digit
# id that must stay a string.
ASSETS = {
    '9q8yyk8ytp': {
        'coordinates': [
            [-122.41, 37.77],
            [-122.40, 37.77],
            [-122.40, 37.78],
            [-122.41, 37.77],
        ],
        'features': {'type': 'building', 'NumberOfStories': 2},
    },
    '9q8yyk8ytp-152': {
        'coordinates': [
            [-122.42, 37.76],
            [-122.41, 37.76],
            [-122.41, 37.77],
            [-122.42, 37.76],
        ],
        'features': {'type': 'building', 'NumberOfStories': 3, 'Name': 'B'},
    },
    '0213456789': {
        'coordinates': [[-122.30, 37.80]],
        'features': {'type': 'building', 'YearBuilt': 1925},
    },
    '9q8yym1bc0': {
        'coordinates': [
            [-122.43, 37.75],
            [-122.42, 37.75],
            [-122.42, 37.76],
            [-122.43, 37.75],
        ],
        'features': {'type': 'building', 'ReplacementCost': 410000.5},
    },
}

SORTED_IDS = sorted(ASSETS)


def make_inventory(order: list[str] | None = None) -> AssetInventory:
    """Build the test inventory in ``order``, or in ``ASSETS`` order if ``None``."""
    inventory = AssetInventory()
    for asset_id in order if order is not None else list(ASSETS):
        record = copy.deepcopy(ASSETS[asset_id])
        inventory.add_asset(
            asset_id, Asset(asset_id, record['coordinates'], record['features'])
        )
    return inventory


def write_integer(
    tmp_path: Path, inventory: AssetInventory
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write with integer ids and return the returned and the on-disk GeoJSON."""
    output = tmp_path / 'inventory.geojson'
    returned = inventory.write_to_geojson(str(output), integer_ids=True)
    with output.open(encoding='utf-8') as f:
        return returned, json.load(f)


def read_table(tmp_path: Path) -> list[dict[str, str]]:
    """Read the id table written beside ``inventory.geojson``."""
    with (tmp_path / 'inventory_ids.csv').open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


class TestIntegerIdWrite:
    """Layout of the file written with integer ids."""

    def test_ids_are_contiguous_and_follow_asset_id_order(
        self, tmp_path: Path
    ) -> None:
        """Ids run 1 to n at the top level in ascending asset-id order."""
        _, written = write_integer(tmp_path, make_inventory())
        features = written['features']
        assert [f['id'] for f in features] == list(range(1, len(ASSETS) + 1))
        assert all(type(f['id']) is int for f in features)
        assert [f['properties']['unique_id'] for f in features] == SORTED_IDS

    def test_properties_carry_type_and_no_reserved_names(
        self, tmp_path: Path
    ) -> None:
        """Every feature has a type, no AIM_id, and no id in its properties."""
        _, written = write_integer(tmp_path, make_inventory())
        for feature in written['features']:
            properties = feature['properties']
            assert properties['type'] == 'building'
            assert 'AIM_id' not in properties
            assert 'id' not in properties
            expected = ASSETS[properties['unique_id']]['features']
            assert {
                k: v for k, v in properties.items() if k != 'unique_id'
            } == expected

    def test_all_digit_asset_id_is_written_as_a_string(self, tmp_path: Path) -> None:
        """An all-digit asset id keeps its string form and leading zero."""
        _, written = write_integer(tmp_path, make_inventory())
        assert written['features'][0]['properties']['unique_id'] == '0213456789'

    def test_returned_dictionary_matches_the_file(self, tmp_path: Path) -> None:
        """The returned GeoJSON carries the same features as the file."""
        returned, written = write_integer(tmp_path, make_inventory())
        assert json.loads(json.dumps(returned['features'])) == written['features']

    def test_insertion_order_does_not_change_the_file(self, tmp_path: Path) -> None:
        """Two insertion orders give the same features and the same table."""
        first = tmp_path / 'first'
        second = tmp_path / 'second'
        first.mkdir()
        second.mkdir()
        _, written_first = write_integer(first, make_inventory())
        _, written_second = write_integer(
            second, make_inventory(list(reversed(SORTED_IDS)))
        )
        assert written_first['features'] == written_second['features']
        assert read_table(first) == read_table(second)

    def test_asset_without_geometry_receives_no_number(self, tmp_path: Path) -> None:
        """A skipped asset leaves no gap in the ids and no row in the table."""
        inventory = make_inventory()
        inventory.add_asset(
            '9q8yyk0000', Asset('9q8yyk0000', [], {'type': 'building'})
        )
        _, written = write_integer(tmp_path, inventory)
        assert [f['id'] for f in written['features']] == [1, 2, 3, 4]
        assert [row['unique_id'] for row in read_table(tmp_path)] == SORTED_IDS

    def test_ids_are_numbered_within_each_type(self, tmp_path: Path) -> None:
        """Each type is numbered 1 to k, and the file is ordered by type."""
        inventory = make_inventory()
        for asset_id in ('9q8yyk8ytp', '9q8yym1bc0'):
            inventory.add_asset_features(
                asset_id, {'type': 'bridge'}, overwrite=True
            )
        _, written = write_integer(tmp_path, inventory)
        assert [
            (f['properties']['type'], f['id'], f['properties']['unique_id'])
            for f in written['features']
        ] == [
            ('bridge', 1, '9q8yyk8ytp'),
            ('bridge', 2, '9q8yym1bc0'),
            ('building', 1, '0213456789'),
            ('building', 2, '9q8yyk8ytp-152'),
        ]
        assert [
            (row['type'], int(row['id']), row['unique_id'])
            for row in read_table(tmp_path)
        ] == [
            (f['properties']['type'], f['id'], f['properties']['unique_id'])
            for f in written['features']
        ]

    def test_integer_asset_ids_are_numbered_in_string_order(
        self, tmp_path: Path
    ) -> None:
        """Integer keys 1, 2 and 10 are ordered as '1', '10' and '2'."""
        inventory = AssetInventory()
        for asset_id in (1, 2, 10):
            inventory.add_asset(
                asset_id, Asset(asset_id, [[-122.3, 37.8]], {'type': 'building'})
            )
        _, written = write_integer(tmp_path, inventory)
        assert [
            (f['id'], f['properties']['unique_id']) for f in written['features']
        ] == [
            (1, '1'),
            (2, '10'),
            (3, '2'),
        ]
        restored = AssetInventory()
        restored.read_from_geojson(
            str(tmp_path / 'inventory.geojson'), key_by_unique_id=True
        )
        assert list(restored.inventory) == ['1', '10', '2']

    def test_no_file_writes_no_table(self, tmp_path: Path) -> None:
        """Without an output file the ids are returned and nothing is written."""
        returned = make_inventory().write_to_geojson(integer_ids=True)
        assert [f['id'] for f in returned['features']] == [1, 2, 3, 4]
        assert list(tmp_path.iterdir()) == []


class TestIdTable:
    """The sidecar table maps each integer id back to its asset id."""

    def test_table_matches_the_file_row_for_row(self, tmp_path: Path) -> None:
        """One row per feature, in file order, with the same type and ids."""
        _, written = write_integer(tmp_path, make_inventory())
        rows = read_table(tmp_path)
        assert [(row['type'], int(row['id']), row['unique_id']) for row in rows] == [
            (f['properties']['type'], f['id'], f['properties']['unique_id'])
            for f in written['features']
        ]

    def test_table_header_and_name(self, tmp_path: Path) -> None:
        """The table sits beside the file as ``<stem>_ids.csv``."""
        write_integer(tmp_path, make_inventory())
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            'inventory.geojson',
            'inventory_ids.csv',
        ]
        text = (tmp_path / 'inventory_ids.csv').read_text(encoding='utf-8')
        assert text.splitlines()[0] == 'type,id,unique_id'
        assert text.splitlines()[1] == 'building,1,0213456789'

    def test_existing_files_are_overwritten(self, tmp_path: Path) -> None:
        """A second export replaces both the file and the table."""
        write_integer(tmp_path, make_inventory())
        inventory = make_inventory()
        inventory.remove_asset('9q8yym1bc0')
        _, written = write_integer(tmp_path, inventory)
        assert len(written['features']) == 3
        assert len(read_table(tmp_path)) == 3

    def test_failed_write_leaves_the_old_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A write that fails leaves the earlier files and no partial file."""
        write_integer(tmp_path, make_inventory())
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

        def fail(*_args: object, **_kwargs: object) -> None:
            raise OSError('disk full')

        monkeypatch.setattr(asset_inventory_module.json, 'dump', fail)
        inventory = make_inventory()
        inventory.remove_asset('9q8yym1bc0')
        with pytest.raises(OSError, match='disk full'):
            inventory.write_to_geojson(
                str(tmp_path / 'inventory.geojson'), integer_ids=True
            )
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

    @pytest.mark.parametrize('existing', ['pair', 'none'])
    def test_failed_second_rename_leaves_the_old_pair(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
    ) -> None:
        """A failed GeoJSON rename after the table's leaves the old files."""
        if existing == 'pair':
            write_integer(tmp_path, make_inventory())
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        geojson_path = tmp_path / 'inventory.geojson'
        replace = type(geojson_path).replace

        def fail_on_geojson(self: Path, target: Path) -> Path:
            if target == geojson_path:
                raise OSError('target busy')
            return replace(self, target)

        monkeypatch.setattr(type(geojson_path), 'replace', fail_on_geojson)
        inventory = make_inventory()
        inventory.remove_asset('9q8yym1bc0')
        with pytest.raises(OSError, match='target busy'):
            inventory.write_to_geojson(str(geojson_path), integer_ids=True)
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


class TestIntegerIdRejects:
    """The export refuses an inventory it would misrepresent."""

    @pytest.mark.parametrize('reserved', ['unique_id', 'AIM_id'])
    def test_reserved_feature_raises(self, tmp_path: Path, reserved: str) -> None:
        """An asset that already carries a reserved name raises."""
        inventory = make_inventory()
        inventory.add_asset_features('9q8yym1bc0', {reserved: 'x'})
        with pytest.raises(ValueError, match=reserved):
            inventory.write_to_geojson(
                str(tmp_path / 'inventory.geojson'), integer_ids=True
            )
        assert list(tmp_path.iterdir()) == []

    def test_missing_type_raises(self) -> None:
        """An asset without a type feature raises."""
        inventory = make_inventory()
        inventory.remove_features(['type'])
        with pytest.raises(ValueError, match="'type'"):
            inventory.write_to_geojson(integer_ids=True)

    @pytest.mark.parametrize('blank', [None, float('nan'), ''])
    def test_blank_type_raises_naming_the_asset(
        self, tmp_path: Path, blank: object
    ) -> None:
        """A type that is None, NaN or empty raises and names the asset."""
        inventory = make_inventory()
        inventory.add_asset_features('9q8yym1bc0', {'type': blank}, overwrite=True)
        with pytest.raises(ValueError, match="'9q8yym1bc0' has no 'type'"):
            inventory.write_to_geojson(
                str(tmp_path / 'inventory.geojson'), integer_ids=True
            )
        assert list(tmp_path.iterdir()) == []

    def test_ids_with_the_same_string_form_raise(self) -> None:
        """The integer 7 and the string '7' cannot both be exported."""
        inventory = AssetInventory()
        for asset_id in (7, '7'):
            inventory.add_asset(
                asset_id, Asset(asset_id, [[-122.3, 37.8]], {'type': 'building'})
            )
        with pytest.raises(ValueError, match="7 and '7' have the same string form"):
            inventory.write_to_geojson(integer_ids=True)


class TestReadByUniqueId:
    """Reading by unique_id is the inverse of the integer-id export."""

    def test_round_trip_reproduces_the_inventory(self, tmp_path: Path) -> None:
        """Asset ids, coordinates and features come back unchanged."""
        original = make_inventory()
        write_integer(tmp_path, original)
        restored = AssetInventory()
        assert restored.read_from_geojson(
            str(tmp_path / 'inventory.geojson'), key_by_unique_id=True
        )
        assert sorted(restored.inventory) == SORTED_IDS
        for asset_id, asset in original.inventory.items():
            copy_back = restored.inventory[asset_id]
            assert copy_back.coordinates == asset.coordinates
            assert copy_back.features == asset.features

    def test_round_trip_keeps_each_type(self, tmp_path: Path) -> None:
        """Each asset comes back with its own type, not ``asset_type``."""
        original = make_inventory()
        original.add_asset_features('9q8yyk8ytp', {'type': 'bridge'}, overwrite=True)
        write_integer(tmp_path, original)
        restored = AssetInventory()
        restored.read_from_geojson(
            str(tmp_path / 'inventory.geojson'), key_by_unique_id=True
        )
        assert {k: a.features for k, a in restored.inventory.items()} == {
            k: a.features for k, a in original.inventory.items()
        }
        assert restored.inventory['9q8yyk8ytp'].features['type'] == 'bridge'

    def test_all_digit_unique_id_stays_a_string(self, tmp_path: Path) -> None:
        """No asset id is converted to an integer."""
        write_integer(tmp_path, make_inventory())
        restored = AssetInventory()
        restored.read_from_geojson(
            str(tmp_path / 'inventory.geojson'), key_by_unique_id=True
        )
        assert all(isinstance(k, str) for k in restored.inventory)
        assert '0213456789' in restored.inventory

    def test_default_read_keys_by_the_integer_id(self, tmp_path: Path) -> None:
        """Without the option the integer id keys and unique_id stays a feature."""
        write_integer(tmp_path, make_inventory())
        restored = AssetInventory()
        restored.read_from_geojson(str(tmp_path / 'inventory.geojson'))
        assert sorted(restored.inventory) == [1, 2, 3, 4]
        assert restored.inventory[1].features['unique_id'] == '0213456789'

    @staticmethod
    def _write_features(tmp_path: Path, unique_ids: list[Any]) -> str:
        """Write point features with the given unique_id values (None: absent)."""
        features = []
        for number, unique_id in enumerate(unique_ids, start=1):
            properties: dict[str, Any] = {'type': 'building'}
            if unique_id is not None:
                properties['unique_id'] = unique_id
            features.append(
                {
                    'type': 'Feature',
                    'id': number,
                    'properties': properties,
                    'geometry': {'type': 'Point', 'coordinates': [-122.3, 37.8]},
                }
            )
        path = tmp_path / 'handmade.geojson'
        path.write_text(
            json.dumps({'type': 'FeatureCollection', 'features': features}),
            encoding='utf-8',
        )
        return str(path)

    def test_missing_unique_id_raises_and_adds_nothing(self, tmp_path: Path) -> None:
        """A feature without unique_id raises before any asset is added."""
        path = self._write_features(tmp_path, ['a', None])
        inventory = AssetInventory()
        with pytest.raises(ValueError, match='index 1'):
            inventory.read_from_geojson(path, key_by_unique_id=True)
        assert inventory.inventory == {}

    def test_empty_unique_id_raises_and_adds_nothing(self, tmp_path: Path) -> None:
        """An empty unique_id counts as missing."""
        path = self._write_features(tmp_path, ['a', ''])
        inventory = AssetInventory()
        with pytest.raises(ValueError, match='index 1'):
            inventory.read_from_geojson(path, key_by_unique_id=True)
        assert inventory.inventory == {}

    def test_unique_id_already_in_the_inventory_raises(self, tmp_path: Path) -> None:
        """A unique_id that already keys an asset raises before any is added."""
        path = self._write_features(tmp_path, ['a', 'b'])
        inventory = AssetInventory()
        inventory.add_asset('b', Asset('b', [[-122.4, 37.7]], {'type': 'building'}))
        with pytest.raises(ValueError, match='already keys an asset'):
            inventory.read_from_geojson(path, key_by_unique_id=True)
        assert list(inventory.inventory) == ['b']

    def test_duplicate_unique_id_raises_and_adds_nothing(
        self, tmp_path: Path
    ) -> None:
        """Two features with the same unique_id raise."""
        path = self._write_features(tmp_path, ['a', 'b', 'a'])
        inventory = AssetInventory()
        with pytest.raises(ValueError, match='more than one feature'):
            inventory.read_from_geojson(path, key_by_unique_id=True)
        assert inventory.inventory == {}

    def test_id_column_with_key_by_unique_id_raises(self, tmp_path: Path) -> None:
        """The two ways of choosing the key cannot be combined."""
        path = self._write_features(tmp_path, ['a'])
        with pytest.raises(ValueError, match="id_column='unique_id'"):
            AssetInventory().read_from_geojson(
                path, id_column='unique_id', key_by_unique_id=True
            )


class _FixedDatetime(dt.datetime):
    """A datetime whose ``now`` is fixed, so the written file is repeatable."""

    @classmethod
    def now(cls, tz: dt.tzinfo | None = None) -> _FixedDatetime:
        """Return 2026-01-01 00:00 in ``tz``."""
        return cls(2026, 1, 1, tzinfo=tz)


class TestDefaultWriteUnchanged:
    """A call without ``integer_ids`` writes the string-id file."""

    def test_default_file_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A call without ``integer_ids`` writes string ids and no table."""
        monkeypatch.setattr(asset_inventory_module, 'datetime', _FixedDatetime)
        monkeypatch.setattr(asset_inventory_module, 'version', lambda _: '9.9')
        inventory = AssetInventory()
        inventory.add_asset(
            'b', Asset('b', [[-122.3, 37.8]], {'type': 'building', 'id': 'x'})
        )
        inventory.add_asset(3, Asset(3, [[-122.4, 37.7]], {'Name': 'A'}))
        output = tmp_path / 'default.geojson'
        inventory.write_to_geojson(str(output))
        expected = {
            'type': 'FeatureCollection',
            'generated': '2026-01-01 00:00:00+00:00',
            'brails_version': '9.9',
            'crs': {
                'type': 'name',
                'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'},
            },
            'features': [
                {
                    'type': 'Feature',
                    'properties': {'type': 'building'},
                    'geometry': {'type': 'Point', 'coordinates': [-122.3, 37.8]},
                    'id': 'b',
                },
                {
                    'type': 'Feature',
                    'properties': {'Name': 'A'},
                    'geometry': {'type': 'Point', 'coordinates': [-122.4, 37.7]},
                    'id': '3',
                },
            ],
        }
        assert output.read_text(encoding='utf-8') == json.dumps(expected, indent=2)
        assert [p.name for p in tmp_path.iterdir()] == ['default.geojson']
