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
import datetime as dt
import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
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
        """Two insertion orders give the same features."""
        first = tmp_path / 'first'
        second = tmp_path / 'second'
        first.mkdir()
        second.mkdir()
        _, written_first = write_integer(first, make_inventory())
        _, written_second = write_integer(
            second, make_inventory(list(reversed(SORTED_IDS)))
        )
        assert written_first['features'] == written_second['features']

    def test_asset_without_geometry_receives_no_number(self, tmp_path: Path) -> None:
        """A skipped asset leaves no gap in the ids."""
        inventory = make_inventory()
        inventory.add_asset(
            '9q8yyk0000', Asset('9q8yyk0000', [], {'type': 'building'})
        )
        _, written = write_integer(tmp_path, inventory)
        assert [f['id'] for f in written['features']] == [1, 2, 3, 4]
        assert [
            f['properties']['unique_id'] for f in written['features']
        ] == SORTED_IDS

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

    def test_integer_asset_ids_are_numbered_in_numeric_order(
        self, tmp_path: Path
    ) -> None:
        """Integer keys 10, 2 and 1 are ordered as 1, 2 and 10."""
        inventory = AssetInventory()
        for asset_id in (10, 2, 1):
            inventory.add_asset(
                asset_id, Asset(asset_id, [[-122.3, 37.8]], {'type': 'building'})
            )
        _, written = write_integer(tmp_path, inventory)
        assert [
            (f['id'], f['properties']['unique_id']) for f in written['features']
        ] == [
            (1, '1'),
            (2, '2'),
            (3, '10'),
        ]

    def test_numpy_integer_asset_ids_are_numbered_in_numeric_order(
        self, tmp_path: Path
    ) -> None:
        """A numpy integer key counts as an integer."""
        inventory = AssetInventory()
        for asset_id in (np.int64(10), np.int64(2), 1):
            inventory.add_asset(
                asset_id, Asset(asset_id, [[-122.3, 37.8]], {'type': 'building'})
            )
        _, written = write_integer(tmp_path, inventory)
        assert [f['properties']['unique_id'] for f in written['features']] == [
            '1',
            '2',
            '10',
        ]

    def test_bool_asset_id_is_not_an_integer(self, tmp_path: Path) -> None:
        """With a bool key among integer keys, all are ordered as strings."""
        inventory = AssetInventory()
        for asset_id in (True, 10, 2):
            inventory.add_asset(
                asset_id, Asset(asset_id, [[-122.3, 37.8]], {'type': 'building'})
            )
        _, written = write_integer(tmp_path, inventory)
        assert [f['properties']['unique_id'] for f in written['features']] == [
            '10',
            '2',
            'True',
        ]

    def test_mixed_asset_ids_are_numbered_in_string_order(
        self, tmp_path: Path
    ) -> None:
        """With one string key among integer keys, all are ordered as strings."""
        inventory = AssetInventory()
        for asset_id in (2, 10, 'a'):
            inventory.add_asset(
                asset_id, Asset(asset_id, [[-122.3, 37.8]], {'type': 'building'})
            )
        _, written = write_integer(tmp_path, inventory)
        assert [f['properties']['unique_id'] for f in written['features']] == [
            '10',
            '2',
            'a',
        ]

    def test_no_file_writes_nothing(self, tmp_path: Path) -> None:
        """Without an output file the ids are returned and nothing is written."""
        returned = make_inventory().write_to_geojson(integer_ids=True)
        assert [f['id'] for f in returned['features']] == [1, 2, 3, 4]
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('ids', ['integer', 'string'])
class TestOutputFile:
    """Both export paths replace the output file only after a full write."""

    def test_existing_file_is_overwritten(self, tmp_path: Path, ids: str) -> None:
        """A second export replaces the file and leaves no other file."""
        output = str(tmp_path / 'inventory.geojson')
        make_inventory().write_to_geojson(output, integer_ids=ids == 'integer')
        inventory = make_inventory()
        inventory.remove_asset('9q8yym1bc0')
        inventory.write_to_geojson(output, integer_ids=ids == 'integer')
        written = json.loads((tmp_path / 'inventory.geojson').read_text('utf-8'))
        assert len(written['features']) == 3
        assert [p.name for p in tmp_path.iterdir()] == ['inventory.geojson']

    def test_failed_write_leaves_the_old_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ids: str
    ) -> None:
        """A write that fails leaves the earlier file and no partial file."""
        output = str(tmp_path / 'inventory.geojson')
        make_inventory().write_to_geojson(output, integer_ids=ids == 'integer')
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

        def fail(*_args: object, **_kwargs: object) -> None:
            raise OSError('disk full')

        monkeypatch.setattr(asset_inventory_module.json, 'dump', fail)
        inventory = make_inventory()
        inventory.remove_asset('9q8yym1bc0')
        with pytest.raises(OSError, match='disk full'):
            inventory.write_to_geojson(output, integer_ids=ids == 'integer')
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

    @pytest.mark.parametrize('existing', ['file', 'none'])
    def test_failed_replace_leaves_the_old_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ids: str,
        existing: str,
    ) -> None:
        """A failed replace leaves the earlier file, or none, and no partial."""
        geojson_path = tmp_path / 'inventory.geojson'
        if existing == 'file':
            make_inventory().write_to_geojson(
                str(geojson_path), integer_ids=ids == 'integer'
            )
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

        def fail(*_args: object) -> Path:
            raise OSError('target busy')

        monkeypatch.setattr(type(geojson_path), 'replace', fail)
        inventory = make_inventory()
        inventory.remove_asset('9q8yym1bc0')
        with pytest.raises(OSError, match='target busy'):
            inventory.write_to_geojson(
                str(geojson_path), integer_ids=ids == 'integer'
            )
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

    @pytest.mark.skipif(os.name == 'nt', reason='POSIX file modes')
    def test_file_has_the_mode_of_a_new_file(self, tmp_path: Path, ids: str) -> None:
        """The written file is not restricted to its owner."""
        output = tmp_path / 'inventory.geojson'
        umask = os.umask(0o022)
        try:
            make_inventory().write_to_geojson(
                str(output), integer_ids=ids == 'integer'
            )
        finally:
            os.umask(umask)
        assert output.stat().st_mode & 0o777 == 0o644

    def test_nan_is_written_as_null(self, tmp_path: Path, ids: str) -> None:
        """A NaN feature value is written as ``null``, and the file is JSON."""
        inventory = make_inventory()
        inventory.add_asset_features(
            '9q8yym1bc0', {'ReplacementCost': float('nan')}, overwrite=True
        )
        output = tmp_path / 'inventory.geojson'
        inventory.write_to_geojson(str(output), integer_ids=ids == 'integer')
        text = output.read_text(encoding='utf-8')
        assert 'NaN' not in text
        written = json.loads(text)
        (feature,) = (
            f for f in written['features'] if 'ReplacementCost' in f['properties']
        )
        assert feature['properties']['ReplacementCost'] is None

    @pytest.mark.parametrize('existing', ['file', 'none'])
    def test_infinite_value_raises_and_writes_nothing(
        self, tmp_path: Path, ids: str, existing: str
    ) -> None:
        """An infinite value raises and leaves the earlier file, or none."""
        output = tmp_path / 'inventory.geojson'
        if existing == 'file':
            make_inventory().write_to_geojson(
                str(output), integer_ids=ids == 'integer'
            )
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        inventory = make_inventory()
        inventory.add_asset_features(
            '9q8yym1bc0', {'ReplacementCost': float('-inf')}, overwrite=True
        )
        with pytest.raises(ValueError, match='infinite'):
            inventory.write_to_geojson(str(output), integer_ids=ids == 'integer')
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


class TestAssetType:
    """asset_type gives a type to the assets without one, in the file only."""

    @pytest.mark.parametrize('blank', [None, float('nan'), ''])
    def test_blank_and_missing_types_take_asset_type(self, blank: object) -> None:
        """A missing or blank type is written as asset_type; others are kept."""
        inventory = make_inventory()
        inventory.add_asset_features('9q8yym1bc0', {'type': blank}, overwrite=True)
        inventory.inventory['0213456789'].features.pop('type')
        inventory.add_asset_features(
            '9q8yyk8ytp', {'type': 'bridge'}, overwrite=True
        )
        before = copy.deepcopy(
            {k: a.features for k, a in inventory.inventory.items()}
        )
        written = inventory.write_to_geojson(integer_ids=True, asset_type='building')
        types = {
            f['properties']['unique_id']: f['properties']['type']
            for f in written['features']
        }
        assert types['9q8yym1bc0'] == 'building'
        assert types['0213456789'] == 'building'
        assert types['9q8yyk8ytp'] == 'bridge'
        after = {k: a.features for k, a in inventory.inventory.items()}
        assert repr(after) == repr(before)

    def test_same_file_as_types_set_in_the_inventory(self) -> None:
        """asset_type writes what setting the type in the inventory writes."""
        given = make_inventory()
        given.remove_features(['type'])
        set_first = make_inventory()
        assert (
            given.write_to_geojson(integer_ids=True, asset_type='building')[
                'features'
            ]
            == set_first.write_to_geojson(integer_ids=True)['features']
        )

    def test_refused_without_integer_ids(self) -> None:
        """asset_type is never ignored: without integer_ids it is refused."""
        with pytest.raises(ValueError, match='only with integer_ids=True'):
            make_inventory().write_to_geojson(asset_type='building')

    @pytest.mark.parametrize(
        'value', ['', ' ', 'nan', 'NaN', 'None', 'null', 7, float('nan')]
    )
    def test_blank_or_other_asset_type_refused(self, value: object) -> None:
        """asset_type must be a string that names a type."""
        with pytest.raises(ValueError, match='a string that names a type'):
            make_inventory().write_to_geojson(integer_ids=True, asset_type=value)


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

    @pytest.mark.parametrize(
        ('asset_ids', 'keys'),
        [
            ((10, 2, 1), [1, 2, 10]),
            (('10', '2', '1'), [1, 2, 10]),
            ((10, 2, -3), ['-3', '10', '2']),
        ],
    )
    def test_export_read_and_reexport_number_alike(
        self, tmp_path: Path, asset_ids: tuple[Any, ...], keys: list[Any]
    ) -> None:
        """Plain-digit keys come back as ints, others as strings in string order."""
        inventory = AssetInventory()
        for asset_id in asset_ids:
            inventory.add_asset(
                asset_id, Asset(asset_id, [[-122.3, 37.8]], {'type': 'building'})
            )
        first = tmp_path / 'first'
        second = tmp_path / 'second'
        first.mkdir()
        second.mkdir()
        _, written_first = write_integer(first, inventory)
        restored = AssetInventory()
        restored.read_from_geojson(
            str(first / 'inventory.geojson'), key_by_unique_id=True
        )
        assert list(restored.inventory) == keys
        _, written_second = write_integer(second, restored)
        assert written_second['features'] == written_first['features']

    @pytest.mark.parametrize(
        ('unique_ids', 'keys'),
        [
            (['0', '7', '10'], [0, 7, 10]),
            (['07', '7'], ['07', '7']),
            (['7', '-7'], ['7', '-7']),
            (['7', 'a'], ['7', 'a']),
        ],
    )
    def test_keys_are_ints_only_when_every_unique_id_is_plain_digits(
        self, tmp_path: Path, unique_ids: list[str], keys: list[Any]
    ) -> None:
        """One unique_id with a leading zero, a sign or a letter keeps strings."""
        path = self._write_features(tmp_path, unique_ids)
        restored = AssetInventory()
        restored.read_from_geojson(path, key_by_unique_id=True)
        assert list(restored.inventory) == keys

    def test_integer_unique_id_already_in_the_inventory_raises(
        self, tmp_path: Path
    ) -> None:
        """An integer key that already keys an asset raises."""
        path = self._write_features(tmp_path, ['4', '5'])
        inventory = AssetInventory()
        inventory.add_asset(5, Asset(5, [[-122.4, 37.7]], {'type': 'building'}))
        with pytest.raises(ValueError, match="'5' already keys an asset"):
            inventory.read_from_geojson(path, key_by_unique_id=True)
        assert list(inventory.inventory) == [5]

    def test_mixed_unique_ids_stay_strings(self, tmp_path: Path) -> None:
        """Unique_ids not all plain digits keep every key a string."""
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
        """A call without ``integer_ids`` writes string ids and no other file."""
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

    def test_whole_number_coordinates_are_written_as_stored(
        self, tmp_path: Path
    ) -> None:
        """Coordinates are tuples in the payload, so -122.0 stays a float."""
        inventory = AssetInventory()
        inventory.add_asset(
            'b',
            Asset(
                'b',
                [[-122.0, 37.0], [-121.0, 37.0], [-121.0, 38.0], [-122.0, 37.0]],
                {'type': 'building'},
            ),
        )
        output = tmp_path / 'default.geojson'
        inventory.write_to_geojson(str(output))
        written = json.loads(output.read_text(encoding='utf-8'))
        ring = written['features'][0]['geometry']['coordinates'][0]
        assert ring[0] == [-122.0, 37.0]
        assert all(type(value) is float for point in ring for value in point)
