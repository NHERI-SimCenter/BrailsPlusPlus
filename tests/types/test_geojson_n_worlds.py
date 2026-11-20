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


"""Tests for the number of possible worlds a GeoJSON inventory states."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from brails.types.asset_inventory import Asset, AssetInventory, check_n_worlds

if TYPE_CHECKING:
    from pathlib import Path

MEMBERS = ['type', 'generated', 'brails_version', 'n_worlds', 'crs', 'features']


pytestmark = pytest.mark.usefixtures('no_network')


def inventory(features: dict[str, dict[str, Any]], n_pw: int = 1) -> AssetInventory:
    """
    Build an inventory of point assets with the default vector features.

    Args:
        features (dict[str, dict[str, Any]]):
            Asset id to its features.
        n_pw (int, optional):
            The number of worlds the inventory states. Defaults to 1.

    Returns:
        AssetInventory:
            The inventory, its assets a meter or so apart.
    """
    inv = AssetInventory()
    for index, (asset_id, values) in enumerate(features.items()):
        point = [[-122.0 + index * 0.00001, 38.0]]
        inv.add_asset(asset_id, Asset(asset_id, point, dict(values)))
    inv.n_pw = n_pw
    return inv


def written(inv: AssetInventory, path: Path, **options: object) -> dict[str, Any]:
    """
    Write an inventory to a file and return the file's content.

    Args:
        inv (AssetInventory):
            The inventory to write.
        path (Path):
            The file to write.
        **options (object):
            Keyword arguments for ``write_to_geojson``.

    Returns:
        dict[str, Any]:
            The file's ``FeatureCollection``, as parsed.
    """
    inv.write_to_geojson(str(path), **options)
    return json.loads(path.read_text(encoding='utf-8'))


def state(inv: AssetInventory) -> tuple[int, dict[str, dict[str, Any]]]:
    """
    Return what a refused read must leave as it was.

    Args:
        inv (AssetInventory):
            The inventory.

    Returns:
        tuple[int, dict[str, dict[str, Any]]]:
            The inventory's ``n_pw`` and a copy of each asset's features.
    """
    return inv.n_pw, {key: dict(a.features) for key, a in inv.inventory.items()}


THREE_WORLDS = {
    'a': {'StructureType': ['W1', 'W2', 'C1'], 'PlanArea': 10.0},
    'b': {'StructureType': ['S1', 'S1', 'S2'], 'HousingUnits': [2, 3]},
}


class TestWrite:
    """write_to_geojson writes n_worlds after brails_version."""

    @pytest.mark.parametrize(
        'options', [{}, {'integer_ids': True, 'asset_type': 'building'}]
    )
    def test_member_follows_brails_version(
        self, tmp_path: Path, options: dict[str, object]
    ) -> None:
        """Both id forms place n_worlds after brails_version, in file and dict."""
        inv = inventory(THREE_WORLDS, n_pw=3)
        collection = written(inv, tmp_path / 'worlds.geojson', **options)
        assert list(collection) == MEMBERS
        assert collection['n_worlds'] == 3
        assert list(inv.write_to_geojson(**options)) == MEMBERS

    @pytest.mark.parametrize(
        ('features', 'n_pw'),
        [
            ({'a': {'PlanArea': 10.0}}, 1),
            ({'a': {'PlanArea': 10.0}}, 3),
            ({'a': {'StructureType': ['W1', 'W2', 'C1', 'C1']}}, 4),
            ({'a': {'HousingUnits': [2, 3]}}, 1),
        ],
        ids=['one world', 'no lists', 'lists of n_pw', 'vector'],
    )
    def test_count_is_n_pw(self, tmp_path: Path, features: dict, n_pw: int) -> None:
        """The member is n_pw."""
        path = tmp_path / 'inventory.geojson'
        assert written(inventory(features, n_pw), path)['n_worlds'] == n_pw

    @pytest.mark.parametrize(
        ('features', 'n_pw', 'message'),
        [
            (
                {'a': {'X': [1, 2, 3, 4]}},
                1,
                "'a' holds 4 values under 'X', but n_pw is 1",
            ),
            (
                {'a': {'X': [1, 2, 3]}, 'b': {'X': [1, 2]}},
                3,
                "'b' holds 2 values under 'X', but n_pw is 3",
            ),
            ({'a': {'X': []}}, 1, "'a' holds 0 values under 'X', but n_pw is 1"),
        ],
        ids=['lists under n_pw 1', 'one list short', 'empty list'],
    )
    def test_list_of_another_length_refused(
        self, tmp_path: Path, features: dict, n_pw: int, message: str
    ) -> None:
        """A list of another length than n_pw raises, and no file is written."""
        with pytest.raises(ValueError, match=message):
            inventory(features, n_pw).write_to_geojson(str(tmp_path / 'x.geojson'))
        assert list(tmp_path.iterdir()) == []

    def test_count_written_as_an_int(self, tmp_path: Path) -> None:
        """An n_pw held as a NumPy integer is written as a plain integer."""
        inv = inventory(THREE_WORLDS, n_pw=3)
        inv.n_pw = np.int64(3)
        collection = written(inv, tmp_path / 'numpy.geojson')
        assert type(collection['n_worlds']) is int
        assert collection['n_worlds'] == 3

    @pytest.mark.parametrize('n_pw', [0, np.int64(0), True, 3.0])
    def test_n_pw_that_is_no_count_refused(self, n_pw: object) -> None:
        """An n_pw that is not an integer of at least 1 raises."""
        with pytest.raises(
            ValueError, match='which is not an integer of at least 1'
        ):
            inventory({'a': {}}, n_pw=n_pw).write_to_geojson()


class TestRead:
    """read_from_geojson checks n_worlds against the file and the inventory."""

    @pytest.mark.parametrize(
        ('options', 'read_options'),
        [
            ({}, {}),
            (
                {'integer_ids': True, 'asset_type': 'building'},
                {'key_by_unique_id': True},
            ),
        ],
    )
    def test_empty_inventory_takes_the_count(
        self, tmp_path: Path, options: dict, read_options: dict
    ) -> None:
        """A file of three worlds read into an empty inventory gives n_pw 3."""
        path = tmp_path / 'worlds.geojson'
        inventory(THREE_WORLDS, n_pw=3).write_to_geojson(str(path), **options)
        back = AssetInventory()
        back.read_from_geojson(str(path), **read_options)
        assert back.n_pw == 3
        back.validate_possible_worlds()
        world = back.get_world_realization(2)
        assert world.inventory['a'].features['StructureType'] == 'C1'

    @pytest.mark.parametrize(
        ('features', 'n_pw'),
        [(THREE_WORLDS, 3), ({'a': {'PlanArea': 10.0}}, 1)],
        ids=['lists', 'no lists'],
    )
    def test_file_without_the_member_counts_its_lists(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        features: dict,
        n_pw: int,
    ) -> None:
        """Without n_worlds, the file's lists give its number of worlds."""
        collection = inventory(features, n_pw=n_pw).write_to_geojson()
        del collection['n_worlds']
        path = tmp_path / 'old.geojson'
        path.write_text(json.dumps(collection), encoding='utf-8')
        back = AssetInventory()
        back.read_from_geojson(str(path))
        assert back.n_pw == n_pw
        back.validate_possible_worlds()
        printed = (
            f'{path} gives no n_worlds, so it is read as 3 possible worlds, the '
            "number of values its lists hold, first under 'StructureType'. If "
            "those lists are not possible worlds, pass 'StructureType' in "
            'vector_features, with DEFAULT_VECTOR_FEATURES.'
        )
        assert (printed in capsys.readouterr().out) == (n_pw > 1)

    def test_count_given_by_the_member_not_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A file that gives n_worlds is read without the line on its lists."""
        path = tmp_path / 'worlds.geojson'
        inventory(THREE_WORLDS, n_pw=3).write_to_geojson(str(path))
        AssetInventory().read_from_geojson(str(path))
        assert 'gives no n_worlds' not in capsys.readouterr().out

    def test_vertex_elevations_are_one_world(self, tmp_path: Path) -> None:
        """Per-vertex all_elevations lists are written and read back as 1 world."""
        features = {
            'a': {'all_elevations': [1.0, 2.0, 3.0, 4.0]},
            'b': {'all_elevations': [1.0, 2.0, 3.0, 4.0, 5.0]},
            'c': {'all_elevations': [5.0, 6.0, 7.0, 8.0, 9.0]},
        }
        collection = written(inventory(features), tmp_path / 'written.geojson')
        assert collection['n_worlds'] == 1
        del collection['n_worlds']
        path = tmp_path / 'old.geojson'
        path.write_text(json.dumps(collection), encoding='utf-8')
        back = AssetInventory()
        back.read_from_geojson(str(path))
        assert back.n_pw == 1
        assert back.inventory['b'].features['all_elevations'] == [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]

    @pytest.mark.parametrize(
        ('values', 'message'),
        [
            (
                ['S1', 'S2'],
                "feature 0 holds 3 under 'StructureType' and feature 1 holds 2 "
                "under 'StructureType'. Pass the vector features",
            ),
            ([], "Feature 1 .* holds an empty list under 'StructureType'"),
        ],
        ids=['two lengths', 'empty list'],
    )
    def test_file_without_the_member_refused_for_its_lists(
        self, tmp_path: Path, values: list, message: str
    ) -> None:
        """Without n_worlds, lists of two lengths or an empty list are refused."""
        collection = inventory(THREE_WORLDS, n_pw=3).write_to_geojson()
        del collection['n_worlds']
        collection['features'][1]['properties']['StructureType'] = values
        path = tmp_path / 'old.geojson'
        path.write_text(json.dumps(collection), encoding='utf-8')
        target = AssetInventory()
        with pytest.raises(ValueError, match=message):
            target.read_from_geojson(str(path))
        assert state(target) == (1, {})

    def test_member_not_a_count_refused(self, tmp_path: Path) -> None:
        """A member that is not an integer of at least 1 is refused."""
        collection = inventory(THREE_WORLDS, n_pw=3).write_to_geojson()
        collection['n_worlds'] = 0
        path = tmp_path / 'bad.geojson'
        path.write_text(json.dumps(collection), encoding='utf-8')
        target = inventory({'z': {'X': [1, 2, 3]}}, n_pw=3)
        before = state(target)
        with pytest.raises(ValueError, match='gives n_worlds as'):
            target.read_from_geojson(str(path))
        assert state(target) == before

    def test_list_of_another_length_refused(self, tmp_path: Path) -> None:
        """A list not of the stated length is refused, and nothing is changed."""
        collection = inventory(THREE_WORLDS, n_pw=3).write_to_geojson()
        collection['features'][1]['properties']['StructureType'] = ['S1', 'S2']
        path = tmp_path / 'short.geojson'
        path.write_text(json.dumps(collection), encoding='utf-8')
        target = AssetInventory()
        with pytest.raises(
            ValueError, match=r"Feature 1 .* 2 values under 'StructureType'"
        ):
            target.read_from_geojson(str(path))
        assert state(target) == (1, {})

    @pytest.mark.parametrize(
        ('held', 'n_pw', 'n_worlds', 'member'),
        [
            ({'X': [1, 2, 3]}, 3, 3, True),
            ({'X': [1, 2, 3]}, 3, 3, False),
            ({'X': 1}, 1, 3, True),
            ({'X': 1}, 1, 3, False),
            ({'X': [1, 2, 3]}, 3, 1, True),
            ({'X': [1, 2, 3]}, 3, 5, True),
        ],
        ids=[
            'equal',
            'equal by its lists',
            'inventory of 1',
            'inventory of 1 by its lists',
            'file of 1',
            'two counts',
        ],
    )
    def test_count_against_the_inventory(
        self,
        tmp_path: Path,
        held: dict[str, Any],
        n_pw: int,
        n_worlds: int,
        *,
        member: bool,
    ) -> None:
        """An inventory with assets takes a file of its own number of worlds only."""
        features = {
            'a': {'StructureType': ['W1'] * n_worlds} if n_worlds > 1 else {}
        }
        collection = inventory(features, n_worlds).write_to_geojson()
        if not member:
            del collection['n_worlds']
        path = tmp_path / 'file.geojson'
        path.write_text(json.dumps(collection), encoding='utf-8')
        target = inventory({'z': held}, n_pw)
        before = state(target)
        if n_worlds == n_pw:
            target.read_from_geojson(str(path))
            assert target.n_pw == n_pw
            assert sorted(target.inventory) == ['a', 'z']
        else:
            noun = 'world' if n_worlds == 1 else 'worlds'
            with pytest.raises(
                ValueError,
                match=f'{n_worlds} possible {noun} and the inventory holds {n_pw}. '
                'Read the file into an empty inventory.',
            ):
                target.read_from_geojson(str(path))
            assert state(target) == before


@pytest.mark.parametrize('value', [True, 0, -1, '3', 3.0, None])
def test_check_n_worlds_refuses_what_is_no_count(value: object) -> None:
    """A bool, an integer below 1 or a value that is no integer is refused."""
    with pytest.raises(ValueError, match='which is not an integer of at least 1'):
        check_n_worlds(value, 'file.geojson')


def test_check_n_worlds_returns_the_count() -> None:
    """An integer of at least 1 is the count."""
    assert check_n_worlds(10, 'file.geojson') == 10
