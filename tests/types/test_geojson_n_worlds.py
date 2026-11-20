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
        ('features', 'n_pw', 'n_worlds'),
        [
            ({'a': {'PlanArea': 10.0}}, 1, 1),
            ({'a': {'PlanArea': 10.0}}, 3, 3),
            ({'a': {'StructureType': ['W1', 'W2', 'C1', 'C1']}}, 4, 4),
            ({'a': {'StructureType': ['W1', 'W2', 'C1', 'C1']}}, 1, 4),
            ({'a': {'HousingUnits': [2, 3]}}, 1, 1),
        ],
        ids=[
            'one world',
            'no lists',
            'lists of n_pw',
            'lists under n_pw 1',
            'vector',
        ],
    )
    def test_count(
        self, tmp_path: Path, features: dict, n_pw: int, n_worlds: int
    ) -> None:
        """The count is n_pw, or under n_pw 1 the length the lists share."""
        path = tmp_path / 'inventory.geojson'
        assert written(inventory(features, n_pw), path)['n_worlds'] == n_worlds

    @pytest.mark.parametrize(
        ('features', 'n_pw', 'lengths'),
        [
            ({'a': {'X': [1, 2, 3]}, 'b': {'X': [1, 2]}}, 3, '2 and 3'),
            ({'a': {'X': [1, 2, 3]}, 'b': {'X': [1, 2]}}, 1, '2 and 3'),
            ({'a': {'X': [1, 2]}}, 3, '2'),
            ({'a': {'X': []}}, 1, '0'),
        ],
    )
    def test_left_out_where_lists_disagree(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        features: dict,
        n_pw: int,
        lengths: str,
    ) -> None:
        """Lists that give no count leave the member out, with one line."""
        path = tmp_path / 'mixed.geojson'
        collection = written(inventory(features, n_pw), path)
        assert 'n_worlds' not in collection
        assert len(collection['features']) == len(features)
        printed = capsys.readouterr().out.splitlines()
        assert printed[0] == (
            f'n_worlds is left out of {path}: n_pw is {n_pw} and the lists hold '
            f'{lengths} values. validate_possible_worlds names the assets whose '
            'lists do not hold n_pw values.'
        )

    def test_count_written_as_an_int(self, tmp_path: Path) -> None:
        """An n_pw held as a NumPy integer is written as a plain integer."""
        inv = inventory(THREE_WORLDS, n_pw=3)
        inv.n_pw = np.int64(3)
        collection = written(inv, tmp_path / 'numpy.geojson')
        assert type(collection['n_worlds']) is int
        assert collection['n_worlds'] == 3

    @pytest.mark.parametrize(
        ('n_pw', 'shown'),
        [(0, '0'), (np.int64(0), '0'), (True, 'True'), (3.0, '3.0')],
    )
    def test_no_member_for_what_is_no_count(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        n_pw: object,
        shown: str,
    ) -> None:
        """An n_pw that is not an integer of at least 1 writes no member."""
        path = tmp_path / 'zero.geojson'
        assert 'n_worlds' not in written(inventory({'a': {}}, n_pw=n_pw), path)
        assert capsys.readouterr().out.splitlines()[0] == (
            f'n_worlds is left out of {path}: n_pw is {shown}, which is not an '
            'integer of at least 1.'
        )


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

    def test_file_without_the_member_as_before(self, tmp_path: Path) -> None:
        """Without n_worlds, the inventory keeps its n_pw."""
        collection = inventory(THREE_WORLDS, n_pw=3).write_to_geojson()
        del collection['n_worlds']
        path = tmp_path / 'old.geojson'
        path.write_text(json.dumps(collection), encoding='utf-8')
        empty = AssetInventory()
        empty.read_from_geojson(str(path))
        assert empty.n_pw == 1
        five = inventory({'z': {'X': [1, 2, 3, 4, 5]}}, n_pw=5)
        five.read_from_geojson(str(path))
        assert five.n_pw == 5

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
        ('held', 'n_pw', 'n_worlds', 'result'),
        [
            ({'X': [1, 2, 3]}, 3, 3, 3),
            ({'X': 1}, 1, 3, 3),
            ({'X': [1, 2]}, 1, 3, r"asset 'z' .* 2 values under 'X'"),
            ({'X': [1, 2, 3]}, 3, 5, '5 possible worlds and the inventory holds 3'),
        ],
        ids=['equal', 'inventory of 1', 'own list too short', 'two'],
    )
    def test_count_against_the_inventory(
        self,
        tmp_path: Path,
        held: dict[str, Any],
        n_pw: int,
        n_worlds: int,
        result: int | str,
    ) -> None:
        """An inventory with assets takes a file's count only where they agree."""
        features = {
            'a': {'StructureType': ['W1'] * n_worlds} if n_worlds > 1 else {}
        }
        path = tmp_path / 'file.geojson'
        assert written(inventory(features, n_worlds), path)['n_worlds'] == n_worlds
        target = inventory({'z': held}, n_pw)
        before = state(target)
        if isinstance(result, int):
            target.read_from_geojson(str(path))
            assert target.n_pw == result
            assert sorted(target.inventory) == ['a', 'z']
        else:
            with pytest.raises(ValueError, match=result):
                target.read_from_geojson(str(path))
            assert state(target) == before

    def test_file_of_one_world_taken_for_every_world(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A file of 1 world without lists joins an inventory of 3, with a line."""
        path = tmp_path / 'one.geojson'
        written(inventory({'a': {'PlanArea': 10.0}}), path)
        capsys.readouterr()
        target = inventory({'z': {'X': [1, 2, 3]}}, n_pw=3)
        target.read_from_geojson(str(path))
        assert target.n_pw == 3
        target.validate_possible_worlds()
        assert capsys.readouterr().out == (
            f'{path} holds 1 possible world, so its values are taken for every '
            "one of the inventory's 3 worlds.\n"
        )

    def test_file_of_one_world_with_lists_refused(self, tmp_path: Path) -> None:
        """A file of 1 world holding one-value lists is refused at 3 worlds."""
        path = tmp_path / 'one.geojson'
        written(inventory({'a': {'X': [7]}}), path)
        target = inventory({'z': {'X': [1, 2, 3]}}, n_pw=3)
        before = state(target)
        with pytest.raises(ValueError, match="list of one value under 'X'"):
            target.read_from_geojson(str(path))
        assert state(target) == before

    def test_inventory_of_one_world_takes_more_with_a_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An inventory of 1 world without lists takes a file of 3, with a line."""
        path = tmp_path / 'three.geojson'
        written(inventory(THREE_WORLDS, n_pw=3), path)
        capsys.readouterr()
        target = inventory({'z': {'PlanArea': 10.0}})
        target.read_from_geojson(str(path))
        assert target.n_pw == 3
        assert capsys.readouterr().out == (
            'The inventory holds 1 possible world and no lists, so the values of '
            f'its assets are taken for every one of the 3 worlds of {path}.\n'
        )


@pytest.mark.parametrize('value', [True, 0, -1, '3', 3.0, None])
def test_check_n_worlds_refuses_what_is_no_count(value: object) -> None:
    """A bool, an integer below 1 or a value that is no integer is refused."""
    with pytest.raises(ValueError, match='which is not an integer of at least 1'):
        check_n_worlds(value, 'file.geojson')


def test_check_n_worlds_returns_the_count() -> None:
    """An integer of at least 1 is the count."""
    assert check_n_worlds(10, 'file.geojson') == 10
