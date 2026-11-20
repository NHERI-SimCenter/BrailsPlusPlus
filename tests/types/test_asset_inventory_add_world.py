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


"""Tests for adding worlds to an inventory with AssetInventory.add_world."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from brails.constants import OCCUPANCY_KEY, STRUCTURE_TYPE_KEY, YEAR_BUILT_KEY
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import (
    SOURCES_KEY,
)
from brails.types.asset_inventory import (
    DEFAULT_VECTOR_FEATURES,
    Asset,
    AssetInventory,
)

POINT = {'a': [[-122.40, 37.70]], 'b': [[-122.41, 37.70]], 'c': [[-122.42, 37.70]]}


def inventory(
    records: dict[str, dict[str, Any]],
    *,
    n_pw: int = 1,
    vector_features: list[str] | None = None,
) -> AssetInventory:
    """
    Build an inventory of point assets.

    Args:
        records (dict[str, dict[str, Any]]):
            Features per asset id, a key of ``POINT``.
        n_pw (int, optional):
            The number of possible worlds to declare. Defaults to 1.
        vector_features (list[str], optional):
            The features whose values are vectors. ``None`` takes the inventory's
            default.

    Returns:
        AssetInventory:
            The inventory, its assets inserted in the order given.
    """
    inv = AssetInventory(vector_features=vector_features)
    for asset_id, features in records.items():
        inv.add_asset(asset_id, Asset(asset_id, POINT[asset_id], dict(features)))
    inv.n_pw = n_pw
    return inv


def features(source: AssetInventory) -> dict[str, dict[str, Any]]:
    """
    Return the features of every asset of an inventory.

    Args:
        source (AssetInventory):
            The inventory whose features are returned, shared and not copied.

    Returns:
        dict[str, dict[str, Any]]:
            Features per asset id.
    """
    return {asset_id: asset.features for asset_id, asset in source.inventory.items()}


def test_collapse_and_expand() -> None:
    """Equal values stay one value, and the first difference makes a list."""
    assembled = AssetInventory()
    realizations = [
        inventory(
            {
                'b': {OCCUPANCY_KEY: 'RES1', YEAR_BUILT_KEY: year},
                'a': {OCCUPANCY_KEY: 'COM1', STRUCTURE_TYPE_KEY: kind},
            }
        )
        for year, kind in [(1950, 'W1'), (1950, 'C1'), (1960, 'W1')]
    ]
    for realization in realizations:
        assembled.add_world(realization)
    assert features(realizations[0]) == {
        'b': {OCCUPANCY_KEY: 'RES1', YEAR_BUILT_KEY: 1950},
        'a': {OCCUPANCY_KEY: 'COM1', STRUCTURE_TYPE_KEY: 'W1'},
    }
    assert features(assembled) == {
        'a': {OCCUPANCY_KEY: 'COM1', STRUCTURE_TYPE_KEY: ['W1', 'C1', 'W1']},
        'b': {OCCUPANCY_KEY: 'RES1', YEAR_BUILT_KEY: [1950, 1950, 1960]},
    }
    assert list(assembled.inventory) == ['a', 'b']
    assert assembled.n_pw == 3
    assembled.validate_possible_worlds()


def test_the_first_world_is_copied() -> None:
    """World 0 is held as copies of its assets and of their feature dicts."""
    first = inventory({'a': {YEAR_BUILT_KEY: 1950}}, vector_features=[])
    assembled = AssetInventory()
    assembled.add_world(first)
    assert assembled.n_pw == 1
    assert assembled.vector_features == []
    assert assembled.inventory['a'] is not first.inventory['a']
    assert assembled.inventory['a'].features is not first.inventory['a'].features
    assembled.add_world(inventory({'a': {YEAR_BUILT_KEY: 1960}}))
    assert first.inventory['a'].features == {YEAR_BUILT_KEY: 1950}
    assert assembled.n_pw == 2
    assert assembled.vector_features == list(DEFAULT_VECTOR_FEATURES)


def test_absent_key_and_markers() -> None:
    """An absent key is None there, and a marker is a value like any other."""
    assembled = AssetInventory()
    assembled.add_world(
        inventory({'a': {YEAR_BUILT_KEY: 1950, 'Height': None, 'Area': 'NA'}})
    )
    assembled.add_world(
        inventory({'a': {STRUCTURE_TYPE_KEY: 'W1', 'Height': None, 'Roof': None}})
    )
    assert features(assembled)['a'] == {
        YEAR_BUILT_KEY: [1950, None],
        'Height': None,
        'Area': ['NA', None],
        STRUCTURE_TYPE_KEY: [None, 'W1'],
    }


def test_nan_arrays_and_vector_features() -> None:
    """NaN equals NaN, arrays compare by item, a vector value compares whole."""
    assembled = AssetInventory()
    for units in ([1, 2], [3]):
        assembled.add_world(
            inventory(
                {
                    'a': {
                        'Area': float('nan'),
                        SOURCES_KEY: np.array([{'dataset': 'OSM'}], dtype=object),
                        'HousingUnits': [1, 2],
                    },
                    'b': {'HousingUnits': units},
                },
                vector_features=['HousingUnits'],
            )
        )
    held = features(assembled)
    assert np.isnan(held['a']['Area'])
    assert isinstance(held['a'][SOURCES_KEY], np.ndarray)
    assert held['a']['HousingUnits'] == [1, 2]
    assert held['b']['HousingUnits'] == [[1, 2], [3]]
    assembled.validate_possible_worlds()


def test_a_list_outside_a_vector_feature_is_refused() -> None:
    """A realization holding a list for a feature that is not a vector is refused."""
    assembled = AssetInventory()
    with pytest.raises(ValueError, match=r"asset 'a', feature 'StructureType'"):
        assembled.add_world(inventory({'a': {STRUCTURE_TYPE_KEY: ['C1']}}))
    assert not assembled.inventory
    assembled.add_world(inventory({'a': {STRUCTURE_TYPE_KEY: 'C1'}}))
    with pytest.raises(ValueError, match='not vector features'):
        assembled.add_world(inventory({'a': {STRUCTURE_TYPE_KEY: ['C1', 'C2']}}))
    assert assembled.n_pw == 1
    assert features(assembled)['a'] == {STRUCTURE_TYPE_KEY: 'C1'}


def test_lists_and_vector_features() -> None:
    """A vector value that changes must be a list in world 0."""
    assembled = AssetInventory()
    assembled.add_world(inventory({'a': {'HousingUnits': [1, 2]}, 'b': {}}))
    with pytest.raises(ValueError, match=r"asset 'b', feature 'HousingUnits'"):
        assembled.add_world(inventory({'a': {}, 'b': {'HousingUnits': [1, 2]}}))
    assembled.add_world(inventory({'a': {}, 'b': {}}))
    assert features(assembled) == {
        'a': {'HousingUnits': [[1, 2], None]},
        'b': {},
    }
    # A feature held per world before a later realization declares it a vector.
    assembled = AssetInventory()
    assembled.add_world(inventory({'a': {'Tags': 'x'}}, vector_features=[]))
    assembled.add_world(inventory({'a': {'Tags': 'y'}}, vector_features=[]))
    later = inventory({'a': {'Tags': ['z1', 'z2']}}, vector_features=['Tags'])
    with pytest.raises(ValueError, match=r"asset 'a', feature 'Tags'"):
        assembled.add_world(later)
    assert assembled.n_pw == 2


def test_other_assets_or_coordinates_raise() -> None:
    """A later world with other assets or coordinates raises and leaves the inventory unchanged."""
    assembled = AssetInventory()
    assembled.add_world(inventory({'a': {YEAR_BUILT_KEY: 1950}, 'b': {}}))
    with pytest.raises(ValueError, match=r"'b' missing, 'c' not in world 0"):
        assembled.add_world(inventory({'a': {YEAR_BUILT_KEY: 1960}, 'c': {}}))
    moved = inventory({'a': {YEAR_BUILT_KEY: 1960}, 'b': {}})
    moved.inventory['b'].coordinates = POINT['c']
    with pytest.raises(ValueError, match=r"coordinates other than in world 0: 'b'"):
        assembled.add_world(moved)
    assembled.add_world(inventory({'a': {YEAR_BUILT_KEY: 1950}, 'b': {}}))
    assert features(assembled) == {'a': {YEAR_BUILT_KEY: 1950}, 'b': {}}
    assert assembled.n_pw == 2


def test_a_realization_holds_one_world() -> None:
    """A realization of several worlds, or one that is not an inventory, is refused."""
    assembled = AssetInventory()
    with pytest.raises(ValueError, match='one world, not 2'):
        assembled.add_world(inventory({'a': {}}, n_pw=2))
    with pytest.raises(TypeError, match='AssetInventory'):
        assembled.add_world({'a': {}})
    assert not assembled.inventory


def test_realizations_without_assets_stay_one_world() -> None:
    """An inventory without assets takes each realization as its first world."""
    assembled = AssetInventory()
    assembled.add_world(AssetInventory())
    assembled.add_world(AssetInventory())
    assert not assembled.inventory
    assert assembled.n_pw == 1


def test_the_realizations_are_left_unchanged() -> None:
    """Adding three worlds changes neither the first realization nor its lists."""
    first = inventory({'a': {'HousingUnits': [1, 2], YEAR_BUILT_KEY: 1950}})
    assembled = AssetInventory()
    assembled.add_world(first)
    for units, year in (([3], 1960), ([4], 1970)):
        assembled.add_world(
            inventory({'a': {'HousingUnits': units, YEAR_BUILT_KEY: year}})
        )
    assert features(first)['a'] == {'HousingUnits': [1, 2], YEAR_BUILT_KEY: 1950}
    assert features(assembled)['a'] == {
        'HousingUnits': [[1, 2], [3], [4]],
        YEAR_BUILT_KEY: [1950, 1960, 1970],
    }


def test_a_key_missing_from_a_later_world_is_none_there() -> None:
    """A key held so far and missing from world 2 is None in world 2."""
    assembled = AssetInventory()
    for year in (1950, 1960):
        assembled.add_world(inventory({'a': {YEAR_BUILT_KEY: year}}))
    assembled.add_world(inventory({'a': {}}))
    assert features(assembled)['a'] == {YEAR_BUILT_KEY: [1950, 1960, None]}


def test_ids_of_mixed_types_are_ordered_as_strings() -> None:
    """World 0's assets are ordered by the asset id as a string, then its type."""
    realization = AssetInventory()
    for asset_id in ('b', 10, 2, '2'):
        realization.add_asset(asset_id, Asset(asset_id, POINT['a'], {}))
    assembled = AssetInventory()
    assembled.add_world(realization)
    assert list(assembled.inventory) == [10, 2, '2', 'b']


def test_world_0_refuses_a_list_its_realization_does_not_declare() -> None:
    """World 0 refuses a list in a feature its realization does not declare."""
    assembled = AssetInventory()
    realization = inventory({'a': {'HousingUnits': [1, 2]}}, vector_features=[])
    with pytest.raises(ValueError, match="asset 'a', feature 'HousingUnits'"):
        assembled.add_world(realization)
    assert not assembled.inventory
    assert assembled.vector_features == list(DEFAULT_VECTOR_FEATURES)


def test_a_list_of_lists_in_a_vector_feature_is_refused() -> None:
    """A list of lists under a vector feature is refused with a message naming it."""
    with pytest.raises(
        ValueError, match="or a list of lists.*asset 'a', feature 'Pairs'"
    ):
        AssetInventory().add_world(
            inventory({'a': {'Pairs': [[1, 2], [3, 4]]}}, vector_features=['Pairs'])
        )
