# Copyright (c) 2025 The Regents of the University of California
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

"""Tests for possible-worlds handling in brails.types.asset_inventory."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from brails.exceptions import BrailsError
from brails.types.asset_inventory import (
    DEFAULT_VECTOR_FEATURES,
    MAX_REPORTED_WORLD_PROBLEMS,
    Asset,
    AssetInventory,
    is_probabilistic_feature,
)
from brails.types.housing_unit_inventory import HousingUnit, HousingUnitInventory

VECTOR = 'HousingUnits'

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------
# is_probabilistic_feature
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ('key', 'value', 'expected'),
    [
        ('StructureType', 'C1', False),
        ('StructureType', ['C1', 'C2'], True),
        ('NumberOfStories', 3, False),
        # A plain list is one value of a vector feature, not one entry per world.
        (VECTOR, [1, 2, 3], False),
        # A list of lists is one vector per world.
        (VECTOR, [[1, 2], [3, 4]], True),
        (VECTOR, [], False),
        ('StructureType', [], True),
    ],
)
def test_is_probabilistic_feature(
    key: str,
    value: Any,
    expected: bool,  # noqa: FBT001
) -> None:
    """Vector features follow a different rule from ordinary ones."""
    assert is_probabilistic_feature(key, value, [VECTOR]) is expected


def test_is_probabilistic_feature_without_vector_features() -> None:
    """With no vector features declared, any list is probabilistic."""
    assert is_probabilistic_feature(VECTOR, [1, 2, 3], []) is True


# ---------------------------------------------------------------------
# vector_features configuration
# ---------------------------------------------------------------------


def test_default_vector_features_applied() -> None:
    """An inventory defaults to the package-level vector features."""
    assert AssetInventory().vector_features == list(DEFAULT_VECTOR_FEATURES)


def test_vector_features_are_configurable() -> None:
    """A caller can declare its own vector features."""
    inventory = AssetInventory(vector_features=['Footprint'])
    assert inventory.vector_features == ['Footprint']


def test_vector_features_are_copied_not_aliased() -> None:
    """Mutating an inventory's list does not affect the caller's."""
    declared = ['Footprint']
    inventory = AssetInventory(vector_features=declared)
    inventory.vector_features.append('Other')
    assert declared == ['Footprint']


def test_asset_add_features_honors_default_vector_features() -> None:
    """A direct Asset.add_features call uses the package default."""
    asset = Asset('A1', [[-122.4, 37.8]])
    _, n_pw = asset.add_features({VECTOR: [1, 2, 3]})
    # Without the default this would be read as three possible worlds.
    assert n_pw == 1


# ---------------------------------------------------------------------
# validate_possible_worlds
# ---------------------------------------------------------------------


def test_validate_accepts_scalars(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Scalar features are shared by every world."""
    inventory = build_inventory(
        {'A1': {'StructureType': 'C1'}, 'A2': {'StructureType': 'W1'}},
        n_pw=3,
    )
    inventory.validate_possible_worlds()


def test_validate_accepts_matching_list_lengths(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Lists of exactly n_pw realizations are accepted."""
    inventory = build_inventory(
        {
            'A1': {'StructureType': ['C1', 'C2'], 'DesignLevel': 'Low-Code'},
            'A2': {'StructureType': ['W1', 'W2']},
        },
        n_pw=2,
    )
    inventory.validate_possible_worlds()


def test_validate_rejects_wrong_length(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A feature with the wrong number of realizations is rejected."""
    inventory = build_inventory(
        {'A1': {'StructureType': ['C1', 'C2', 'C3']}}, n_pw=2
    )
    with pytest.raises(BrailsError) as excinfo:
        inventory.validate_possible_worlds()

    message = str(excinfo.value)
    assert 'A1' in message
    assert 'StructureType' in message
    assert '3 realizations' in message


def test_validate_rejects_single_element_list(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A length-one list is an error when the inventory has many worlds."""
    inventory = build_inventory({'A1': {'StructureType': ['C1']}}, n_pw=3)
    with pytest.raises(BrailsError, match='StructureType'):
        inventory.validate_possible_worlds()


def test_validate_accepts_single_element_list_when_one_world(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A length-one list is fine when there is only one world."""
    inventory = build_inventory({'A1': {'StructureType': ['C1']}}, n_pw=1)
    inventory.validate_possible_worlds()


def test_validate_exempts_vector_features(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A plain list on a vector feature is one value, not many worlds."""
    inventory = build_inventory({'A1': {VECTOR: [1, 2, 3, 4, 5]}}, n_pw=2)
    inventory.validate_possible_worlds()


def test_validate_accepts_vertex_elevations_in_one_world(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Per-vertex all_elevations lists are one value in an inventory of 1 world."""
    inventory = build_inventory(
        {
            'A1': {'all_elevations': [1.0, 2.0, 3.0, 4.0]},
            'A2': {'all_elevations': [1.0, 2.0, 3.0, 4.0, 5.0]},
        },
        n_pw=1,
    )
    inventory.validate_possible_worlds()


def test_validate_checks_vector_features_of_lists(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A vector feature given per world must still supply n_pw entries."""
    inventory = build_inventory({'A1': {VECTOR: [[1, 2], [3, 4], [5, 6]]}}, n_pw=2)
    with pytest.raises(BrailsError, match=VECTOR):
        inventory.validate_possible_worlds()


def test_validate_reports_every_offender_and_truncates(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """All problems are counted; the listing itself is bounded."""
    offenders = MAX_REPORTED_WORLD_PROBLEMS + 5
    inventory = build_inventory(
        {
            f'A{index}': {'StructureType': ['C1', 'C2', 'C3']}
            for index in range(offenders)
        },
        n_pw=2,
    )
    with pytest.raises(BrailsError) as excinfo:
        inventory.validate_possible_worlds()

    message = str(excinfo.value)
    assert f'{offenders} feature value(s)' in message
    assert '... and 5 more.' in message


def test_validate_passes_on_empty_inventory() -> None:
    """An inventory with no assets has nothing to contradict n_pw."""
    AssetInventory().validate_possible_worlds()


# ---------------------------------------------------------------------
# get_multi_keys
# ---------------------------------------------------------------------


def test_get_multi_keys_excludes_vector_features(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A plain list on a vector feature is not a multi-world feature."""
    inventory = build_inventory(
        {
            'A1': {
                'StructureType': ['C1', 'C2'],
                VECTOR: [1, 2],
                'DesignLevel': 'Low-Code',
            }
        },
        n_pw=2,
    )
    multi_keys, all_keys = inventory.get_multi_keys()

    assert multi_keys == ['StructureType']
    assert set(all_keys) == {'StructureType', VECTOR, 'DesignLevel'}


def test_get_multi_keys_includes_vector_features_of_lists(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A vector feature supplied per world is a multi-world feature."""
    inventory = build_inventory({'A1': {VECTOR: [[1, 2], [3, 4]]}}, n_pw=2)
    multi_keys, _ = inventory.get_multi_keys()

    assert multi_keys == [VECTOR]


# ---------------------------------------------------------------------
# get_world_realization
# ---------------------------------------------------------------------


def test_get_world_realization_preserves_vector_features(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A vector feature is carried across whole, not indexed into."""
    inventory = build_inventory(
        {'A1': {'StructureType': ['C1', 'C2'], VECTOR: [1, 2]}}, n_pw=2
    )
    realization = inventory.get_world_realization(1)
    _, features = realization.get_asset_features('A1')

    assert features['StructureType'] == 'C2'
    assert features[VECTOR] == [1, 2]


def test_get_world_realization_indexes_vector_features_of_lists(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A vector feature given per world is indexed like any other."""
    inventory = build_inventory({'A1': {VECTOR: [[1, 2], [3, 4]]}}, n_pw=2)
    realization = inventory.get_world_realization(1)
    _, features = realization.get_asset_features('A1')

    assert features[VECTOR] == [3, 4]


@pytest.mark.parametrize('world', [0, 1, 5])
def test_get_world_realization_of_one_world_accepts_any_index(
    build_inventory: Callable[..., AssetInventory], world: int
) -> None:
    """A single-world inventory returns its one world for any index."""
    inventory = build_inventory(
        {'A1': {'StructureType': 'C1', 'YearBuilt': [1950], VECTOR: [1, 2]}}
    )
    realization = inventory.get_world_realization(world)
    _, features = realization.get_asset_features('A1')

    assert realization.n_pw == 1
    assert features == {'StructureType': 'C1', 'YearBuilt': 1950, VECTOR: [1, 2]}


@pytest.mark.parametrize(
    ('world', 'message'),
    [(2, 'smaller than the existing number of worlds 2'), (3, 'contains only 2')],
)
def test_get_world_realization_refuses_a_missing_world(
    build_inventory: Callable[..., AssetInventory], world: int, message: str
) -> None:
    """An index past the worlds of a multi-world inventory raises."""
    inventory = build_inventory({'A1': {'StructureType': ['C1', 'C2']}}, n_pw=2)

    with pytest.raises(Exception, match=message):
        inventory.get_world_realization(world)


def test_get_world_realization_has_one_world_and_a_dataframe(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A realization holding a plain vector feature converts to a dataframe."""
    inventory = build_inventory(
        {'A1': {'StructureType': ['C1', 'C2'], VECTOR: [1, 2]}}, n_pw=2
    )
    realization = inventory.get_world_realization(1)
    properties, _, count = realization.get_dataframe()

    assert realization.n_pw == 1
    assert count == 1
    assert properties.loc['A1', 'StructureType'] == 'C2'
    assert properties.loc['A1', VECTOR] == [1, 2]


def test_get_world_realization_reads_back_a_multi_world_file(tmp_path: Any) -> None:
    """Without n_worlds, a multi-world file takes its worlds from its lists."""
    inventory = AssetInventory()
    inventory.add_asset(
        'A1', Asset('A1', [[-122.4, 37.8]], {'YearBuilt': [1950, 1960, 1970]})
    )
    inventory.n_pw = 3
    path = tmp_path / 'worlds.geojson'
    collection = inventory.write_to_geojson()
    del collection['n_worlds']
    path.write_text(json.dumps(collection), encoding='utf-8')
    back = AssetInventory()
    back.read_from_geojson(str(path))

    assert back.n_pw == 3
    assert back.get_world_realization(2).get_asset_features('A1')[1] == {
        'YearBuilt': 1970,
        'type': 'building',
    }


def test_get_world_realization_keeps_the_class() -> None:
    """The realization of a subclass instance is an instance of that subclass."""

    class Subclass(AssetInventory):
        pass

    inventory = Subclass()
    inventory.add_asset('A1', Asset('A1', [[-122.4, 37.8]], {'YearBuilt': 1950}))

    assert type(inventory.get_world_realization(0)) is Subclass


def test_get_world_realization_leaves_the_source_unchanged() -> None:
    """Changing a realization's geometry or features leaves the source as it was."""
    inventory = AssetInventory(vector_features=['w'])
    inventory.add_asset(
        'A1',
        Asset(
            'A1',
            [[-122.4, 37.8]],
            {'StructureType': ['C1', 'C2'], 'w': [1], 'meta': {'a': 1}},
        ),
    )
    inventory.n_pw = 2
    units = HousingUnitInventory()
    units.add_housing_unit('1', HousingUnit({'income': 50000}))
    inventory.set_housing_unit_inventory(units, validate=False)

    realization = inventory.get_world_realization(0)
    realization.inventory['A1'].coordinates[0][0] = 0.0
    realization.add_asset_features('A1', {'StructureType': 'W1'}, overwrite=True)
    realization.inventory['A1'].features['w'].append(2)
    realization.inventory['A1'].features['meta']['a'] = 2
    realization.housing_unit_inventory.inventory['1'].features['income'] = 1

    assert realization.vector_features == ['w']
    assert inventory.inventory['A1'].coordinates == [[-122.4, 37.8]]
    assert inventory.get_asset_features('A1')[1] == {
        'StructureType': ['C1', 'C2'],
        'w': [1],
        'meta': {'a': 1},
    }
    assert units.inventory['1'].features == {'income': 50000}
