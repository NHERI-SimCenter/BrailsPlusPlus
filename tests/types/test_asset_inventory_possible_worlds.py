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

from typing import Any

import pytest

from brails.exceptions import BrailsError
from brails.types.asset_inventory import (
    DEFAULT_VECTOR_FEATURES,
    MAX_REPORTED_WORLD_PROBLEMS,
    Asset,
    AssetInventory,
    is_probabilistic_feature,
)

VECTOR = 'HousingUnits'


def build_inventory(
    features_by_asset: dict[str, dict[str, Any]],
    n_pw: int = 1,
    vector_features: list[str] | None = None,
) -> AssetInventory:
    """
    Build an inventory with the given features and number of worlds.

    Assets are added directly so that ``n_pw`` can be set explicitly,
    rather than being inferred from whichever feature was added last.

    Args:
        features_by_asset (dict[str, dict[str, Any]]):
            Features keyed by asset id.
        n_pw (int):
            The number of possible worlds to declare.
        vector_features (list[str], optional):
            Vector feature names to pass to the inventory.

    Returns:
        AssetInventory:
            The populated inventory.
    """
    inventory = AssetInventory(vector_features=vector_features)
    for asset_id, features in features_by_asset.items():
        inventory.add_asset(
            asset_id, Asset(asset_id, [[-122.4, 37.8]], dict(features))
        )
    inventory.n_pw = n_pw
    return inventory


# ---------------------------------------------------------------------
# is_probabilistic_feature
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ('key', 'value', 'expected'),
    [
        ('StructureType', 'C1', False),
        ('StructureType', ['C1', 'C2'], True),
        ('NumberOfStories', 3, False),
        # A plain list is one value of a vector feature, not one entry
        # per world.
        (VECTOR, [1, 2, 3], False),
        # A list of lists is one vector per world.
        (VECTOR, [[1, 2], [3, 4]], True),
        (VECTOR, [], False),
        ('StructureType', [], True),
    ],
)
def test_is_probabilistic_feature(
    key: str, value: Any, expected: bool  # noqa: FBT001
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


def test_validate_accepts_scalars() -> None:
    """Scalar features are shared by every world."""
    inventory = build_inventory(
        {'A1': {'StructureType': 'C1'}, 'A2': {'StructureType': 'W1'}},
        n_pw=3,
    )
    inventory.validate_possible_worlds()


def test_validate_accepts_matching_list_lengths() -> None:
    """Lists of exactly n_pw realizations are accepted."""
    inventory = build_inventory(
        {
            'A1': {'StructureType': ['C1', 'C2'], 'DesignLevel': 'Low-Code'},
            'A2': {'StructureType': ['W1', 'W2']},
        },
        n_pw=2,
    )
    inventory.validate_possible_worlds()


def test_validate_rejects_wrong_length() -> None:
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


def test_validate_rejects_single_element_list() -> None:
    """A length-one list is an error when the inventory has many worlds."""
    inventory = build_inventory({'A1': {'StructureType': ['C1']}}, n_pw=3)
    with pytest.raises(BrailsError, match='StructureType'):
        inventory.validate_possible_worlds()


def test_validate_accepts_single_element_list_when_one_world() -> None:
    """A length-one list is fine when there is only one world."""
    inventory = build_inventory({'A1': {'StructureType': ['C1']}}, n_pw=1)
    inventory.validate_possible_worlds()


def test_validate_exempts_vector_features() -> None:
    """A plain list on a vector feature is one value, not many worlds."""
    inventory = build_inventory({'A1': {VECTOR: [1, 2, 3, 4, 5]}}, n_pw=2)
    inventory.validate_possible_worlds()


def test_validate_checks_vector_features_of_lists() -> None:
    """A vector feature given per world must still supply n_pw entries."""
    inventory = build_inventory(
        {'A1': {VECTOR: [[1, 2], [3, 4], [5, 6]]}}, n_pw=2
    )
    with pytest.raises(BrailsError, match=VECTOR):
        inventory.validate_possible_worlds()


def test_validate_reports_every_offender_and_truncates() -> None:
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


def test_get_multi_keys_excludes_vector_features() -> None:
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


def test_get_multi_keys_includes_vector_features_of_lists() -> None:
    """A vector feature supplied per world is a multi-world feature."""
    inventory = build_inventory(
        {'A1': {VECTOR: [[1, 2], [3, 4]]}}, n_pw=2
    )
    multi_keys, _ = inventory.get_multi_keys()

    assert multi_keys == [VECTOR]


# ---------------------------------------------------------------------
# get_world_realization
# ---------------------------------------------------------------------


def test_get_world_realization_preserves_vector_features() -> None:
    """A vector feature is carried across whole, not indexed into."""
    inventory = build_inventory(
        {'A1': {'StructureType': ['C1', 'C2'], VECTOR: [1, 2]}}, n_pw=2
    )
    realization = inventory.get_world_realization(1)
    _, features = realization.get_asset_features('A1')

    assert features['StructureType'] == 'C2'
    assert features[VECTOR] == [1, 2]


def test_get_world_realization_indexes_vector_features_of_lists() -> None:
    """A vector feature given per world is indexed like any other."""
    inventory = build_inventory(
        {'A1': {VECTOR: [[1, 2], [3, 4]]}}, n_pw=2
    )
    realization = inventory.get_world_realization(1)
    _, features = realization.get_asset_features('A1')

    assert features[VECTOR] == [3, 4]
