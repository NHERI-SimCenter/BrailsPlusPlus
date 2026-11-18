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

"""Tests for the composition of one source of one world from frozen stages."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from brails.compilers.world_assembly import compose_source_parts
from brails.constants import (
    DROPPED_KEY,
    OCCUPANCY_KEY,
    YEAR_BUILT_KEY,
)
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils.stage_freeze import StageFreeze

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
            Passed to the inventory. Defaults to the inventory's default.

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
            The inventory whose features are returned, shared and not
            copied.

    Returns:
        dict[str, dict[str, Any]]:
            Features per asset id.
    """
    return {asset_id: asset.features for asset_id, asset in source.inventory.items()}


@pytest.fixture
def freeze() -> StageFreeze:
    """
    Return a freeze of two one-world stages and five per-world ones.

    Returns:
        StageFreeze:
            ``spine`` and ``dropped`` with one world; ``drawn``, ``later``,
            ``clashing``, ``units`` and ``withheld`` with two.
    """
    stages = StageFreeze()
    stages.freeze(
        'spine',
        inventory(
            {
                'a': {OCCUPANCY_KEY: 'RES1', 'Area': 'NA'},
                'b': {OCCUPANCY_KEY: 'COM1'},
                'c': {OCCUPANCY_KEY: 'RES3A', 'Area': None},
            }
        ),
    )
    stages.freeze('dropped', inventory({'b': {DROPPED_KEY: True}}))
    stages.freeze(
        'drawn',
        inventory(
            {'a': {YEAR_BUILT_KEY: [1950, None], 'Area': 'NA'}, 'b': {}, 'c': {}},
            n_pw=2,
        ),
    )
    stages.freeze(
        'later',
        inventory(
            {
                'a': {YEAR_BUILT_KEY: [None, 1960]},
                'b': {},
                'c': {'Area': [120.0, 90.0]},
            },
            n_pw=2,
        ),
    )
    stages.freeze(
        'clashing',
        inventory({'a': {YEAR_BUILT_KEY: [None, 1965]}, 'b': {}, 'c': {}}, n_pw=2),
    )
    stages.freeze(
        'units',
        inventory(
            {'a': {'HousingUnits': [[1, 2], [3]]}, 'b': {'HousingUnits': [1, 2]}},
            n_pw=2,
            vector_features=['HousingUnits'],
        ),
    )
    stages.freeze(
        'withheld',
        inventory(
            {
                'a': {OCCUPANCY_KEY: ['nsi', None], YEAR_BUILT_KEY: 'derived'},
                'b': {},
                'c': {OCCUPANCY_KEY: [None, 'nsi']},
            },
            n_pw=2,
        ),
    )
    return stages


class TestComposeSource:
    """A source is the union of frozen parts, less the dropped assets and the withheld values."""

    def test_union_across_worlds(self, freeze: StageFreeze) -> None:
        """Parts holding a feature in different worlds unite, markers kept."""
        world_0, world_1 = (
            features(
                compose_source_parts(
                    freeze, ['spine', 'drawn', 'later'], world=w, label='nsi'
                )
            )
            for w in (0, 1)
        )
        assert world_0 == {
            'a': {OCCUPANCY_KEY: 'RES1', 'Area': 'NA', YEAR_BUILT_KEY: 1950},
            'b': {OCCUPANCY_KEY: 'COM1'},
            'c': {OCCUPANCY_KEY: 'RES3A', 'Area': 120.0},
        }
        assert world_1['a'][YEAR_BUILT_KEY] == 1960
        assert world_1['c']['Area'] == 90.0
        units = [
            features(compose_source_parts(freeze, ['units'], world=w, label='nsi'))
            for w in (np.int64(0), 1)
        ]
        assert units == [
            {'a': {'HousingUnits': [1, 2]}, 'b': {'HousingUnits': [1, 2]}},
            {'a': {'HousingUnits': [3]}, 'b': {'HousingUnits': [1, 2]}},
        ]

    def test_two_values_in_one_world_raise(self, freeze: StageFreeze) -> None:
        """Two parts holding a value for one feature in one world raise."""
        parts = ['drawn', 'later', 'clashing']
        with pytest.raises(
            ValueError,
            match=r"1 feature.* world 1: asset 'a', feature 'YearBuilt' \('clashing'",
        ):
            compose_source_parts(freeze, parts, world=1, label='nsi')
        world_0 = features(compose_source_parts(freeze, parts, world=0, label='nsi'))
        assert world_0['a'] == {YEAR_BUILT_KEY: 1950, 'Area': 'NA'}

    def test_withheld_values_are_removed_by_label_and_world(
        self, freeze: StageFreeze
    ) -> None:
        """Only the values withheld from the label in that world are removed."""
        composed = {
            (label, world): features(
                compose_source_parts(
                    freeze,
                    ['spine', 'drawn', 'later'],
                    world=world,
                    withheld=['withheld'],
                    label=label,
                )
            )
            for label in ('nsi', 'derived')
            for world in (0, 1)
        }
        assert composed['nsi', 0]['a'] == {'Area': 'NA', YEAR_BUILT_KEY: 1950}
        assert composed['nsi', 0]['c'][OCCUPANCY_KEY] == 'RES3A'
        assert composed['nsi', 1]['a'][OCCUPANCY_KEY] == 'RES1'
        assert OCCUPANCY_KEY not in composed['nsi', 1]['c']
        assert composed['derived', 0]['a'] == {
            OCCUPANCY_KEY: 'RES1',
            'Area': 'NA',
        }
        assert composed['derived', 1]['a'] == {
            OCCUPANCY_KEY: 'RES1',
            'Area': 'NA',
        }
        assert composed['derived', 0]['c'] == {
            OCCUPANCY_KEY: 'RES3A',
            'Area': 120.0,
        }

    def test_dropped(self, freeze: StageFreeze) -> None:
        """An asset of the dropped record leaves the source."""
        composed = compose_source_parts(
            freeze,
            ['spine', 'drawn', 'later'],
            world=0,
            label='imputed',
            dropped='dropped',
        )
        assert features(composed) == {
            'a': {OCCUPANCY_KEY: 'RES1', 'Area': 'NA', YEAR_BUILT_KEY: 1950},
            'c': {OCCUPANCY_KEY: 'RES3A', 'Area': 120.0},
        }

    def test_shares_and_leaves_stages_unchanged(self, freeze: StageFreeze) -> None:
        """Unchanged assets are the frozen records, and no stage changes."""
        manifest = freeze.manifest()
        spine = freeze.read('spine')
        alone = compose_source_parts(freeze, ['spine'], world=1, label='nsi')
        assert all(alone.inventory[i] is spine.inventory[i] for i in 'abc')
        unchanged = compose_source_parts(
            freeze, ['spine', 'drawn'], world=0, label='nsi'
        )
        assert unchanged.inventory['b'] is spine.inventory['b']
        assert unchanged.inventory['c'] is spine.inventory['c']
        composed = compose_source_parts(
            freeze,
            ['spine', 'drawn', 'later'],
            world=0,
            withheld=['withheld'],
            label='nsi',
            dropped='dropped',
        )
        assert composed.inventory['a'] is not spine.inventory['a']
        merged = spine.merge(
            {'nsi': composed}, ['footprint', 'nsi'], name='footprint'
        )
        assert merged.get_asset_features('a')[1][YEAR_BUILT_KEY] == 1950
        assert freeze.manifest() == manifest

    @pytest.mark.parametrize(
        ('arguments', 'error', 'message'),
        [
            ({'parts': [AssetInventory()]}, TypeError, 'by name only'),
            ({'parts': 'spine'}, TypeError, 'sequence of stage names'),
            ({'withheld': [None]}, TypeError, 'by name only'),
            ({'dropped': AssetInventory()}, TypeError, 'by name only'),
            (
                {'parts': ['spine', 'unfrozen']},
                LookupError,
                "'unfrozen' is not frozen",
            ),
            ({'label': None}, TypeError, 'label must be a string'),
            ({'world': 1.0}, TypeError, 'world must be an integer'),
            ({'world': -1}, ValueError, 'not be negative'),
            ({'parts': ['drawn'], 'world': 2}, ValueError, "'drawn' holds 2 worlds"),
            ({'parts': []}, ValueError, 'at least one stage, each once'),
            ({'parts': ['spine', 'spine']}, ValueError, 'each once'),
        ],
    )
    def test_refusals(
        self,
        freeze: StageFreeze,
        arguments: dict[str, Any],
        error: type[Exception],
        message: str,
    ) -> None:
        """Only names of frozen stages enter, and the arguments are checked."""
        given = {'parts': ['spine'], 'world': 0, 'label': 'nsi', **arguments}
        with pytest.raises(error, match=message):
            compose_source_parts(freeze, **given)
