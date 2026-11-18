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

"""Tests for the composition of one world's sources and the assembly of worlds."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import brails.compilers
from brails.compilers.world_assembly import WorldAssembler, compose_source
from brails.constants import (
    DROPPED_KEY,
    OCCUPANCY_KEY,
    STRUCTURE_TYPE_KEY,
    YEAR_BUILT_KEY,
)
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import (
    SOURCES_KEY,
)
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils import Importer
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


class TestWorldAssembler:
    """Realizations assemble into one inventory over the worlds."""

    def test_collapse_and_expand(self) -> None:
        """Equal values stay one value, and the first difference makes a list."""
        assembler = WorldAssembler(np.int64(3))
        realizations = [
            inventory(
                {
                    'b': {OCCUPANCY_KEY: 'RES1', YEAR_BUILT_KEY: year},
                    'a': {OCCUPANCY_KEY: 'COM1', STRUCTURE_TYPE_KEY: kind},
                }
            )
            for year, kind in [(1950, 'W1'), (1950, 'C1'), (1960, 'W1')]
        ]
        for world, realization in enumerate(realizations):
            assembler.add(np.int64(world), realization)
        result = assembler.result()
        assert features(realizations[0]) == {
            'b': {OCCUPANCY_KEY: 'RES1', YEAR_BUILT_KEY: 1950},
            'a': {OCCUPANCY_KEY: 'COM1', STRUCTURE_TYPE_KEY: 'W1'},
        }
        assert features(result) == {
            'a': {OCCUPANCY_KEY: 'COM1', STRUCTURE_TYPE_KEY: ['W1', 'C1', 'W1']},
            'b': {OCCUPANCY_KEY: 'RES1', YEAR_BUILT_KEY: [1950, 1950, 1960]},
        }
        assert list(result.inventory) == ['a', 'b']
        assert result.n_pw == 3
        result.validate_possible_worlds()
        assert assembler.result() is result

    def test_absent_key_and_markers(self) -> None:
        """An absent key is None there, and a marker is a value like any other."""
        assembler = WorldAssembler(2)
        assembler.add(
            0, inventory({'a': {YEAR_BUILT_KEY: 1950, 'Height': None, 'Area': 'NA'}})
        )
        assembler.add(
            1,
            inventory(
                {'a': {STRUCTURE_TYPE_KEY: 'W1', 'Height': None, 'Roof': None}}
            ),
        )
        assert features(assembler.result())['a'] == {
            YEAR_BUILT_KEY: [1950, None],
            'Height': None,
            'Area': ['NA', None],
            STRUCTURE_TYPE_KEY: [None, 'W1'],
        }

    def test_nan_arrays_and_vector_features(self) -> None:
        """NaN equals NaN, arrays compare by item, a vector value compares whole."""
        assembler = WorldAssembler(2)
        for world, units in enumerate([[1, 2], [3]]):
            assembler.add(
                world,
                inventory(
                    {
                        'a': {
                            'Area': float('nan'),
                            SOURCES_KEY: np.array(
                                [{'dataset': 'OSM'}], dtype=object
                            ),
                            'HousingUnits': [1, 2],
                        },
                        'b': {'HousingUnits': units},
                    },
                    vector_features=['HousingUnits'],
                ),
            )
        result = assembler.result()
        held = features(result)
        assert np.isnan(held['a']['Area'])
        assert isinstance(held['a'][SOURCES_KEY], np.ndarray)
        assert held['a']['HousingUnits'] == [1, 2]
        assert held['b']['HousingUnits'] == [[1, 2], [3]]
        result.validate_possible_worlds()

    def test_lists_and_vector_features(self) -> None:
        """A list is one value, and a vector value that changes must be a list at position 0."""
        assembler = WorldAssembler(2)
        assembler.add(0, inventory({'a': {STRUCTURE_TYPE_KEY: ['C1']}}))
        assembler.add(1, inventory({'a': {STRUCTURE_TYPE_KEY: 'C1'}}))
        assert features(assembler.result())['a'] == {
            STRUCTURE_TYPE_KEY: [['C1'], 'C1']
        }
        assembler = WorldAssembler(2)
        assembler.add(0, inventory({'a': {'HousingUnits': [1, 2]}, 'b': {}}))
        with pytest.raises(ValueError, match=r"asset 'b', feature 'HousingUnits'"):
            assembler.add(1, inventory({'a': {}, 'b': {'HousingUnits': [1, 2]}}))
        assembler.add(1, inventory({'a': {}, 'b': {}}))
        assert features(assembler.result()) == {
            'a': {'HousingUnits': [[1, 2], None]},
            'b': {},
        }
        # A feature listed before a later realization declares it a vector.
        assembler = WorldAssembler(3)
        assembler.add(0, inventory({'a': {'Tags': 'x'}}, vector_features=[]))
        assembler.add(1, inventory({'a': {'Tags': 'y'}}, vector_features=[]))
        later = inventory({'a': {'Tags': ['z1', 'z2']}}, vector_features=['Tags'])
        with pytest.raises(ValueError, match=r"asset 'a', feature 'Tags'"):
            assembler.add(2, later)

    def test_other_assets_or_coordinates_raise(self) -> None:
        """A later position with other assets or coordinates raises, unapplied."""
        assembler = WorldAssembler(2)
        assembler.add(0, inventory({'a': {YEAR_BUILT_KEY: 1950}, 'b': {}}))
        with pytest.raises(ValueError, match=r"'b' missing, 'c' not at position 0"):
            assembler.add(1, inventory({'a': {YEAR_BUILT_KEY: 1960}, 'c': {}}))
        moved = inventory({'a': {YEAR_BUILT_KEY: 1960}, 'b': {}})
        moved.inventory['b'].coordinates = POINT['c']
        with pytest.raises(
            ValueError, match=r"coordinates other than at position 0: 'b'"
        ):
            assembler.add(1, moved)
        assembler.add(1, inventory({'a': {YEAR_BUILT_KEY: 1950}, 'b': {}}))
        assert features(assembler.result()) == {'a': {YEAR_BUILT_KEY: 1950}, 'b': {}}

    def test_positions_in_order(self) -> None:
        """Positions go in order from 0, each one world, and all before result."""
        one = inventory({'a': {}})
        assembler = WorldAssembler(2)
        with pytest.raises(ValueError, match='position 0 is next'):
            assembler.add(1, one)
        with pytest.raises(ValueError, match='one world, not 2'):
            assembler.add(0, inventory({'a': {}}, n_pw=2))
        with pytest.raises(TypeError, match='AssetInventory'):
            assembler.add(0, {'a': {}})
        assembler.add(0, one)
        with pytest.raises(ValueError, match='1 of 2 positions'):
            assembler.result()
        assembler.add(1, one)
        with pytest.raises(ValueError, match='All 2 positions'):
            assembler.add(2, one)
        with pytest.raises(ValueError, match='at least 1'):
            WorldAssembler(0)
        with pytest.raises(TypeError, match='integer'):
            WorldAssembler(2.0)


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
    """A source composes from frozen stages by union and three removals."""

    def test_union_across_worlds(self, freeze: StageFreeze) -> None:
        """Parts holding a feature in different worlds unite, markers kept."""
        world_0, world_1 = (
            features(
                compose_source(
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
            features(compose_source(freeze, ['units'], world=w, label='nsi'))
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
            compose_source(freeze, parts, world=1, label='nsi')
        world_0 = features(compose_source(freeze, parts, world=0, label='nsi'))
        assert world_0['a'] == {YEAR_BUILT_KEY: 1950, 'Area': 'NA'}

    def test_withheld_values_are_removed_by_label_and_world(
        self, freeze: StageFreeze
    ) -> None:
        """Only the values withheld from the label in that world are removed."""
        composed = {
            (label, world): features(
                compose_source(
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

    def test_dropped_and_excluded(self, freeze: StageFreeze) -> None:
        """Dropped assets leave, and excluded ones lose only the named part's values."""
        composed = compose_source(
            freeze,
            ['spine', 'drawn', 'later'],
            world=0,
            label='imputed',
            dropped='dropped',
            excluded={'c', 'b'},
            excluded_parts=['later'],
        )
        assert features(composed) == {
            'a': {OCCUPANCY_KEY: 'RES1', 'Area': 'NA', YEAR_BUILT_KEY: 1950},
            'c': {OCCUPANCY_KEY: 'RES3A', 'Area': None},
        }

    def test_shares_and_leaves_stages_unchanged(self, freeze: StageFreeze) -> None:
        """Unchanged assets are the frozen records, and no stage changes."""
        manifest = freeze.manifest()
        spine = freeze.read('spine')
        alone = compose_source(freeze, ['spine'], world=1, label='nsi')
        assert all(alone.inventory[i] is spine.inventory[i] for i in 'abc')
        unchanged = compose_source(freeze, ['spine', 'drawn'], world=0, label='nsi')
        assert unchanged.inventory['b'] is spine.inventory['b']
        assert unchanged.inventory['c'] is spine.inventory['c']
        composed = compose_source(
            freeze,
            ['spine', 'drawn', 'later'],
            world=0,
            withheld=['withheld'],
            label='nsi',
            dropped='dropped',
            excluded={'c'},
            excluded_parts=['later'],
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
            ({'excluded': ['c']}, TypeError, 'set of asset ids'),
            ({'excluded_parts': 'spine'}, TypeError, 'sequence of stage names'),
            ({'parts': []}, ValueError, 'at least one stage, each once'),
            ({'parts': ['spine', 'spine']}, ValueError, 'each once'),
            ({'excluded_parts': ['drawn']}, ValueError, 'must be among the parts'),
            ({'excluded': {'c'}}, ValueError, 'excluded_parts names no part'),
            (
                {'excluded': {'c', 'x', 7}, 'excluded_parts': ['spine']},
                ValueError,
                r"2 excluded .* no dropped record: 7, 'x'",
            ),
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
            compose_source(freeze, **given)


def test_discovered_and_exported() -> None:
    """The Importer finds the class, and the package exports it."""
    assert Importer().get_class('WorldAssembler') is WorldAssembler
    assert 'WorldAssembler' in brails.compilers.__all__
    assert brails.compilers.WorldAssembler is WorldAssembler
