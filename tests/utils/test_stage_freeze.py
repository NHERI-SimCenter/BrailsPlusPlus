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


"""Tests for the freeze of stage outputs."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import numpy as np
import pytest

import brails.utils
from brails.types.asset_inventory import Asset, AssetInventory
from brails.types.housing_unit_inventory import HousingUnit, HousingUnitInventory
from brails.utils import Importer
from brails.utils.stage_freeze import DIGEST_ALGORITHM, StageFreeze
from tests.imputers.donor_fixtures import features_of

N_WORLDS = 3

# Digest of sample(), computed outside the module.
GOLDEN_DIGEST = '644e1a9b3ca5cd503daccb9b7592254e'

ALGORITHM = {'digest_algorithm': DIGEST_ALGORITHM}


def inventory(
    records: list[tuple[Any, list, dict[str, Any]]],
    *,
    n_pw: int = 1,
    vector_features: list[str] | None = None,
) -> AssetInventory:
    """
    Build an inventory from ``(asset_id, coordinates, features)`` tuples.

    Args:
        records (list[tuple[Any, list, dict[str, Any]]]):
            One tuple per asset, inserted in the order given.
        n_pw (int, optional):
            The number of possible worlds to declare. Defaults to 1.
        vector_features (list[str], optional):
            Passed to the inventory. Defaults to the inventory's default.

    Returns:
        AssetInventory:
            The inventory.
    """
    inv = AssetInventory(vector_features=vector_features)
    for asset_id, coordinates, features in records:
        inv.add_asset(asset_id, Asset(asset_id, coordinates, dict(features)))
    inv.n_pw = n_pw
    return inv


def sample() -> AssetInventory:
    """Return a small two-asset inventory with mixed feature types."""
    return inventory(
        [
            (
                '9q8yyk8yuv',
                [[-122.4194, 37.7749]],
                {
                    'OccupancyClass': 'RES1',
                    'NumberOfStories': 2,
                    'PlanArea': 151.25,
                },
            ),
            (
                '9q8yyk8yuw',
                [[-122.4190, 37.7750]],
                {
                    'OccupancyClass': 'COM1',
                    'NumberOfStories': 1,
                    'PlanArea': None,
                },
            ),
        ]
    )


def housing_units(income: int = 50000) -> HousingUnitInventory:
    """Return a one-unit housing-unit inventory."""
    units = HousingUnitInventory()
    units.add_housing_unit('1', HousingUnit({'income': income, 'size': 3}))
    return units


def digest_of(inv: AssetInventory) -> str:
    """Freeze ``inv`` alone and return its digest."""
    return StageFreeze().freeze('stage', inv)


class TestFrozenStages:
    """Only a frozen stage can be read, and each stage is frozen once."""

    def test_reading_an_unfrozen_stage_raises(self) -> None:
        """Reading an unfrozen stage raises."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        with pytest.raises(LookupError, match="'imputed' is not frozen"):
            freeze.read('imputed')
        with pytest.raises(LookupError, match='not frozen'):
            freeze.digest('imputed')

    def test_a_stage_name_is_frozen_once(self) -> None:
        """A stage name is frozen once."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        with pytest.raises(ValueError, match='already frozen'):
            freeze.freeze('observed', sample())
        assert freeze.order == ('observed',)


class TestHeldByReference:
    """Frozen outputs are held by reference, and the manifest catches a change."""

    def test_read_returns_the_frozen_object(self) -> None:
        """Freezing keeps the caller's object, and every read returns it."""
        freeze = StageFreeze()
        original = sample()
        freeze.freeze('observed', original)
        assert freeze.read('observed') is original
        assert freeze.read('observed') is freeze.read('observed')

    @pytest.mark.parametrize(
        'change', ['feature value', 'geometry in place', 'asset removed']
    )
    def test_a_change_through_read_fails_the_manifest(self, change: str) -> None:
        """Changing a frozen output through read() makes manifest() raise."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        digest = freeze.freeze('residential', sample())
        assert freeze.manifest()['stages'][1]['digest'] == digest

        held = freeze.read('residential')
        if change == 'feature value':
            held.add_asset_features(
                '9q8yyk8yuv', {'OccupancyClass': 'IND1'}, overwrite=True
            )
        elif change == 'geometry in place':
            held.inventory['9q8yyk8yuw'].coordinates[0][0] = 0.0
        else:
            held.remove_asset('9q8yyk8yuw')

        with pytest.raises(RuntimeError, match="'residential' changed after"):
            freeze.manifest()
        assert freeze.digest('residential') == digest == digest_of(sample())

    def test_a_change_to_the_original_fails_the_manifest(self) -> None:
        """Changing the frozen object through the caller's own name is caught too."""
        freeze = StageFreeze()
        original = sample()
        freeze.freeze('observed', original)
        original.add_asset('extra', Asset('extra', [[-122.0, 37.0]]))
        with pytest.raises(RuntimeError, match="'observed' changed after"):
            freeze.manifest()

    def test_a_value_that_cannot_be_digested_fails_the_manifest(self) -> None:
        """A change that leaves no canonical form also names the stage."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        freeze.read('observed').inventory['9q8yyk8yuv'].features['x'] = {1, 2}
        with pytest.raises(RuntimeError, match="'observed' changed after"):
            freeze.manifest()

    def test_save_manifest_writes_nothing_after_a_change(
        self, tmp_path: Any
    ) -> None:
        """save_manifest raises before it writes the file."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        freeze.read('observed').n_pw = 4
        path = tmp_path / 'manifest.json'
        with pytest.raises(RuntimeError, match="'observed' changed after"):
            freeze.save_manifest(path)
        assert not path.exists()


class TestRecordedOrder:
    """The order of freezing and the digests are kept and read back."""

    def build(self) -> StageFreeze:
        """
        Freeze three stages.

        Returns:
            StageFreeze:
                The freeze, holding ``observed``, ``residential`` and
                ``nonresidential`` in that order.
        """
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        freeze.freeze('residential', sample())
        freeze.freeze('nonresidential', sample())
        return freeze

    def test_order_is_the_order_of_freezing(self) -> None:
        """The order is the order of freezing."""
        assert self.build().order == ('observed', 'residential', 'nonresidential')

    def test_manifest_records_order_and_digests(self) -> None:
        """The manifest records the order and the digests."""
        freeze = self.build()
        manifest = freeze.manifest()
        assert manifest['digest_algorithm'] == DIGEST_ALGORITHM == 'blake2b-128'
        assert [entry['stage'] for entry in manifest['stages']] == list(freeze.order)
        for entry in manifest['stages']:
            assert set(entry) == {'stage', 'digest'}
            assert entry['digest'] == freeze.digest(entry['stage'])

    def test_manifest_round_trips_through_a_file(self, tmp_path: Any) -> None:
        """The manifest round-trips through a file."""
        freeze = self.build()
        path = tmp_path / 'manifest.json'
        freeze.save_manifest(path)
        assert StageFreeze.load_manifest(path) == freeze.manifest()
        assert StageFreeze.load_manifest(str(path)) == freeze.manifest()

    def test_manifest_is_a_copy(self) -> None:
        """Changing a returned manifest leaves the record unchanged."""
        freeze = self.build()
        manifest = freeze.manifest()
        manifest['stages'][1]['digest'] = 'changed'
        recorded = freeze.manifest()['stages'][1]['digest']
        assert recorded == freeze.digest('residential')

    @pytest.mark.parametrize(
        ('content', 'message'),
        [
            ([], 'does not hold a stage manifest'),
            ({'stages': {}}, 'does not hold a stage manifest'),
            ({'stages': []}, "algorithm None, not 'blake2b-128'"),
            (
                {'digest_algorithm': 'md5', 'stages': []},
                "algorithm 'md5', not 'blake2b-128'",
            ),
            (
                {**ALGORITHM, 'stages': [{'stage': 'a'}]},
                'malformed stage entry',
            ),
            (
                {**ALGORITHM, 'stages': [{'stage': 'a', 'digest': 1}]},
                'malformed stage entry',
            ),
            (
                {**ALGORITHM, 'stages': [{'stage': 1, 'digest': 'x'}]},
                'malformed stage entry',
            ),
            (
                {
                    **ALGORITHM,
                    'stages': [
                        {'stage': 'a', 'digest': 'x'},
                        {'stage': 'a', 'digest': 'y'},
                    ],
                },
                "'a' more than once",
            ),
        ],
    )
    def test_load_refuses_a_file_that_is_not_a_manifest(
        self, tmp_path: Any, content: Any, message: str
    ) -> None:
        """``load_manifest`` refuses a file that is not a manifest."""
        path = tmp_path / 'manifest.json'
        path.write_text(json.dumps(content), encoding='utf-8')
        with pytest.raises(ValueError, match=message):
            StageFreeze.load_manifest(path)


FOOTPRINT = [[-122.42, 37.77], [-122.41, 37.77], [-122.41, 37.78], [-122.42, 37.77]]
CENTROID = [[-122.415, 37.775]]

# One inventory of each shape a stage can take, the compiler's among them.
STAGE_SHAPES = {
    'spine part': inventory(
        [
            (
                '9q8yyk8yuv',
                FOOTPRINT,
                {
                    'height': 7.5,
                    'overture_id': 'a1',
                    'sources': [{'dataset': 'OpenStreetMap', 'confidence': None}],
                },
            ),
            (
                '9q8yyk8yuw',
                FOOTPRINT,
                {'height': None, 'overture_id': 'a2', 'sources': []},
            ),
        ]
    ),
    'per-world part, padded': inventory(
        [
            ('9q8yyk8yuv', CENTROID, {'YearBuilt': [1921, 1934, 1921]}),
            ('9q8yyk8yuw', CENTROID, {}),
            ('9q8yyk8yux', CENTROID, {'YearBuilt': 1950}),
        ],
        n_pw=3,
    ),
    'withheld record': inventory(
        [
            (
                '9q8yyk8yuv',
                CENTROID,
                {'GrossArea': 'nsi', 'ReplacementCost': 'derived'},
            ),
            ('9q8yyk8yuw', CENTROID, {'NumberOfStories': ['nsi', None, 'nsi']}),
        ],
        n_pw=3,
    ),
    'dropped record': inventory([('9q8yyk8yuw', CENTROID, {})]),
    'view': inventory(
        [
            (
                '9q8yyk8yuv',
                FOOTPRINT,
                {
                    'OccupancyClass': ['RES1', 'RES3A', 'RES1'],
                    'NumberOfStories': 2,
                    'PlanArea': float('nan'),
                    'GrossArea': 2.0,
                    'ReplacementCost': np.float32(1.5e5),
                    'StructureType': 'W1',
                    'Excluded': [False, True, False],
                    'Name': 'Caf\u00e9',
                    'Plan': {'Rings': [3, 4], 'Source': 'footprint'},
                    'Vector': [0.5, 0.25],
                    'Pairs': [1, 2],
                },
            ),
            (17, CENTROID, {'OccupancyClass': 'COM1', 'NumberOfStories': None}),
        ],
        n_pw=3,
        vector_features=['Vector', 'Pairs'],
    ),
    'empty stage': AssetInventory(),
}
STAGE_SHAPES['housing units'] = sample()
# Three units, inserted in an order other than the sorted one ('1', '10', '2').
UNITS = HousingUnitInventory()
for unit_id in ('2', '10', '1'):
    UNITS.add_housing_unit(unit_id, HousingUnit({'income': 50000, 'size': 3}))
STAGE_SHAPES['housing units'].set_housing_unit_inventory(UNITS, validate=False)
# Their digests, computed outside the module.
PINNED_DIGESTS = {
    'dropped record': 'dc927b313fb500e6e4056454846a3b05',
    'empty stage': 'b13019be762b43a841145038cec74e3b',
    'housing units': 'bcb1f6b42c1407059968acd7eb788845',
    'per-world part, padded': '71fbefc960fcae2ef418a9919290b729',
    'spine part': 'f7a607d5b291c4ced9ca43816e8a77ec',
    'view': '57aa2756b68bc2efcb1f53b1b344c4e6',
    'withheld record': 'ed653a8009a74eb3c10eb60f533119e1',
}


class TestDigest:
    """The digest is a function of the content alone, in canonical form."""

    def test_golden_value(self) -> None:
        """The digest of the sample matches its pinned value."""
        assert digest_of(sample()) == GOLDEN_DIGEST

    @pytest.mark.parametrize('shape', sorted(STAGE_SHAPES))
    def test_each_stage_shape_keeps_its_pinned_digest(self, shape: str) -> None:
        """Each shape of stage keeps its pinned digest."""
        assert digest_of(STAGE_SHAPES[shape]) == PINNED_DIGESTS[shape]

    def test_insertion_order_of_assets_and_features_does_not_matter(self) -> None:
        """The insertion order of assets and features does not matter."""
        records = [
            (
                '9q8yyk8yuw',
                [[-122.4190, 37.7750]],
                {
                    'PlanArea': None,
                    'NumberOfStories': 1,
                    'OccupancyClass': 'COM1',
                },
            ),
            (
                '9q8yyk8yuv',
                [[-122.4194, 37.7749]],
                {
                    'PlanArea': 151.25,
                    'OccupancyClass': 'RES1',
                    'NumberOfStories': 2,
                },
            ),
        ]
        assert digest_of(inventory(records)) == digest_of(sample())

    def test_numpy_values_digest_as_python_values(self) -> None:
        """Numpy values digest as the equal Python values."""
        plain = inventory(
            [
                (
                    'a',
                    [[-122.4, 37.7]],
                    {'n': 3, 'x': 0.5, 'w': [1.5, 2.5], 'b': True},
                )
            ]
        )
        numpy_values = inventory(
            [
                (
                    'a',
                    [[-122.4, 37.7]],
                    {
                        'n': np.int64(3),
                        'x': np.float64(0.5),
                        'w': np.array([1.5, 2.5]),
                        'b': np.True_,
                    },
                )
            ]
        )
        # Asset refuses array coordinates, so set them after construction.
        numpy_values.inventory['a'].coordinates = np.array([[-122.4, 37.7]])
        assert digest_of(numpy_values) == digest_of(plain)

    def test_nan_is_a_value_and_equal_to_itself(self) -> None:
        """NaN is a value and digests equal to itself."""
        first = inventory([('a', [[-122.4, 37.7]], {'x': float('nan')})])
        second = inventory([('a', [[-122.4, 37.7]], {'x': np.float64('nan')})])
        assert digest_of(first) == digest_of(second)
        assert digest_of(first) != digest_of(
            inventory([('a', [[-122.4, 37.7]], {'x': None})])
        )

    def test_int_and_float_of_equal_value_differ(self) -> None:
        """An int and a float of equal value differ."""
        as_int = inventory([('a', [[-122.4, 37.7]], {'n': 3})])
        as_float = inventory([('a', [[-122.4, 37.7]], {'n': 3.0})])
        assert digest_of(as_int) != digest_of(as_float)

    def test_float32_keeps_its_rounding(self) -> None:
        """A float32 keeps its rounding."""
        as_32 = inventory([('a', [[-122.4, 37.7]], {'x': np.float32(0.1)})])
        as_64 = inventory([('a', [[-122.4, 37.7]], {'x': 0.1})])
        rounded = inventory([('a', [[-122.4, 37.7]], {'x': float(np.float32(0.1))})])
        assert digest_of(as_32) != digest_of(as_64)
        assert digest_of(as_32) == digest_of(rounded)

    def test_integer_and_string_ids_are_distinct(self) -> None:
        """Integer and string ids are distinct."""
        as_int = inventory([(1, [[-122.4, 37.7]], {})])
        as_str = inventory([('1', [[-122.4, 37.7]], {})])
        both = inventory([('1', [[-122.4, 37.7]], {}), (1, [[-122.4, 37.7]], {})])
        reversed_both = inventory(
            [(1, [[-122.4, 37.7]], {}), ('1', [[-122.4, 37.7]], {})]
        )
        assert digest_of(as_int) != digest_of(as_str)
        assert digest_of(both) == digest_of(reversed_both)

    @pytest.mark.parametrize(
        'change',
        [
            'geometry',
            'feature value',
            'feature added',
            'asset removed',
            'n_pw',
            'vector features',
        ],
    )
    def test_each_kind_of_change_moves_the_digest(self, change: str) -> None:
        """Each kind of change moves the digest."""
        inv = sample()
        if change == 'geometry':
            inv.inventory['9q8yyk8yuv'].coordinates = [[-122.4194, 37.77491]]
        elif change == 'feature value':
            inv.add_asset_features(
                '9q8yyk8yuv', {'NumberOfStories': 3}, overwrite=True
            )
        elif change == 'feature added':
            inv.add_asset_features('9q8yyk8yuv', {'YearBuilt': 1950})
        elif change == 'asset removed':
            inv.remove_asset('9q8yyk8yuw')
        elif change == 'n_pw':
            inv.n_pw = 2
        else:
            inv.vector_features = [*inv.vector_features, 'Extra']
        assert digest_of(inv) != digest_of(sample())

    def test_an_attached_housing_unit_inventory_is_digested(self) -> None:
        """Attaching a housing-unit inventory, or changing it, moves the digest."""
        with_units = sample()
        with_units.set_housing_unit_inventory(housing_units(), validate=False)
        other_units = sample()
        other_units.set_housing_unit_inventory(housing_units(60000), validate=False)
        assert digest_of(with_units) != digest_of(sample())
        assert digest_of(with_units) != digest_of(other_units)

        freeze = StageFreeze()
        digest = freeze.freeze('households', with_units)
        units = freeze.read('households').housing_unit_inventory
        units.inventory['1'].features['size'] = 4
        with pytest.raises(RuntimeError, match="'households' changed after"):
            freeze.manifest()
        assert digest == freeze.digest('households')

    def test_order_of_vector_features_does_not_matter(self) -> None:
        """The order of vector features does not matter."""
        records = [('a', [[-122.4, 37.7]], {'w': [1, 2]})]
        first = inventory(records, vector_features=['w', 'v'])
        second = inventory(records, vector_features=['v', 'w'])
        assert digest_of(first) == digest_of(second)

    @pytest.mark.parametrize(
        'value', [{1, 2}, object(), np.datetime64('2026-01-01'), b'bytes']
    )
    def test_an_unsupported_value_raises_and_freezes_nothing(
        self, value: Any
    ) -> None:
        """An unsupported value raises and freezes nothing."""
        freeze = StageFreeze()
        bad = inventory([('a', [[-122.4, 37.7]], {'x': value})])
        with pytest.raises(TypeError, match="features of asset 'a'"):
            freeze.freeze('observed', bad)
        assert freeze.order == ()

    def test_non_string_feature_key_raises(self) -> None:
        """A feature key that is not a string raises."""
        bad = inventory([('a', [[-122.4, 37.7]], {})])
        bad.inventory['a'].features[1] = 'x'
        with pytest.raises(TypeError, match='has type int, not str'):
            digest_of(bad)


class TestArguments:
    """Arguments of the wrong kind are refused before anything is frozen."""

    @pytest.mark.parametrize(('stage', 'error'), [('', ValueError), (1, TypeError)])
    def test_bad_stage_name_raises(self, stage: Any, error: type) -> None:
        """An empty or non-string stage name raises."""
        with pytest.raises(error):
            StageFreeze().freeze(stage, sample())

    def test_output_must_be_an_inventory(self) -> None:
        """An output that is not an inventory raises."""
        with pytest.raises(TypeError, match='AssetInventory, not dict'):
            StageFreeze().freeze('observed', {})


# Longitudes of the batch test's donors and targets.
BATCH_LONGITUDES = {
    'd1': -122.40,
    'd2': -122.46,
    't1': -122.41,
    't2': -122.45,
    't3': -122.411,
    't4': -122.451,
    't5': -122.4115,
}


def nearest_donor(
    target_lon: float, pool: AssetInventory, world: int
) -> tuple[str, int]:
    """
    Stand in for a donor draw: take the nearest donor's year, plus ``world + 1``.

    Args:
        target_lon (float):
            The target's longitude. Distance is in longitude alone, and a tie
            goes to the lower asset id.
        pool (AssetInventory):
            One world's donors, observed and imputed.
        world (int):
            The world the pool holds, added to the year so that each world's
            value differs.

    Returns:
        tuple[str, int]:
            The donor's id and the year given to the target.
    """
    donor = min(
        pool.get_asset_ids(),
        key=lambda aid: (
            abs(pool.get_asset_coordinates(aid)[1][0][0] - target_lon),
            str(aid),
        ),
    )
    year = pool.get_asset_features(donor)[1]['YearBuilt']
    return donor, year + world + 1


class TestImputerBatchOption:
    """The calls a donor imputer that runs in batches makes."""

    BATCHES = (('t1', 't2'), ('t3',), ('t4', 't5'))

    def run(
        self, *, reverse: bool = False
    ) -> tuple[StageFreeze, list[set[str]], dict[str, list[str]]]:
        """
        Run the stages.

        Args:
            reverse (bool, optional):
                Insert the observed donors in reverse order. Defaults to
                False.

        Returns:
            tuple[StageFreeze, list[set[str]], dict[str, list[str]]]:
                The freeze, each batch's donor ids, and each target's donor
                in every world.
        """
        # two observed donors, one year in every world
        records = [
            ('d1', [[BATCH_LONGITUDES['d1'], 37.77]], {'YearBuilt': 1950}),
            ('d2', [[BATCH_LONGITUDES['d2'], 37.77]], {'YearBuilt': 1970}),
        ]
        freeze = StageFreeze()
        freeze.freeze(
            'observed',
            inventory(records[::-1] if reverse else records, n_pw=N_WORLDS),
        )
        pools = []
        donors_used: dict[str, list[str]] = {}
        for j, targets in enumerate(self.BATCHES, start=1):
            earlier = tuple(f'imputed_batch{i}' for i in range(1, j))
            frozen = [freeze.read(name) for name in ('observed', *earlier)]
            years: dict[str, list[int]] = {target: [] for target in targets}
            for w in range(N_WORLDS):
                pool = AssetInventory()
                for stage_output in frozen:
                    realization = stage_output.get_world_realization(w)
                    for aid in realization.get_asset_ids():
                        pool.add_asset(aid, realization.inventory[aid])
                for target in targets:
                    donor, year = nearest_donor(BATCH_LONGITUDES[target], pool, w)
                    donors_used.setdefault(target, []).append(donor)
                    years[target].append(year)
            pools.append(set(pool.get_asset_ids()))
            batch = inventory(
                [
                    (t, [[BATCH_LONGITUDES[t], 37.77]], {'YearBuilt': years[t]})
                    for t in targets
                ],
                n_pw=N_WORLDS,
            )
            freeze.freeze(f'imputed_batch{j}', batch)
        return freeze, pools, donors_used

    def test_a_batch_draws_only_from_donors_and_earlier_batches(self) -> None:
        """A batch draws only from donors and earlier batches."""
        _, pools, _ = self.run()
        assert pools == [
            {'d1', 'd2'},
            {'d1', 'd2', 't1', 't2'},
            {'d1', 'd2', 't1', 't2', 't3'},
        ]

    def test_a_later_batch_draws_from_an_earlier_batch_in_each_world(self) -> None:
        """A batch-2 target takes a batch-1 target's value in the same world."""
        freeze, _, donors_used = self.run()
        assert donors_used['t3'] == ['t1'] * N_WORLDS
        assert donors_used['t4'] == ['t2'] * N_WORLDS
        assert donors_used['t5'] == ['t3'] * N_WORLDS
        t1 = freeze.read('imputed_batch1').get_asset_features('t1')[1]['YearBuilt']
        t3 = freeze.read('imputed_batch2').get_asset_features('t3')[1]['YearBuilt']
        assert t1 == [1951, 1952, 1953]
        assert t3 == [t1[w] + w + 1 for w in range(N_WORLDS)] == [1952, 1954, 1956]

    def test_reading_a_batch_not_yet_frozen_raises(self) -> None:
        """Reading a batch not yet frozen raises."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        freeze.freeze('imputed_batch1', sample())
        with pytest.raises(LookupError):
            freeze.read('imputed_batch2')

    def test_manifest_lists_the_batches_in_order(self) -> None:
        """The manifest lists the batches in order."""
        freeze, _, _ = self.run()
        names = ['observed', 'imputed_batch1', 'imputed_batch2', 'imputed_batch3']
        assert freeze.manifest()['stages'] == [
            {'stage': name, 'digest': freeze.digest(name)} for name in names
        ]

    def test_a_frozen_batch_keeps_its_worlds(self) -> None:
        """A frozen batch keeps its worlds."""
        freeze, _, _ = self.run()
        batch = freeze.read('imputed_batch1')
        assert batch.n_pw == N_WORLDS
        assert batch.get_asset_features('t2')[1]['YearBuilt'] == [1971, 1972, 1973]

    def test_the_manifest_does_not_depend_on_donor_order(self) -> None:
        """Donors inserted in another order give the same manifest."""
        first, _, _ = self.run()
        second, _, _ = self.run(reverse=True)
        assert first.manifest() == second.manifest()


class TestNonResidentialReadOfResidentialWorlds:
    """The calls a non-residential stage makes on frozen stages."""

    def residential(self) -> AssetInventory:
        """Return two residential assets with years in three worlds."""
        return inventory(
            [
                (
                    'r1',
                    [[-122.41, 37.77]],
                    {
                        'OccupancyClass': 'RES1',
                        'YearBuilt': [1921, 1935, 1948],
                        'NumberOfStories': 2,
                    },
                ),
                (
                    'r2',
                    [[-122.40, 37.77]],
                    {
                        'OccupancyClass': 'RES3A',
                        'YearBuilt': [1960, 1962, 1958],
                        'NumberOfStories': 3,
                    },
                ),
            ],
            n_pw=N_WORLDS,
        )

    def run(self) -> tuple[StageFreeze, str]:
        """
        Run the observed, residential and non-residential stages.

        Returns:
            tuple[StageFreeze, str]:
                The freeze and the residential stage's digest.
        """
        freeze = StageFreeze()
        freeze.freeze(
            'observed',
            inventory([('o1', [[-122.405, 37.771]], {'OccupancyClass': 'COM4'})]),
        )
        digest = freeze.freeze('residential', self.residential())
        observed = freeze.read('observed')
        occupancy = observed.get_asset_features('o1')[1]['OccupancyClass']
        # This stage changes what it read, so it reads a copy.
        worlds = deepcopy(freeze.read('residential'))
        assert worlds.n_pw == N_WORLDS
        years = []
        for w in range(N_WORLDS):
            realization = worlds.get_world_realization(w)
            donor_years = [
                realization.get_asset_features(aid)[1]['YearBuilt']
                for aid in sorted(realization.get_asset_ids(), key=str)
            ]
            years.append(min(donor_years))
        worlds.add_asset_features('r1', {'YearBuilt': 2020}, overwrite=True)
        nonresidential = inventory(
            [
                (
                    'n1',
                    [[-122.405, 37.77]],
                    {'OccupancyClass': occupancy, 'YearBuilt': years},
                )
            ],
            n_pw=N_WORLDS,
        )
        freeze.freeze('nonresidential', nonresidential)
        return freeze, digest

    def test_each_world_reads_its_own_realization(self) -> None:
        """Each world reads its own realization, and the class comes from NSI."""
        freeze, _ = self.run()
        out = freeze.read('nonresidential').get_asset_features('n1')[1]
        assert out['YearBuilt'] == [1921, 1935, 1948]
        assert out['OccupancyClass'] == 'COM4'

    def test_residential_worlds_are_unchanged_by_the_later_stage(self) -> None:
        """A later stage that changes a copy leaves the frozen stage as it was."""
        freeze, digest = self.run()
        frozen = freeze.read('residential')
        assert (
            freeze.digest('residential') == digest == digest_of(self.residential())
        )
        assert features_of(frozen) == features_of(self.residential())
        assert frozen.n_pw == N_WORLDS
        assert frozen.vector_features == self.residential().vector_features

    def test_manifest_passes_the_guard(self) -> None:
        """The manifest lists the three stages, none of them changed."""
        freeze, _ = self.run()
        assert freeze.order == ('observed', 'residential', 'nonresidential')
        assert [entry['stage'] for entry in freeze.manifest()['stages']] == list(
            freeze.order
        )


class TestDiscovery:
    """The class is exported and registered with the Importer."""

    def test_exported_from_the_package(self) -> None:
        """The class is exported from brails.utils."""
        assert brails.utils.StageFreeze is StageFreeze

    def test_importer_finds_the_class(self) -> None:
        """The Importer registers the class."""
        assert Importer().get_class('StageFreeze') is StageFreeze
