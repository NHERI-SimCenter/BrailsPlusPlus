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


"""Tests for the one-way freeze of stage outputs."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

import brails.compilers
from brails.compilers.stage_freeze import DIGEST_ALGORITHM, StageFreeze
from brails.types.asset_inventory import Asset, AssetInventory
from brails.types.housing_unit_inventory import HousingUnit, HousingUnitInventory
from brails.utils import Importer

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


def point(lon: float, lat: float) -> list[list[float]]:
    """Return point coordinates."""
    return [[lon, lat]]


def sample() -> AssetInventory:
    """Return a small two-asset inventory with mixed feature types."""
    return inventory(
        [
            (
                '9q8yyk8yuv',
                point(-122.4194, 37.7749),
                {
                    'OccupancyClass': 'RES1',
                    'NumberOfStories': 2,
                    'PlanArea': 151.25,
                },
            ),
            (
                '9q8yyk8yuw',
                point(-122.4190, 37.7750),
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


def features_of(inv: AssetInventory) -> dict[Any, dict[str, Any]]:
    """Return every asset's features keyed by asset id."""
    return {aid: dict(inv.get_asset_features(aid)[1]) for aid in inv.get_asset_ids()}


def digest_of(inv: AssetInventory) -> str:
    """Freeze ``inv`` alone and return its digest."""
    return StageFreeze().freeze('stage', inv)


class TestOneWay:
    """A stage reads only stages frozen before it, and cannot change them."""

    def test_reading_an_unfrozen_stage_raises(self) -> None:
        """Reading an unfrozen stage raises."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        with pytest.raises(LookupError, match="'imputed' is not frozen"):
            freeze.read('imputed')
        with pytest.raises(LookupError, match='not frozen'):
            freeze.digest('imputed')

    def test_declaring_an_unfrozen_read_raises_and_freezes_nothing(self) -> None:
        """Declaring an unfrozen read raises and freezes nothing."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        with pytest.raises(
            LookupError, match=r"declares reads of unfrozen stages \['later'\]"
        ):
            freeze.freeze('imputed', sample(), reads=('observed', 'later'))
        assert freeze.order == ('observed',)
        with pytest.raises(LookupError):
            freeze.read('imputed')

    def test_a_stage_cannot_declare_that_it_reads_itself(self) -> None:
        """A stage cannot declare that it reads itself."""
        freeze = StageFreeze()
        with pytest.raises(LookupError):
            freeze.freeze('observed', sample(), reads=('observed',))
        assert freeze.order == ()

    def test_an_undeclared_read_raises_and_freezes_nothing(self) -> None:
        """A read the next stage does not declare raises and freezes nothing."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        freeze.freeze('residential', sample())
        freeze.read('observed')
        freeze.read('residential')
        with pytest.raises(ValueError, match=r"does not declare \['residential'\]"):
            freeze.freeze('nonresidential', sample(), reads=('observed',))
        assert freeze.order == ('observed', 'residential')
        freeze.freeze('nonresidential', sample(), reads=('observed', 'residential'))
        assert freeze.manifest()['stages'][2]['reads'] == ['observed', 'residential']

    def test_reads_are_tracked_only_since_the_previous_freeze(self) -> None:
        """A read before the previous freeze need not be declared again."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        freeze.freeze('imputed', freeze.read('observed'), reads=('observed',))
        freeze.freeze('report', sample())
        assert freeze.manifest()['stages'][2]['reads'] == []

    def test_changing_the_original_after_freezing_leaves_the_freeze_unchanged(
        self,
    ) -> None:
        """Changing the original after freezing leaves the freeze unchanged."""
        freeze = StageFreeze()
        original = sample()
        digest = freeze.freeze('observed', original)
        expected = features_of(sample())

        original.add_asset_features(
            '9q8yyk8yuv', {'NumberOfStories': 9}, overwrite=True
        )
        original.inventory['9q8yyk8yuw'].coordinates[0][0] = 0.0
        original.add_asset('extra', Asset('extra', point(-122.0, 37.0)))

        assert features_of(freeze.read('observed')) == expected
        assert freeze.read('observed').get_asset_coordinates('9q8yyk8yuw')[1] == (
            point(-122.4190, 37.7750)
        )
        assert freeze.digest('observed') == digest == digest_of(sample())

    def test_changing_a_read_copy_leaves_the_freeze_unchanged(self) -> None:
        """Changing a read copy leaves the freeze unchanged."""
        freeze = StageFreeze()
        digest = freeze.freeze('observed', sample())

        later = freeze.read('observed')
        later.add_asset_features(
            '9q8yyk8yuv', {'OccupancyClass': 'IND1'}, overwrite=True
        )
        later.remove_asset('9q8yyk8yuw')
        later.n_pw = 4
        freeze.freeze('later', later, reads=('observed',))

        again = freeze.read('observed')
        assert features_of(again) == features_of(sample())
        assert again.n_pw == 1
        assert freeze.digest('observed') == digest == digest_of(again)
        assert freeze.read('observed') is not again

    def test_a_stage_name_is_frozen_once(self) -> None:
        """A stage name is frozen once."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        with pytest.raises(ValueError, match='already frozen'):
            freeze.freeze('observed', sample())
        assert freeze.order == ('observed',)


class TestRecordedOrder:
    """The order of freezing and the reads are kept and read back."""

    def build(self) -> StageFreeze:
        """Freeze three stages, each reading earlier ones."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        freeze.read('observed')
        freeze.freeze('residential', sample(), reads=['observed'])
        freeze.read('residential')
        freeze.read('observed')
        freeze.freeze('nonresidential', sample(), reads=('residential', 'observed'))
        return freeze

    def test_order_is_the_order_of_freezing(self) -> None:
        """The order is the order of freezing."""
        assert self.build().order == ('observed', 'residential', 'nonresidential')

    def test_manifest_records_order_reads_and_digests(self) -> None:
        """The manifest records the order, the reads and the digests."""
        freeze = self.build()
        manifest = freeze.manifest()
        assert manifest['digest_algorithm'] == DIGEST_ALGORITHM == 'blake2b-128'
        assert [entry['stage'] for entry in manifest['stages']] == list(freeze.order)
        assert [entry['reads'] for entry in manifest['stages']] == [
            [],
            ['observed'],
            ['residential', 'observed'],
        ]
        for entry in manifest['stages']:
            assert entry['digest'] == freeze.digest(entry['stage'])

    def test_manifest_records_reads_as_declared(self) -> None:
        """A declared stage is recorded whether or not read() was called."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        freeze.freeze('residential', sample())
        freeze.read('residential')
        freeze.freeze('later', sample(), reads=('observed', 'residential'))
        assert freeze.manifest()['stages'][2]['reads'] == [
            'observed',
            'residential',
        ]

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
        manifest['stages'][1]['reads'].append('nonresidential')
        assert freeze.manifest()['stages'][1]['reads'] == ['observed']

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
                {**ALGORITHM, 'stages': [{'stage': 'a', 'reads': []}]},
                'malformed stage entry',
            ),
            (
                {
                    **ALGORITHM,
                    'stages': [{'stage': 'a', 'reads': [['b']], 'digest': 'x'}],
                },
                'malformed stage entry',
            ),
            (
                {
                    **ALGORITHM,
                    'stages': [
                        {'stage': 'a', 'reads': [], 'digest': 'x'},
                        {'stage': 'b', 'reads': ['a', 'a'], 'digest': 'y'},
                    ],
                },
                'malformed stage entry',
            ),
            (
                {
                    **ALGORITHM,
                    'stages': [
                        {'stage': 'a', 'reads': [], 'digest': 'x'},
                        {'stage': 'a', 'reads': [], 'digest': 'y'},
                    ],
                },
                "'a' more than once",
            ),
            (
                {
                    **ALGORITHM,
                    'stages': [
                        {'stage': 'a', 'reads': ['b'], 'digest': 'x'},
                        {'stage': 'b', 'reads': [], 'digest': 'y'},
                    ],
                },
                r"reading \['b'\], which were not frozen before it",
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


class TestDigest:
    """The digest is a function of the content alone, in canonical form."""

    def test_golden_value(self) -> None:
        """The digest of the sample matches its pinned value."""
        assert digest_of(sample()) == GOLDEN_DIGEST

    def test_insertion_order_of_assets_and_features_does_not_matter(self) -> None:
        """The insertion order of assets and features does not matter."""
        records = [
            (
                '9q8yyk8yuw',
                point(-122.4190, 37.7750),
                {
                    'PlanArea': None,
                    'NumberOfStories': 1,
                    'OccupancyClass': 'COM1',
                },
            ),
            (
                '9q8yyk8yuv',
                point(-122.4194, 37.7749),
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
                    point(-122.4, 37.7),
                    {'n': 3, 'x': 0.5, 'w': [1.5, 2.5], 'b': True},
                )
            ]
        )
        numpy_values = inventory(
            [
                (
                    'a',
                    point(-122.4, 37.7),
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
        first = inventory([('a', point(-122.4, 37.7), {'x': float('nan')})])
        second = inventory([('a', point(-122.4, 37.7), {'x': np.float64('nan')})])
        assert digest_of(first) == digest_of(second)
        assert digest_of(first) != digest_of(
            inventory([('a', point(-122.4, 37.7), {'x': None})])
        )

    def test_int_and_float_of_equal_value_differ(self) -> None:
        """An int and a float of equal value differ."""
        as_int = inventory([('a', point(-122.4, 37.7), {'n': 3})])
        as_float = inventory([('a', point(-122.4, 37.7), {'n': 3.0})])
        assert digest_of(as_int) != digest_of(as_float)

    def test_float32_keeps_its_rounding(self) -> None:
        """A float32 keeps its rounding."""
        as_32 = inventory([('a', point(-122.4, 37.7), {'x': np.float32(0.1)})])
        as_64 = inventory([('a', point(-122.4, 37.7), {'x': 0.1})])
        rounded = inventory(
            [('a', point(-122.4, 37.7), {'x': float(np.float32(0.1))})]
        )
        assert digest_of(as_32) != digest_of(as_64)
        assert digest_of(as_32) == digest_of(rounded)

    def test_integer_and_string_ids_are_distinct(self) -> None:
        """Integer and string ids are distinct."""
        as_int = inventory([(1, point(-122.4, 37.7), {})])
        as_str = inventory([('1', point(-122.4, 37.7), {})])
        both = inventory(
            [('1', point(-122.4, 37.7), {}), (1, point(-122.4, 37.7), {})]
        )
        reversed_both = inventory(
            [(1, point(-122.4, 37.7), {}), ('1', point(-122.4, 37.7), {})]
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
            inv.inventory['9q8yyk8yuv'].coordinates = point(-122.4194, 37.77491)
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
        with_units.housing_unit_inventory.inventory['1'].features['size'] = 4
        frozen = freeze.read('households').housing_unit_inventory
        assert frozen.inventory['1'].features == {'income': 50000, 'size': 3}
        assert digest == freeze.digest('households')

    def test_order_of_vector_features_does_not_matter(self) -> None:
        """The order of vector features does not matter."""
        records = [('a', point(-122.4, 37.7), {'w': [1, 2]})]
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
        bad = inventory([('a', point(-122.4, 37.7), {'x': value})])
        with pytest.raises(TypeError, match="features of asset 'a'"):
            freeze.freeze('observed', bad)
        assert freeze.order == ()

    def test_non_string_feature_key_raises(self) -> None:
        """A feature key that is not a string raises."""
        bad = inventory([('a', point(-122.4, 37.7), {})])
        bad.inventory['a'].features[1] = 'x'
        with pytest.raises(TypeError, match='has type int, not str'):
            digest_of(bad)


class TestArguments:
    """Arguments of the wrong kind are refused before anything is frozen."""

    def test_reads_given_as_a_single_string_raises(self) -> None:
        """``reads`` given as a single string raises."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        with pytest.raises(TypeError, match=r"Pass \('observed',\)"):
            freeze.freeze('imputed', sample(), reads='observed')

    def test_reads_naming_a_stage_twice_raises(self) -> None:
        """``reads`` naming a stage twice raises and names it."""
        freeze = StageFreeze()
        freeze.freeze('observed', sample())
        with pytest.raises(ValueError, match=r"\['observed'\] more than once"):
            freeze.freeze('imputed', sample(), reads=('observed', 'observed'))

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

    def donors(self, *, reverse: bool = False) -> AssetInventory:
        """Return the observed donors, which hold one year in every world."""
        records = [
            ('d1', point(BATCH_LONGITUDES['d1'], 37.77), {'YearBuilt': 1950}),
            ('d2', point(BATCH_LONGITUDES['d2'], 37.77), {'YearBuilt': 1970}),
        ]
        return inventory(records[::-1] if reverse else records, n_pw=N_WORLDS)

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
        freeze = StageFreeze()
        freeze.freeze('observed', self.donors(reverse=reverse))
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
                    (t, point(BATCH_LONGITUDES[t], 37.77), {'YearBuilt': years[t]})
                    for t in targets
                ],
                n_pw=N_WORLDS,
            )
            freeze.freeze(f'imputed_batch{j}', batch, reads=('observed', *earlier))
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
        freeze.freeze('imputed_batch1', sample(), reads=('observed',))
        with pytest.raises(LookupError):
            freeze.read('imputed_batch2')
        with pytest.raises(LookupError):
            freeze.freeze(
                'imputed_batch3',
                sample(),
                reads=('observed', 'imputed_batch1', 'imputed_batch2'),
            )

    def test_manifest_lists_the_batches_in_order_with_their_reads(self) -> None:
        """The manifest lists the batches in order with their reads."""
        freeze, _, _ = self.run()
        assert freeze.manifest()['stages'] == [
            {'stage': 'observed', 'reads': [], 'digest': freeze.digest('observed')},
            {
                'stage': 'imputed_batch1',
                'reads': ['observed'],
                'digest': freeze.digest('imputed_batch1'),
            },
            {
                'stage': 'imputed_batch2',
                'reads': ['observed', 'imputed_batch1'],
                'digest': freeze.digest('imputed_batch2'),
            },
            {
                'stage': 'imputed_batch3',
                'reads': ['observed', 'imputed_batch1', 'imputed_batch2'],
                'digest': freeze.digest('imputed_batch3'),
            },
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

    def observed(self) -> AssetInventory:
        """Return one NSI-observed non-residential donor."""
        return inventory(
            [('o1', point(-122.405, 37.771), {'OccupancyClass': 'COM4'})]
        )

    def residential(self) -> AssetInventory:
        """Return two residential assets with years in three worlds."""
        return inventory(
            [
                (
                    'r1',
                    point(-122.41, 37.77),
                    {
                        'OccupancyClass': 'RES1',
                        'YearBuilt': [1921, 1935, 1948],
                        'NumberOfStories': 2,
                    },
                ),
                (
                    'r2',
                    point(-122.40, 37.77),
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
        freeze.freeze('observed', self.observed())
        digest = freeze.freeze('residential', self.residential())
        observed = freeze.read('observed')
        occupancy = observed.get_asset_features('o1')[1]['OccupancyClass']
        worlds = freeze.read('residential')
        assert worlds.n_pw == N_WORLDS
        years = []
        for w in range(N_WORLDS):
            realization = worlds.get_world_realization(w)
            donor_years = [
                realization.get_asset_features(aid)[1]['YearBuilt']
                for aid in sorted(realization.get_asset_ids(), key=str)
            ]
            years.append(min(donor_years))
        # The later stage may change what it read without reaching the freeze.
        worlds.add_asset_features('r1', {'YearBuilt': 2020}, overwrite=True)
        nonresidential = inventory(
            [
                (
                    'n1',
                    point(-122.405, 37.77),
                    {'OccupancyClass': occupancy, 'YearBuilt': years},
                )
            ],
            n_pw=N_WORLDS,
        )
        freeze.freeze(
            'nonresidential', nonresidential, reads=('observed', 'residential')
        )
        return freeze, digest

    def test_each_world_reads_its_own_realization(self) -> None:
        """Each world reads its own realization, and the class comes from NSI."""
        freeze, _ = self.run()
        out = freeze.read('nonresidential').get_asset_features('n1')[1]
        assert out['YearBuilt'] == [1921, 1935, 1948]
        assert out['OccupancyClass'] == 'COM4'

    def test_residential_worlds_are_unchanged_by_the_later_stage(self) -> None:
        """The residential worlds are unchanged by the later stage."""
        freeze, digest = self.run()
        frozen = freeze.read('residential')
        assert (
            freeze.digest('residential') == digest == digest_of(self.residential())
        )
        assert features_of(frozen) == features_of(self.residential())
        assert frozen.n_pw == N_WORLDS
        assert frozen.vector_features == self.residential().vector_features

    def test_manifest_records_both_reads(self) -> None:
        """The manifest records the observed and residential reads."""
        freeze, _ = self.run()
        assert freeze.order == ('observed', 'residential', 'nonresidential')
        assert freeze.manifest()['stages'][2]['reads'] == ['observed', 'residential']


class TestDiscovery:
    """The class is exported and registered with the Importer."""

    def test_exported_from_the_package(self) -> None:
        """The class is exported from brails.compilers."""
        assert brails.compilers.StageFreeze is StageFreeze
        assert brails.compilers.__all__ == ['StageFreeze']

    def test_importer_finds_the_class(self) -> None:
        """The Importer registers the class."""
        assert Importer().get_class('StageFreeze') is StageFreeze
