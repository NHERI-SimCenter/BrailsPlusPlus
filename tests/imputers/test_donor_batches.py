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


"""Tests for the batch feedback option of the observed-donor imputer."""

from __future__ import annotations

from typing import Any

import pytest

from brails.imputers import ObservedDonorImputer
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    OUTCOME_EXHAUSTED,
    ROUND_DONOR,
    ROUND_REPAIR,
)
from brails.utils.stage_freeze import StageFreeze
from tests.imputers.donor_fixtures import (
    OTHER_SEED,
    asset,
    features_of,
    imputer,
    inventory,
)

DONOR_STAGE = 'observed'

# Golden batch numbers of GOLDEN_IDS with SEED and three batches.
GOLDEN_IDS = ('a', 'b', 'c', 'd')
GOLDEN_BATCHES = [3, 2, 3, 2]


def observed() -> dict[str, tuple[list, dict[str, Any]]]:
    """
    Return six observed donors 1.5 km east of the origin.

    Returns:
        dict[str, tuple[list, dict[str, Any]]]:
            ``o0`` to ``o5``, residential and non-residential.
    """
    return {
        f'o{i}': asset(
            1500.0 + 20.0 * i,
            0.0,
            OccupancyClass=('RES1', 'RES3A', 'COM1')[i % 3],
            NumberOfStories=1 + i % 3,
            NumberOfUnits=i,
        )
        for i in range(6)
    }


def clustered_targets(n: int = 24) -> dict[str, tuple[list, dict[str, Any]]]:
    """
    Return targets on a line west of the donors, missing every feature.

    Args:
        n (int, optional):
            The number of targets. Defaults to 24.

    Returns:
        dict[str, tuple[list, dict[str, Any]]]:
            ``t00`` onward at 25 m steps from the origin.
    """
    return {f't{i:02d}': asset(25.0 * i, 0.0) for i in range(n)}


def frozen_donors(records: dict[str, tuple[list, dict[str, Any]]]) -> StageFreeze:
    """
    Return a freeze holding the observed donors.

    Args:
        records (dict[str, tuple[list, dict[str, Any]]]):
            The donors.

    Returns:
        StageFreeze:
            A freeze with one stage, :data:`DONOR_STAGE`.
    """
    freeze = StageFreeze()
    freeze.freeze(DONOR_STAGE, inventory(records))
    return freeze


class SpyFreeze(StageFreeze):
    """A freeze that records each freeze in an event list."""

    def __init__(self, events: list[tuple[str, Any]]) -> None:
        """
        Create the freeze.

        Args:
            events (list[tuple[str, Any]]):
                The list the events are appended to.
        """
        super().__init__()
        self.events = events

    def freeze(self, stage: str, output: Any) -> str:
        """
        Record and freeze a stage.

        Args:
            stage (str):
                The stage.
            output (Any):
                Its output.

        Returns:
            str:
                The digest.
        """
        self.events.append(('freeze', stage))
        return super().freeze(stage, output)


class DroppingFreeze(StageFreeze):
    """A freeze that silently drops the first batch."""

    def freeze(self, stage: str, output: Any) -> str:
        """
        Freeze a stage unless it is the first batch.

        Args:
            stage (str):
                The stage.
            output (Any):
                Its output.

        Returns:
            str:
                The digest, empty for the dropped stage.
        """
        if stage.endswith('_batch1'):
            return ''
        return super().freeze(stage, output)


class TestOptionOff:
    """The option is off by default and one batch is the plain run."""

    def test_default_is_one_batch(self) -> None:
        """The default is one batch."""
        assert ObservedDonorImputer().n_batches == 1

    def test_one_batch_equals_the_plain_run(self) -> None:
        """One batch through a freeze equals the run without one."""
        targets = inventory(clustered_targets())
        plain = imputer().impute_stage(targets, inventory(observed()), n_worlds=3)
        freeze = frozen_donors(observed())
        batched = imputer().impute_stage(
            targets, freeze=freeze, donor_stage=DONOR_STAGE, n_worlds=3
        )
        assert batched.batches == ('imputed_batch1',)
        assert features_of(batched.imputed) == features_of(plain.imputed)
        assert features_of(freeze.read('imputed_batch1')) == features_of(
            plain.imputed
        )
        assert batched.log['donor_id'].tolist() == plain.log['donor_id'].tolist()


class TestBatchNumbers:
    """Batch numbers come from the target's keyed uniform at world 0."""

    def test_golden(self) -> None:
        """The batch numbers of four targets are pinned."""
        assert imputer(n_batches=3)._batch_numbers(GOLDEN_IDS) == GOLDEN_BATCHES

    def test_seed_changes_a_number(self) -> None:
        """A second seed changes at least one number."""
        other = imputer(n_batches=3, seed=OTHER_SEED)._batch_numbers(GOLDEN_IDS)
        assert other != GOLDEN_BATCHES

    def test_reordering_and_subsetting(self) -> None:
        """Numbers follow the id, not its position."""
        imp = imputer(n_batches=3)
        assert imp._batch_numbers(GOLDEN_IDS[::-1]) == GOLDEN_BATCHES[::-1]
        assert imp._batch_numbers(['b', 'd']) == [
            GOLDEN_BATCHES[1],
            GOLDEN_BATCHES[3],
        ]

    def test_same_in_every_world(self) -> None:
        """Batch membership does not depend on the number of worlds."""
        targets = inventory(clustered_targets())
        memberships = []
        for n_worlds in (1, 3):
            freeze = frozen_donors(observed())
            imputer(n_batches=3).impute_stage(
                targets, freeze=freeze, donor_stage=DONOR_STAGE, n_worlds=n_worlds
            )
            memberships.append(
                [
                    sorted(freeze.read(f'imputed_batch{j}').get_asset_ids())
                    for j in (1, 2, 3)
                ]
            )
        assert memberships[0] == memberships[1]


class TestFrozenReads:
    """A batch reads only the observed donors and frozen earlier batches."""

    def run(self, n_worlds: int = 4) -> tuple[Any, StageFreeze, dict[str, int]]:
        """
        Run three batches on the clustered targets.

        Args:
            n_worlds (int, optional):
                The number of worlds. Defaults to 4.

        Returns:
            tuple[Any, StageFreeze, dict[str, int]]:
                The result, the freeze and each target's batch.
        """
        imp = imputer(n_batches=3)
        freeze = frozen_donors(observed())
        targets = clustered_targets()
        result = imp.impute_stage(
            inventory(targets),
            freeze=freeze,
            donor_stage=DONOR_STAGE,
            n_worlds=n_worlds,
        )
        batches = dict(zip(sorted(targets), imp._batch_numbers(sorted(targets))))
        return result, freeze, batches

    def test_later_batch_is_never_a_donor(self) -> None:
        """A target of batch 2 or 3 is never a donor in batch 2."""
        result, _, batches = self.run()
        assert set(batches.values()) == {1, 2, 3}
        rows = result.log[result.log['round'] == ROUND_DONOR]
        second = rows[[batches[a] == 2 for a in rows['asset_id']]]
        donors = set(second['donor_id'])
        assert not {d for d in donors if batches.get(d, 0) >= 2}
        assert {d for d in donors if batches.get(d) == 1}

    def test_earlier_batch_supplies_its_world_values(self) -> None:
        """A batch-1 donor supplies in world w its values in world w."""
        result, freeze, batches = self.run()
        first = features_of(freeze.read('imputed_batch1'))
        rows = result.log[result.log['round'] == ROUND_DONOR]
        checked = 0
        for row in rows.itertuples(index=False):
            if batches[row.asset_id] == 2 and batches.get(row.donor_id) == 1:
                for feature, value in row.values.items():
                    donor_value = first[row.donor_id][feature]
                    if isinstance(donor_value, list):
                        donor_value = donor_value[row.world]
                    assert value == donor_value
                    checked += 1
        assert checked

    def test_missing_earlier_batch_raises(self) -> None:
        """Reading through a freeze that lacks an earlier batch raises."""
        freeze = DroppingFreeze()
        freeze.freeze(DONOR_STAGE, inventory(observed()))
        with pytest.raises(LookupError):
            imputer(n_batches=2).impute_stage(
                inventory(clustered_targets()),
                freeze=freeze,
                donor_stage=DONOR_STAGE,
            )

    def test_manifest(self) -> None:
        """The donor stage and the batches appear in order, and nothing else."""
        _, freeze, _ = self.run()
        stages = freeze.manifest()['stages']
        assert [s['stage'] for s in stages] == [
            DONOR_STAGE,
            'imputed_batch1',
            'imputed_batch2',
            'imputed_batch3',
        ]
        assert all(s['digest'] == freeze.digest(s['stage']) for s in stages)


class TestValidatedBeforeFrozen:
    """Each batch is validated and repaired before it is frozen."""

    def test_order_and_repair(self) -> None:
        """Validation of a batch precedes its freeze; repairs reach the freeze."""
        events: list[tuple[str, Any]] = []
        freeze = SpyFreeze(events)
        records = observed()
        records['tall'] = asset(
            1450.0, 0.0, OccupancyClass='RES1', NumberOfStories=3, NumberOfUnits=9
        )
        freeze.freeze(DONOR_STAGE, inventory(records))
        imp = imputer(n_batches=2)
        targets = clustered_targets()
        batches = dict(zip(sorted(targets), imp._batch_numbers(sorted(targets))))

        def validate(
            asset_id: Any,
            world: int,  # noqa: ARG001
            features: dict[str, Any],
        ) -> bool:
            """
            Record the call and judge the pair.

            Args:
                asset_id (Any):
                    The asset.
                world (int):
                    The world.
                features (dict[str, Any]):
                    Its features.

            Returns:
                bool:
                    Whether the pair is plausible.
            """
            events.append(('validate', asset_id))
            # a RES1 building of three or more stories is implausible
            return not (
                features.get('OccupancyClass') == 'RES1'
                and features.get('NumberOfStories', 0) >= 3
            )

        result = imp.impute_stage(
            inventory(targets),
            freeze=freeze,
            donor_stage=DONOR_STAGE,
            n_worlds=6,
            validate=validate,
        )
        position = {e: i for i, e in enumerate(events) if e[0] == 'freeze'}
        for i, (kind, asset_id) in enumerate(events):
            if kind == 'validate':
                batch = batches[asset_id]
                assert i < position[('freeze', f'imputed_batch{batch}')]
                if batch == 2:
                    assert i > position[('freeze', 'imputed_batch1')]
        repairs = result.log[
            (result.log['round'] == ROUND_REPAIR)
            & (result.log['outcome'] != OUTCOME_EXHAUSTED)
        ]
        assert not repairs.empty
        for row in repairs.itertuples(index=False):
            stored = features_of(
                freeze.read(f'imputed_batch{batches[row.asset_id]}')
            )[row.asset_id]
            for feature, value in row.values.items():
                held = stored[feature]
                assert (held[row.world] if isinstance(held, list) else held) == value


class TestLocalityAcrossBatches:
    """With earlier batches fixed, inputs beyond the cap do not move a draw."""

    def test_distant_inputs(self) -> None:
        """Donors more than a cap from every target change no batch."""
        imp = imputer(n_batches=2)
        targets = inventory(clustered_targets())
        changed = observed()
        for i in range(5):
            changed[f'far{i}'] = asset(
                -4000.0 - 10.0 * i, 0.0, OccupancyClass='RES1', NumberOfStories=1
            )
        first = frozen_donors(observed())
        second = frozen_donors(changed)
        a = imp.impute_stage(
            targets, freeze=first, donor_stage=DONOR_STAGE, n_worlds=3
        )
        b = imp.impute_stage(
            targets, freeze=second, donor_stage=DONOR_STAGE, n_worlds=3
        )
        assert first.digest('imputed_batch1') == second.digest('imputed_batch1')
        assert first.digest('imputed_batch2') == second.digest('imputed_batch2')
        assert a.log['donor_id'].tolist() == b.log['donor_id'].tolist()


class TestArguments:
    """Argument combinations that cannot run raise."""

    def test_batches_without_freeze(self) -> None:
        """Several batches without a freeze raise."""
        with pytest.raises(ValueError, match='freeze'):
            imputer(n_batches=2).impute_stage(
                inventory(clustered_targets()), inventory(observed())
            )

    def test_donors_and_freeze(self) -> None:
        """Donors together with a freeze raise."""
        with pytest.raises(ValueError, match='not both'):
            imputer().impute_stage(
                inventory(clustered_targets()),
                inventory(observed()),
                freeze=frozen_donors(observed()),
                donor_stage=DONOR_STAGE,
            )

    def test_freeze_without_stage_and_no_donors(self) -> None:
        """A freeze without a donor stage, or no donors at all, raises."""
        with pytest.raises(ValueError, match='donor_stage'):
            imputer().impute_stage(
                inventory(clustered_targets()), freeze=frozen_donors(observed())
            )
        with pytest.raises(ValueError, match='observed donors'):
            imputer().impute_stage(inventory(clustered_targets()))

    def test_status_name_clashing_with_the_batch_label(self) -> None:
        """A status name whose round label is the batch label raises."""
        with pytest.raises(ValueError, match='status_name'):
            imputer(status_name='batch')


class TestDonorAlsoTarget:
    """An observed donor that is also a target enters every pool once."""

    def test_overlapping_donor_and_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pools hold each id once, and a frozen target replaces its observed record."""
        records = {
            f'o{i}': asset(
                100.0 + 20.0 * i,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=1,
                NumberOfUnits=i + 1,
            )
            for i in range(3)
        }
        # 'x' holds a class and a story count, so it is a donor, and lacks its unit
        # count, so it is also a target.
        records['x'] = asset(50.0, 0.0, OccupancyClass='RES1', NumberOfStories=1)
        targets = dict(records)
        targets.update(
            {
                f't{i:02d}': asset(
                    10.0 * i, 30.0, OccupancyClass='RES1', NumberOfStories=1
                )
                for i in range(12)
            }
        )
        imp = imputer(n_batches=3)
        pools = []
        build = imp._build_pool

        def spy(*args: Any) -> Any:
            """
            Record each pool the stage builds.

            Args:
                *args (Any):
                    The arguments of ``_build_pool``.

            Returns:
                Any:
                    The pool.
            """
            pool = build(*args)
            pools.append(pool)
            return pool

        monkeypatch.setattr(imp, '_build_pool', spy)
        freeze = frozen_donors(records)
        imp.impute_stage(
            inventory(targets), freeze=freeze, donor_stage=DONOR_STAGE, n_worlds=2
        )
        assert len(pools) == 3
        for pool in pools:
            assert len(set(pool.donor_ids)) == len(pool.donor_ids)
            assert 'x' in pool.donor_ids
        batch_of_x = imp._batch_numbers(['x'])[0]
        assert batch_of_x < 3
        last = pools[-1]
        units = last.view(0).donor_values[last.donor_index['x']].get('NumberOfUnits')
        assert units is not None
