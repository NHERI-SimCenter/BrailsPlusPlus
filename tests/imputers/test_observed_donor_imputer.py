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

"""Tests for the observed-donor imputer."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from brails.constants import BLOCK_GROUP_KEY, EARTH_RADIUS_M
from brails.imputers import ObservedDonorImputer
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    OUTCOME_BLOCK_GROUP,
    OUTCOME_EXHAUSTED,
    OUTCOME_FEWER_THAN_K,
    OUTCOME_NEAREST_ADMITTED,
    OUTCOME_NO_AREA,
    OUTCOME_NO_DONOR,
    OUTCOME_NONE_ADMITTED,
    OUTCOME_RELAXED,
    PURPOSE_RESIDENTIAL,
    PURPOSE_RESIDENTIAL_FALLBACK,
    ROUND_DONOR,
    ROUND_REPAIR,
    ROUND_STATUS,
    DonorDraw,
)
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils import Importer
from brails.utils.stage_freeze import StageFreeze
from brails.utils.unit_count_rules import UNIT_COUNT_CLASSES
from tests.imputers.donor_fixtures import (
    BG_A,
    BG_B,
    LAT0,
    LON0,
    OTHER_SEED,
    SEED,
    asset,
    donors_from_log,
    imputer,
    inventory,
    square,
    world_value,
)

# Golden values, computed with SEED on the fixtures below.
GOLDEN_DONORS = {
    ('t1', 0): 'd12',
    ('t1', 1): 'd15',
    ('t2', 0): 'd14',
    ('t2', 1): 'd23',
    ('t3', 0): 'd27',
    ('t3', 1): 'd20',
}
GOLDEN_STATUS = {'r1': False, 'r2': True}
GOLDEN_FALLBACK = 'b1'
GOLDEN_RESIDENTIAL_FALLBACK = {'f0': 'c_n', 'f1': 'a_r1', 'f2': 'c_n'}


def donor_grid() -> dict[str, tuple[list, dict[str, Any]]]:
    """
    Return 30 residential and non-residential donors on a 100 m grid.

    Returns:
        dict[str, tuple[list, dict[str, Any]]]:
            Ids ``d00`` to ``d29``; every donor's ``NumberOfUnits`` is its index, so
            a copied value names its donor.
    """
    classes = ('RES1', 'RES3A', 'COM1')
    records = {}
    for i in range(30):
        records[f'd{i:02d}'] = asset(
            100.0 * (i % 6),
            100.0 * (i // 6),
            8.0 + i % 4,
            OccupancyClass=classes[i % 3],
            NumberOfStories=1 + i % 3,
            NumberOfUnits=i,
            BLOCKGROUP_GEOID=BG_A,
        )
    return records


def grid_targets() -> dict[str, tuple[list, dict[str, Any]]]:
    """
    Return three targets among the grid donors.

    Returns:
        dict[str, tuple[list, dict[str, Any]]]:
            ``t1`` and ``t2`` with a known class, ``t3`` with none.
    """
    return {
        't1': asset(120.0, 130.0, 9.0, OccupancyClass='RES1'),
        't2': asset(330.0, 260.0, 10.0, OccupancyClass='COM1'),
        't3': asset(250.0, 410.0, 11.0),
    }


class TestDonorsAreObserved:
    """Donors are observed buildings and never the target itself."""

    def test_target_is_never_its_own_donor(self) -> None:
        """An asset missing a feature never copies from itself."""
        records = donor_grid()
        coordinates, features = records['d07']
        features.pop('NumberOfUnits')
        records['d07'] = (coordinates, features)
        result = imputer().impute_stage(
            inventory(records), inventory(records), n_worlds=20
        )
        rows = result.log[result.log['asset_id'] == 'd07']
        assert not rows.empty
        assert 'd07' not in set(rows['donor_id'])

    def test_donor_lacking_a_required_feature_is_never_drawn(self) -> None:
        """An asset without a story count is never a donor."""
        records = donor_grid()
        for _, features in records.values():
            features.pop('NumberOfStories')
        records['only'] = asset(
            500.0, 500.0, OccupancyClass='RES1', NumberOfStories=2, NumberOfUnits=99
        )
        targets = {'t': asset(120.0, 130.0, OccupancyClass='RES1')}
        result = imputer().impute_stage(
            inventory(targets), inventory(records), n_worlds=10
        )
        assert set(donors_from_log(result).values()) == {'only'}

    def test_imputed_value_never_feeds_another_target(self) -> None:
        """A target copying from another target gets only observed values."""
        records = {
            'a': asset(0.0, 0.0, OccupancyClass='RES1', NumberOfStories=1),
            'b': asset(20.0, 0.0, OccupancyClass='RES1', NumberOfStories=1),
            'c': asset(
                3000.0,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=1,
                NumberOfUnits=1,
            ),
        }
        inv = inventory(records)
        result = imputer().impute_stage(inv, inv, n_worlds=4)
        donors = donors_from_log(result)
        assert donors[('a', 0)] == 'b'
        assert donors[('b', 0)] == 'a'
        assert result.imputed.get_asset_ids() == []


class TestFilters:
    """Known categorical features filter the donors."""

    def test_known_class_draws_only_that_class(self) -> None:
        """A target with a known class draws only donors of that class."""
        result = imputer().impute_stage(
            inventory(grid_targets()), inventory(donor_grid()), n_worlds=30
        )
        grid = donor_grid()
        for (target, _), donor in donors_from_log(result).items():
            if target == 't1':
                assert grid[donor][1]['OccupancyClass'] == 'RES1'
            if target == 't2':
                assert grid[donor][1]['OccupancyClass'] == 'COM1'

    def test_relaxation_drops_the_last_filter_first(self) -> None:
        """Below k matching donors, the last soft filter is relaxed first."""
        records = {
            f'd{i}': asset(
                30.0 * i,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=1,
                BuildingType='W' if i < 2 else 'M',
                Roof='gable' if i < 6 else 'flat',
            )
            for i in range(8)
        }
        targets = {
            't': asset(
                0.0, 40.0, OccupancyClass='RES1', BuildingType='W', Roof='gable'
            )
        }
        imp = imputer(soft_filters=('OccupancyClass', 'Roof', 'BuildingType'))
        result = imp.impute_stage(inventory(targets), inventory(records), n_worlds=5)
        rows = result.log[result.log['round'] == ROUND_DONOR]
        assert set(rows['level']) == {1}
        assert set(rows['outcome']) == {OUTCOME_RELAXED}
        assert set(rows['n_candidates']) == {6}

    def test_residential_status_is_never_relaxed(self) -> None:
        """A residential target never draws a non-residential donor."""
        records = {
            f'c{i}': asset(30.0 * i, 0.0, OccupancyClass='COM1', NumberOfStories=1)
            for i in range(8)
        }
        records['r'] = asset(60.0, 30.0, OccupancyClass='RES1', NumberOfStories=1)
        targets = {'t': asset(0.0, 40.0, OccupancyClass='RES3A')}
        result = imputer().impute_stage(
            inventory(targets), inventory(records), n_worlds=10
        )
        rows = result.log[result.log['round'] == ROUND_DONOR]
        assert set(rows['donor_id']) == {'r'}
        assert set(rows['outcome']) == {OUTCOME_FEWER_THAN_K}


class TestDistance:
    """The distance combines location, log plan area and story count."""

    def test_pinned_distance(self) -> None:
        """One pair's distance equals the formula computed independently."""
        donor_coords = square(300.0, 400.0, 20.0)
        target_coords = square(0.0, 0.0, 10.0)
        records = {
            'd': (
                donor_coords,
                {'OccupancyClass': 'RES1', 'NumberOfStories': 3},
            )
        }
        targets = {
            't': (
                target_coords,
                {'OccupancyClass': 'RES1', 'NumberOfStories': 1},
            )
        }
        result = imputer().impute_stage(inventory(targets), inventory(records))
        row = result.log.iloc[0]
        lon1, lat1, area1 = AssetInventory.centroid_and_plan_area(target_coords)
        lon2, lat2, area2 = AssetInventory.centroid_and_plan_area(donor_coords)
        p1, p2 = math.radians(lat1), math.radians(lat2)
        a = (
            math.sin((p2 - p1) / 2) ** 2
            + math.cos(p1)
            * math.cos(p2)
            * math.sin(math.radians(lon2 - lon1) / 2) ** 2
        )
        g = 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))
        area = math.log(area1) - math.log(area2)
        expected = math.sqrt((g / 100.0) ** 2 + area**2 + (1 - 3) ** 2)
        assert row['distance_m'] == pytest.approx(g, rel=1e-9)
        assert row['distance'] == pytest.approx(expected, rel=1e-9)
        assert row['distance'] == pytest.approx(5.560699, abs=1e-6)

    def test_ranking_follows_area(self) -> None:
        """With the location fixed, the nearest log area is drawn."""
        records = {
            f'a{side}': asset(
                50.0, 0.0, side, OccupancyClass='RES1', NumberOfStories=1
            )
            for side in (5, 12, 30)
        }
        targets = {'t': asset(0.0, 0.0, 11.0, OccupancyClass='RES1')}
        result = imputer(k=1).impute_stage(inventory(targets), inventory(records))
        assert result.log.iloc[0]['donor_id'] == 'a12'

    def test_ranking_follows_stories(self) -> None:
        """With location and area fixed, the nearest story count is drawn."""
        records = {
            f's{n}': asset(
                50.0,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=n,
            )
            for n in (1, 3, 6)
        }
        targets = {'t': asset(0.0, 0.0, OccupancyClass='RES1', NumberOfStories=4)}
        result = imputer(k=1).impute_stage(inventory(targets), inventory(records))
        assert result.log.iloc[0]['donor_id'] == 's3'


class TestKeyedDraws:
    """One donor per world by the target's keyed uniform."""

    def run(
        self,
        seed: int = SEED,
        targets: dict[str, Any] | None = None,
        donor_order: list[str] | None = None,
        target_order: list[str] | None = None,
    ) -> dict[tuple[str, int], str]:
        """
        Run the grid fixture.

        Args:
            seed (int, optional):
                The seed.
            targets (dict, optional):
                The targets. Defaults to :func:`grid_targets`.
            donor_order (list[str], optional):
                Insertion order of the donors.
            target_order (list[str], optional):
                Insertion order of the targets.

        Returns:
            dict[tuple[str, int], str]:
                The donor of each target and world.
        """
        targets = targets or grid_targets()
        result = imputer(seed=seed).impute_stage(
            inventory(targets, target_order),
            inventory(donor_grid(), donor_order),
            n_worlds=2,
        )
        return donors_from_log(result)

    def test_golden_donors(self) -> None:
        """The donors of three targets in two worlds are pinned."""
        assert self.run() == GOLDEN_DONORS

    def test_seed_changes_a_donor(self) -> None:
        """A second seed changes at least one donor."""
        assert self.run(OTHER_SEED) != GOLDEN_DONORS

    def test_reordering_changes_nothing(self) -> None:
        """Shuffling targets and donors changes no donor."""
        donors = list(donor_grid())
        random.Random(3).shuffle(donors)
        assert (
            self.run(donor_order=donors, target_order=['t3', 't1', 't2'])
            == GOLDEN_DONORS
        )

    def test_subsetting_targets_changes_nothing(self) -> None:
        """Dropping targets leaves the others' donors unchanged."""
        targets = grid_targets()
        del targets['t2']
        expected = {key: v for key, v in GOLDEN_DONORS.items() if key[0] != 't2'}
        assert self.run(targets=targets) == expected


def mixed_neighbors() -> dict[str, tuple[list, dict[str, Any]]]:
    """
    Return residential and non-residential donors mixed around the origin.

    Returns:
        dict[str, tuple[list, dict[str, Any]]]:
            Six donors, alternating RES1 and COM1.
    """
    return {
        f'm{i}': asset(
            40.0 * i,
            20.0,
            OccupancyClass='RES1' if i % 2 else 'COM1',
            NumberOfStories=1,
            NumberOfUnits=i,
        )
        for i in range(6)
    }


class TestTwoRounds:
    """Round one decides the residential status, round two the donor."""

    def statuses(self, seed: int, order: list[str] | None = None) -> dict[str, bool]:
        """
        Return the round-one status of the targets ``r1`` and ``r2``.

        Args:
            seed (int):
                The seed.
            order (list[str], optional):
                Insertion order of the donors.

        Returns:
            dict[str, bool]:
                Target to residential status in world 0.
        """
        grid = mixed_neighbors()
        targets = {'r1': asset(10.0, 0.0), 'r2': asset(150.0, 0.0)}
        result = imputer(seed=seed).impute_stage(
            inventory(targets), inventory(grid, order)
        )
        rows = result.log[result.log['round'] == ROUND_STATUS]
        return {
            row.asset_id: grid[row.donor_id][1]['OccupancyClass']
            in UNIT_COUNT_CLASSES
            for row in rows.itertuples(index=False)
        }

    def test_golden_status(self) -> None:
        """The round-one statuses of two targets are pinned."""
        assert self.statuses(SEED) == GOLDEN_STATUS

    def test_seed_changes_a_status(self) -> None:
        """A second seed changes at least one status."""
        assert any(self.statuses(s) != GOLDEN_STATUS for s in (OTHER_SEED, 11, 12))

    def test_reordering_changes_no_status(self) -> None:
        """Reordering the donors changes no status."""
        assert (
            self.statuses(SEED, ['m5', 'm0', 'm3', 'm1', 'm4', 'm2'])
            == GOLDEN_STATUS
        )

    def test_round_two_draws_from_the_status(self) -> None:
        """In every world, the copied class has the round-one status."""
        grid = mixed_neighbors()
        targets = {'r': asset(100.0, 0.0)}
        result = imputer().impute_stage(
            inventory(targets), inventory(grid), n_worlds=40
        )
        log = result.log
        one = log[log['round'] == ROUND_STATUS].set_index('world')
        two = log[log['round'] == ROUND_DONOR].set_index('world')
        seen = set()
        for world in range(40):
            status = (
                grid[one.loc[world, 'donor_id']][1]['OccupancyClass']
                in UNIT_COUNT_CLASSES
            )
            copied = two.loc[world, 'values']['OccupancyClass']
            assert (copied in UNIT_COUNT_CLASSES) == status
            seen.add(status)
        assert seen == {True, False}

    def test_known_status_replaces_round_one(self) -> None:
        """A status passed in skips round one."""
        targets = {'r': asset(100.0, 0.0)}
        result = imputer().impute_stage(
            inventory(targets),
            inventory(mixed_neighbors()),
            n_worlds=10,
            status={'r': False},
        )
        assert ROUND_STATUS not in set(result.log['round'])
        assert {v['OccupancyClass'] for v in result.log['values']} == {'COM1'}


def fallback_fixture() -> dict[str, tuple[list, dict[str, Any]]]:
    """
    Return three donors of block group A, all beyond the cap of the origin.

    Returns:
        dict[str, tuple[list, dict[str, Any]]]:
            ``b1`` to ``b3`` at 2.5 to 3.5 km, and ``x`` in block group B.
    """
    records = {
        f'b{i}': asset(
            2000.0 + 500.0 * i,
            0.0,
            OccupancyClass='RES1',
            NumberOfStories=1,
            NumberOfUnits=i,
            BLOCKGROUP_GEOID=BG_A,
        )
        for i in (1, 2, 3)
    }
    records['x'] = asset(
        -2500.0, 0.0, OccupancyClass='RES1', NumberOfStories=1, BLOCKGROUP_GEOID=BG_B
    )
    return records


class TestFallback:
    """With no donor within the cap, one donor of the block group is drawn."""

    def fallback(self, seed: int = SEED, order: list[str] | None = None) -> str:
        """
        Return the fallback donor of a target in block group A.

        Args:
            seed (int, optional):
                The seed.
            order (list[str], optional):
                Insertion order of the donors.

        Returns:
            str:
                The donor id.
        """
        targets = {
            't': asset(0.0, 0.0, OccupancyClass='RES1', BLOCKGROUP_GEOID=BG_A)
        }
        result = imputer(seed=seed).impute_stage(
            inventory(targets), inventory(fallback_fixture(), order)
        )
        row = result.log.iloc[0]
        assert row['outcome'] == OUTCOME_BLOCK_GROUP
        assert row['n_candidates'] == 3
        return row['donor_id']

    def test_golden_fallback(self) -> None:
        """The fallback donor is pinned."""
        assert self.fallback() == GOLDEN_FALLBACK

    def test_seed_changes_the_fallback(self) -> None:
        """Another seed picks another of the three block-group donors."""
        assert {self.fallback(s) for s in (OTHER_SEED, 11, 12, 13)} - {
            GOLDEN_FALLBACK
        }

    def test_reordering_changes_nothing(self) -> None:
        """Reordering the donors leaves the fallback unchanged."""
        assert self.fallback(order=['x', 'b3', 'b1', 'b2']) == GOLDEN_FALLBACK

    def test_known_class_filters_block_group_donors(self) -> None:
        """A known class filters the block-group donors."""
        records = fallback_fixture()
        records['com'] = asset(
            2200.0,
            0.0,
            OccupancyClass='COM1',
            NumberOfStories=1,
            BLOCKGROUP_GEOID=BG_A,
        )
        targets = {
            't': asset(0.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_A)
        }
        result = imputer().impute_stage(
            inventory(targets), inventory(records), n_worlds=5
        )
        assert set(result.log['donor_id']) == {'com'}

    def test_no_donor_leaves_features_missing(self) -> None:
        """Without a block-group donor the outcome is no_donor."""
        targets = {
            't': asset(0.0, 0.0, OccupancyClass='RES1', BLOCKGROUP_GEOID=BG_B)
        }
        records = {k: v for k, v in fallback_fixture().items() if k != 'x'}
        result = imputer().impute_stage(inventory(targets), inventory(records))
        assert result.log.iloc[0]['outcome'] == OUTCOME_NO_DONOR
        assert result.imputed.get_asset_ids() == []


class TestLocality:
    """Donors beyond the cap matter only through the block-group fallback."""

    def test_region_wide_pool(self) -> None:
        """A donor in another block group within the cap is drawn."""
        records = {
            'near': asset(
                1500.0,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=1,
                BLOCKGROUP_GEOID=BG_B,
            )
        }
        targets = {
            't': asset(0.0, 0.0, OccupancyClass='RES1', BLOCKGROUP_GEOID=BG_A)
        }
        result = imputer().impute_stage(inventory(targets), inventory(records))
        assert result.log.iloc[0]['donor_id'] == 'near'
        assert result.log.iloc[0]['outcome'] == OUTCOME_FEWER_THAN_K

    def test_distant_donors_change_no_draw_with_candidates(self) -> None:
        """Changing donors beyond 2 km leaves draws with candidates unchanged."""
        base = donor_grid()
        changed = dict(base)
        for i in range(10):
            changed[f'far{i}'] = asset(
                5000.0 + 10.0 * i,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=1,
                BLOCKGROUP_GEOID=BG_A,
            )
        targets = grid_targets()
        first = imputer().impute_stage(
            inventory(targets), inventory(base), n_worlds=6
        )
        second = imputer().impute_stage(
            inventory(targets), inventory(changed), n_worlds=6
        )
        assert donors_from_log(first) == donors_from_log(second)

    def test_distant_donors_outside_block_group_change_nothing(self) -> None:
        """Donors beyond 2 km and outside the block group change no draw."""
        targets = {
            't': asset(0.0, 0.0, OccupancyClass='RES1', BLOCKGROUP_GEOID=BG_A)
        }
        base = fallback_fixture()
        changed = dict(base)
        changed['y'] = asset(
            -2400.0,
            10.0,
            OccupancyClass='RES1',
            NumberOfStories=3,
            BLOCKGROUP_GEOID=BG_B,
        )
        first = imputer().impute_stage(
            inventory(targets), inventory(base), n_worlds=6
        )
        second = imputer().impute_stage(
            inventory(targets), inventory(changed), n_worlds=6
        )
        assert donors_from_log(first) == donors_from_log(second)

    def test_donor_just_past_the_cap(self) -> None:
        """A donor 2,001 m away is drawn only as the block group's fallback."""
        targets = {
            't': asset(0.0, 0.0, OccupancyClass='RES1', BLOCKGROUP_GEOID=BG_A)
        }
        other = {
            'p': asset(
                2001.0,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=1,
                BLOCKGROUP_GEOID=BG_B,
            )
        }
        result = imputer().impute_stage(inventory(targets), inventory(other))
        assert result.log.iloc[0]['outcome'] == OUTCOME_NO_DONOR
        same = {
            'p': asset(
                2001.0,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=1,
                BLOCKGROUP_GEOID=BG_A,
            )
        }
        result = imputer().impute_stage(inventory(targets), inventory(same))
        row = result.log.iloc[0]
        assert row['outcome'] == OUTCOME_BLOCK_GROUP
        assert row['donor_id'] == 'p'
        assert row['distance_m'] > 2000.0


class TestWorlds:
    """Worlds are independent and the copy is joint."""

    def test_world_two_is_the_same_in_any_run(self) -> None:
        """World 2 of a three-world run equals world 2 of five and draw(world=2)."""
        imp = imputer()
        targets = inventory(grid_targets())
        donors = inventory(donor_grid())
        three = donors_from_log(imp.impute_stage(targets, donors, n_worlds=3))
        five = donors_from_log(imp.impute_stage(targets, donors, n_worlds=5))
        pool = imp.pool(donors, targets)
        for target, (_, features) in grid_targets().items():
            assert three[(target, 2)] == five[(target, 2)]
            assert (
                imp.draw(pool, target, features, world=2).donor_id
                == three[(target, 2)]
            )

    def test_joint_copy(self) -> None:
        """In every world, every copied feature is the logged donor's."""
        grid = donor_grid()
        targets = {
            't': asset(220.0, 150.0),
            'u': asset(310.0, 90.0, NumberOfStories=2),
        }
        result = imputer().impute_stage(
            inventory(targets), inventory(grid), n_worlds=8
        )
        donors = donors_from_log(result)
        for target in targets:
            features = result.imputed.get_asset_features(target)[1]
            for world in range(8):
                donor = grid[donors[(target, world)]][1]
                for feature, value in features.items():
                    assert world_value(value, world) == donor[feature]
        assert 'NumberOfStories' not in result.imputed.get_asset_features('u')[1]

    def test_donor_with_worlds_raises(self) -> None:
        """A donor inventory with several worlds raises."""
        donors = inventory(donor_grid())
        donors.n_pw = 2
        with pytest.raises(ValueError, match='possible worlds'):
            imputer().impute_stage(inventory(grid_targets()), donors)


class TestStories:
    """The story term compares the counts of the stories key."""

    def distance(
        self,
        target: dict[str, Any],
        donor: dict[str, Any],
        **options: Any,
    ) -> float:
        """
        Return the distance between a target and a donor of equal geometry.

        Args:
            target (dict[str, Any]):
                The target's story features.
            donor (dict[str, Any]):
                The donor's story features.
            **options (Any):
                Constructor arguments of the imputer. Donors need not hold a count
                unless ``required`` is given.

        Returns:
            float:
                The donor distance.
        """
        records = {'d': asset(0.0, 0.0, OccupancyClass='RES1', **donor)}
        targets = {'t': asset(0.0, 0.0, OccupancyClass='RES1', **target)}
        options = {'required': ('OccupancyClass',), **options}
        result = imputer(**options).impute_stage(
            inventory(targets), inventory(records)
        )
        return float(result.log.iloc[0]['distance'])

    def test_default_reads_number_of_stories(self) -> None:
        """By default the term compares the counts of NumberOfStories."""
        records = {'d': asset(0.0, 0.0, OccupancyClass='RES1', NumberOfStories=1)}
        targets = {
            't': asset(0.0, 0.0, OccupancyClass='RES1', NumberOfStories=4),
        }
        result = imputer().impute_stage(inventory(targets), inventory(records))
        assert result.log.iloc[0]['distance'] == pytest.approx(3.0)

    def test_difference_of_counts(self) -> None:
        """Two counts enter the distance as their difference."""
        assert self.distance(
            {'NumberOfStories': 4}, {'NumberOfStories': 1}
        ) == pytest.approx(3.0)

    def test_a_side_without_a_count_adds_nothing(self) -> None:
        """A target or a donor without a count compares on location and area only."""
        assert self.distance({}, {'NumberOfStories': 5}) == pytest.approx(0.0)
        assert self.distance({'NumberOfStories': 5}, {}) == pytest.approx(0.0)

    @pytest.mark.parametrize('value', ['3', True, None, 'NA', math.nan, [2, 3]])
    def test_a_value_that_is_not_a_count_adds_nothing(self, value: Any) -> None:
        """A string, a bool, a missing value or a list is no count."""
        assert self.distance(
            {'NumberOfStories': 4}, {'NumberOfStories': value}
        ) == pytest.approx(0.0)

    def test_the_configured_key_is_read(self) -> None:
        """With another stories key, the term reads that feature."""
        target = {'NumberOfStories': 4, 'Floors': 2}
        donor = {'NumberOfStories': 1, 'Floors': 3}
        assert self.distance(target, donor, stories_key='Floors') == pytest.approx(
            1.0
        )
        assert self.distance(target, donor, stories_key=None) == pytest.approx(0.0)

    def test_a_targets_own_count_is_kept(self) -> None:
        """A target holding a count receives only the features it is missing."""
        records = {
            'd': asset(
                10.0,
                0.0,
                OccupancyClass='RES3A',
                NumberOfStories=2,
                NumberOfUnits=4,
            )
        }
        targets = {'t': asset(0.0, 0.0, NumberOfStories=3)}
        result = imputer().impute_stage(inventory(targets), inventory(records))
        assert result.imputed.get_asset_features('t')[1] == {
            'OccupancyClass': 'RES3A',
            'NumberOfUnits': 4,
        }
        merged = imputer().impute(inventory({**records, **targets}))
        assert merged.get_asset_features('t')[1]['NumberOfStories'] == 3


class TestStoriesPerWorld:
    """A story count held per world enters the distance of its world."""

    def inputs(
        self, near_class: str = 'COM1', stories: tuple[int, int] = (1, 9)
    ) -> tuple[AssetInventory, AssetInventory]:
        """
        Return a near one-story donor, a far nine-story donor and a target.

        Args:
            near_class (str, optional):
                The near donor's occupancy class. Defaults to COM1, the target's.
            stories (tuple[int, int], optional):
                The target's story count in worlds 0 and 1. Defaults to (1, 9).

        Returns:
            tuple[AssetInventory, AssetInventory]:
                The donors, and the target in two worlds. With ``k=1`` and a story
                weight of 100, the target's count picks the donor.
        """
        donors = {
            'near': asset(
                0.0,
                10.0,
                OccupancyClass=near_class,
                NumberOfStories=1,
                NumberOfUnits=1,
            ),
            'far': asset(
                0.0, 200.0, OccupancyClass='COM1', NumberOfStories=9, NumberOfUnits=9
            ),
        }
        targets = inventory(
            {
                't': asset(
                    0.0, 0.0, OccupancyClass='COM1', NumberOfStories=list(stories)
                )
            }
        )
        targets.n_pw = 2
        return inventory(donors), targets

    def test_a_targets_count_per_world(self) -> None:
        """Each world of a run picks the donor a run of that world alone picks."""
        donors, targets = self.inputs()
        imp = imputer(k=1, stories_weight=100.0)
        both = donors_from_log(imp.impute_stage(targets, donors, n_worlds=2))
        alone = donors_from_log(
            imp.impute_stage(targets.get_world_realization(1), donors)
        )
        assert both == {('t', 0): 'near', ('t', 1): 'far'}
        assert alone[('t', 0)] == both[('t', 1)]

    def test_draw_reads_the_worlds_count(self) -> None:
        """A draw in world 1 picks the donor the stage picks there."""
        donors, targets = self.inputs()
        imp = imputer(k=1, stories_weight=100.0)
        pool = imp.pool(donors, targets)
        features = {'OccupancyClass': 'COM1', 'NumberOfStories': 9}
        assert imp.draw(pool, 't', features, world=1).donor_id == 'far'

    def test_nearest_admitted_reads_the_worlds_count(self) -> None:
        """The nearest-first scan in world 1 ranks by that world's count."""
        donors, targets = self.inputs()
        imp = imputer(k=1, stories_weight=100.0)
        pool = imp.pool(donors, targets)
        features = {'OccupancyClass': 'COM1', 'NumberOfStories': 9}
        scan = imp.nearest_admitted(
            pool, 't', features, ['NumberOfUnits'], world=1, admit=lambda _: True
        )
        assert scan.donor_id == 'far'

    def test_status_probability_reads_world_zeros_count(self) -> None:
        """The chance of the status reads the target's count in world 0."""
        donors, targets = self.inputs(near_class='RES1', stories=(9, 1))
        imp = imputer(k=1, stories_weight=100.0)
        pool = imp.pool(donors, targets)
        assert imp.status_probability(pool, ['t']) == {'t': 0.0}

    def test_a_fed_back_donors_count_per_world(self) -> None:
        """A later batch's distance reads the count an earlier batch copied."""
        observed = {
            f'd{i:02d}': asset(
                30.0 * i,
                300.0,
                OccupancyClass='COM1',
                NumberOfStories=1 + 7 * i % 9,
                NumberOfUnits=1,
            )
            for i in range(20)
        }
        # Odd targets hold a count and compare; even ones lack it, receive one per
        # world and donate. Equal footprints leave no area term.
        records = {
            f't{i:02d}': asset(
                15.0 * i,
                0.0,
                OccupancyClass='COM1',
                **(
                    {'NumberOfStories': 1 + i % 5} if i % 2 else {'NumberOfUnits': 1}
                ),
            )
            for i in range(40)
        }
        freeze = StageFreeze()
        freeze.freeze('observed', inventory(observed))
        targets = inventory(records)
        targets.n_pw = 2
        result = imputer(
            n_batches=2, location_scale_m=100.0, stories_weight=1.0
        ).impute_stage(targets, freeze=freeze, donor_stage='observed', n_worlds=2)
        first = freeze.read(result.batches[0]).inventory
        copied = {
            asset_id: imputed.features['NumberOfStories']
            for asset_id, imputed in first.items()
            if isinstance(imputed.features.get('NumberOfStories'), list)
        }
        log = result.log
        rows = log[
            (log['round'] == ROUND_DONOR)
            & log['donor_id'].isin(list(copied))
            & ~log['asset_id'].isin(list(first))
        ]
        pairs = [
            (row, records[row.asset_id][1].get('NumberOfStories'))
            for row in rows.itertuples(index=False)
        ]
        pairs = [(row, own) for row, own in pairs if own is not None]
        assert any(own != copied[row.donor_id][row.world] for row, own in pairs)
        for row, own in pairs:
            assert row.distance == pytest.approx(
                math.hypot(
                    row.distance_m / 100.0, own - copied[row.donor_id][row.world]
                )
            )


class TestEntryPoint:
    """draw is the stage's path, one asset at a time."""

    def test_draw_reproduces_the_stage(self) -> None:
        """At iteration 0, draw gives the stage's donor and values."""
        imp = imputer()
        targets = inventory(grid_targets())
        donors = inventory(donor_grid())
        result = imp.impute_stage(targets, donors, n_worlds=3)
        log = result.log[result.log['round'] == ROUND_DONOR]
        for row in log.itertuples(index=False):
            features = grid_targets()[row.asset_id][1]
            draw = imp.draw(result.pool, row.asset_id, features, world=row.world)
            assert draw.donor_id == row.donor_id
            assert dict(draw.values) == row.values

    def test_redraw_limits_the_copy_and_does_not_filter(self) -> None:
        """The redraw argument names the copied features, which count as missing."""
        imp = imputer()
        donors = inventory(donor_grid())
        features = {
            'OccupancyClass': 'RES1',
            'NumberOfStories': 1,
            'NumberOfUnits': 1,
        }
        targets = inventory({'t': asset(120.0, 130.0, **features)})
        pool = imp.pool(donors, targets)
        only = imp.draw(pool, 't', features, ['NumberOfStories'], world=0)
        assert set(only.values) == {'NumberOfStories'}
        classes = {
            imp.draw(
                pool, 't', features, ['OccupancyClass'], world=w, status=True
            ).values['OccupancyClass']
            for w in range(30)
        }
        assert classes == {'RES1', 'RES3A'}

    def test_zero_area_asset(self) -> None:
        """A zero-area asset gives no_area and copies nothing."""
        flat = [[LON0, LAT0], [LON0 + 0.001, LAT0], [LON0, LAT0]]
        targets = AssetInventory()
        targets.add_asset('z', Asset('z', flat, {}))
        imp = imputer()
        result = imp.impute_stage(targets, inventory(donor_grid()))
        assert result.log.iloc[0]['outcome'] == OUTCOME_NO_AREA
        assert imp.draw(result.pool, 'z', {}, world=0).outcome == OUTCOME_NO_AREA

    def test_unknown_asset_raises(self) -> None:
        """An asset the pool lacks raises LookupError."""
        imp = imputer()
        pool = imp.pool(inventory(donor_grid()))
        with pytest.raises(LookupError):
            imp.draw(pool, 'nowhere', {}, world=0)


def labeled_donors() -> dict[str, tuple[list, dict[str, Any]]]:
    """
    Return eight residential donors on a line, labeled by their units.

    Returns:
        dict[str, tuple[list, dict[str, Any]]]:
            ``e0`` to ``e7`` at 50 m steps; ``NumberOfUnits`` is the index.
    """
    return {
        f'e{i}': asset(
            50.0 * (i + 1),
            0.0,
            OccupancyClass='RES1',
            NumberOfStories=1,
            NumberOfUnits=i,
        )
        for i in range(8)
    }


class TestRepairs:
    """With admit, draw makes the attempts and then the nearest-first scan."""

    def setup_draw(self) -> tuple[ObservedDonorImputer, Any, dict[str, Any]]:
        """
        Build an imputer, a pool and a target on the labeled donors.

        Returns:
            tuple[ObservedDonorImputer, Any, dict[str, Any]]:
                The imputer, the pool and the target's features.
        """
        imp = imputer()
        features = {'OccupancyClass': 'RES1', 'NumberOfStories': 1}
        targets = inventory({'t': asset(0.0, 0.0, **features)})
        return imp, imp.pool(inventory(labeled_donors()), targets), features

    def test_accept_all_is_the_first_pick(self) -> None:
        """An admit that accepts everything returns the first pick, attempts 1."""
        imp, pool, features = self.setup_draw()
        plain = imp.draw(pool, 't', features, world=0, iteration=3)
        repaired = imp.draw(
            pool, 't', features, world=0, iteration=3, admit=lambda _values: True
        )
        assert repaired.donor_id == plain.donor_id
        assert repaired.attempts == 1
        assert repaired.iteration == 3

    def test_attempts_never_retry_and_use_consecutive_iterations(self) -> None:
        """Rejected donors are not retried; attempt a runs at iteration + a - 1."""
        imp, pool, features = self.setup_draw()
        seen: list[int] = []

        def admit(values: dict[str, Any]) -> bool:
            """
            Accept the fourth donor seen.

            Args:
                values (dict[str, Any]):
                    The candidate's values.

            Returns:
                bool:
                    Whether this is the fourth call.
            """
            seen.append(values['NumberOfUnits'])
            return len(seen) == 4

        draw = imp.draw(
            pool, 't', features, ['NumberOfUnits'], world=1, iteration=2, admit=admit
        )
        assert len(set(seen)) == 4
        assert draw.attempts == 4
        assert draw.iteration == 5
        assert draw.values == {'NumberOfUnits': seen[-1]}

    def test_nearest_admitted_after_the_attempts(self) -> None:
        """Rejecting the nearest donors gives the scan's nearest admitted one."""
        _, _, features = self.setup_draw()
        imp = imputer(k=1)
        pool = imp.pool(
            inventory(labeled_donors()),
            inventory({'t': asset(0.0, 0.0, **features)}),
        )
        draw = imp.draw(
            pool,
            't',
            features,
            ['NumberOfUnits'],
            world=0,
            admit=lambda v: v['NumberOfUnits'] >= 6,
        )
        assert draw.outcome == OUTCOME_NEAREST_ADMITTED
        assert draw.donor_id == 'e6'
        assert draw.attempts == imp.max_attempts

    def test_exhausted(self) -> None:
        """Rejecting every donor, or spending the attempts without the scan, gives exhausted."""
        imp, pool, features = self.setup_draw()
        none = imp.draw(
            pool,
            't',
            features,
            ['NumberOfUnits'],
            world=0,
            admit=lambda _values: False,
        )
        assert none.outcome == OUTCOME_EXHAUSTED
        assert none.donor_id is None
        assert dict(none.values) == {}
        off = imp.draw(
            pool,
            't',
            features,
            ['NumberOfUnits'],
            world=0,
            admit=lambda v: v['NumberOfUnits'] == 7,
            scan_nearest=False,
        )
        assert off.outcome == OUTCOME_EXHAUSTED

    def test_attempts_reach_the_fallback(self) -> None:
        """The attempts apply to the block group's donors too."""
        imp = imputer()
        features = {
            'OccupancyClass': 'RES1',
            BLOCK_GROUP_KEY: BG_A,
            'NumberOfStories': 1,
        }
        targets = inventory({'t': asset(0.0, 0.0, **features)})
        pool = imp.pool(inventory(fallback_fixture()), targets)
        first = imp.draw(pool, 't', features, ['NumberOfUnits'], world=0)
        draw = imp.draw(
            pool,
            't',
            features,
            ['NumberOfUnits'],
            world=0,
            admit=lambda v: v['NumberOfUnits'] != first.values['NumberOfUnits'],
        )
        assert draw.outcome == OUTCOME_BLOCK_GROUP
        assert draw.attempts == 2
        assert draw.donor_id != first.donor_id


def implausible(asset_id: Any, world: int, features: dict[str, Any]) -> bool:  # noqa: ARG001
    """
    Reject a RES1 building of three or more stories.

    Args:
        asset_id (Any):
            The asset.
        world (int):
            The world.
        features (dict[str, Any]):
            Its features after the copy.

    Returns:
        bool:
            Whether the pair is plausible.
    """
    return not (
        features.get('OccupancyClass') == 'RES1' and features['NumberOfStories'] >= 3
    )


class TestRepairSplit:
    """Validation re-draws only the values this run copied."""

    def donors(self) -> dict[str, tuple[list, dict[str, Any]]]:
        """
        Return donors whose nearest one carries an implausible story count.

        Returns:
            dict[str, tuple[list, dict[str, Any]]]:
                ``tall`` nearest with three stories, then two-story donors.
        """
        records = {
            'tall': asset(
                10.0, 0.0, OccupancyClass='RES1', NumberOfStories=3, NumberOfUnits=1
            )
        }
        for i in range(6):
            records[f'ok{i}'] = asset(
                400.0 + 10.0 * i,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=2,
                NumberOfUnits=1,
            )
        return records

    def test_input_class_copied_stories(self) -> None:
        """A known class stays and the copied story count is re-drawn."""
        result = imputer(k=1).impute_stage(
            inventory(
                {'t': asset(0.0, 0.0, OccupancyClass='RES1', NumberOfUnits=1)}
            ),
            inventory(self.donors()),
            validate=implausible,
        )
        repair = result.log[result.log['round'] == ROUND_REPAIR].iloc[0]
        assert set(repair['values']) == {'NumberOfStories'}
        features = result.imputed.get_asset_features('t')[1]
        assert features == {'NumberOfStories': 2}

    def test_copied_class_input_stories(self) -> None:
        """A copied class against an input story count is re-drawn alone."""
        records = {
            'nonres': asset(
                10.0, 0.0, OccupancyClass='RES1', NumberOfStories=1, NumberOfUnits=1
            ),
        }
        for i in range(6):
            records[f'r{i}'] = asset(
                400.0 + 10.0 * i,
                0.0,
                OccupancyClass='RES3A',
                NumberOfStories=1,
                NumberOfUnits=1,
            )
        imp = imputer(k=1)
        result = imp.impute_stage(
            inventory({'t': asset(0.0, 0.0, NumberOfStories=4, NumberOfUnits=1)}),
            inventory(records),
            validate=implausible,
            status={'t': True},
        )
        repair = result.log[result.log['round'] == ROUND_REPAIR].iloc[0]
        assert set(repair['values']) == {'OccupancyClass'}
        assert result.imputed.get_asset_features('t')[1] == {
            'OccupancyClass': 'RES3A'
        }

    def test_still_rejected_keeps_first_values(self) -> None:
        """A copy rejected after every attempt keeps its first values and is logged exhausted."""
        imp = imputer(k=1)
        records = {
            f'tall{i}': asset(
                10.0 * (i + 1),
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=3,
                NumberOfUnits=1,
            )
            for i in range(8)
        }
        result = imp.impute_stage(
            inventory(
                {'t': asset(0.0, 0.0, OccupancyClass='RES1', NumberOfUnits=1)}
            ),
            inventory(records),
            validate=implausible,
        )
        repair = result.log[result.log['round'] == ROUND_REPAIR].iloc[0]
        assert repair['outcome'] == OUTCOME_EXHAUSTED
        assert repair['attempts'] == imp.max_attempts
        assert result.imputed.get_asset_features('t')[1] == {'NumberOfStories': 3}

    def test_no_validation_without_a_copied_class_or_story(self) -> None:
        """A target that copied neither is not validated."""
        calls: list[Any] = []

        def spy(asset_id: Any, world: int, features: dict[str, Any]) -> bool:  # noqa: ARG001
            """
            Record the call.

            Args:
                asset_id (Any):
                    The asset.
                world (int):
                    The world.
                features (dict[str, Any]):
                    Its features.

            Returns:
                bool:
                    Always ``True``.
            """
            calls.append(asset_id)
            return True

        imputer().impute_stage(
            inventory(
                {'t': asset(0.0, 0.0, OccupancyClass='RES1', NumberOfStories=1)}
            ),
            inventory(self.donors()),
            validate=spy,
        )
        assert calls == []


class TestNearestAdmitted:
    """The scan covers the donors within the cap, nearest first."""

    def test_nearest_admitted_within_the_cap(self) -> None:
        """The scan returns the nearest admitted donor and copies only redraw."""
        imp = imputer()
        features = {'OccupancyClass': 'RES1', 'NumberOfStories': 1}
        records = labeled_donors()
        records['far'] = asset(
            2500.0,
            0.0,
            OccupancyClass='RES1',
            NumberOfStories=1,
            NumberOfUnits=99,
            BLOCKGROUP_GEOID=BG_A,
        )
        targets = inventory(
            {'t': asset(0.0, 0.0, BLOCKGROUP_GEOID=BG_A, **features)}
        )
        pool = imp.pool(inventory(records), targets)
        draw = imp.nearest_admitted(
            pool,
            't',
            features,
            ['NumberOfUnits'],
            world=0,
            admit=lambda v: v['NumberOfUnits'] >= 3,
        )
        assert (draw.outcome, draw.donor_id, dict(draw.values)) == (
            OUTCOME_NEAREST_ADMITTED,
            'e3',
            {'NumberOfUnits': 3},
        )
        none = imp.nearest_admitted(
            pool,
            't',
            features,
            ['NumberOfUnits'],
            world=0,
            admit=lambda v: v['NumberOfUnits'] == 99,
        )
        assert none.outcome == OUTCOME_NONE_ADMITTED

    def test_protected_overlap_raises(self) -> None:
        """A feature both re-drawn and protected raises in the scan and draw."""
        imp = imputer()
        features = {'OccupancyClass': 'RES1', 'NumberOfStories': 1}
        pool = imp.pool(
            inventory(labeled_donors()),
            inventory({'t': asset(0.0, 0.0, **features)}),
        )
        with pytest.raises(ValueError, match='protected'):
            imp.nearest_admitted(
                pool,
                't',
                features,
                ['NumberOfStories'],
                world=0,
                admit=lambda _values: True,
                protected=['NumberOfStories'],
            )
        with pytest.raises(ValueError, match='protected'):
            imp.draw(
                pool,
                't',
                features,
                ['NumberOfStories'],
                world=0,
                protected=['NumberOfStories'],
            )


class TestParameters:
    """Each constructor parameter changes the draw as its docstring states."""

    def test_k_one_takes_the_nearest(self) -> None:
        """With k 1 every world takes the nearest donor."""
        result = imputer(k=1).impute_stage(
            inventory({'t': asset(0.0, 0.0, OccupancyClass='RES1')}),
            inventory(labeled_donors()),
            n_worlds=6,
        )
        assert set(result.log['donor_id']) == {'e0'}

    def test_cap(self) -> None:
        """A smaller cap leaves out donors beyond it."""
        result = imputer(cap_m=120.0).impute_stage(
            inventory({'t': asset(0.0, 0.0, OccupancyClass='RES1')}),
            inventory(labeled_donors()),
            n_worlds=6,
        )
        assert set(result.log['donor_id']) <= {'e0', 'e1'}
        assert set(result.log['outcome']) == {OUTCOME_FEWER_THAN_K}

    def test_location_scale_and_area_weight(self) -> None:
        """The location scale and area weight trade distance against area."""
        records = {
            'near_small': asset(
                50.0, 0.0, 5.0, OccupancyClass='RES1', NumberOfStories=1
            ),
            'far_same': asset(
                300.0, 0.0, 10.0, OccupancyClass='RES1', NumberOfStories=1
            ),
        }
        targets = inventory({'t': asset(0.0, 0.0, 10.0, OccupancyClass='RES1')})
        near = imputer(k=1).impute_stage(targets, inventory(records))
        assert near.log.iloc[0]['donor_id'] == 'near_small'
        area = imputer(k=1, area_weight=100.0).impute_stage(
            targets, inventory(records)
        )
        assert area.log.iloc[0]['donor_id'] == 'far_same'
        scale = imputer(k=1, location_scale_m=10000.0).impute_stage(
            targets, inventory(records)
        )
        assert scale.log.iloc[0]['donor_id'] == 'far_same'

    def test_stories_weight(self) -> None:
        """The stories weight scales the story term."""
        records = {
            'd': asset(
                0.0,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=1,
            )
        }
        targets = inventory(
            {'t': asset(0.0, 0.0, OccupancyClass='RES1', NumberOfStories=3)}
        )
        result = imputer(stories_weight=0.25).impute_stage(
            targets, inventory(records)
        )
        assert result.log.iloc[0]['distance'] == pytest.approx(1.0)


class TestAttributesAndImporter:
    """Public attributes and Importer discovery."""

    def test_attributes(self) -> None:
        """The attributes keep the arguments, and k and the stories key have their defaults."""
        default = ObservedDonorImputer()
        assert (default.k, default.stories_key) == (20, 'NumberOfStories')
        imp = ObservedDonorImputer(
            seed=1, stories_key='Floors', stories_weight=2.0, max_attempts=3
        )
        assert (imp.stories_key, imp.stories_weight, imp.max_attempts, imp.seed) == (
            'Floors',
            2.0,
            3,
            1,
        )

    def test_seed_is_drawn_and_recorded(self) -> None:
        """Without a seed, one is drawn and recorded."""
        imp = ObservedDonorImputer()
        assert isinstance(imp.seed, int)
        result = imp.impute_stage(
            inventory({'t': asset(0.0, 0.0, OccupancyClass='RES1')}),
            inventory(labeled_donors()),
        )
        assert result.seed == imp.seed

    def test_importer_resolves_the_class(self) -> None:
        """Importer builds and resolves ObservedDonorImputer."""
        assert Importer().get_class('ObservedDonorImputer') is ObservedDonorImputer

    def test_draw_carries_its_interface_fields(self) -> None:
        """DonorDraw carries the donor, values, outcome and attempt count."""
        assert {'values', 'outcome', 'attempts', 'donor_id'} <= set(
            DonorDraw._fields
        )

    def test_impute_fills_the_inventory(self) -> None:
        """The impute method fills missing values in a copy and leaves the input unchanged."""
        records = labeled_donors()
        records['t'] = asset(0.0, 0.0, OccupancyClass='RES1', NumberOfStories=1)
        inv = inventory(records)
        out = imputer().impute(inv, n_worlds=2)
        assert 'NumberOfUnits' not in inv.get_asset_features('t')[1]
        assert 'NumberOfUnits' in out.get_asset_features('t')[1]
        assert out.n_pw == 2


def block_group_mix() -> dict[str, tuple[list, dict[str, Any]]]:
    """
    Return three donors of block group A beyond the cap of the origin.

    Returns:
        dict[str, tuple[list, dict[str, Any]]]:
            ``a_r1`` and ``b_r2``, residential, and ``c_n``, non-residential, in
            ascending id order.
    """
    return {
        'a_r1': asset(
            3000.0,
            0.0,
            OccupancyClass='RES1',
            NumberOfStories=1,
            BLOCKGROUP_GEOID=BG_A,
        ),
        'b_r2': asset(
            3100.0,
            0.0,
            OccupancyClass='RES1',
            NumberOfStories=1,
            BLOCKGROUP_GEOID=BG_A,
        ),
        'c_n': asset(
            3200.0,
            0.0,
            OccupancyClass='COM1',
            NumberOfStories=1,
            BLOCKGROUP_GEOID=BG_A,
        ),
    }


class TestFallbackRounds:
    """The residential round and the donor round fall back independently."""

    def test_round_two_fallback_is_not_tied_to_round_one(self) -> None:
        """After a residential fallback, both residential donors are drawn evenly."""
        targets = {
            f't{i:04d}': asset(0.0, 0.0, BLOCKGROUP_GEOID=BG_A) for i in range(1200)
        }
        result = imputer().impute_stage(
            inventory(targets), inventory(block_group_mix())
        )
        log = result.log
        one = log[log['round'] == ROUND_STATUS]
        assert set(one['outcome']) == {OUTCOME_BLOCK_GROUP}
        two = log[log['round'] == ROUND_DONOR]
        counts = two['donor_id'].value_counts()
        residential = counts['a_r1'] + counts['b_r2']
        # A uniform shared with round one would split them about 75 to 25.
        assert abs(counts['a_r1'] - counts['b_r2']) < 0.1 * residential

    def statuses(self, seed: int, order: list[str] | None = None) -> dict[str, str]:
        """
        Return the residential-round fallback donors of three targets.

        Args:
            seed (int):
                The seed.
            order (list[str], optional):
                Insertion order of the donors.

        Returns:
            dict[str, str]:
                Target to round-one donor id.
        """
        targets = {
            f'f{i}': asset(0.0, 10.0 * i, BLOCKGROUP_GEOID=BG_A) for i in range(3)
        }
        result = imputer(seed=seed).impute_stage(
            inventory(targets), inventory(block_group_mix(), order)
        )
        rows = result.log[result.log['round'] == ROUND_STATUS]
        return dict(zip(rows['asset_id'], rows['donor_id']))

    def test_golden_residential_fallback(self) -> None:
        """The residential round's fallback donors are pinned."""
        assert self.statuses(SEED) == GOLDEN_RESIDENTIAL_FALLBACK

    def test_seed_changes_a_residential_fallback(self) -> None:
        """A second seed changes at least one."""
        assert self.statuses(OTHER_SEED) != GOLDEN_RESIDENTIAL_FALLBACK

    def test_reordering_changes_no_residential_fallback(self) -> None:
        """Reordering the donors changes none."""
        assert (
            self.statuses(SEED, ['c_n', 'a_r1', 'b_r2'])
            == GOLDEN_RESIDENTIAL_FALLBACK
        )

    def test_repair_reaches_the_block_group_past_the_cap(self) -> None:
        """Once the donors within the cap are rejected, a repair falls back."""
        records = {
            'near': asset(
                100.0,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=3,
                NumberOfUnits=1,
                BLOCKGROUP_GEOID=BG_A,
            ),
            'far': asset(
                5000.0,
                0.0,
                OccupancyClass='RES1',
                NumberOfStories=1,
                NumberOfUnits=2,
                BLOCKGROUP_GEOID=BG_A,
            ),
        }
        targets = {
            't': asset(
                0.0,
                0.0,
                OccupancyClass='RES1',
                NumberOfUnits=1,
                BLOCKGROUP_GEOID=BG_A,
            )
        }
        result = imputer().impute_stage(
            inventory(targets), inventory(records), validate=implausible
        )
        repair = result.log[result.log['round'] == ROUND_REPAIR].iloc[0]
        assert (repair['donor_id'], repair['outcome']) == (
            'far',
            OUTCOME_BLOCK_GROUP,
        )
        assert repair['distance_m'] > 2000.0
        assert result.imputed.get_asset_features('t')[1] == {'NumberOfStories': 1}


class TestPurposePrefix:
    """A caller's purpose prefix gives draws distinct from the stage's."""

    def test_prefix_changes_the_uniforms(self) -> None:
        """Without a prefix draw equals the stage; two prefixes give other picks."""
        imp = imputer()
        targets = inventory(grid_targets())
        pool = imp.pool(inventory(donor_grid()), targets)
        features = grid_targets()['t3'][1]
        picks = {
            prefix: [
                imp.draw(
                    pool, 't3', features, world=w, purpose_prefix=prefix
                ).donor_id
                for w in range(40)
            ]
            for prefix in (None, 'first', 'second')
        }
        stage = donors_from_log(
            imp.impute_stage(targets, inventory(donor_grid()), n_worlds=40)
        )
        assert picks[None] == [stage[('t3', w)] for w in range(40)]
        assert picks['first'] != picks[None]
        assert picks['second'] != picks['first']
        same = sum(a == b for a, b in zip(picks['first'], picks['second']))
        assert same < 40

    def test_prefix_reaches_the_residential_round(self) -> None:
        """A prefix also changes the residential round's picks."""
        imp = imputer()
        targets = inventory(grid_targets())
        pool = imp.pool(inventory(donor_grid()), targets)
        features = grid_targets()['t3'][1]
        rounds = {
            prefix: [
                imp._draw(
                    pool, 't3', features, None, world=w, purpose_prefix=prefix
                ).round_one.donor_id
                for w in range(40)
            ]
            for prefix in (None, 'first')
        }
        assert rounds['first'] != rounds[None]


def material_neighbors() -> dict[str, tuple[list, dict[str, Any]]]:
    """
    Return residential donors of two materials mixed around the origin.

    Returns:
        dict[str, tuple[list, dict[str, Any]]]:
            Six RES1 donors, ``w0`` to ``w5``, alternating ``Material`` ``'C'`` and
            ``'W'``.
    """
    return {
        f'w{i}': asset(
            40.0 * i,
            20.0,
            OccupancyClass='RES1',
            Material='W' if i % 2 else 'C',
            NumberOfStories=1,
            NumberOfUnits=i,
        )
        for i in range(6)
    }


def material_imputer(**kwargs: Any) -> ObservedDonorImputer:
    """
    Build an imputer whose status is a wood material.

    Args:
        **kwargs (Any):
            Further constructor arguments, or overrides of these.

    Returns:
        ObservedDonorImputer:
            The imputer, requiring ``Material`` of every donor.
    """
    options = {
        'required': ('OccupancyClass', 'NumberOfStories', 'Material'),
        'status_key': 'Material',
        'status_values': ('W',),
        'status_name': 'wood',
    }
    options.update(kwargs)
    return imputer(**options)


class TestStatusPartition:
    """The status the rounds partition donors by is a constructor argument."""

    def test_defaults_are_the_residential_partition(self) -> None:
        """The defaults are the residential labels and values."""
        assert PURPOSE_RESIDENTIAL == 'observed_donor_residential'
        assert PURPOSE_RESIDENTIAL_FALLBACK == 'observed_donor_residential_fallback'
        imp = imputer()
        assert (
            imp.status_key,
            imp.status_values,
            imp.status_name,
            imp.validated,
        ) == (
            'OccupancyClass',
            UNIT_COUNT_CLASSES,
            'residential',
            ('OccupancyClass', 'NumberOfStories'),
        )

    def test_status_name_labels_the_round(self) -> None:
        """Another status name gives the round other uniforms."""
        targets = inventory({'r': asset(100.0, 0.0)})

        def round_one(imp: ObservedDonorImputer) -> list[Any]:
            """
            Return the round-one donors of ``r`` in 40 worlds.

            Args:
                imp (ObservedDonorImputer):
                    The imputer.

            Returns:
                list[Any]:
                    The donors in world order.
            """
            log = imp.impute_stage(
                targets, inventory(mixed_neighbors()), n_worlds=40
            ).log
            return list(log[log['round'] == ROUND_STATUS]['donor_id'])

        assert round_one(imputer(status_name='housing')) != round_one(imputer())

    def test_round_one_reads_the_status_key(self) -> None:
        """A target with a class but no material runs round one on material."""
        grid = material_neighbors()
        targets = inventory({'t': asset(100.0, 0.0, OccupancyClass='RES1')})
        assert ROUND_STATUS not in set(
            imputer()
            .impute_stage(targets, inventory(grid), n_worlds=10)
            .log['round']
        )
        log = (
            material_imputer()
            .impute_stage(targets, inventory(grid), n_worlds=40)
            .log
        )
        one = log[log['round'] == ROUND_STATUS].set_index('world')
        two = log[log['round'] == ROUND_DONOR].set_index('world')
        seen = set()
        for world in range(40):
            material = grid[one.loc[world, 'donor_id']][1]['Material']
            assert grid[two.loc[world, 'donor_id']][1]['Material'] == material
            seen.add(material)
        assert seen == {'C', 'W'}

    def test_known_value_filters_the_donors(self) -> None:
        """A target's own material is its status, and donors match it."""
        grid = material_neighbors()
        for material in ('C', 'W'):
            targets = inventory({'t': asset(100.0, 0.0, Material=material)})
            log = (
                material_imputer()
                .impute_stage(targets, inventory(grid), n_worlds=20)
                .log
            )
            assert ROUND_STATUS not in set(log['round'])
            assert {grid[d][1]['Material'] for d in log['donor_id']} == {material}

    def test_given_status_and_scan_follow_the_values(self) -> None:
        """A given status and the scan's default status use the configured values."""
        grid = material_neighbors()
        targets = inventory({'t': asset(100.0, 0.0, Material='C')})
        imp = material_imputer()
        log = imp.impute_stage(
            targets, inventory(grid), n_worlds=20, status={'t': True}
        ).log
        assert {grid[d][1]['Material'] for d in log['donor_id']} == {'W'}
        pool = imp.pool(inventory(grid), targets)
        draw = imp.nearest_admitted(
            pool,
            't',
            {'Material': 'C'},
            ['NumberOfUnits'],
            world=0,
            admit=lambda values: True,  # noqa: ARG005
        )
        assert draw.outcome == OUTCOME_NEAREST_ADMITTED
        assert grid[draw.donor_id][1]['Material'] == 'C'

    @pytest.mark.parametrize(
        'arguments',
        [
            {'status_key': 'Material'},
            {'status_values': 'RES1'},
            {'status_values': ()},
            {'status_name': ''},
            {'status_name': 'fallback'},
            {'status_name': 3},
            {'validated': 'NumberOfUnits'},
        ],
    )
    def test_invalid_status_arguments_raise(self, arguments: dict[str, Any]) -> None:
        """An unlisted key, bad values, a bad name or bad validated features raise."""
        with pytest.raises(ValueError, match='status_|validated'):
            imputer(**arguments)

    def test_default_validated_is_the_class_and_stories(self) -> None:
        """By default the class and the story count trigger validation."""
        assert imputer().validated == ('OccupancyClass', 'NumberOfStories')

    def test_default_validated_without_stories(self) -> None:
        """Without the story count among the features, only the filled ones trigger."""
        filled = imputer(features=('OccupancyClass', 'NumberOfUnits'))
        assert filled.validated == ('OccupancyClass',)
        assert imputer(features=('NumberOfUnits',)).validated == ()

    def test_explicit_validated_outside_features_raises(self) -> None:
        """An explicit validated feature the imputer does not fill raises."""
        with pytest.raises(ValueError, match='validated'):
            imputer(validated=('Material',))

    def test_status_values_from_a_set(self) -> None:
        """A set of status values is held as a tuple and draws the same."""
        grid = material_neighbors()
        targets = inventory({'t': asset(100.0, 0.0, OccupancyClass='RES1')})
        logs = []
        for values in (('W',), frozenset({'W'})):
            imp = material_imputer(status_values=values)
            assert imp.status_values == ('W',)
            logs.append(imp.impute_stage(targets, inventory(grid), n_worlds=10).log)
        assert logs[0].equals(logs[1])

    def test_pool_of_another_imputer_raises(self) -> None:
        """A pool another imputer built is refused by draw and the scan."""
        grid = material_neighbors()
        targets = inventory({'t': asset(100.0, 0.0, Material='C')})
        pool = imputer().pool(inventory(grid), targets)
        imp = material_imputer()
        with pytest.raises(ValueError, match='pool was built'):
            imp.draw(pool, 't', {'Material': 'C'}, world=0)
        with pytest.raises(ValueError, match='pool was built'):
            imp.nearest_admitted(
                pool,
                't',
                {'Material': 'C'},
                ['NumberOfUnits'],
                world=0,
                admit=lambda values: True,  # noqa: ARG005
            )

    def test_pool_of_an_equal_imputer_raises(self) -> None:
        """A pool built by another imputer of the same settings is refused too."""
        targets = inventory({'t': asset(100.0, 0.0, Material='C')})
        pool = material_imputer().pool(inventory(material_neighbors()), targets)
        assert pool.imputer.draw(pool, 't', {'Material': 'C'}, world=0).donor_id
        with pytest.raises(ValueError, match='pool was built'):
            material_imputer().draw(pool, 't', {'Material': 'C'}, world=0)

    def test_validated_features_trigger_validation(self) -> None:
        """Validation runs on the configured features, not the occupancy pair."""
        grid = material_neighbors()
        targets = inventory(
            {
                't': asset(
                    100.0,
                    0.0,
                    OccupancyClass='RES1',
                    NumberOfStories=1,
                    Material='W',
                )
            }
        )
        calls = []

        def validate(asset_id: Any, world: int, features: dict[str, Any]) -> bool:
            """
            Record the call and accept.

            Args:
                asset_id (Any):
                    The target.
                world (int):
                    The world.
                features (dict[str, Any]):
                    Its features after the copy.

            Returns:
                bool:
                    ``True``.
            """
            calls.append((asset_id, world, features['NumberOfUnits']))
            return True

        material_imputer().impute_stage(
            targets, inventory(grid), n_worlds=3, validate=validate
        )
        assert calls == []
        material_imputer(validated=('NumberOfUnits',)).impute_stage(
            targets, inventory(grid), n_worlds=3, validate=validate
        )
        assert [(a, w) for a, w, _ in calls] == [('t', 0), ('t', 1), ('t', 2)]
