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

"""Tests for the imputation donors built from an observed-donor stage log."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from brails.imputers.observed_donor_imputer.donor_records import (
    LOG_COLUMNS,
    OUTCOME_DONOR,
    OUTCOME_EXHAUSTED,
    OUTCOME_NO_DONOR,
    ROUND_DONOR,
    ROUND_REPAIR,
    ROUND_STATUS,
)
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    IMPUTATION_DONOR_KEY,
    imputation_donors_from_log,
)
from brails.utils.stage_freeze import StageFreeze
from tests.imputers.donor_fixtures import asset, imputer, inventory, world_value

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory


def row(
    asset_id: str,
    world: int,
    round_number: int,
    donor_id: str | None,
    outcome: str = OUTCOME_DONOR,
) -> dict[str, Any]:
    """
    Return one log row, copying the occupancy class where a donor is named.

    Args:
        asset_id (str):
            The target.
        world (int):
            The world.
        round_number (int):
            The round.
        donor_id (str | None):
            The donor, or None.
        outcome (str, optional):
            The outcome. Defaults to ``'donor'``.

    Returns:
        dict[str, Any]:
            The row, keyed by the log columns.
    """
    return {
        'asset_id': asset_id,
        'world': world,
        'iteration': int(round_number == ROUND_REPAIR),
        'donor_id': donor_id,
        'values': {} if donor_id is None else {'OccupancyClass': 'COM1'},
        'level': 0,
        'n_candidates': 3,
        'distance': math.nan if donor_id is None else 1.0,
        'distance_m': math.nan if donor_id is None else 100.0,
        'attempts': 0,
        'outcome': outcome,
        'round': round_number,
    }


def targets(*ids: str) -> AssetInventory:
    """
    Return an inventory of targets east of the origin.

    Args:
        *ids (str):
            The target ids.

    Returns:
        AssetInventory:
            One footprint per id, 100 m apart.
    """
    return inventory({a: asset(100.0 * i, 0.0) for i, a in enumerate(ids)})


def donors_of(rows: list[dict[str, Any]], n_worlds: int, *ids: str) -> dict:
    """
    Return the imputation donors' value per target for a list of log rows.

    Args:
        rows (list[dict[str, Any]]):
            The log rows.
        n_worlds (int):
            The number of worlds.
        *ids (str):
            The target ids.

    Returns:
        dict:
            Target id to the stored value of the donor key.
    """
    stage = imputation_donors_from_log(
        pd.DataFrame(rows, columns=list(LOG_COLUMNS)),
        targets(*ids),
        n_worlds=n_worlds,
    )
    return {a: stage.inventory[a].features[IMPUTATION_DONOR_KEY] for a in ids}


class TestDonorKept:
    """The donor named is the one whose values the target kept."""

    def test_repair_after_rejection_names_the_redraw(self) -> None:
        """A repair row logged after the donor round names the re-drawn donor."""
        rows = [
            row('t', 0, ROUND_DONOR, 'first'),
            row('t', 0, ROUND_REPAIR, 'second'),
        ]
        assert donors_of(rows, 1, 't') == {'t': 'second'}

    def test_exhausted_repair_keeps_the_donor_round(self) -> None:
        """An exhausted repair copies nothing, so round two's donor stands."""
        rows = [
            row('t', 0, ROUND_DONOR, 'first'),
            row('t', 0, ROUND_REPAIR, None, OUTCOME_EXHAUSTED),
        ]
        assert donors_of(rows, 1, 't') == {'t': 'first'}

    def test_no_donor_gives_none(self) -> None:
        """A donor round without a donor leaves the target without one."""
        rows = [row('t', 0, ROUND_DONOR, None, OUTCOME_NO_DONOR)]
        assert donors_of(rows, 2, 't') == {'t': [None, None]}

    def test_status_round_never_wins(self) -> None:
        """A status-round row names no donor, even one logged with values."""
        rows = [
            row('t', 0, ROUND_DONOR, None, OUTCOME_NO_DONOR),
            row('t', 0, ROUND_STATUS, 'status_donor'),
            row('u', 0, ROUND_STATUS, 'status_donor'),
            row('u', 0, ROUND_DONOR, 'donor'),
        ]
        assert donors_of(rows, 1, 't', 'u') == {'t': [None], 'u': 'donor'}

    def test_absent_target_and_world(self) -> None:
        """A target or world without a row gets None; agreeing worlds collapse."""
        rows = [
            row('t', 0, ROUND_DONOR, 'a'),
            row('t', 2, ROUND_DONOR, 'b'),
            row('u', 0, ROUND_DONOR, 'c'),
            row('u', 1, ROUND_DONOR, 'c'),
            row('u', 2, ROUND_DONOR, 'c'),
        ]
        assert donors_of(rows, 3, 't', 'u', 'v') == {
            't': ['a', None, 'b'],
            'u': 'c',
            'v': [None, None, None],
        }

    def test_stage_run_with_repairs(self) -> None:
        """On a real stage run, every named donor holds the class the target kept."""
        donors = inventory(
            {
                f'd{i}': asset(
                    30.0 * i,
                    40.0,
                    OccupancyClass='COM1' if i % 2 else 'COM4',
                    NumberOfStories=1,
                    NumberOfUnits=1,
                )
                for i in range(1, 7)
            }
        )
        stage_targets = inventory({'t': asset(0.0, 0.0), 'u': asset(50.0, 0.0)})

        def validate(asset_id: Any, _world: int, features: Any) -> bool:
            """
            Reject COM1 for ``t`` and every class for ``u``.

            Args:
                asset_id (Any):
                    The target.
                _world (int):
                    The world.
                features (Any):
                    Its features after the copy.

            Returns:
                bool:
                    Whether the copy stands.
            """
            return asset_id == 't' and features['OccupancyClass'] != 'COM1'

        n_worlds = 30
        result = imputer().impute_stage(
            stage_targets,
            donors,
            n_worlds=n_worlds,
            status={'t': False, 'u': False},
            validate=validate,
        )
        stage = imputation_donors_from_log(result.log, stage_targets, n_worlds=n_worlds)
        repaired = result.log[result.log['round'] == ROUND_REPAIR]
        # t has repairs that found a donor, u only exhausted ones
        assert set(repaired.loc[repaired['asset_id'] == 't', 'outcome']) - {
            OUTCOME_EXHAUSTED
        }
        assert set(repaired.loc[repaired['asset_id'] == 'u', 'outcome']) == {
            OUTCOME_EXHAUSTED
        }
        for target in ('t', 'u'):
            named = stage.inventory[target].features[IMPUTATION_DONOR_KEY]
            kept = result.imputed.inventory[target].features['OccupancyClass']
            for world in range(n_worlds):
                donor = world_value(named, world)
                assert donors.inventory[donor].features[
                    'OccupancyClass'
                ] == world_value(kept, world)


class TestStage:
    """The stage's shape, its refusals and its freeze."""

    def test_coordinates_copied(self) -> None:
        """Each target keeps its coordinates, copied, and the stage its worlds."""
        stage_targets = targets('t', 'a')
        stage = imputation_donors_from_log(
            pd.DataFrame(columns=list(LOG_COLUMNS)), stage_targets, n_worlds=4
        )
        assert stage.get_asset_ids() == ['a', 't']
        assert stage.n_pw == 4
        for asset_id in ('t', 'a'):
            copied = stage.inventory[asset_id].coordinates
            original = stage_targets.inventory[asset_id].coordinates
            assert copied == original
            assert copied is not original

    @pytest.mark.parametrize(
        ('rows', 'n_worlds', 'columns', 'message'),
        [
            ([row('t', 0, ROUND_DONOR, 'a')], 1, LOG_COLUMNS[:-1], 'lacks the col'),
            ([row('t', 2, ROUND_DONOR, 'a')], 2, LOG_COLUMNS, 'not integers'),
            ([row('t', -1, ROUND_DONOR, 'a')], 2, LOG_COLUMNS, 'not integers'),
            ([row('t', 0.5, ROUND_DONOR, 'a')], 2, LOG_COLUMNS, 'not integers'),
            ([row('x', 0, ROUND_DONOR, 'a')], 1, LOG_COLUMNS, 'targets lack'),
            ([], 0, LOG_COLUMNS, 'positive integer'),
            ([], True, LOG_COLUMNS, 'positive integer'),
        ],
    )
    def test_refusals(
        self,
        rows: list[dict[str, Any]],
        n_worlds: Any,
        columns: tuple[str, ...],
        message: str,
    ) -> None:
        """A missing column, a world out of range or an unknown asset raises."""
        log = pd.DataFrame(rows, columns=list(LOG_COLUMNS))[list(columns)]
        with pytest.raises(ValueError, match=message):
            imputation_donors_from_log(log, targets('t'), n_worlds=n_worlds)

    def test_rebuilt_stage_has_the_same_digest(self) -> None:
        """A stage rebuilt from the same log freezes to the same digest."""
        rows = [row('t', 0, ROUND_DONOR, 'a'), row('t', 1, ROUND_DONOR, 'b')]
        log = pd.DataFrame(rows, columns=list(LOG_COLUMNS))
        digests = [
            StageFreeze().freeze(
                'imputation_donors',
                imputation_donors_from_log(log, targets('t'), n_worlds=2),
            )
            for _ in range(2)
        ]
        assert digests[0] == digests[1]
