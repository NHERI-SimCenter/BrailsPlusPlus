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


"""
Draw outcomes, log rows and the stage result of the observed-donor imputer.

.. autosummary::

      DonorDraw
      DonorImputationResult
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from brails.imputers.observed_donor_imputer.donor_pool import DonorPool
    from brails.types.asset_inventory import AssetInventory

# Outcomes of a draw.
OUTCOME_DONOR = 'donor'
OUTCOME_RELAXED = 'relaxed'
OUTCOME_FEWER_THAN_K = 'fewer_than_k'
OUTCOME_BLOCK_GROUP = 'block_group'
OUTCOME_NO_DONOR = 'no_donor'
OUTCOME_NO_AREA = 'no_area'
OUTCOME_EXHAUSTED = 'exhausted'
OUTCOME_NEAREST_ADMITTED = 'nearest_admitted'
OUTCOME_NONE_ADMITTED = 'none_admitted'

# Values of the ``round`` column of the log: 1 for the status round
# (residential by default), 2 for the donor round and 3 for a repair.
ROUND_STATUS = 1
ROUND_DONOR = 2
ROUND_REPAIR = 3


class DonorDraw(NamedTuple):
    """
    The outcome of one donor draw for one asset in one world.

    Attributes:
        asset_id (Hashable):
            The id of the target the draw is for.
        world (int):
            The index of the possible world the draw is for, from 0.
        iteration (int):
            The iteration of the accepted pick, or of the first pick.
        donor_id (Any):
            The donor whose values were copied, or ``None``.
        values (Mapping[str, Any]):
            The copied features, empty when ``donor_id`` is ``None``.
        level (int | None):
            The number of soft filters relaxed, ``None`` where no filter
            level applies.
        n_candidates (int):
            The number of candidates at the level used, or in the
            target's block group for :data:`OUTCOME_BLOCK_GROUP`.
        distance (float):
            The donor distance, NaN without a donor.
        distance_m (float):
            The great-circle distance to the donor in meters, NaN without
            one.
        attempts (int):
            The number of picks ``admit`` judged, 0 without it.
        outcome (str):
            One of :data:`OUTCOME_DONOR`, :data:`OUTCOME_RELAXED`,
            :data:`OUTCOME_FEWER_THAN_K`, :data:`OUTCOME_BLOCK_GROUP`,
            :data:`OUTCOME_NO_DONOR`, :data:`OUTCOME_NO_AREA`,
            :data:`OUTCOME_EXHAUSTED`, :data:`OUTCOME_NEAREST_ADMITTED` and
            :data:`OUTCOME_NONE_ADMITTED`.
    """

    asset_id: Any
    world: int
    iteration: int
    donor_id: Any
    values: Mapping[str, Any]
    level: int | None
    n_candidates: int
    distance: float
    distance_m: float
    attempts: int
    outcome: str


# Columns of the log of ObservedDonorImputer.impute_stage.
LOG_COLUMNS = (*DonorDraw._fields, 'round')


class DonorImputationResult(NamedTuple):
    """
    The result of a stage run of the observed-donor imputer.

    :meth:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.ObservedDonorImputer.impute_stage`
    returns it.

    Attributes:
        imputed (AssetInventory):
            The targets that received a value, holding only the copied
            features.
        log (pandas.DataFrame):
            One row per target, world and round, with the columns of
            :data:`LOG_COLUMNS`.
        pool (DonorPool):
            The pool the draws read; with a freeze, the pool of the last
            batch, which holds the sites of that batch's targets only.
        seed (int):
            The seed of the draws.
        batches (tuple[str, ...]):
            The names of the frozen batches, in order; empty when no
            freeze was given.
    """

    imputed: AssetInventory
    log: pd.DataFrame
    pool: DonorPool
    seed: int
    batches: tuple[str, ...] = ()


def _log_row(draw: DonorDraw, round_number: int) -> dict[str, Any]:
    """
    Turn a draw into a log row.

    Args:
        draw (DonorDraw):
            The draw to log. Its fields become the row's columns, and its ``values``
            mapping is copied into a new dict.
        round_number (int):
            The round the draw belongs to, written to the row's ``round`` column:
            :data:`ROUND_STATUS`, :data:`ROUND_DONOR` or :data:`ROUND_REPAIR`.

    Returns:
        dict[str, Any]:
            The row, keyed by :data:`LOG_COLUMNS`.
    """
    row = draw._asdict()
    row['values'] = dict(draw.values)
    row['round'] = round_number
    return row
