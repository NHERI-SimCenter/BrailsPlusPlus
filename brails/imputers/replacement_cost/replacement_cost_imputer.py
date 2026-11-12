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
Name the observed donor each imputed building took its values from.

.. autosummary::

      donor_stage_from_log
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pandas as pd

from brails.imputers.observed_donor_imputer.donor_records import (
    LOG_COLUMNS as DONOR_LOG_COLUMNS,
)
from brails.imputers.observed_donor_imputer.donor_records import ROUND_STATUS
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils.world_values import collapse_realizations

if TYPE_CHECKING:
    from collections.abc import Hashable

# Feature of the donor stage holding, per world, the id of the donor whose
# values a target kept, and the name the stage is frozen under.
IMPUTATION_DONOR_KEY = 'ImputationDonor'
DONOR_STAGE = 'imputation_donors'


def donor_stage_from_log(
    log: pd.DataFrame, targets: AssetInventory, *, n_worlds: int
) -> AssetInventory:
    """
    Build the donor stage from the log of an observed-donor stage run.

    For each target and world the donor named is that of the log row with
    the highest ``round`` whose ``values`` are not empty, the status round
    excluded. The status round copies nothing, a repair carries a higher
    round than the donor round it replaces, and an exhausted repair copies
    nothing, so the donor named is the one whose values the target kept.

    Args:
        log (pandas.DataFrame):
            The log of
            :meth:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.ObservedDonorImputer.impute_stage`,
            with the columns of
            :data:`~brails.imputers.observed_donor_imputer.donor_records.LOG_COLUMNS`.
        targets (AssetInventory):
            The targets of that run. Not modified.
        n_worlds (int):
            The number of worlds of that run.

    Returns:
        AssetInventory:
            One asset per target, in ascending order of the id as a string,
            with the target's coordinates and :data:`IMPUTATION_DONOR_KEY`
            holding the donor id per world, ``None`` in a world where the
            target kept no donor's values or has no log row. A donor equal
            in every world is stored as a scalar, otherwise as a list, so a
            target without a donor in any world holds a list of ``None``.
            The inventory declares ``n_worlds`` possible worlds.

    Raises:
        ValueError:
            If ``n_worlds`` is not a positive integer, the log lacks a
            column, a row's world is not an integer below ``n_worlds``, or
            a row names an asset that ``targets`` lacks.
    """
    if isinstance(n_worlds, bool) or not isinstance(n_worlds, int) or n_worlds < 1:
        msg = f'n_worlds must be a positive integer, not {n_worlds!r}.'
        raise ValueError(msg)
    missing = [column for column in DONOR_LOG_COLUMNS if column not in log.columns]
    if missing:
        msg = f'The log lacks the columns {missing}.'
        raise ValueError(msg)
    bad_worlds = sorted(
        {
            repr(world)
            for world in log['world']
            if not pd.api.types.is_integer(world) or not 0 <= world < n_worlds
        }
    )
    if bad_worlds:
        msg = (
            f'The log holds worlds {bad_worlds}, which are not integers from 0 '
            f'to {n_worlds - 1}.'
        )
        raise ValueError(msg)
    unknown = sorted(
        {repr(a) for a in log['asset_id'] if a not in targets.inventory}
    )
    if unknown:
        msg = f'The log names assets that the targets lack: {unknown}.'
        raise ValueError(msg)

    # (target, world) to the round and id of the donor kept so far
    kept: dict[tuple[Hashable, int], tuple[int, Any]] = {}
    for asset_id, world, round_number, values, donor_id in zip(
        log['asset_id'], log['world'], log['round'], log['values'], log['donor_id']
    ):
        if round_number == ROUND_STATUS or not values:
            continue
        key = (asset_id, int(world))
        if key not in kept or round_number >= kept[key][0]:
            kept[key] = (round_number, donor_id)

    stage = AssetInventory()
    for asset_id in sorted(targets.inventory, key=str):
        donors = [
            kept[(asset_id, world)][1] if (asset_id, world) in kept else None
            for world in range(n_worlds)
        ]
        coordinates = deepcopy(targets.inventory[asset_id].coordinates)
        stage.add_asset(
            asset_id,
            Asset(
                asset_id,
                coordinates,
                {IMPUTATION_DONOR_KEY: collapse_realizations(donors)},
            ),
        )
    stage.n_pw = n_worlds
    return stage
