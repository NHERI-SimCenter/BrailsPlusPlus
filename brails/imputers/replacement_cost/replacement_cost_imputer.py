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
Impute the replacement cost of a building from the rate of a donor.

.. autosummary::

      ReplacementCostImputer
      ReplacementCostResult
      donor_stage_from_log
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from brails.aggregators.points_to_polygons.nsi_footprint_join import (
    GROSS_AREA,
    REPLACEMENT_COST,
)
from brails.aggregators.unit_budget.unit_budget import (
    AREA_FROM_GROSS_AREA,
    AREA_FROM_PLAN_AREA,
    AREA_FROM_STORIES,
)
from brails.imputers.observed_donor_imputer.donor_records import (
    LOG_COLUMNS as DONOR_LOG_COLUMNS,
)
from brails.imputers.observed_donor_imputer.donor_records import (
    OUTCOME_NO_AREA,
    ROUND_STATUS,
)
from brails.types.asset_inventory import (
    HAZUS_EQ_SOURCE_FEATURES,
    SOURCE_CODES,
    SOURCE_STRING_FEATURE,
    Asset,
    AssetInventory,
    is_missing_value,
)
from brails.utils.unit_converter import UNIT_TO_METER
from brails.utils.world_values import (
    OCCUPANCY_KEY,
    STORIES_KEY,
    collapse_realizations,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from brails.compilers.stage_freeze import StageFreeze

# Feature of the donor stage holding, per world, the id of the donor whose
# values a target kept, and the name the stage is frozen under.
IMPUTATION_DONOR_KEY = 'ImputationDonor'
DONOR_STAGE = 'imputation_donors'

# Source codes of a value NSI states: one record's value, or the sum of the
# values of several records of one footprint.
NSI_VALUE_CODES = (SOURCE_CODES['nsi'], SOURCE_CODES['derived'])

# Outcomes of a target, besides the imputer's OUTCOME_NO_AREA.
OUTCOME_IMPUTATION_DONOR = 'imputation_donor'
OUTCOME_NO_CLASS = 'no_class'
OUTCOME_NO_DONOR_RATE = 'no_donor_rate'

# Columns of the log of ReplacementCostImputer.impute_stage. donor_id is the
# donor the donor stage names, with or without a rate.
LOG_COLUMNS = (
    'asset_id',
    'world',
    'outcome',
    'donor_id',
    'distance_m',
    'rate',
    'floor_area_sqft',
    'area_source',
)


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


class ReplacementCostResult(NamedTuple):
    """
    The result of a stage run of the replacement cost imputer.

    Attributes:
        imputed (AssetInventory):
            The targets that received a value in some world, holding only
            the value feature, with ``None`` in a world without one.
        log (pandas.DataFrame):
            One row per target and world, with the columns of
            :data:`LOG_COLUMNS`.
    """

    imputed: AssetInventory
    log: pd.DataFrame


class ReplacementCostImputer:
    """
    Give a building without a replacement cost the value of its donor's rate.

    A building has a rate where the source code of its ``value_key`` is
    among :data:`NSI_VALUE_CODES` and its ``value_key`` and
    ``gross_area_key`` are finite numbers above zero: the value divided by
    the gross floor area, in the value's units per square foot. The gross
    area carries no source code of its own, so a building's gross area is
    taken to be NSI's whenever its value is. In each world of a frozen
    stage:

    1. A building holding a ``value_key`` is no target and keeps its value.
    2. A target whose ``occupancy_key`` is missing or not a string is
       logged :data:`OUTCOME_NO_CLASS` and gets no value.
    3. The target's floor area in square feet is its ``gross_area_key``
       where that is a finite number above zero, else its plan area times
       its ``stories_key`` where that is a finite number above zero, else
       its plan area as one story. A target with neither a gross area nor
       a plan area is logged
       :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_NO_AREA`
       and gets no value.
    4. Where the donor stage names the donor whose values the observed-donor
       imputer copied to the target in this world, and that donor has a rate
       in the same world of the stage, the target's value is that rate times
       its floor area, logged :data:`OUTCOME_IMPUTATION_DONOR`. The value is
       copied with the features that donor supplied, and the donor's class
       is not compared with the target's.
    5. Any other target is logged :data:`OUTCOME_NO_DONOR_RATE` and gets no
       value.

    Every constructor argument is stored as an attribute of the same name,
    read-only by convention.
    """

    def __init__(
        self,
        *,
        value_key: str = REPLACEMENT_COST,
        gross_area_key: str = GROSS_AREA,
        occupancy_key: str = OCCUPANCY_KEY,
        stories_key: str = STORIES_KEY,
        source_string_key: str = SOURCE_STRING_FEATURE,
        source_features: Sequence[str] = HAZUS_EQ_SOURCE_FEATURES,
    ) -> None:
        """
        Configure the imputer.

        Args:
            value_key (str, optional):
                The replacement cost feature, the one imputed. Defaults to
                :data:`~brails.aggregators.points_to_polygons.nsi_footprint_join.REPLACEMENT_COST`.
            gross_area_key (str, optional):
                The gross floor area feature, in square feet. Defaults to
                :data:`~brails.aggregators.points_to_polygons.nsi_footprint_join.GROSS_AREA`.
            occupancy_key (str, optional):
                The occupancy class feature. Defaults to
                :data:`~brails.utils.world_values.OCCUPANCY_KEY`.
            stories_key (str, optional):
                The story count feature. Defaults to
                :data:`~brails.utils.world_values.STORIES_KEY`.
            source_string_key (str, optional):
                The feature holding the source string. Defaults to
                :data:`~brails.types.asset_inventory.SOURCE_STRING_FEATURE`.
            source_features (Sequence[str], optional):
                The features the source string covers, in its order.
                Defaults to
                :data:`~brails.types.asset_inventory.HAZUS_EQ_SOURCE_FEATURES`.

        Raises:
            TypeError:
                If ``source_features`` is a single string.
            ValueError:
                If ``source_features`` repeats a feature or lacks the
                occupancy class or the value feature.
        """
        if isinstance(source_features, str):
            msg = 'source_features must be a sequence of feature names.'
            raise TypeError(msg)
        source_features = tuple(source_features)
        if len(set(source_features)) != len(source_features):
            msg = 'source_features repeats a feature.'
            raise ValueError(msg)
        missing = [
            key for key in (occupancy_key, value_key) if key not in source_features
        ]
        if missing:
            msg = f'source_features lacks {missing}.'
            raise ValueError(msg)
        self.value_key = value_key
        self.gross_area_key = gross_area_key
        self.occupancy_key = occupancy_key
        self.stories_key = stories_key
        self.source_string_key = source_string_key
        self.source_features = source_features

    def impute_stage(
        self,
        freeze: StageFreeze,
        stage: str,
        donor_stage: str,
        *,
        world: int | None = None,
    ) -> ReplacementCostResult:
        """
        Impute the replacement cost of a frozen stage's targets.

        Neither frozen stage is changed. With ``world=None`` every world is
        computed, the number of worlds being the larger of the two stages'.
        With a ``world``, that world's realization of each stage is read,
        or world 0 of a stage holding one world.

        Args:
            freeze (StageFreeze):
                The freeze holding both stages.
            stage (str):
                The frozen stage holding the buildings, their classes,
                stories, geometry, values and source strings.
            donor_stage (str):
                The frozen stage that :func:`donor_stage_from_log` built for
                the observed-donor imputer's run on these buildings.
            world (int, optional):
                The one world to compute. Defaults to ``None``, every world.

        Returns:
            ReplacementCostResult:
                The values, one possible world with a ``world`` given and
                otherwise the number computed, a value equal in every world
                stored as a scalar, and the log.

        Raises:
            TypeError:
                If ``donor_stage`` is ``None``.
            LookupError:
                If either stage is not frozen.
            ValueError:
                If a stage holds a number of worlds other than 1 and the
                other's, the donor stage holds an asset the stage lacks or
                names a donor the stage lacks, ``world`` is not a world of
                the stages, a donor's source string is missing or of the
                wrong length, or a target's coordinates cannot be read.
        """
        if donor_stage is None:
            msg = (
                'impute_stage needs the donor stage that donor_stage_from_log '
                'builds; without it no target would take the rate of its donor.'
            )
            raise TypeError(msg)
        buildings = freeze.read(stage)
        donors = freeze.read(donor_stage)
        stray = sorted(
            repr(asset_id)
            for asset_id in donors.inventory
            if asset_id not in buildings.inventory
        )
        if stray:
            msg = f'The donor stage holds assets the stage lacks: {stray}.'
            raise ValueError(msg)
        n_worlds = max(buildings.n_pw, donors.n_pw)
        if {buildings.n_pw, donors.n_pw} - {1, n_worlds}:
            msg = (
                f'The stage holds {buildings.n_pw} possible worlds and the '
                f'donor stage {donors.n_pw}.'
            )
            raise ValueError(msg)
        if world is not None and (
            isinstance(world, bool)
            or not isinstance(world, int)
            or world < 0
            or (n_worlds > 1 and world >= n_worlds)
        ):
            msg = f'world {world!r} is not a world of stages with {n_worlds}.'
            raise ValueError(msg)
        worlds = list(range(n_worlds)) if world is None else [world]

        values: dict[Hashable, dict[int, float]] = {}
        rows: list[dict[str, Any]] = []
        for position, w in enumerate(worlds):
            world_values, world_rows = self._impute_world(
                buildings.get_world_realization(w if buildings.n_pw > 1 else 0),
                donors.get_world_realization(w if donors.n_pw > 1 else 0),
                w,
            )
            for asset_id, value in world_values.items():
                values.setdefault(asset_id, {})[position] = value
            rows.extend(world_rows)

        imputed = AssetInventory(vector_features=buildings.vector_features)
        for asset_id in sorted(values, key=str):
            realizations = [values[asset_id].get(p) for p in range(len(worlds))]
            imputed.add_asset(
                asset_id,
                Asset(
                    asset_id,
                    deepcopy(buildings.inventory[asset_id].coordinates),
                    {self.value_key: collapse_realizations(realizations)},
                ),
            )
        imputed.n_pw = len(worlds)
        return ReplacementCostResult(
            imputed, pd.DataFrame(rows, columns=list(LOG_COLUMNS))
        )

    def _impute_world(
        self, buildings: AssetInventory, donors: AssetInventory, world: int
    ) -> tuple[dict[Hashable, float], list[dict[str, Any]]]:
        """
        Impute the targets of one world.

        Args:
            buildings (AssetInventory):
                The stage's realization of the world.
            donors (AssetInventory):
                The donor stage's realization of the world.
            world (int):
                The world, as logged.

        Returns:
            tuple[dict[Hashable, float], list[dict[str, Any]]]:
                The value per target that received one, and the log rows in
                ascending order of the target id as a string.

        Raises:
            ValueError:
                As :meth:`impute_stage` raises.
        """
        values: dict[Hashable, float] = {}
        rows = []
        for asset_id in sorted(buildings.inventory, key=str):
            asset = buildings.inventory[asset_id]
            if not is_missing_value(asset.features.get(self.value_key)):
                continue
            row = dict.fromkeys(LOG_COLUMNS)
            row.update(
                asset_id=asset_id,
                world=world,
                distance_m=math.nan,
                rate=math.nan,
                floor_area_sqft=math.nan,
            )
            rows.append(row)
            occupancy = asset.features.get(self.occupancy_key)
            if not isinstance(occupancy, str) or is_missing_value(occupancy):
                row['outcome'] = OUTCOME_NO_CLASS
                continue
            area, row['area_source'] = self._floor_area(asset_id, asset)
            if area is None:
                row['outcome'] = OUTCOME_NO_AREA
                continue
            row['floor_area_sqft'] = area
            named = donors.inventory.get(asset_id)
            donor_id = (
                None if named is None else named.features.get(IMPUTATION_DONOR_KEY)
            )
            rate = None
            if donor_id is not None:
                row['donor_id'] = donor_id
                if donor_id not in buildings.inventory:
                    msg = (
                        f'The donor stage names donor {donor_id!r} for asset '
                        f'{asset_id!r} in world {world}, which the stage lacks.'
                    )
                    raise ValueError(msg)
                rate = self._rate(donor_id, buildings.inventory[donor_id].features)
            if rate is None:
                row['outcome'] = OUTCOME_NO_DONOR_RATE
                continue
            row.update(outcome=OUTCOME_IMPUTATION_DONOR, rate=rate)
            values[asset_id] = rate * area
        return values, rows

    def _floor_area(
        self, asset_id: Hashable, asset: Asset
    ) -> tuple[float | None, str | None]:
        """
        Return a target's gross floor area and where it comes from.

        Args:
            asset_id (Hashable):
                The target, for the error message.
            asset (Asset):
                The target in one world.

        Returns:
            tuple[float | None, str | None]:
                The floor area in square feet and one of
                :data:`~brails.aggregators.unit_budget.unit_budget.AREA_FROM_GROSS_AREA`,
                :data:`~brails.aggregators.unit_budget.unit_budget.AREA_FROM_STORIES`
                and
                :data:`~brails.aggregators.unit_budget.unit_budget.AREA_FROM_PLAN_AREA`,
                or ``(None, None)`` without a gross area or a plan area.

        Raises:
            ValueError:
                If the coordinates cannot be read as a footprint.
        """
        # The gross floor area of BlockGroupUnitBudget, which must agree.
        gross = _positive(asset.features.get(self.gross_area_key))
        if gross is not None:
            return gross, AREA_FROM_GROSS_AREA
        try:
            with_area = AssetInventory.has_area(asset.coordinates)
        except ValueError as error:
            msg = f'Asset {asset_id!r} has no readable footprint: {error}'
            raise ValueError(msg) from error
        if not with_area:
            return None, None
        plan = (
            AssetInventory.centroid_and_plan_area(asset.coordinates)[2]
            / UNIT_TO_METER['ft'] ** 2
        )
        stories = _positive(asset.features.get(self.stories_key))
        if stories is None:
            return plan, AREA_FROM_PLAN_AREA
        return plan * stories, AREA_FROM_STORIES

    def _rate(self, asset_id: Hashable, features: Mapping[str, Any]) -> float | None:
        """
        Return a building's value per square foot of gross floor area.

        Args:
            asset_id (Hashable):
                The building, for the error message.
            features (Mapping[str, Any]):
                Its features in one world.

        Returns:
            float | None:
                The rate, or ``None`` where the building has none.

        Raises:
            ValueError:
                If the source string is missing or of the wrong length.
        """
        codes = features.get(self.source_string_key)
        if not isinstance(codes, str) or len(codes) != len(self.source_features):
            msg = (
                f'Asset {asset_id!r} carries no source string of '
                f'{len(self.source_features)} codes in '
                f'{self.source_string_key!r}: {codes!r}.'
            )
            raise ValueError(msg)
        if codes[self.source_features.index(self.value_key)] not in NSI_VALUE_CODES:
            return None
        value = _positive(features.get(self.value_key))
        gross = _positive(features.get(self.gross_area_key))
        if value is None or gross is None:
            return None
        return value / gross


def _positive(value: object) -> float | None:
    """
    Read a feature value as a finite number above zero.

    Args:
        value (object):
            The value.

    Returns:
        float | None:
            The value as a float, or ``None`` where it is not a finite
            number above zero. A boolean is not a number here.
    """
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None
