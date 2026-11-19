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
Value a building as a rate times the gross area computed from its footprint.

.. autosummary::

      ReplacementCostImputer
      ReplacementCostResult
      footprint_gross_area
"""

from __future__ import annotations

import math
import secrets
from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from brails.constants import (
    AREA_FROM_FEATURE,
    AREA_FROM_PLAN_AREA,
    AREA_FROM_STORIES,
    BLOCK_GROUP_KEY,
    GROSS_AREA_KEY,
    IMPUTATION_DONOR_KEY,
    OCCUPANCY_KEY,
    REPLACEMENT_COST_KEY,
    SOURCE_STRING_KEY,
    STORIES_KEY,
)
from brails.imputers.imputation import Imputation
from brails.imputers.observed_donor_imputer.donor_records import (
    LOG_COLUMNS as DONOR_LOG_COLUMNS,
)
from brails.imputers.observed_donor_imputer.donor_records import (
    OUTCOME_BLOCK_GROUP,
    OUTCOME_DONOR,
    OUTCOME_FEWER_THAN_K,
    OUTCOME_NO_AREA,
)
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    DEFAULT_AREA_WEIGHT,
    DEFAULT_DONOR_CAP_M,
    DEFAULT_DONOR_K,
    DEFAULT_LOCATION_SCALE_M,
    DEFAULT_STORIES_WEIGHT,
    ObservedDonorImputer,
    imputation_donors_from_log,
)
from brails.types.asset_inventory import (
    HAZUS_EQ_SOURCE_FEATURES,
    SOURCE_CODE_NONE,
    SOURCE_CODES,
    Asset,
    AssetInventory,
    is_missing_value,
    read_source_string,
)
from brails.utils.stage_freeze import StageFreeze
from brails.utils.unit_converter import UNIT_TO_METER
from brails.utils.world_values import collapse_realizations

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

# Name under which the donor stage that imputation_donors_from_log builds is
# frozen.
DONOR_STAGE = 'imputation_donors'

# Source codes of a value that gives a rate over its GrossArea: local, NSI
# and derived. A derived value is the sum of the values of several NSI records
# of one footprint, and it is also the code this imputer writes for a value
# from a building's own rate. README.rst beside this module, "The value and
# the rate".
RATE_VALUE_CODES = (
    SOURCE_CODES['local'],
    SOURCE_CODES['nsi'],
    SOURCE_CODES['derived'],
)

# Source codes a candidate's occupancy class may carry in the same-class
# neighbor draw: local, footprint, NSI or derived, never imputed or drawn.
# README.rst beside this module, "Scope".
NEIGHBOR_CLASS_CODES = tuple(
    SOURCE_CODES[label] for label in ('local', 'footprint', 'nsi', 'derived')
)

# Prefix of the purpose labels of the same-class neighbor draw, which the
# observed-donor imputer's draw extends to 'same_class_neighbor:observed_donor'
# and 'same_class_neighbor:observed_donor_fallback'.
NEIGHBOR_PURPOSE_PREFIX = 'same_class_neighbor'

# Outcomes of a building outside the same-class neighbor draw.
# OUTCOME_NO_AREA and the draw's own outcomes are the observed-donor imputer's.
OUTCOME_LOCAL_VALUE = 'local_value'
OUTCOME_OWN_RATE = 'own_rate'
OUTCOME_IMPUTATION_DONOR = 'imputation_donor'
OUTCOME_NO_CLASS = 'no_class'

# Source code of a building's value by its outcome: local where the value is
# kept, derived for the building's own rate, imputed for another building's.
# An outcome not listed gives no value. README.rst beside this module,
# "Procedure".
OUTCOME_VALUE_CODES = MappingProxyType(
    {
        OUTCOME_LOCAL_VALUE: SOURCE_CODES['local'],
        OUTCOME_OWN_RATE: SOURCE_CODES['derived'],
        OUTCOME_IMPUTATION_DONOR: SOURCE_CODES['imputed'],
        OUTCOME_DONOR: SOURCE_CODES['imputed'],
        OUTCOME_FEWER_THAN_K: SOURCE_CODES['imputed'],
        OUTCOME_BLOCK_GROUP: SOURCE_CODES['imputed'],
    }
)

# Feature holding a building's rate in the neighbor draw's own inventories.
_RATE_KEY = 'ReplacementCostRate'

# Columns of the log of ReplacementCostImputer.impute_stage. donor_id names the
# donor or neighbor whose rate a building took, and is missing otherwise (test
# with pandas.isna). rate is the rate the value is computed from, and on the row
# of a kept local value the building's own rate, where it has one.
# footprint_gross_area_sqft is the gross area the rate multiplies, computed from
# the footprint unless the caller names a feature that holds it, and
# area_source says where it came from.
LOG_COLUMNS = (
    'asset_id',
    'world',
    'outcome',
    'donor_id',
    'distance_m',
    'rate',
    'footprint_gross_area_sqft',
    'area_source',
)


class ReplacementCostResult(NamedTuple):
    """
    The result of a stage run of the replacement cost imputer.

    Attributes:
        costs (AssetInventory):
            Every value the run holds, own-rate and local values included:
            the buildings with a value in some world, holding only the value
            feature, with ``None`` in a world without one.
        log (pandas.DataFrame):
            One row per building and world, with the columns of
            :data:`LOG_COLUMNS`, whose ``outcome`` gives the value's source
            code through :data:`OUTCOME_VALUE_CODES`.
        seed (int):
            The seed of the same-class neighbor draw.
    """

    costs: AssetInventory
    log: pd.DataFrame
    seed: int


class ReplacementCostImputer(Imputation):
    """
    Value every building as a rate times the gross area computed from its footprint.

    In each world, a building's gross area is computed from its footprint:
    its plan area times its ``stories_key``, or its plan area alone without
    a positive story count. It is not read from ``gross_area_key``. A caller
    holding the gross areas can name the feature that holds them as
    ``target_area_key`` of :meth:`impute` and :meth:`impute_stage`. Its rate
    is
    the first that exists of three: its own, ``value_key`` over
    ``gross_area_key`` where the value's source code is among
    :data:`RATE_VALUE_CODES` and both are numbers above zero; that of the
    donor the donor stage names; that of a building of its occupancy class
    drawn by the same-class neighbor draw. Every rate is read from the
    stage, never from a value the imputer wrote. A value coded local is kept
    and gives a rate as a value coded NSI does, and every other stated value
    is replaced. A building without a rate or a gross area gets no value,
    and the log says why. The section "Procedure" of the README beside this
    module describes the run.

    Every constructor argument is stored as an attribute of the same name,
    read-only by convention, and ``seed`` holds the seed used.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        k: int = DEFAULT_DONOR_K,
        cap_m: float = DEFAULT_DONOR_CAP_M,
        location_scale_m: float = DEFAULT_LOCATION_SCALE_M,
        area_weight: float = DEFAULT_AREA_WEIGHT,
        stories_weight: float = DEFAULT_STORIES_WEIGHT,
        value_key: str = REPLACEMENT_COST_KEY,
        gross_area_key: str = GROSS_AREA_KEY,
        occupancy_key: str = OCCUPANCY_KEY,
        stories_key: str = STORIES_KEY,
        source_string_key: str = SOURCE_STRING_KEY,
        source_features: Sequence[str] = HAZUS_EQ_SOURCE_FEATURES,
    ) -> None:
        """
        Configure the imputer.

        Args:
            seed (int, optional):
                The seed of the same-class neighbor draw. ``None`` draws one
                from system entropy.
            k (int, optional):
                The number of nearest buildings the neighbor draw picks
                among. Defaults to
                :data:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.DEFAULT_DONOR_K`.
            cap_m (float, optional):
                The neighbor draw's distance cap in meters. Defaults to
                :data:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.DEFAULT_DONOR_CAP_M`.
            location_scale_m (float, optional):
                Meters of separation counted as one unit of the neighbor
                draw's distance. Defaults to
                :data:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.DEFAULT_LOCATION_SCALE_M`.
            area_weight (float, optional):
                Weight of the squared difference of log plan areas in that
                distance. Defaults to
                :data:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.DEFAULT_AREA_WEIGHT`.
            stories_weight (float, optional):
                Weight of the squared difference of the ``stories_key``
                counts in that distance. Defaults to
                :data:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.DEFAULT_STORIES_WEIGHT`.
            value_key (str, optional):
                The replacement cost feature. Defaults to
                :data:`~brails.constants.REPLACEMENT_COST_KEY`.
            gross_area_key (str, optional):
                The feature a building's own value is divided by, in square
                feet. Defaults to
                :data:`~brails.constants.GROSS_AREA_KEY`.
            occupancy_key (str, optional):
                The occupancy class feature. Defaults to
                :data:`~brails.constants.OCCUPANCY_KEY`.
            stories_key (str, optional):
                The story count feature. Defaults to
                :data:`~brails.constants.STORIES_KEY`.
            source_string_key (str, optional):
                The feature holding the source string. Defaults to
                :data:`~brails.constants.SOURCE_STRING_KEY`.
            source_features (Sequence[str], optional):
                The features the source string covers, in its order.
                Defaults to
                :data:`~brails.types.asset_inventory.HAZUS_EQ_SOURCE_FEATURES`.

        Raises:
            TypeError:
                If ``source_features`` is a single string.
            ValueError:
                If ``source_features`` repeats a feature or lacks the
                occupancy class or the value feature, ``k`` is below 1,
                ``cap_m`` or ``location_scale_m`` is not positive, or a
                weight is negative.
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
        self.seed = secrets.randbits(64) if seed is None else int(seed)
        # The defaults of the draw's five options: README.rst beside this
        # module, "Scope".
        self.k = k
        self.cap_m = cap_m
        self.location_scale_m = location_scale_m
        self.area_weight = area_weight
        self.stories_weight = stories_weight
        self._imputer = ObservedDonorImputer(
            features=(_RATE_KEY,),
            soft_filters=(),
            required=(OCCUPANCY_KEY, _RATE_KEY),
            k=k,
            cap_m=cap_m,
            location_scale_m=location_scale_m,
            area_weight=area_weight,
            stories_key=STORIES_KEY,
            stories_weight=stories_weight,
            seed=self.seed,
        )

    def impute(
        self, input_inventory: AssetInventory, *, target_area_key: str | None = None
    ) -> AssetInventory:
        """
        Value an inventory's buildings, without a donor stage.

        No donor is named, so a building without its own rate takes the
        same-class neighbor draw. Each building's value is replaced by the
        imputer's, or removed where it gets none, and the value's code in
        each source string is rewritten from the outcome by
        :data:`OUTCOME_VALUE_CODES`, or set to ``-`` without a value.

        Args:
            input_inventory (AssetInventory):
                The inventory. It is not modified.
            target_area_key (str, optional):
                The feature of the inventory that holds each building's
                gross area in square feet, which the rate then multiplies,
                as :meth:`impute_stage` describes. Defaults to ``None``, the
                gross area computed from the footprint.

        Returns:
            AssetInventory:
                A copy of the inventory with the values and codes written.

        Raises:
            ValueError:
                As :meth:`impute_stage` raises, or if a building's source
                string is not one that
                :func:`~brails.types.asset_inventory.read_source_string`
                reads.
        """
        freeze = StageFreeze()
        freeze.freeze('inventory', input_inventory)
        freeze.freeze(
            DONOR_STAGE,
            imputation_donors_from_log(
                pd.DataFrame(columns=list(DONOR_LOG_COLUMNS)),
                AssetInventory(),
                n_worlds=input_inventory.n_pw,
            ),
        )
        result = self.impute_stage(
            freeze, 'inventory', DONOR_STAGE, target_area_key=target_area_key
        )
        outcomes = dict(
            zip(
                zip(result.log['asset_id'], result.log['world']),
                result.log['outcome'],
            )
        )
        position = self.source_features.index(self.value_key)
        output = deepcopy(input_inventory)
        # each building's value, and the value's code in each world
        for asset_id, asset in output.inventory.items():
            held = result.costs.inventory.get(asset_id)
            if held is None:
                asset.features.pop(self.value_key, None)
            else:
                asset.features[self.value_key] = held.features[self.value_key]
            strings = asset.features.get(self.source_string_key)
            if strings is None:
                continue
            if not isinstance(strings, list):
                strings = [strings] * output.n_pw
            rewritten = []
            for world, string in enumerate(strings):
                codes = list(
                    read_source_string(
                        asset_id,
                        {self.source_string_key: string},
                        source_string_key=self.source_string_key,
                        source_features=self.source_features,
                    )
                )
                codes[position] = OUTCOME_VALUE_CODES.get(
                    outcomes[(asset_id, world)], SOURCE_CODE_NONE
                )
                rewritten.append(''.join(codes))
            asset.features[self.source_string_key] = collapse_realizations(rewritten)
        return output

    def impute_stage(  # noqa: C901
        self,
        freeze: StageFreeze,
        stage: str,
        donor_stage: str,
        *,
        world: int | None = None,
        target_area_key: str | None = None,
        area_stage: str | None = None,
        areas: AssetInventory | None = None,
    ) -> ReplacementCostResult:
        """
        Value the buildings of a frozen stage.

        No frozen stage is changed. With ``world=None`` every world is
        computed, the number of worlds being the largest of the stages'.
        With a ``world``, that world's realization of each stage is read,
        or world 0 of a stage holding one world, and the neighbor draw
        draws with that world's index.

        With a ``target_area_key``, a building's gross area is read from
        that feature, of ``area_stage`` or ``areas`` where one is given and
        otherwise of ``stage``, and logged as
        :data:`~brails.constants.AREA_FROM_FEATURE`. The footprint is then
        not read for the area, and a building without a finite number above
        zero there gets no value, logged as
        :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_NO_AREA`.
        The same-class neighbor draw still locates a building by its
        footprint, so a building that would take a neighbor's rate and whose
        coordinates enclose no area cannot be located and gets no value. Its
        log row holds that outcome, with no area and no area source.

        Args:
            freeze (StageFreeze):
                The freeze holding the stages.
            stage (str):
                The frozen stage holding the buildings, their occupancy
                classes, stories, geometry, values and source strings.
            donor_stage (str):
                The frozen stage that
                :func:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.imputation_donors_from_log`
                built for
                the observed-donor imputer's run on these buildings.
            world (int, optional):
                The one world to compute. Defaults to ``None``, every world.
            target_area_key (str, optional):
                The feature that holds each building's gross area in square
                feet, which the rate then multiplies. Defaults to ``None``,
                the gross area computed from the footprint.
            area_stage (str, optional):
                The frozen stage that holds ``target_area_key``, for a
                ``stage`` frozen before the areas were computed. Defaults to
                ``None``, which reads the feature from ``stage``.
            areas (AssetInventory, optional):
                An inventory that holds ``target_area_key`` and is not frozen,
                read in place of ``area_stage``, such as the areas of the one
                world a caller computes before valuing it. Defaults to
                ``None``.

        Returns:
            ReplacementCostResult:
                The values, one possible world with a ``world`` given and
                otherwise the number computed, a value equal in every world
                stored as a scalar, and the log.

        Raises:
            TypeError:
                If ``donor_stage`` is ``None``.
            LookupError:
                If a named stage is not frozen.
            ValueError:
                If ``area_stage`` or ``areas`` is given without a
                ``target_area_key``, both are given, a stage holds a number
                of worlds other than 1 and the largest, the donor stage or
                the areas hold an asset the stage lacks, the donor stage
                names a donor the stage lacks, ``world`` is not a world of
                the stages, a building holding a value carries no source
                string that
                :func:`~brails.types.asset_inventory.read_source_string`
                reads, or a building's coordinates cannot be read.
        """
        if donor_stage is None:
            msg = (
                'impute_stage needs the donor stage that '
                'imputation_donors_from_log builds; without it no target would '
                'take the rate of its donor.'
            )
            raise TypeError(msg)
        if area_stage is not None and areas is not None:
            msg = 'impute_stage takes area_stage or areas, not both.'
            raise ValueError(msg)
        if target_area_key is None and (area_stage is not None or areas is not None):
            given = (
                'areas are given'
                if area_stage is None
                else f'area_stage {area_stage!r} is named'
            )
            msg = f'{given} without the target_area_key feature to read from them.'
            raise ValueError(msg)
        buildings = freeze.read(stage)
        # The other inventories, keyed by the name the error messages give them.
        others = {'donor stage': freeze.read(donor_stage)}
        if area_stage is not None:
            areas = others['area stage'] = freeze.read(area_stage)
        elif areas is not None:
            others['area inventory'] = areas
        donors = others['donor stage']
        for name, held in others.items():
            stray = sorted(
                repr(asset_id)
                for asset_id in held.inventory
                if asset_id not in buildings.inventory
            )
            if stray:
                msg = f'The {name} holds assets the stage lacks: {stray}.'
                raise ValueError(msg)
        counts = {name: held.n_pw for name, held in others.items()}
        n_worlds = max(buildings.n_pw, *counts.values())
        if {buildings.n_pw, *counts.values()} - {1, n_worlds}:
            msg = (
                f'The stage holds {buildings.n_pw} possible worlds, '
                + ', '.join(f'the {name} {count}' for name, count in counts.items())
                + '. Each must hold 1 or the largest number.'
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

        values: dict[Hashable, dict[int, Any]] = {}
        rows: list[dict[str, Any]] = []
        for position, w in enumerate(worlds):
            world_values, world_rows = self._impute_world(
                buildings.get_world_realization(w if buildings.n_pw > 1 else 0),
                donors.get_world_realization(w if donors.n_pw > 1 else 0),
                w,
                target_area_key,
                None
                if areas is None
                else areas.get_world_realization(w if areas.n_pw > 1 else 0),
            )
            for asset_id, value in world_values.items():
                values.setdefault(asset_id, {})[position] = value
            rows.extend(world_rows)

        costs = AssetInventory(vector_features=buildings.vector_features)
        for asset_id in sorted(values, key=str):
            realizations = [values[asset_id].get(p) for p in range(len(worlds))]
            costs.add_asset(
                asset_id,
                Asset(
                    asset_id,
                    deepcopy(buildings.inventory[asset_id].coordinates),
                    {self.value_key: collapse_realizations(realizations)},
                ),
            )
        costs.n_pw = len(worlds)
        return ReplacementCostResult(
            costs, pd.DataFrame(rows, columns=list(LOG_COLUMNS)), self.seed
        )

    def _impute_world(  # noqa: C901
        self,
        buildings: AssetInventory,
        donors: AssetInventory,
        world: int,
        target_area_key: str | None,
        areas: AssetInventory | None,
    ) -> tuple[dict[Hashable, Any], list[dict[str, Any]]]:
        """
        Value the buildings of one world.

        Args:
            buildings (AssetInventory):
                The stage's realization of the world.
            donors (AssetInventory):
                The donor stage's realization of the world.
            world (int):
                The world, as logged.
            target_area_key (str | None):
                The feature holding each building's gross area, or ``None``
                to compute the area from the footprint.
            areas (AssetInventory | None):
                The area stage's realization of the world, or ``None`` to
                read ``target_area_key`` from ``buildings``.

        Returns:
            tuple[dict[Hashable, Any], list[dict[str, Any]]]:
                The value per building that holds one, and the log rows
                in ascending order of the building id as a string.

        Raises:
            ValueError:
                As :meth:`impute_stage` raises.
        """
        # every building's own rate, read before any value is written
        rates: dict[Hashable, float] = {}
        for asset_id, asset in buildings.inventory.items():
            rate = self._rate(asset_id, asset.features)
            if rate is not None:
                rates[asset_id] = rate

        values: dict[Hashable, Any] = {}
        rows: dict[Hashable, dict[str, Any]] = {}
        # occupancy class to the buildings of the neighbor draw, with their
        # log rows
        pending: dict[str, list[tuple[Hashable, dict[str, Any]]]] = {}
        value_position = self.source_features.index(self.value_key)
        for asset_id in sorted(buildings.inventory, key=str):
            asset = buildings.inventory[asset_id]
            row = dict.fromkeys(LOG_COLUMNS)
            row.update(
                asset_id=asset_id,
                world=world,
                distance_m=math.nan,
                rate=math.nan,
                footprint_gross_area_sqft=math.nan,
            )
            rows[asset_id] = row
            stated = asset.features.get(self.value_key)
            if (
                not is_missing_value(stated)
                and read_source_string(
                    asset_id,
                    asset.features,
                    source_string_key=self.source_string_key,
                    source_features=self.source_features,
                )[value_position]
                == SOURCE_CODES['local']
            ):
                row.update(
                    outcome=OUTCOME_LOCAL_VALUE,
                    rate=rates.get(asset_id, math.nan),
                )
                values[asset_id] = stated
                continue
            if target_area_key is None:
                area, row['area_source'] = footprint_gross_area(
                    asset_id, asset, stories_key=self.stories_key
                )
            else:
                holder = asset if areas is None else areas.inventory.get(asset_id)
                area = (
                    None
                    if holder is None
                    else _positive(holder.features.get(target_area_key))
                )
                if area is not None:
                    row['area_source'] = AREA_FROM_FEATURE
            if area is None:
                row['outcome'] = OUTCOME_NO_AREA
                continue
            row['footprint_gross_area_sqft'] = area
            if asset_id in rates:
                row.update(outcome=OUTCOME_OWN_RATE, rate=rates[asset_id])
                values[asset_id] = rates[asset_id] * area
                continue
            named = donors.inventory.get(asset_id)
            donor_id = (
                None if named is None else named.features.get(IMPUTATION_DONOR_KEY)
            )
            if donor_id is not None and donor_id not in buildings.inventory:
                msg = (
                    f'The donor stage names donor {donor_id!r} for asset '
                    f'{asset_id!r} in world {world}, which the stage lacks.'
                )
                raise ValueError(msg)
            if donor_id in rates:
                row.update(
                    outcome=OUTCOME_IMPUTATION_DONOR,
                    donor_id=donor_id,
                    rate=rates[donor_id],
                )
                values[asset_id] = rates[donor_id] * area
                continue
            occupancy = asset.features.get(self.occupancy_key)
            if not isinstance(occupancy, str) or is_missing_value(occupancy):
                row['outcome'] = OUTCOME_NO_CLASS
                continue
            pending.setdefault(occupancy, []).append((asset_id, row))
        if pending:
            drawn = self._same_class_neighbor_rate(buildings, rates, pending, world)
            for asset_id, rate in drawn.items():
                values[asset_id] = rate * rows[asset_id]['footprint_gross_area_sqft']
        return values, list(rows.values())

    def _same_class_neighbor_rate(
        self,
        buildings: AssetInventory,
        rates: Mapping[Hashable, float],
        pending: Mapping[str, list[tuple[Hashable, dict[str, Any]]]],
        world: int,
    ) -> dict[Hashable, float]:
        """
        Draw a building of its occupancy class for each building without a rate yet.

        Each occupancy class builds one pool over the world's buildings of
        that occupancy class that have their own rate and whose occupancy
        class carries a source code among :data:`NEIGHBOR_CLASS_CODES`, and
        records the sites of the buildings that draw.

        Args:
            buildings (AssetInventory):
                The stage's realization of the world.
            rates (Mapping[Hashable, float]):
                Each building's own rate in the world, where it has one.
            pending (Mapping[str, list[tuple[Hashable, dict[str, Any]]]]):
                Per occupancy class, each building that draws, with its log
                row, which the draw completes.
            world (int):
                The world the draw reads.

        Returns:
            dict[Hashable, float]:
                The neighbor's rate per building that drew one.
        """
        position = self.source_features.index(self.occupancy_key)
        candidates: dict[str, dict[Hashable, float]] = {}
        for asset_id in sorted(rates, key=str):
            features = buildings.inventory[asset_id].features
            occupancy = features.get(self.occupancy_key)
            if (
                isinstance(occupancy, str)
                and occupancy in pending
                and read_source_string(
                    asset_id,
                    features,
                    source_string_key=self.source_string_key,
                    source_features=self.source_features,
                )[position]
                in NEIGHBOR_CLASS_CODES
            ):
                candidates.setdefault(occupancy, {})[asset_id] = rates[asset_id]

        drawn = {}
        for occupancy in sorted(pending):
            drawing = _draw_inventory(
                buildings,
                occupancy,
                dict.fromkeys(asset_id for asset_id, _ in pending[occupancy]),
                self.stories_key,
            )
            pool = self._imputer.pool(
                _draw_inventory(
                    buildings,
                    occupancy,
                    candidates.get(occupancy, {}),
                    self.stories_key,
                ),
                drawing,
            )
            for asset_id, row in pending[occupancy]:
                draw = self._imputer.draw(
                    pool,
                    asset_id,
                    drawing.inventory[asset_id].features,
                    (_RATE_KEY,),
                    world=world,
                    status=None,
                    purpose_prefix=NEIGHBOR_PURPOSE_PREFIX,
                )
                row.update(
                    outcome=draw.outcome,
                    donor_id=draw.donor_id,
                    distance_m=draw.distance_m,
                )
                # A building the draw cannot locate gets no value, and its log
                # row holds no area.
                if draw.outcome == OUTCOME_NO_AREA:
                    row.update(area_source=None, footprint_gross_area_sqft=math.nan)
                if draw.donor_id is not None:
                    row['rate'] = dict(draw.values)[_RATE_KEY]
                    drawn[asset_id] = row['rate']
        return drawn

    def _rate(self, asset_id: Hashable, features: Mapping[str, Any]) -> float | None:
        """
        Return a building's own value per square foot of its gross area.

        A value gives a rate where its source code is among
        :data:`RATE_VALUE_CODES` and it and ``gross_area_key`` are numbers
        above zero.

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
                If the building holds a value and no source string that
                :func:`~brails.types.asset_inventory.read_source_string`
                reads.
        """
        if is_missing_value(features.get(self.value_key)):
            return None
        codes = read_source_string(
            asset_id,
            features,
            source_string_key=self.source_string_key,
            source_features=self.source_features,
        )
        if codes[self.source_features.index(self.value_key)] not in RATE_VALUE_CODES:
            return None
        value = _positive(features.get(self.value_key))
        gross = _positive(features.get(self.gross_area_key))
        if value is None or gross is None:
            return None
        return value / gross


def footprint_gross_area(
    asset_id: Hashable, asset: Asset, *, stories_key: str = STORIES_KEY
) -> tuple[float | None, str | None]:
    """
    Return a building's gross area from its footprint and how it was computed.

    The gross area is the plan area of the building's footprint times its
    story count where that is a finite number above zero, and the plan area
    alone otherwise.

    Args:
        asset_id (Hashable):
            The building, for the error message.
        asset (Asset):
            The building in one world.
        stories_key (str, optional):
            The story count feature. Defaults to
            :data:`~brails.constants.STORIES_KEY`.

    Returns:
        tuple[float | None, str | None]:
            The gross area in square feet and one of
            :data:`~brails.constants.AREA_FROM_STORIES`
            and
            :data:`~brails.constants.AREA_FROM_PLAN_AREA`,
            or ``(None, None)`` without a plan area.

    Raises:
        ValueError:
            If the coordinates cannot be read as a footprint.
    """
    # ACSUnitBudget's gross area on a footprint without a GrossArea;
    # the budget reads GrossArea first, and this never does.
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
    stories = _positive(asset.features.get(stories_key))
    if stories is None:
        return plan, AREA_FROM_PLAN_AREA
    return plan * stories, AREA_FROM_STORIES


def _draw_inventory(
    buildings: AssetInventory,
    occupancy: str,
    members: Mapping[Hashable, float | None],
    stories_key: str,
) -> AssetInventory:
    """
    Write buildings of one occupancy class in the features the draw reads.

    Args:
        buildings (AssetInventory):
            The stage's realization of the world.
        occupancy (str):
            The occupancy class.
        members (Mapping[Hashable, float | None]):
            The buildings, each with its rate, or ``None`` for one that draws.
        stories_key (str):
            The story count feature of ``buildings``.

    Returns:
        AssetInventory:
            One asset per member, sharing its coordinates, with the
            occupancy class, the story count and the block group where the
            building holds them, and the rate.
    """
    inventory = AssetInventory()
    for asset_id, rate in members.items():
        asset = buildings.inventory[asset_id]
        features: dict[str, Any] = {OCCUPANCY_KEY: occupancy}
        if stories_key in asset.features:
            features[STORIES_KEY] = asset.features[stories_key]
        if BLOCK_GROUP_KEY in asset.features:
            features[BLOCK_GROUP_KEY] = asset.features[BLOCK_GROUP_KEY]
        if rate is not None:
            features[_RATE_KEY] = rate
        inventory.add_asset(asset_id, Asset(asset_id, asset.coordinates, features))
    return inventory


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
