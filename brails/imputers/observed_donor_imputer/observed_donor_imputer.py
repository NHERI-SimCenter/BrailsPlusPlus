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
Fill missing building features by copying them from one observed donor.

``DonorPool`` (from
:mod:`~brails.imputers.observed_donor_imputer.donor_pool`), ``DonorDraw``
and ``DonorImputationResult`` (from
:mod:`~brails.imputers.observed_donor_imputer.donor_records`) can also be
imported from this module.

.. autosummary::

      ObservedDonorImputer
"""

from __future__ import annotations

import math
import secrets
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric

from brails.aggregators.points_to_polygons.nsi_footprint_join import UNIT_CLASSES
from brails.imputers.imputation import Imputation
from brails.imputers.observed_donor_imputer.donor_pool import (
    _STATUS_INSIDE,
    _STATUS_OUTSIDE,
    _STATUS_UNKNOWN,
    DonorPool,
    _DonorNeighborhood,
    _DonorSite,
    _DonorWorldView,
    _is_missing,
    _status_code,
    _world_features,
)
from brails.imputers.observed_donor_imputer.donor_records import (  # re-exports
    LOG_COLUMNS,
    OUTCOME_BLOCK_GROUP,
    OUTCOME_DONOR,
    OUTCOME_EXHAUSTED,
    OUTCOME_FEWER_THAN_K,
    OUTCOME_NEAREST_ADMITTED,
    OUTCOME_NO_AREA,
    OUTCOME_NO_DONOR,
    OUTCOME_NONE_ADMITTED,
    OUTCOME_RELAXED,
    ROUND_DONOR,
    ROUND_REPAIR,
    ROUND_STATUS,
    DonorDraw,
    DonorImputationResult,
    _log_row,
)
from brails.imputers.observed_donor_imputer.donor_search import (
    DonorSearch,
    _DonorPick,
)
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils.asset_id import EARTH_RADIUS_M
from brails.utils.deterministic_sampling import asset_key_digest, keyed_uniforms
from brails.utils.world_values import (
    OCCUPANCY_KEY,
    STORIES_KEY,
    UNITS_KEY,
    collapse_realizations,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    from brails.compilers.stage_freeze import StageFreeze

__all__ = [
    'DonorDraw',
    'DonorImputationResult',
    'DonorPool',
    'ObservedDonorImputer',
]

# Features the imputer fills by default.
OCCUPANCY_UNITS_AND_STORIES = (OCCUPANCY_KEY, UNITS_KEY, STORIES_KEY)

# Features a donor must hold.
REQUIRED_DONOR_FEATURES = (OCCUPANCY_KEY, STORIES_KEY)

# Features whose copy makes a target subject to validation by default, of
# those the imputer fills.
VALIDATED_FEATURES = (OCCUPANCY_KEY, STORIES_KEY)

# Occupancy classes that count as residential, the default status values:
# the classes of the single and multi-family walk groups of NSIFootprintJoin.
RESIDENTIAL_CLASSES = UNIT_CLASSES

# Name of the default status, which enters the purpose labels of its round.
RESIDENTIAL_STATUS_NAME = 'residential'

# Purpose labels of the keyed uniforms.
PURPOSE_DONOR = 'observed_donor'
PURPOSE_FALLBACK = 'observed_donor_fallback'
PURPOSE_BATCH = 'observed_donor_batch'

# The great-circle distance the donor pool's tree measures, for one pair.
_HAVERSINE = DistanceMetric.get_metric('haversine')


def _status_purposes(status_name: str) -> tuple[str, str]:
    """
    Return the purpose labels of the status round and of its fallback.

    Args:
        status_name (str):
            The name of the status.

    Returns:
        tuple[str, str]:
            The two labels, formed as the ``status_name`` entry of
            :class:`ObservedDonorImputer` states.
    """
    label = f'{PURPOSE_DONOR}_{status_name}'
    return label, f'{label}_fallback'


# Purpose labels of the status round and its fallback under the default
# status name.
PURPOSE_RESIDENTIAL, PURPOSE_RESIDENTIAL_FALLBACK = _status_purposes(
    RESIDENTIAL_STATUS_NAME
)


class _DonorOutcome(NamedTuple):
    """
    Everything one call of the draw path produced.

    Attributes:
        round_one (DonorDraw | None):
            The status round, when it ran.
        draw (DonorDraw):
            The donor round.
        status_code (int | None):
            The status code the donor round filtered on.
    """

    round_one: DonorDraw | None
    draw: DonorDraw
    status_code: int | None


class ObservedDonorImputer(Imputation):
    """
    Fill missing features of each target from one observed donor per world.

    Donors are buildings with observed values. An asset has the status when
    its value of ``status_key`` is among ``status_values``, by default when
    its occupancy class is residential. For each target and world,
    the donors within ``cap_m`` that match the target's status and its
    known soft-filter features are ranked by a distance over location, log
    plan area and story count. One of the ``k`` nearest donors is picked with
    inverse-distance weights and using the target's keyed uniform, and every
    missing feature is copied from it. A target whose ``status_key`` value
    is unknown and whose status is not given, first takes its status from a
    pick among all donors within the cap. A target with no donor within the
    cap draws one donor of its own block group with equal weights. The donor
    in such cases may lie beyond the cap in a rural block group that covers
    a large area. Every draw depends on the target's id, the seed, the world,
    the iteration and the donors, but never on processing order.

    Every constructor argument is stored as an attribute of the same name,
    read-only by convention, and ``seed`` holds the seed used.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        features: Sequence[str] = OCCUPANCY_UNITS_AND_STORIES,
        soft_filters: Sequence[str] = (OCCUPANCY_KEY,),
        required: Sequence[str] = REQUIRED_DONOR_FEATURES,
        status_key: str = OCCUPANCY_KEY,
        status_values: Sequence[Any] = RESIDENTIAL_CLASSES,
        status_name: str = RESIDENTIAL_STATUS_NAME,
        validated: Sequence[str] | None = None,
        k: int = 20,
        cap_m: float = 2000.0,
        location_scale_m: float = 100.0,
        area_weight: float = 1.0,
        stories_key: str | None = STORIES_KEY,
        stories_weight: float = 1.0,
        max_attempts: int = 5,
        n_batches: int = 1,
        seed: int | None = None,
    ) -> None:
        """
        Configure the imputer.

        Args:
            features (Sequence[str], optional):
                The features filled. Defaults to
                :data:`OCCUPANCY_UNITS_AND_STORIES`.
            soft_filters (Sequence[str], optional):
                Features a donor must match where the target holds them,
                relaxed one at a time from the last when fewer than ``k``
                donors match. Defaults to the occupancy class.
            required (Sequence[str], optional):
                Features a donor must hold. Defaults to
                :data:`REQUIRED_DONOR_FEATURES`.
            status_key (str, optional):
                The feature an asset's status is read from, one of
                ``features``, ``soft_filters`` or ``required``. A donor
                without it is never drawn. Defaults to
                :data:`OCCUPANCY_KEY`.
            status_values (Sequence[Any], optional):
                The values of ``status_key`` that give an asset the status.
                Defaults to :data:`RESIDENTIAL_CLASSES`.
            status_name (str, optional):
                The name of the status, which forms the purpose labels of
                its round, ``f'{PURPOSE_DONOR}_{status_name}'`` and that
                label followed by ``'_fallback'``. Defaults to
                :data:`RESIDENTIAL_STATUS_NAME`, which gives
                :data:`PURPOSE_RESIDENTIAL` and
                :data:`PURPOSE_RESIDENTIAL_FALLBACK`.
            validated (Sequence[str], optional):
                The features, among ``features``, whose copy makes a target
                subject to the validation of :meth:`impute_stage`. ``None``
                takes those of :data:`VALIDATED_FEATURES` that ``features``
                holds, possibly none.
            k (int, optional):
                The number of nearest donors picked among. Defaults to 20.
            cap_m (float, optional):
                The donor distance cap in meters, the same for every target.
                Defaults to 2000.
            location_scale_m (float, optional):
                Meters of separation counted as one unit of distance.
                Defaults to 100.
            area_weight (float, optional):
                Weight of the squared difference of log plan areas.
                Defaults to 1.
            stories_key (str | None, optional):
                The feature holding the story count the distance compares,
                on targets and donors alike. A side without a count there
                adds nothing to the distance, and ``None`` drops the term.
                Defaults to :data:`STORIES_KEY`.
            stories_weight (float, optional):
                Weight of the squared difference of story counts.
                Defaults to 1.
            max_attempts (int, optional):
                Number of repair attempts before giving up or scanning the
                nearest donors. Defaults to 5.
            n_batches (int, optional):
                The number of feedback batches of :meth:`impute_stage`.
                Defaults to 1, no feedback.
            seed (int, optional):
                The seed. ``None`` draws one from system entropy.

        Raises:
            ValueError:
                If ``k``, ``max_attempts`` or ``n_batches`` is below 1,
                ``cap_m`` or ``location_scale_m`` is not positive, a weight
                is negative, ``status_key`` is not among ``features``,
                ``soft_filters`` and ``required``,
                ``status_values`` is a string or empty, ``status_name`` is
                empty or gives a purpose label the imputer uses for another
                draw, or ``validated`` is a string or names a feature not
                in ``features``.
        """
        if k < 1 or max_attempts < 1 or n_batches < 1:
            msg = 'k, max_attempts and n_batches must be at least 1.'
            raise ValueError(msg)
        if cap_m <= 0 or location_scale_m <= 0:
            msg = 'cap_m and location_scale_m must be positive.'
            raise ValueError(msg)
        if min(area_weight, stories_weight) < 0:
            msg = 'The weights must not be negative.'
            raise ValueError(msg)
        self.features = tuple(features)
        self.soft_filters = tuple(soft_filters)
        self.required = tuple(required)
        if not isinstance(status_values, str):
            status_values = tuple(status_values)
        self._check_status(status_key, status_values, status_name)
        self.status_key = status_key
        self.status_values = status_values
        self.status_name = status_name
        self._status_purposes = _status_purposes(status_name)
        if validated is None:
            validated = tuple(f for f in VALIDATED_FEATURES if f in self.features)
        elif not isinstance(validated, str):
            validated = tuple(validated)
        if isinstance(validated, str) or not set(validated) <= set(self.features):
            msg = 'validated must be a sequence of the features filled.'
            raise ValueError(msg)
        self.validated = validated
        self.k = int(k)
        self.cap_m = float(cap_m)
        self.location_scale_m = float(location_scale_m)
        self.area_weight = float(area_weight)
        self.stories_key = stories_key
        self.stories_weight = float(stories_weight)
        self.max_attempts = int(max_attempts)
        self.n_batches = int(n_batches)
        self.seed = secrets.randbits(64) if seed is None else int(seed)
        self._search = DonorSearch(
            k=self.k,
            cap_m=self.cap_m,
            location_scale_m=self.location_scale_m,
            area_weight=self.area_weight,
            stories_key=self.stories_key,
            stories_weight=self.stories_weight,
            seed=self.seed,
        )

    def _check_status(
        self, status_key: str, status_values: Sequence[Any], status_name: str
    ) -> None:
        """
        Check the status arguments of the constructor.

        Args:
            status_key (str):
                The feature the status is read from.
            status_values (Sequence[Any]):
                The values that give the status.
            status_name (str):
                The name of the status.

        Raises:
            ValueError:
                In the status cases the constructor lists.
        """
        if status_key not in (*self.features, *self.soft_filters, *self.required):
            msg = (
                f'status_key {status_key!r} must be one of the features, soft '
                'filters or required features.'
            )
            raise ValueError(msg)
        if isinstance(status_values, str) or not status_values:
            msg = 'status_values must be a non-empty sequence of values.'
            raise ValueError(msg)
        labels = _status_purposes(status_name)
        if (
            not isinstance(status_name, str)
            or not status_name
            or set(labels) & {PURPOSE_DONOR, PURPOSE_FALLBACK, PURPOSE_BATCH}
        ):
            msg = (
                f'status_name {status_name!r} must be a non-empty string whose '
                f'labels {labels} differ from those of the other draws.'
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Public entry points

    def impute(
        self, input_inventory: AssetInventory, n_worlds: int = 1
    ) -> AssetInventory:
        """
        Fill the missing features of an inventory from its own assets.

        Every asset may be a target and a donor. As a donor it supplies
        only values it holds in the input, so no imputed value feeds
        another target.

        Args:
            input_inventory (AssetInventory):
                The inventory, with one possible world. It is not modified.
            n_worlds (int, optional):
                The number of possible worlds drawn. Defaults to 1.

        Returns:
            AssetInventory:
                A copy of the inventory with the copied values filled in,
                declaring ``n_worlds`` possible worlds.

        Raises:
            ValueError:
                If the inventory holds more than one possible world, or
                :attr:`n_batches` is above 1, which needs a freeze.
        """
        result = self.impute_stage(
            input_inventory, input_inventory, n_worlds=n_worlds
        )
        output = deepcopy(input_inventory)
        for asset_id in result.imputed.get_asset_ids():
            output.add_asset_features(
                asset_id, result.imputed.inventory[asset_id].features
            )
        output.n_pw = n_worlds
        return output

    def impute_stage(
        self,
        targets: AssetInventory,
        donors: AssetInventory | None = None,
        *,
        n_worlds: int = 1,
        status: Mapping[Any, bool] | None = None,
        validate: Callable[[Any, int, Mapping[str, Any]], bool] | None = None,
        freeze: StageFreeze | None = None,
        donor_stage: str | None = None,
        batch_prefix: str = 'imputed',
    ) -> DonorImputationResult:
        """
        Fill the missing features of targets from a donor inventory.

        A target is an asset of ``targets`` missing a feature of
        :attr:`features` in a world; it receives only the features it is
        missing, all from one donor in that world. World ``w`` reads the
        targets' realization ``w`` and draws with ``world=w``.

        With ``validate``, each target for which this run copied a feature of
        :attr:`validated` is checked after the copy. A rejected target takes
        one repair :meth:`draw` at iteration 1. The repair re-draws only the
        copied features, excludes the rejected donor, keeps the status of the
        rejected draw and makes no nearest-first scan. Like any draw, it
        falls back to the target's block group, which may reach past the
        cap, once no candidate within the cap is left untried. A target
        still rejected keeps its first values and is logged
        :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_EXHAUSTED`.

        With ``freeze``, the observed donors are read from it as
        ``donor_stage``, and the targets run in :attr:`n_batches` batches,
        one batch when :attr:`n_batches` is 1. A target's batch is ``1 +
        floor(u * n_batches)`` for its keyed uniform under
        :data:`PURPOSE_BATCH` at world 0, so it is the same in every world.
        Batch ``j`` draws from the observed donors and from the frozen
        batches ``1`` to ``j - 1``. A target of an earlier batch donates, in
        world ``w``, its own features with the values copied to it in world
        ``w`` in their place, and it replaces its own observed record when
        it is also an observed donor. Each batch is validated and repaired,
        then frozen as ``f'{batch_prefix}_batch{j}'``, an empty batch
        included. Nothing else is frozen.

        Args:
            targets (AssetInventory):
                The targets, with one possible world or ``n_worlds``. Not
                modified.
            donors (AssetInventory, optional):
                The observed donors, with one possible world. Not modified.
                Required without ``freeze``.
            n_worlds (int, optional):
                The number of worlds. Defaults to 1.
            status (Mapping[Any, bool], optional):
                A status per target id, ``True`` for a value among
                :attr:`status_values`, which replaces the status round for
                those targets.
            validate (Callable[[Any, int, Mapping[str, Any]], bool], optional):
                ``validate(asset_id, world, features)``, called with the
                target's features in that world after the copy.
            freeze (StageFreeze, optional):
                The freeze the donors are read from and the batches written
                to.
            donor_stage (str, optional):
                The frozen stage holding the observed donors. Required with
                ``freeze``.
            batch_prefix (str, optional):
                The prefix of the batch names. Defaults to ``'imputed'``.

        Returns:
            DonorImputationResult:
                The copied values as an inventory with ``n_worlds`` possible
                worlds, assets in ascending order of the id as a string. A
                feature equal in every world is stored as a scalar and
                otherwise as a list with ``None`` where no value was copied.
                With ``freeze``, the inventory is the union of the frozen
                batches. The result also carries the log, the pool, the seed
                and the names of the frozen batches.

        Raises:
            ValueError:
                If ``n_worlds`` is below 1; if the observed donors hold more
                than one possible world; if ``targets`` holds a number of
                worlds other than 1 and ``n_worlds``; if both ``donors`` and
                ``freeze`` are given, or neither; if ``freeze`` is given
                without ``donor_stage``; or if :attr:`n_batches` is above 1
                without ``freeze``.
            LookupError:
                If the freeze lacks the donor stage or an earlier batch.
        """
        self._check_stage_arguments(donors, freeze, donor_stage)
        observed = donors if freeze is None else freeze.read(donor_stage)
        self._check_worlds(targets, observed, n_worlds)
        target_ids = sorted(targets.inventory, key=str)
        if freeze is None:
            groups = [target_ids]
        else:
            numbers = self._batch_numbers(target_ids)
            groups = [
                [a for a, n in zip(target_ids, numbers) if n == batch]
                for batch in range(1, self.n_batches + 1)
            ]
        observed_records = {
            asset_id: (asset_id, asset, 1, observed.vector_features)
            for asset_id, asset in observed.inventory.items()
        }
        names: list[str] = []
        rows: list[dict[str, Any]] = []
        imputed = None
        pool = None
        for batch, members in enumerate(groups, start=1):
            records = dict(observed_records)
            for name in names:
                for record in self._batch_records(
                    freeze.read(name), targets, n_worlds
                ):
                    records[record[0]] = record
            # the batch's targets as their own inventory
            batch_targets = AssetInventory(vector_features=targets.vector_features)
            for asset_id in members:
                batch_targets.inventory[asset_id] = targets.inventory[asset_id]
            batch_targets.n_pw = targets.n_pw
            pool = self._build_pool(list(records.values()), batch_targets)
            copied, log = self._run_targets(
                pool,
                targets,
                members,
                n_worlds=n_worlds,
                status=status,
                validate=validate,
            )
            rows.extend(log)
            imputed = self._imputed_inventory(targets, copied, n_worlds)
            if freeze is not None:
                name = f'{batch_prefix}_batch{batch}'
                freeze.freeze(name, imputed)
                names.append(name)
        if freeze is not None:
            # the batch outputs, merged in id order
            assets = {
                asset_id: asset
                for name in names
                for asset_id, asset in freeze.read(name).inventory.items()
            }
            imputed = AssetInventory(vector_features=targets.vector_features)
            for asset_id in sorted(assets, key=str):
                imputed.add_asset(asset_id, deepcopy(assets[asset_id]))
            imputed.n_pw = n_worlds
        return DonorImputationResult(
            imputed,
            pd.DataFrame(rows, columns=list(LOG_COLUMNS)),
            pool,
            self.seed,
            tuple(names),
        )

    def _check_stage_arguments(
        self,
        donors: AssetInventory | None,
        freeze: StageFreeze | None,
        donor_stage: str | None,
    ) -> None:
        """
        Check that the donor arguments of :meth:`impute_stage` can run.

        Args:
            donors (AssetInventory | None):
                The donors given directly.
            freeze (StageFreeze | None):
                The freeze given.
            donor_stage (str | None):
                The donor stage given.

        Raises:
            ValueError:
                In the argument cases :meth:`impute_stage` lists.
        """
        if donors is not None and freeze is not None:
            msg = 'Pass the donors directly or through a freeze, not both.'
            raise ValueError(msg)
        if freeze is None and self.n_batches > 1:
            msg = (
                f'n_batches={self.n_batches} needs a freeze and a donor_stage '
                'to read earlier batches through.'
            )
            raise ValueError(msg)
        if freeze is not None and donor_stage is None:
            msg = 'A freeze needs the donor_stage the observed donors are frozen as.'
            raise ValueError(msg)
        if freeze is None and donors is None:
            msg = 'Pass the observed donors, or a freeze and a donor_stage.'
            raise ValueError(msg)

    def _batch_numbers(self, asset_ids: Iterable[Hashable]) -> list[int]:
        """
        Return each target's feedback batch.

        Args:
            asset_ids (Iterable[Hashable]):
                The targets.

        Returns:
            list[int]:
                One batch number from 1 to :attr:`n_batches` per target, in
                the order given, as :meth:`impute_stage` describes.
        """
        asset_ids = list(asset_ids)
        if not asset_ids:
            return []
        uniforms = keyed_uniforms(
            asset_key_digest(asset_ids),
            seed=self.seed,
            purpose=PURPOSE_BATCH,
            world=0,
        )
        return [min(1 + int(u * self.n_batches), self.n_batches) for u in uniforms]

    def pool(
        self, donors: AssetInventory, targets: AssetInventory | None = None
    ) -> DonorPool:
        """
        Index donors, and the sites of targets, for :meth:`draw`.

        Args:
            donors (AssetInventory):
                The observed donors, with one possible world. An asset
                lacking a required feature is kept but never drawn.
            targets (AssetInventory, optional):
                The targets whose sites the pool records beside the donors'.

        Returns:
            DonorPool:
                The pool, which only this imputer accepts.

        Raises:
            ValueError:
                If ``donors`` holds more than one possible world.
        """
        if donors.n_pw > 1:
            msg = (
                f'The donor inventory holds {donors.n_pw} possible worlds; '
                'observed donors must hold one.'
            )
            raise ValueError(msg)
        records = [
            (asset_id, donors.inventory[asset_id], 1, donors.vector_features)
            for asset_id in donors.inventory
        ]
        return self._build_pool(records, targets)

    def status_probability(
        self, pool: DonorPool, asset_ids: Iterable[Hashable]
    ) -> dict[Hashable, float]:
        """
        Return the chance that the status round gives each asset the status.

        By default this is the chance of being residential. The chance is that
        of the status round's pick on ``pool`` in world 0 at iteration 0,
        before its uniform is drawn: the inverse-distance weight of the donors
        with the status among the ``k`` nearest donors within :attr:`cap_m`,
        over the weight of all of them. Without a donor within the cap it is
        the share of the donors of the asset's block group with the status,
        which the round's fallback picks with equal weights. Like the round, it
        reads no soft filter and no feature of the asset itself, and the asset
        is never its own donor. Under batch feedback a later batch's round
        reads more donors than an observed pool holds.

        Args:
            pool (DonorPool):
                A pool from :meth:`pool` that records the assets.
            asset_ids (Iterable[Hashable]):
                The assets.

        Returns:
            dict[Hashable, float]:
                The chance per asset, NaN for an asset without area or
                without any donor the round could pick.

        Raises:
            LookupError:
                If the pool does not record an asset.
            ValueError:
                If another imputer built the pool.
        """
        self._check_pool(pool)
        view = pool.view(0)
        chances: dict[Hashable, float] = {}
        for asset_id in asset_ids:
            if asset_id in pool.no_area:
                chances[asset_id] = math.nan
                continue
            site = pool.site(asset_id)
            neighborhood = self._search.neighborhood(pool, site)
            excluded = {asset_id}
            mask = self._search.eligible(
                pool, view, neighborhood.index, excluded, None
            )
            if mask.any():
                ranked, weights = self._search.ranked_weights(neighborhood, mask)
                status = view.status[neighborhood.index[ranked]]
                share = weights[status == _STATUS_INSIDE].sum() / weights.sum()
                chances[asset_id] = float(share)
                continue
            index, base = self._search.block_group_candidates(
                pool, view, site, excluded, None
            )
            members = index[base]
            chances[asset_id] = (
                float(np.mean(view.status[members] == _STATUS_INSIDE))
                if members.size
                else math.nan
            )
        return chances

    def draw(
        self,
        pool: DonorPool,
        asset_id: Hashable,
        features: Mapping[str, Any],
        redraw: Sequence[str] | None = None,
        *,
        world: int,
        iteration: int = 0,
        admit: Callable[[Mapping[str, Any]], bool] | None = None,
        exclude_donors: Iterable[Any] = (),
        status: bool | None = None,
        protected: Iterable[str] = (),
        scan_nearest: bool = True,
        purpose_prefix: str | None = None,
    ) -> DonorDraw:
        """
        Draw one donor for one asset in one world.

        This is the path :meth:`impute_stage` takes for each target, so at
        iteration 0 without ``admit`` it reproduces the stage result,
        including the status round where the stage runs it.

        With ``admit``, the draw makes up to :attr:`max_attempts` attempts.
        Attempt ``a`` picks among the untried candidates, those of the
        block-group fallback included, at iteration ``iteration + a - 1``, and
        ``admit`` judges the picked donor's values for ``redraw``. A rejected
        donor counts as tried. Once no untried candidate lies within the cap,
        an attempt falls back to the block group, whose donors may lie beyond
        the cap. When the attempts are spent or no untried candidate is left,
        :meth:`nearest_admitted` runs if ``scan_nearest`` is set; it scans only
        donors within the cap. Without an admitted donor the outcome is
        :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_EXHAUSTED`,
        with no donor and empty values, and the caller keeps its value.

        Args:
            pool (DonorPool):
                A pool built by this imputer's :meth:`pool`.
            asset_id (Hashable):
                The asset; the pool must record it.
            features (Mapping[str, Any]):
                The asset's scalar features in ``world``.
            redraw (Sequence[str], optional):
                The features to copy, which count as missing for the
                filters. Defaults to the asset's missing features of
                :attr:`features`.
            world (int):
                The world.
            iteration (int, optional):
                The iteration of the first pick. Defaults to 0.
            admit (Callable[[Mapping[str, Any]], bool], optional):
                Judges a candidate's values for ``redraw``.
            exclude_donors (Iterable[Any], optional):
                Donors removed before ranking and before the fallback.
            status (bool, optional):
                The status, ``True`` for a value among
                :attr:`status_values`, which replaces the status round.
            protected (Iterable[str], optional):
                Features the draw may not change.
            scan_nearest (bool, optional):
                Whether a repair ends with :meth:`nearest_admitted`.
                Defaults to ``True``.
            purpose_prefix (str, optional):
                A caller's own prefix for the purpose labels. With it, each
                label becomes ``f'{purpose_prefix}:{label}'``, so the
                caller's uniforms are distinct from the stage's by label.
                Defaults to ``None``, the module's labels.

        Returns:
            DonorDraw:
                The draw.

        Raises:
            LookupError:
                If the pool does not record ``asset_id``.
            ValueError:
                If a feature is both in ``redraw`` and ``protected``, there
                is nothing to draw, or another imputer built the pool.
        """
        self._check_pool(pool)
        return self._draw(
            pool,
            asset_id,
            features,
            redraw,
            world=world,
            iteration=iteration,
            admit=admit,
            exclude_donors=exclude_donors,
            status=status,
            protected=protected,
            scan_nearest=scan_nearest,
            purpose_prefix=purpose_prefix,
        ).draw

    def nearest_admitted(
        self,
        pool: DonorPool,
        asset_id: Hashable,
        features: Mapping[str, Any],
        redraw: Sequence[str],
        *,
        world: int,
        admit: Callable[[Mapping[str, Any]], bool],
        protected: Iterable[str] = (),
        status: bool | None = None,
    ) -> DonorDraw:
        """
        Scan the donors within the cap, nearest first, for the first admitted.

        The scan uses no uniform. It covers every donor within the cap
        that holds the required features and has the target's status,
        ordered by donor distance and then by id as a string, and
        copies only ``redraw``.

        Args:
            pool (DonorPool):
                A pool built by this imputer's :meth:`pool`.
            asset_id (Hashable):
                The asset.
            features (Mapping[str, Any]):
                The asset's scalar features in ``world``.
            redraw (Sequence[str]):
                The features to copy.
            world (int):
                The world.
            admit (Callable[[Mapping[str, Any]], bool]):
                Judges a candidate's values for ``redraw``.
            protected (Iterable[str], optional):
                Features the scan may not change.
            status (bool, optional):
                The status, needed because the scan matches it even when
                :attr:`status_key` is re-drawn. Defaults to the status of
                the asset's :attr:`status_key` value when ``redraw`` does
                not hold the key.

        Returns:
            DonorDraw:
                Outcome
                :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_NEAREST_ADMITTED`
                with the admitted donor, or
                :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_NONE_ADMITTED`.

        Raises:
            LookupError:
                If the pool does not record ``asset_id``.
            ValueError:
                If a feature is both in ``redraw`` and ``protected``, the
                status is neither given nor known, or another imputer built
                the pool.
        """
        self._check_pool(pool)
        redraw = self._check_redraw(features, redraw, protected)
        if asset_id in pool.no_area:
            return _empty_draw(asset_id, world, 0, OUTCOME_NO_AREA)
        site = pool.site(asset_id)
        status_code = self._known_status(features, redraw, status=status)
        if status_code is None:
            msg = (
                f'The scan needs the {self.status_name} status: pass '
                f'status=, or keep {self.status_key!r} out of redraw.'
            )
            raise ValueError(msg)
        return self._nearest_admitted(
            pool,
            pool.view(world),
            site,
            asset_id,
            redraw,
            world=world,
            iteration=0,
            admit=admit,
            status_code=status_code,
            excluded={asset_id},
            attempts=0,
        )

    # ------------------------------------------------------------------
    # Pool construction

    def _build_pool(
        self,
        records: Sequence[tuple[Any, Asset, int, Sequence[str]]],
        targets: AssetInventory | None,
    ) -> DonorPool:
        """
        Build a pool from donor records and an optional target inventory.

        Args:
            records (Sequence[tuple[Any, Asset, int, Sequence[str]]]):
                ``(asset_id, asset, n_pw, vector_features)`` per donor.
            targets (AssetInventory | None):
                The targets.

        Returns:
            DonorPool:
                The pool.
        """
        keep = (*self.features, *self.soft_filters, *self.required)
        donors = []
        no_area = set()
        for asset_id, asset, n_pw, _ in sorted(records, key=lambda r: str(r[0])):
            site = self._search.site(asset)
            if site is None:
                no_area.add(asset_id)
                continue
            raw = {key: asset.features[key] for key in keep if key in asset.features}
            donors.append((asset_id, site, raw, n_pw))
        vector_features = tuple(
            dict.fromkeys(f for record in records for f in record[3])
        )
        target_sites = {}
        if targets is not None:
            donor_ids = {asset_id for asset_id, _, _, _ in donors}
            for asset_id, asset in targets.inventory.items():
                if asset_id in donor_ids or asset_id in no_area:
                    continue
                site = self._search.site(asset)
                if site is None:
                    no_area.add(asset_id)
                else:
                    target_sites[asset_id] = site
        return DonorPool(
            imputer=self,
            donors=donors,
            targets=target_sites,
            no_area=no_area,
            vector_features=vector_features,
            features=self.features,
            soft_filters=self.soft_filters,
            required=self.required,
            status_key=self.status_key,
            status_values=self.status_values,
        )

    def _batch_records(
        self,
        batch: AssetInventory,
        targets: AssetInventory,
        n_worlds: int,
    ) -> list[tuple[Hashable, Asset, int, Sequence[str]]]:
        """
        Turn a frozen batch into donor records.

        Each record holds the target's own features, its copied values
        taking their place in the worlds where a value was copied.

        Args:
            batch (AssetInventory):
                A frozen batch, holding copied values only.
            targets (AssetInventory):
                The targets, for their own features and coordinates.
            n_worlds (int):
                The number of worlds.

        Returns:
            list[tuple[Hashable, Asset, int, Sequence[str]]]:
                ``(asset_id, asset, n_worlds, vector_features)`` per asset
                of the batch.
        """
        records = []
        for asset_id, imputed in batch.inventory.items():
            target = targets.inventory[asset_id]
            features = dict(target.features)
            for key, value in imputed.features.items():
                if not isinstance(value, list):
                    features[key] = value
                    continue
                own = features.get(key)
                features[key] = [
                    copied
                    if copied is not None
                    else _world_features(
                        {key: own}, w, targets.n_pw, targets.vector_features
                    ).get(key)
                    for w, copied in enumerate(value)
                ]
            asset = Asset(asset_id, target.coordinates, features)
            records.append((asset_id, asset, n_worlds, targets.vector_features))
        return records

    # ------------------------------------------------------------------
    # The draw path

    def _check_worlds(
        self, targets: AssetInventory, donors: AssetInventory | None, n_worlds: int
    ) -> None:
        """
        Check the numbers of possible worlds of a stage run.

        Args:
            targets (AssetInventory):
                The targets.
            donors (AssetInventory | None):
                The observed donors, if given directly.
            n_worlds (int):
                The number of worlds requested.

        Raises:
            ValueError:
                In the cases :meth:`impute_stage` lists.
        """
        if (
            isinstance(n_worlds, bool)
            or not isinstance(n_worlds, int)
            or n_worlds < 1
        ):
            msg = f'n_worlds must be a positive integer, not {n_worlds!r}.'
            raise ValueError(msg)
        if donors is not None and donors.n_pw > 1:
            msg = (
                f'The donor inventory holds {donors.n_pw} possible worlds; '
                'observed donors must hold one.'
            )
            raise ValueError(msg)
        if targets.n_pw not in (1, n_worlds):
            msg = (
                f'The targets hold {targets.n_pw} possible worlds; pass '
                f'n_worlds={targets.n_pw} or targets with one world.'
            )
            raise ValueError(msg)

    def _run_targets(
        self,
        pool: DonorPool,
        targets: AssetInventory,
        target_ids: Sequence[Any],
        *,
        n_worlds: int,
        status: Mapping[Any, bool] | None,
        validate: Callable[[Any, int, Mapping[str, Any]], bool] | None,
    ) -> tuple[dict[Any, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
        """
        Draw, copy and validate every target in every world.

        Args:
            pool (DonorPool):
                The pool.
            targets (AssetInventory):
                The targets.
            target_ids (Sequence[Any]):
                The targets to run, in ascending order of the id as a
                string.
            n_worlds (int):
                The number of worlds.
            status (Mapping[Any, bool] | None):
                Known statuses.
            validate (Callable | None):
                The validation test.

        Returns:
            tuple[dict, list[dict[str, Any]]]:
                The copied values per target and world, and the log rows.
        """
        copied: dict[Any, dict[int, dict[str, Any]]] = {}
        log: list[dict[str, Any]] = []
        for asset_id in target_ids:
            raw = targets.inventory[asset_id].features
            neighborhood = None
            for world in range(n_worlds):
                features = _world_features(
                    raw, world, targets.n_pw, targets.vector_features
                )
                missing = [f for f in self.features if _is_missing(features, f)]
                if not missing:
                    continue
                if neighborhood is None and asset_id not in pool.no_area:
                    neighborhood = self._search.neighborhood(
                        pool, pool.site(asset_id)
                    )
                outcome = self._draw(
                    pool,
                    asset_id,
                    features,
                    missing,
                    world=world,
                    iteration=0,
                    status=None if status is None else status.get(asset_id),
                    neighborhood=neighborhood,
                )
                if outcome.round_one is not None:
                    log.append(_log_row(outcome.round_one, ROUND_STATUS))
                log.append(_log_row(outcome.draw, ROUND_DONOR))
                values = dict(outcome.draw.values)
                if (
                    validate is not None
                    and any(feature in values for feature in self.validated)
                    and not validate(asset_id, world, {**features, **values})
                ):
                    repair = self._repair(
                        pool,
                        asset_id,
                        features,
                        outcome,
                        world,
                        validate,
                        neighborhood,
                    )
                    log.append(_log_row(repair, ROUND_REPAIR))
                    if repair.outcome != OUTCOME_EXHAUSTED:
                        values = dict(repair.values)
                if values:
                    copied.setdefault(asset_id, {})[world] = values
        return copied, log

    def _repair(
        self,
        pool: DonorPool,
        asset_id: Hashable,
        features: Mapping[str, Any],
        outcome: _DonorOutcome,
        world: int,
        validate: Callable[[Any, int, Mapping[str, Any]], bool],
        neighborhood: _DonorNeighborhood | None,
    ) -> DonorDraw:
        """
        Re-draw the copied features of a target its validation rejected.

        Args:
            pool (DonorPool):
                The pool.
            asset_id (Hashable):
                The target.
            features (Mapping[str, Any]):
                Its features in ``world`` before the copy.
            outcome (_DonorOutcome):
                The rejected draw.
            world (int):
                The world.
            validate (Callable):
                The validation test.
            neighborhood (_DonorNeighborhood | None):
                The target's donors within the cap, with their distances.

        Returns:
            DonorDraw:
                The repair draw,
                :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_EXHAUSTED`
                when no donor passes.
        """
        first = outcome.draw

        def admit(values: Mapping[str, Any]) -> bool:
            """
            Validate the target with a candidate's values.

            Args:
                values (Mapping[str, Any]):
                    The candidate's values.

            Returns:
                bool:
                    The validation result.
            """
            return bool(validate(asset_id, world, {**features, **values}))

        return self._draw(
            pool,
            asset_id,
            features,
            tuple(first.values),
            world=world,
            iteration=1,
            admit=admit,
            exclude_donors=(first.donor_id,),
            status_code=outcome.status_code,
            scan_nearest=False,
            neighborhood=neighborhood,
        ).draw

    def _check_redraw(
        self,
        features: Mapping[str, Any],
        redraw: Sequence[str] | None,
        protected: Iterable[str],
    ) -> tuple[str, ...]:
        """
        Resolve and check the features to copy.

        Args:
            features (Mapping[str, Any]):
                The asset's features.
            redraw (Sequence[str] | None):
                The requested features, ``None`` for the missing ones.
            protected (Iterable[str]):
                Features that may not change.

        Returns:
            tuple[str, ...]:
                The features to copy.

        Raises:
            ValueError:
                If a feature is in both ``redraw`` and ``protected``, or none
                is left to copy.
        """
        if redraw is None:
            redraw = [f for f in self.features if _is_missing(features, f)]
        elif isinstance(redraw, str):
            redraw = [redraw]
        redraw = tuple(dict.fromkeys(redraw))
        clash = sorted(set(redraw) & set(protected))
        if clash:
            msg = f'Features {clash} are both to be re-drawn and protected.'
            raise ValueError(msg)
        if not redraw:
            msg = (
                'Nothing to draw: redraw is empty, or the asset misses none '
                'of the features.'
            )
            raise ValueError(msg)
        return redraw

    def _check_pool(self, pool: DonorPool) -> None:
        """
        Refuse a pool another imputer built.

        Args:
            pool (DonorPool):
                The pool given to a public method.

        Raises:
            ValueError:
                If :meth:`pool` of another imputer built the pool.
        """
        if pool.imputer is not self:
            msg = 'The pool was built by another imputer; build it with this one.'
            raise ValueError(msg)

    def _known_status(
        self,
        features: Mapping[str, Any],
        redraw: Sequence[str],
        *,
        status: bool | None,
    ) -> int | None:
        """
        Return the status code a draw filters on without the status round.

        Args:
            features (Mapping[str, Any]):
                The asset's features.
            redraw (Sequence[str]):
                The features to copy.
            status (bool | None):
                A given status.

        Returns:
            int | None:
                The status code, ``None`` when neither given nor known.
        """
        if status is not None:
            return _STATUS_INSIDE if status else _STATUS_OUTSIDE
        if self.status_key in redraw:
            return None
        code = _status_code(features, self.status_key, self.status_values)
        return None if code == _STATUS_UNKNOWN else code

    def _draw(  # noqa: C901
        self,
        pool: DonorPool,
        asset_id: Hashable,
        features: Mapping[str, Any],
        redraw: Sequence[str] | None,
        *,
        world: int,
        iteration: int = 0,
        admit: Callable[[Mapping[str, Any]], bool] | None = None,
        exclude_donors: Iterable[Any] = (),
        status: bool | None = None,
        status_code: int | None = None,
        protected: Iterable[str] = (),
        scan_nearest: bool = True,
        neighborhood: _DonorNeighborhood | None = None,
        purpose_prefix: str | None = None,
    ) -> _DonorOutcome:
        """
        Run the draw path of :meth:`draw` and keep its status round.

        Args:
            pool (DonorPool):
                The pool.
            asset_id (Hashable):
                The asset.
            features (Mapping[str, Any]):
                Its scalar features in ``world``.
            redraw (Sequence[str] | None):
                The features to copy.
            world (int):
                The world.
            iteration (int, optional):
                The first iteration.
            admit (Callable | None, optional):
                The repair test.
            exclude_donors (Iterable[Any], optional):
                Donors removed.
            status (bool | None, optional):
                A given status.
            status_code (int | None, optional):
                A given status code, which takes precedence over
                ``status``.
            protected (Iterable[str], optional):
                Features that may not change.
            scan_nearest (bool, optional):
                Whether a repair ends with the nearest-first scan.
            neighborhood (_DonorNeighborhood | None, optional):
                The asset's donors within the cap with their distances,
                computed when not given.
            purpose_prefix (str | None, optional):
                The prefix of the purpose labels, as :meth:`draw` takes it.

        Returns:
            _DonorOutcome:
                The status round, if run, the draw and the status code.

        Raises:
            LookupError:
                If the pool does not record ``asset_id``.
            ValueError:
                In the cases :meth:`draw` lists.
        """
        redraw = self._check_redraw(features, redraw, protected)
        if asset_id in pool.no_area:
            return _DonorOutcome(
                None, _empty_draw(asset_id, world, iteration, OUTCOME_NO_AREA), None
            )
        site = pool.site(asset_id)
        view = pool.view(world)
        excluded = {asset_id, *exclude_donors}
        if neighborhood is None:
            neighborhood = self._search.neighborhood(pool, site)
        key = asset_key_digest([asset_id])
        status_labels, donor_labels = (
            tuple(
                label if purpose_prefix is None else f'{purpose_prefix}:{label}'
                for label in labels
            )
            for labels in (
                self._status_purposes,
                (PURPOSE_DONOR, PURPOSE_FALLBACK),
            )
        )

        round_one = None
        if status_code is None:
            status_code = self._known_status(features, redraw, status=status)
        if status_code is None:
            pick = self._select(
                pool,
                view,
                site,
                key,
                neighborhood,
                excluded=excluded,
                status_code=None,
                soft=(),
                purposes=status_labels,
                world=world,
                iteration=iteration,
            )
            round_one = _as_draw(
                pool, asset_id, world, iteration, pick, values={}, attempts=0
            )
            if pick.index is None:
                return _DonorOutcome(
                    round_one,
                    _empty_draw(asset_id, world, iteration, OUTCOME_NO_DONOR),
                    None,
                )
            status_code = int(view.status[pick.index])

        soft = tuple(
            (f, features[f])
            for f in self.soft_filters
            if f not in redraw and not _is_missing(features, f)
        )

        def values_of(index: int) -> dict[str, Any]:
            """
            Return a donor's values for ``redraw``.

            Args:
                index (int):
                    The donor's index.

            Returns:
                dict[str, Any]:
                    The values it holds.
            """
            held = view.donor_values[index]
            return {f: held[f] for f in redraw if f in held}

        if admit is None:
            pick = self._select(
                pool,
                view,
                site,
                key,
                neighborhood,
                excluded=excluded,
                status_code=status_code,
                soft=soft,
                purposes=donor_labels,
                world=world,
                iteration=iteration,
            )
            values = {} if pick.index is None else values_of(pick.index)
            draw = _as_draw(
                pool, asset_id, world, iteration, pick, values=values, attempts=0
            )
            return _DonorOutcome(round_one, draw, status_code)

        tried: set[Any] = set()
        attempts = 0
        for attempt in range(1, self.max_attempts + 1):
            current = iteration + attempt - 1
            pick = self._select(
                pool,
                view,
                site,
                key,
                neighborhood,
                excluded=excluded | tried,
                status_code=status_code,
                soft=soft,
                purposes=donor_labels,
                world=world,
                iteration=current,
            )
            if pick.index is None:
                break
            attempts = attempt
            values = values_of(pick.index)
            if admit(values):
                draw = _as_draw(
                    pool,
                    asset_id,
                    world,
                    current,
                    pick,
                    values=values,
                    attempts=attempt,
                )
                return _DonorOutcome(round_one, draw, status_code)
            tried.add(pool.donor_ids[pick.index])

        if scan_nearest:
            draw = self._nearest_admitted(
                pool,
                view,
                site,
                asset_id,
                redraw,
                world=world,
                iteration=iteration,
                admit=admit,
                status_code=status_code,
                excluded=excluded,
                attempts=attempts,
                neighborhood=neighborhood,
            )
            if draw.outcome == OUTCOME_NEAREST_ADMITTED:
                return _DonorOutcome(round_one, draw, status_code)
        return _DonorOutcome(
            round_one,
            _empty_draw(asset_id, world, iteration, OUTCOME_EXHAUSTED, attempts),
            status_code,
        )

    def _select(
        self,
        pool: DonorPool,
        view: _DonorWorldView,
        site: _DonorSite,
        key: np.ndarray,
        neighborhood: _DonorNeighborhood,
        *,
        excluded: set[Any],
        status_code: int | None,
        soft: Sequence[tuple[str, Any]],
        purposes: tuple[str, str],
        world: int,
        iteration: int,
    ) -> _DonorPick:
        """
        Pick one donor for a target: the main pick, else the fallback.

        Args:
            pool (DonorPool):
                The pool.
            view (_DonorWorldView):
                The donors in this world.
            site (_DonorSite):
                The target's site.
            key (numpy.ndarray):
                The target's key from :func:`asset_key_digest`.
            neighborhood (_DonorNeighborhood):
                The donors within the cap, with their distances.
            excluded (set[Any]):
                Donors that may not be picked.
            status_code (int | None):
                The status code to match, ``None`` for none.
            soft (Sequence[tuple[str, Any]]):
                The soft filters that apply, with the target's values.
            purposes (tuple[str, str]):
                The purpose labels of the main pick and of the fallback.
            world (int):
                The world.
            iteration (int):
                The iteration.

        Returns:
            _DonorPick:
                The pick of :meth:`DonorSearch.select`, or the fallback's
                when it finds none.
        """
        pick = self._search.select(
            pool,
            view,
            key,
            neighborhood,
            excluded=excluded,
            status_code=status_code,
            soft=soft,
            purpose=purposes[0],
            world=world,
            iteration=iteration,
        )
        if pick is not None:
            return pick
        return self._fallback(
            pool,
            view,
            site,
            key,
            excluded,
            status_code,
            soft,
            purposes[1],
            world,
            iteration,
        )

    def _fallback(
        self,
        pool: DonorPool,
        view: _DonorWorldView,
        site: _DonorSite,
        key: np.ndarray,
        excluded: set[Any],
        status_code: int | None,
        soft: Sequence[tuple[str, Any]],
        purpose: str,
        world: int,
        iteration: int,
    ) -> _DonorPick:
        """
        Pick one donor of the target's block group with equal weights.

        Args:
            pool (DonorPool):
                The pool.
            view (_DonorWorldView):
                The donors in this world.
            site (_DonorSite):
                The target's site.
            key (numpy.ndarray):
                The target's key from :func:`asset_key_digest`.
            excluded (set[Any]):
                Donors removed.
            status_code (int | None):
                The status code to match.
            soft (Sequence[tuple[str, Any]]):
                The soft filters, relaxed one at a time from the last until
                a level holds a donor.
            purpose (str):
                The purpose label of the pick.
            world (int):
                The world.
            iteration (int):
                The iteration.

        Returns:
            _DonorPick:
                Outcome
                :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_BLOCK_GROUP`,
                or
                :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_NO_DONOR`
                without a block group or a donor in it.
        """
        no_donor = _DonorPick(None, None, 0, math.nan, math.nan, OUTCOME_NO_DONOR)
        index, base = self._search.block_group_candidates(
            pool, view, site, excluded, status_code
        )
        if index.size == 0:
            return no_donor
        for level, mask in enumerate(self._search.levels(view, index, base, soft)):
            members = index[mask]
            if members.size:
                u = self._search.uniform(key, purpose, world, iteration)
                pick = int(members[min(int(u * members.size), members.size - 1)])
                # great-circle meters to the picked donor
                meters = (
                    _HAVERSINE.pairwise(
                        np.radians([[site.lat, site.lon]]),
                        np.radians([[pool.lat[pick], pool.lon[pick]]]),
                    )[0]
                    * EARTH_RADIUS_M
                )
                distance = self._search.distances(
                    pool, site, np.array([pick]), meters
                )
                return _DonorPick(
                    pick,
                    level,
                    int(members.size),
                    float(distance[0]),
                    float(meters[0]),
                    OUTCOME_BLOCK_GROUP,
                )
        return no_donor

    def _nearest_admitted(
        self,
        pool: DonorPool,
        view: _DonorWorldView,
        site: _DonorSite,
        asset_id: Hashable,
        redraw: Sequence[str],
        *,
        world: int,
        iteration: int,
        admit: Callable[[Mapping[str, Any]], bool],
        status_code: int,
        excluded: set[Any],
        attempts: int,
        neighborhood: _DonorNeighborhood | None = None,
    ) -> DonorDraw:
        """
        Scan the donors within the cap for the nearest ``admit`` accepts.

        Args:
            pool (DonorPool):
                The pool.
            view (_DonorWorldView):
                The donors in this world.
            site (_DonorSite):
                The target's site.
            asset_id (Hashable):
                The target.
            redraw (Sequence[str]):
                The features to copy.
            world (int):
                The world.
            iteration (int):
                The iteration of the first pick, recorded on the draw.
            admit (Callable[[Mapping[str, Any]], bool]):
                Judges a candidate's values for ``redraw``.
            status_code (int):
                The status code to match.
            excluded (set[Any]):
                Donors removed.
            attempts (int):
                The attempts made before the scan.
            neighborhood (_DonorNeighborhood | None, optional):
                The donors within the cap with their distances, computed
                when not given.

        Returns:
            DonorDraw:
                Outcome
                :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_NEAREST_ADMITTED`
                or
                :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_NONE_ADMITTED`.
        """
        if neighborhood is None:
            neighborhood = self._search.neighborhood(pool, site)
        mask = self._search.eligible(
            pool, view, neighborhood.index, excluded, status_code
        )
        ranked = neighborhood.order[mask[neighborhood.order]]
        for position in ranked.tolist():
            held = view.donor_values[int(neighborhood.index[position])]
            values = {f: held[f] for f in redraw if f in held}
            if admit(values):
                return DonorDraw(
                    asset_id,
                    world,
                    iteration,
                    pool.donor_ids[int(neighborhood.index[position])],
                    values,
                    None,
                    int(ranked.size),
                    float(neighborhood.distance[position]),
                    float(neighborhood.meters[position]),
                    attempts,
                    OUTCOME_NEAREST_ADMITTED,
                )
        return _empty_draw(
            asset_id, world, iteration, OUTCOME_NONE_ADMITTED, attempts
        )

    def _imputed_inventory(
        self,
        targets: AssetInventory,
        copied: Mapping[Any, Mapping[int, Mapping[str, Any]]],
        n_worlds: int,
    ) -> AssetInventory:
        """
        Write the copied values as an inventory.

        Args:
            targets (AssetInventory):
                The targets, for their coordinates.
            copied (Mapping[Any, Mapping[int, Mapping[str, Any]]]):
                The values per target and world.
            n_worlds (int):
                The number of worlds.

        Returns:
            AssetInventory:
                One asset per target that received a value, in ascending
                order of the id as a string.
        """
        imputed = AssetInventory(vector_features=targets.vector_features)
        for asset_id in sorted(copied, key=str):
            by_world = copied[asset_id]
            features = {}
            for feature in self.features:
                realizations = [
                    by_world.get(w, {}).get(feature) for w in range(n_worlds)
                ]
                if all(value is None for value in realizations):
                    continue
                features[feature] = collapse_realizations(realizations)
            coordinates = deepcopy(targets.inventory[asset_id].coordinates)
            imputed.add_asset(asset_id, Asset(asset_id, coordinates, features))
        imputed.n_pw = n_worlds
        return imputed


def _as_draw(
    pool: DonorPool,
    asset_id: Hashable,
    world: int,
    iteration: int,
    pick: _DonorPick,
    *,
    values: Mapping[str, Any],
    attempts: int,
) -> DonorDraw:
    """
    Turn a pick into a draw.

    Args:
        pool (DonorPool):
            The pool.
        asset_id (Hashable):
            The target.
        world (int):
            The world.
        iteration (int):
            The iteration.
        pick (_DonorPick):
            The pick.
        values (Mapping[str, Any]):
            The copied values.
        attempts (int):
            Attempts judged.

    Returns:
        DonorDraw:
            The draw.
    """
    donor = None if pick.index is None else pool.donor_ids[pick.index]
    return DonorDraw(
        asset_id,
        world,
        iteration,
        donor,
        dict(values) if donor is not None else {},
        pick.level,
        pick.n_candidates,
        pick.distance,
        pick.distance_m,
        attempts,
        pick.outcome,
    )


def _empty_draw(
    asset_id: Hashable,
    world: int,
    iteration: int,
    outcome: str,
    attempts: int = 0,
) -> DonorDraw:
    """
    Return a draw that copies nothing.

    Args:
        asset_id (Hashable):
            The target.
        world (int):
            The world.
        iteration (int):
            The iteration.
        outcome (str):
            The outcome code.
        attempts (int, optional):
            Attempts judged.

    Returns:
        DonorDraw:
            A draw without a donor.
    """
    return DonorDraw(
        asset_id,
        world,
        iteration,
        None,
        {},
        None,
        0,
        math.nan,
        math.nan,
        attempts,
        outcome,
    )
