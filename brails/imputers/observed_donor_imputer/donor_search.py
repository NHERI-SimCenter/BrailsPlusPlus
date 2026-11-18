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
Rank and pick the donors of the observed-donor imputer.

.. autosummary::

      DonorSearch
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from brails.constants import BLOCK_GROUP_KEY
from brails.imputers.observed_donor_imputer.donor_pool import (
    _STATUS_UNKNOWN,
    _DonorSite,
    _story_count,
)
from brails.imputers.observed_donor_imputer.donor_records import (
    OUTCOME_DONOR,
    OUTCOME_FEWER_THAN_K,
    OUTCOME_RELAXED,
)
from brails.types.asset_inventory import AssetInventory, _is_missing_value
from brails.utils.deterministic_sampling import keyed_uniforms

if TYPE_CHECKING:
    from collections.abc import Sequence

    from brails.imputers.observed_donor_imputer.donor_pool import (
        DonorPool,
        _DonorNeighborhood,
        _DonorWorldView,
    )
    from brails.types.asset_inventory import Asset

# Distance below which inverse-distance weights stop growing.
MIN_WEIGHT_DISTANCE = 1e-9


class _DonorPick(NamedTuple):
    """
    One pick among the candidates of a target.

    Attributes:
        index (int | None):
            The donor's index in the pool, ``None`` without one.
        level (int | None):
            Soft filters relaxed.
        n_candidates (int):
            The number of candidates at the level used.
        distance (float):
            Donor distance, NaN without a donor.
        distance_m (float):
            Great-circle distance in meters, NaN without a donor.
        outcome (str):
            The outcome code.
    """

    index: int | None
    level: int | None
    n_candidates: int
    distance: float
    distance_m: float
    outcome: str


class DonorSearch:
    """
    The donor distance, the candidate filters and the keyed pick.

    :class:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.ObservedDonorImputer`
    checks its arguments and builds one search, which reads an asset's site,
    finds and ranks the donors within the cap and picks among them. The status
    round, the donor round, the fallback and the nearest-first scan are the
    imputer's.

    Every argument is stored as an attribute of the same name, read-only by
    convention.
    """

    def __init__(
        self,
        *,
        k: int,
        cap_m: float,
        location_scale_m: float,
        area_weight: float,
        stories_key: str | None,
        stories_weight: float,
        seed: int,
    ) -> None:
        """
        Configure the search with the imputer's checked arguments.

        Args:
            k (int):
                The number of nearest donors picked among.
            cap_m (float):
                The donor distance cap in meters.
            location_scale_m (float):
                Meters of separation counted as one unit of distance.
            area_weight (float):
                Weight of the squared difference of log plan areas.
            stories_key (str | None):
                The feature holding the story count the distance compares.
            stories_weight (float):
                Weight of the squared difference of story counts.
            seed (int):
                The seed of the uniforms.
        """
        self.k = k
        self.cap_m = cap_m
        self.location_scale_m = location_scale_m
        self.area_weight = area_weight
        self.stories_key = stories_key
        self.stories_weight = stories_weight
        self.seed = seed

    def neighborhood(
        self, pool: DonorPool, view: _DonorWorldView, site: _DonorSite
    ) -> _DonorNeighborhood:
        """
        Find the donors within the cap of a site with their donor distances.

        The result depends on the site, the pool's geometry and the donors'
        story counts in the view's world.

        Args:
            pool (DonorPool):
                The pool.
            view (_DonorWorldView):
                The donors in the target's world.
            site (_DonorSite):
                The target's site in that world.

        Returns:
            _DonorNeighborhood:
                The donors within :attr:`cap_m`, with distances.
        """
        found = pool.neighborhood(site, self.cap_m)
        distance = self.distances(pool, view, site, found.index, found.meters)
        order = np.lexsort((found.index, distance))
        return found._replace(distance=distance, order=order)

    def distances(
        self,
        pool: DonorPool,
        view: _DonorWorldView,
        site: _DonorSite,
        index: np.ndarray,
        meters: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the donor distance from a site to donors.

        Args:
            pool (DonorPool):
                The pool.
            view (_DonorWorldView):
                The donors in the target's world.
            site (_DonorSite):
                The target's site in that world.
            index (numpy.ndarray):
                Donor indices.
            meters (numpy.ndarray):
                Their great-circle distances in meters.

        Returns:
            numpy.ndarray:
                ``sqrt((g / location_scale_m)**2 + area_weight * (ln A_t -
                ln A_d)**2 + stories_weight * (n_t - n_d)**2)``, with the
                story term zero where either side has no count.
        """
        location = (meters / self.location_scale_m) ** 2
        area = self.area_weight * (site.log_area - pool.log_area[index]) ** 2
        stories = self.stories_weight * (site.stories - view.stories[index]) ** 2
        return np.sqrt(location + area + np.nan_to_num(stories))

    def select(
        self,
        pool: DonorPool,
        view: _DonorWorldView,
        key: np.ndarray,
        neighborhood: _DonorNeighborhood,
        *,
        excluded: set[Any],
        status_code: int | None,
        soft: Sequence[tuple[str, Any]],
        purpose: str,
        world: int,
        iteration: int,
    ) -> _DonorPick | None:
        """
        Pick one donor among a target's candidates within the cap.

        The first filter level with :attr:`k` or more candidates is used,
        else the loosest level with any. One of its :attr:`k` nearest is
        picked with inverse-distance weights and the target's uniform.

        Args:
            pool (DonorPool):
                The pool.
            view (_DonorWorldView):
                The donors in this world.
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
            purpose (str):
                The purpose label of the pick.
            world (int):
                The world.
            iteration (int):
                The iteration.

        Returns:
            _DonorPick | None:
                The pick, ``None`` when no candidate within the cap passes
                the hard filter.
        """
        base = self.eligible(pool, view, neighborhood.index, excluded, status_code)
        levels = self.levels(view, neighborhood.index, base, soft)
        chosen = None
        for level, mask in enumerate(levels):
            if mask.sum() >= self.k:
                chosen = (
                    level,
                    mask,
                    OUTCOME_DONOR if level == 0 else OUTCOME_RELAXED,
                )
                break
        if chosen is None and levels[-1].any():
            chosen = (len(levels) - 1, levels[-1], OUTCOME_FEWER_THAN_K)
        if chosen is not None:
            level, mask, outcome = chosen
            ranked, weights = self.ranked_weights(neighborhood, mask)
            cumulative = np.cumsum(weights)
            u = self.uniform(key, purpose, world, iteration)
            position = int(
                np.searchsorted(cumulative, u * cumulative[-1], side='right')
            )
            pick = ranked[min(position, len(ranked) - 1)]
            return _DonorPick(
                int(neighborhood.index[pick]),
                level,
                int(mask.sum()),
                float(neighborhood.distance[pick]),
                float(neighborhood.meters[pick]),
                outcome,
            )
        return None

    @staticmethod
    def eligible(
        pool: DonorPool,
        view: _DonorWorldView,
        index: np.ndarray,
        excluded: set[Any],
        status_code: int | None,
    ) -> np.ndarray:
        """
        Mark the donors a draw may pick.

        A donor qualifies when it holds the required features, has a known
        status matching ``status_code``, and is not excluded.

        Args:
            pool (DonorPool):
                The pool.
            view (_DonorWorldView):
                The donors in this world.
            index (numpy.ndarray):
                Donor indices.
            excluded (set[Any]):
                Donors removed.
            status_code (int | None):
                The status code to match, ``None`` for none.

        Returns:
            numpy.ndarray:
                One boolean per index.
        """
        mask = view.eligible[index].copy()
        mask &= view.status[index] != _STATUS_UNKNOWN
        if status_code is not None:
            mask &= view.status[index] == status_code
        if excluded:
            removed = [pool.donor_index.get(asset_id) for asset_id in excluded]
            removed = [i for i in removed if i is not None]
            if removed:
                mask &= ~np.isin(index, removed)
        return mask

    @staticmethod
    def levels(
        view: _DonorWorldView,
        index: np.ndarray,
        base: np.ndarray,
        soft: Sequence[tuple[str, Any]],
    ) -> list[np.ndarray]:
        """
        Build the candidate masks from all soft filters to none.

        Args:
            view (_DonorWorldView):
                The donors in this world.
            index (numpy.ndarray):
                Donor indices.
            base (numpy.ndarray):
                The mask after the hard filter.
            soft (Sequence[tuple[str, Any]]):
                The soft filters with the target's values.

        Returns:
            list[numpy.ndarray]:
                Level ``j`` drops the last ``j`` soft filters.
        """
        levels = []
        for relaxed in range(len(soft) + 1):
            mask = base.copy()
            for feature, value in soft[: len(soft) - relaxed]:
                column = view.filters[feature][index]
                mask &= np.array([v == value for v in column], dtype=bool).reshape(
                    mask.shape
                )
            levels.append(mask)
        return levels

    def ranked_weights(
        self, neighborhood: _DonorNeighborhood, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rank the admitted candidates of a pick and weight them.

        Args:
            neighborhood (_DonorNeighborhood):
                The donors within the cap, with distances and their order.
            mask (numpy.ndarray):
                The candidates admitted, one boolean per donor of the
                neighborhood, at least one true.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]:
                The positions into the neighborhood of the :attr:`k`
                nearest admitted candidates, nearest first, and their
                inverse-distance weights.
        """
        ranked = neighborhood.order[mask[neighborhood.order]][: self.k]
        weights = 1.0 / np.maximum(
            neighborhood.distance[ranked], MIN_WEIGHT_DISTANCE
        )
        return ranked, weights

    def block_group_candidates(
        self,
        pool: DonorPool,
        view: _DonorWorldView,
        site: _DonorSite,
        excluded: set[Any],
        status_code: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the donors of a site's block group and those a fallback may pick.

        Args:
            pool (DonorPool):
                The pool.
            view (_DonorWorldView):
                The donors in this world.
            site (_DonorSite):
                The target's site.
            excluded (set[Any]):
                Donors removed.
            status_code (int | None):
                The status code to match, ``None`` for none.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]:
                The donors' indices in ascending order, empty without a
                block group, and the mask of :meth:`eligible` over them.
        """
        if site.block_group is None:
            index = np.zeros(0, dtype=np.int64)
        else:
            index = np.array(
                pool.block_group_donors.get(site.block_group, []), dtype=np.int64
            )
        return index, self.eligible(pool, view, index, excluded, status_code)

    def uniform(
        self, key: np.ndarray, purpose: str, world: int, iteration: int
    ) -> float:
        """
        Return the target's keyed uniform for one draw.

        Args:
            key (numpy.ndarray):
                The target's key from :func:`asset_key_digest`.
            purpose (str):
                The purpose label.
            world (int):
                The world.
            iteration (int):
                The iteration.

        Returns:
            float:
                The uniform in [0, 1).
        """
        return float(
            keyed_uniforms(
                key,
                seed=self.seed,
                purpose=purpose,
                world=world,
                iteration=iteration,
            )[0]
        )

    def site(self, asset: Asset) -> _DonorSite | None:
        """
        Locate an asset and read its story count.

        Args:
            asset (Asset):
                The asset.

        Returns:
            _DonorSite | None:
                Its site, with a NaN story count where the value of
                :attr:`stories_key` is not a finite number, or ``None``
                when its coordinates enclose no area.
        """
        coordinates = asset.coordinates
        if not coordinates:
            return None
        try:
            lon, lat, area = AssetInventory.centroid_and_plan_area(coordinates)
        except ValueError:
            return None
        if area <= 0:
            return None
        features = asset.features
        block_group = features.get(BLOCK_GROUP_KEY)
        if not isinstance(block_group, str) or _is_missing_value(block_group):
            block_group = None
        return _DonorSite(
            lon,
            lat,
            math.log(area),
            _story_count(features.get(self.stories_key)),
            block_group,
        )
