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
Index the donors and targets of the observed-donor imputer.

.. autosummary::

      DonorPool
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from sklearn.neighbors import BallTree

from brails.constants import EARTH_RADIUS_M
from brails.types.asset_inventory import _is_missing_value, is_probabilistic_feature

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
        ObservedDonorImputer,
    )

# Relative margin by which the tree search exceeds the cap before the
# distances it returns are compared with the cap.
TREE_RADIUS_MARGIN = 1e-6

# Status codes of a donor: its value of the status key is missing, outside
# the status values, or among them.
_STATUS_UNKNOWN = -1
_STATUS_OUTSIDE = 0
_STATUS_INSIDE = 1


class _DonorSite(NamedTuple):
    """
    Where an asset is and how it compares with donors.

    Attributes:
        lon (float):
            Longitude of the centroid of the stored coordinates.
        lat (float):
            Latitude of the centroid.
        log_area (float):
            Natural log of the plan area in square meters.
        stories (float):
            The story count the distance compares, NaN without one or
            where it is held per world, which
            :meth:`DonorPool.world_site` gives.
        block_group (str | None):
            The asset's block-group GEOID, if it holds one.
    """

    lon: float
    lat: float
    log_area: float
    stories: float
    block_group: str | None


class _DonorWorldView(NamedTuple):
    """
    The donors' values in one possible world.

    Attributes:
        eligible (numpy.ndarray):
            Whether each donor holds every required feature.
        status (numpy.ndarray):
            Each donor's status code.
        filters (dict[str, numpy.ndarray]):
            Each soft filter's donor values, ``None`` where missing.
        donor_values (list[dict[str, Any]]):
            Each donor's non-missing values of the imputed features.
        stories (numpy.ndarray):
            Each donor's story count for the distance, NaN without one.
    """

    eligible: np.ndarray
    status: np.ndarray
    filters: dict[str, np.ndarray]
    donor_values: list[dict[str, Any]]
    stories: np.ndarray


class _DonorNeighborhood(NamedTuple):
    """
    The donors within the cap of one site.

    Attributes:
        index (numpy.ndarray):
            Donor indices, in the order the tree returns them.
        meters (numpy.ndarray):
            Great-circle distance of each in meters.
        distance (numpy.ndarray | None):
            Donor distance of each, ``None`` until computed.
        order (numpy.ndarray | None):
            Positions into ``index`` sorted by donor distance, then by
            donor index, ``None`` until computed.
    """

    index: np.ndarray
    meters: np.ndarray
    distance: np.ndarray | None = None
    order: np.ndarray | None = None


def _is_missing(features: Mapping[str, Any], key: str) -> bool:
    """
    Check whether a feature is absent or holds a missing value.

    Args:
        features (Mapping[str, Any]):
            An asset's features.
        key (str):
            The feature.

    Returns:
        bool:
            ``True`` when ``key`` is absent or its value is missing as
            :meth:`AssetInventory.merge` reads it.
    """
    return key not in features or _is_missing_value(features[key])


def _status_code(
    features: Mapping[str, Any], key: str, values: Sequence[Any]
) -> int:
    """
    Code an asset's status from its features.

    Args:
        features (Mapping[str, Any]):
            The asset's features in one world.
        key (str):
            The feature the status is read from.
        values (Sequence[Any]):
            The values of ``key`` that give the status.

    Returns:
        int:
            :data:`_STATUS_UNKNOWN` when ``key`` is missing, else
            :data:`_STATUS_INSIDE` or :data:`_STATUS_OUTSIDE`.
    """
    if _is_missing(features, key):
        return _STATUS_UNKNOWN
    return _STATUS_INSIDE if features[key] in values else _STATUS_OUTSIDE


def _story_count(value: object) -> float:
    """
    Read a story count for the donor distance.

    Args:
        value (object):
            A story count in one world.

    Returns:
        float:
            The count, NaN where ``value`` is not a finite number.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
    ):
        return math.nan
    return float(value)


def _world_features(
    features: Mapping[str, Any],
    world: int,
    n_pw: int,
    vector_features: Iterable[str],
) -> dict[str, Any]:
    """
    Reduce an asset's features to their values in one possible world.

    Args:
        features (Mapping[str, Any]):
            The asset's features.
        world (int):
            The world.
        n_pw (int):
            The number of worlds of the asset's inventory. With one world,
            every world reads realization 0.
        vector_features (Iterable[str]):
            Features whose plain list values are single values.

    Returns:
        dict[str, Any]:
            One value per feature.
    """
    vector_features = tuple(vector_features)
    reduced = {}
    for key, value in features.items():
        if not is_probabilistic_feature(key, value, vector_features):
            reduced[key] = value
        elif not value:
            reduced[key] = None
        else:
            reduced[key] = value[world if n_pw > 1 else 0]
    return reduced


class DonorPool:
    """
    The donors and targets of one imputation, indexed for the donor search.

    A pool holds the centroid, log plan area, story count and block group of
    every donor and target with area, the donors' feature values and statuses,
    and a haversine ``BallTree`` over the donors. Donors are ordered by their
    id as a string. Build one with
    :meth:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.ObservedDonorImputer.pool`;
    only the imputer that built it accepts it.

    Attributes:
        donor_ids (tuple):
            The donors, in ascending order of the id as a string.
        lon (numpy.ndarray):
            Donor centroid longitudes.
        lat (numpy.ndarray):
            Donor centroid latitudes.
        log_area (numpy.ndarray):
            Natural log of each donor's plan area in square meters.
        stories (numpy.ndarray):
            Each donor's story count for the distance, NaN without one or
            where it is held per world; :meth:`view` gives each world's.
        block_group (tuple):
            Each donor's block-group GEOID, or ``None``.
        no_area (frozenset):
            Assets of the donor and target inventories whose coordinates
            enclose no area. The pool holds no site for them and never
            draws them.
        tree (sklearn.neighbors.BallTree | None):
            The search tree, ``None`` without donors.
        imputer (ObservedDonorImputer):
            The imputer that built the pool.
        features (tuple):
            The features a donor supplies.
        soft_filters (tuple):
            The features filtered on where the target holds them.
        required (tuple):
            The features a donor must hold.
        status_key (str):
            The feature a donor's status is read from.
        status_values (tuple):
            The values of ``status_key`` that give a donor the status.
        stories_key (str | None):
            The feature a donor's story count is read from in each world.
        donor_index (dict):
            Each donor's index in :attr:`donor_ids`, by id.
        block_group_donors (dict):
            The indices of each block group's donors in ascending order, by
            GEOID.
    """

    def __init__(
        self,
        *,
        imputer: ObservedDonorImputer,
        donors: Sequence[tuple[Any, _DonorSite, Mapping[str, Any], int]],
        targets: Mapping[Any, _DonorSite],
        no_area: Iterable[Any],
        vector_features: Sequence[str],
        features: Sequence[str],
        soft_filters: Sequence[str],
        required: Sequence[str],
        status_key: str,
        status_values: Sequence[Any],
        stories_key: str | None = None,
        target_stories: Mapping[Any, Sequence[float]] | None = None,
    ) -> None:
        """
        Index prepared donors and targets.

        Args:
            imputer (ObservedDonorImputer):
                The imputer building the pool.
            donors (Sequence[tuple[Any, _DonorSite, Mapping[str, Any], int]]):
                ``(asset_id, site, features, n_pw)`` per donor, in ascending
                order of the id as a string, each id once.
            targets (Mapping[Any, _DonorSite]):
                The sites of the targets that are not donors.
            no_area (Iterable[Any]):
                Assets without area.
            vector_features (Sequence[str]):
                Features whose plain lists are single values.
            features (Sequence[str]):
                The features a donor supplies.
            soft_filters (Sequence[str]):
                The features filtered on where the target holds them.
            required (Sequence[str]):
                The features a donor must hold.
            status_key (str):
                The feature a donor's status is read from.
            status_values (Sequence[Any]):
                The values of ``status_key`` that give a donor the status.
            stories_key (str, optional):
                The feature a donor's story count is read from in each
                world, for a donor of several worlds.
            target_stories (Mapping[Any, Sequence[float]], optional):
                The story count in each world of every target that holds
                one per world, by id.

        Raises:
            ValueError:
                If a donor id occurs twice.
        """
        self.imputer = imputer
        self.donor_ids = tuple(asset_id for asset_id, _, _, _ in donors)
        if len(set(self.donor_ids)) != len(self.donor_ids):
            msg = 'A donor pool holds each donor id once.'
            raise ValueError(msg)
        sites = [site for _, site, _, _ in donors]
        self.lon = np.array([s.lon for s in sites], dtype=float)
        self.lat = np.array([s.lat for s in sites], dtype=float)
        self.log_area = np.array([s.log_area for s in sites], dtype=float)
        self.stories = np.array([s.stories for s in sites], dtype=float)
        self.block_group = tuple(s.block_group for s in sites)
        self.no_area = frozenset(no_area)
        self.tree = (
            BallTree(
                np.radians(np.column_stack([self.lat, self.lon])), metric='haversine'
            )
            if sites
            else None
        )
        self._donor_features = [dict(f) for _, _, f, _ in donors]
        self._donor_worlds = [n for _, _, _, n in donors]
        self._per_world = any(n > 1 for n in self._donor_worlds)
        self._vector_features = tuple(vector_features)
        self.features = tuple(features)
        self.soft_filters = tuple(soft_filters)
        self.required = tuple(required)
        self.status_key = status_key
        self.status_values = tuple(status_values)
        self.stories_key = stories_key
        self._target_stories = dict(target_stories or {})
        self._sites = dict(targets)
        self._sites.update(zip(self.donor_ids, sites))
        self.donor_index = {asset_id: i for i, asset_id in enumerate(self.donor_ids)}
        self.block_group_donors: dict[str, list[int]] = {}
        for index, block_group in enumerate(self.block_group):
            if block_group is not None:
                self.block_group_donors.setdefault(block_group, []).append(index)
        self._views: dict[int, _DonorWorldView] = {}

    def __contains__(self, asset_id: object) -> bool:
        """Return whether the pool records ``asset_id``."""
        return asset_id in self._sites or asset_id in self.no_area

    def __len__(self) -> int:
        """Return the number of donors."""
        return len(self.donor_ids)

    def site(self, asset_id: Hashable) -> _DonorSite:
        """
        Return an asset's site.

        Args:
            asset_id (Hashable):
                A donor or target with area.

        Returns:
            _DonorSite:
                Its site.

        Raises:
            LookupError:
                If the pool does not hold the asset with area.
        """
        try:
            return self._sites[asset_id]
        except KeyError:
            msg = f'The pool holds no asset {asset_id!r} with area.'
            raise LookupError(msg) from None

    def world_site(self, asset_id: Hashable, world: int) -> _DonorSite:
        """
        Return an asset's site with its story count in one world.

        Args:
            asset_id (Hashable):
                A donor or target with area.
            world (int):
                The world.

        Returns:
            _DonorSite:
                The site of :meth:`site`, itself where the asset holds no
                story count per world.

        Raises:
            LookupError:
                If the pool does not hold the asset with area.
        """
        site = self.site(asset_id)
        counts = self._target_stories.get(asset_id)
        if counts is not None:
            return site._replace(stories=counts[world])
        index = self.donor_index.get(asset_id)
        if index is None or self._donor_worlds[index] == 1:
            return site
        return site._replace(stories=self.view(world).stories[index])

    def neighborhood(self, site: _DonorSite, cap_m: float) -> _DonorNeighborhood:
        """
        Find the donors within a distance of a site.

        The tree searches a slightly larger radius, and the great-circle
        distance it returns for each pair is compared with ``cap_m``.

        Args:
            site (_DonorSite):
                The site.
            cap_m (float):
                The distance in meters.

        Returns:
            _DonorNeighborhood:
                The donors within ``cap_m``, by index, with their distances.
        """
        if self.tree is None:
            return _DonorNeighborhood(np.zeros(0, dtype=np.int64), np.zeros(0))
        radius = cap_m / EARTH_RADIUS_M * (1.0 + TREE_RADIUS_MARGIN)
        found, radians = self.tree.query_radius(
            np.radians([[site.lat, site.lon]]), r=radius, return_distance=True
        )
        index = found[0].astype(np.int64)
        meters = radians[0] * EARTH_RADIUS_M
        keep = meters <= cap_m
        return _DonorNeighborhood(index[keep], meters[keep])

    def view(self, world: int) -> _DonorWorldView:
        """
        Return the donors' values in one world.

        Args:
            world (int):
                The world.

        Returns:
            _DonorWorldView:
                Eligibility, status, soft-filter values, imputed-feature
                values and story count of every donor in that world.
        """
        key = world if self._per_world else 0
        if key in self._views:
            return self._views[key]
        n = len(self.donor_ids)
        eligible = np.zeros(n, dtype=bool)
        status = np.full(n, _STATUS_UNKNOWN, dtype=np.int8)
        filters = {f: np.empty(n, dtype=object) for f in self.soft_filters}
        values = []
        stories = self.stories.copy()
        for i, (raw, n_pw) in enumerate(
            zip(self._donor_features, self._donor_worlds)
        ):
            features = _world_features(raw, world, n_pw, self._vector_features)
            eligible[i] = all(not _is_missing(features, f) for f in self.required)
            status[i] = _status_code(features, self.status_key, self.status_values)
            for f in self.soft_filters:
                filters[f][i] = None if _is_missing(features, f) else features[f]
            values.append(
                {
                    f: features[f]
                    for f in self.features
                    if not _is_missing(features, f)
                }
            )
            if n_pw > 1:
                stories[i] = _story_count(features.get(self.stories_key))
        view = _DonorWorldView(eligible, status, filters, values, stories)
        self._views[key] = view
        return view
