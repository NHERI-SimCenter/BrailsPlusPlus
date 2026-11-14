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
Copy construction years to buildings from their residential neighbors.

.. autosummary::

      NeighborYearSampler
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from brails.constants import BLOCK_GROUP_KEY
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    DEFAULT_DONOR_CAP_M,
    DEFAULT_DONOR_K,
    OUTCOME_BLOCK_GROUP,
    OUTCOME_DONOR,
    OUTCOME_FEWER_THAN_K,
    ObservedDonorImputer,
)
from brails.inferers.acs_year_built_sampler.acs_year_built_sampler import (
    RESIDENTIAL_CLASSES,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INFEASIBLE,
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
    diagnosis_frame,
    preview_failures,
)
from brails.inferers.inference_engine import InferenceEngine
from brails.types.asset_inventory import AssetInventory, is_probabilistic_feature
from brails.utils.deterministic_sampling import (
    DEFAULT_ITERATION,
    DEFAULT_WORLD,
    require_run_seed,
)
from brails.utils.stage_freeze import StageFreeze

if TYPE_CHECKING:
    from collections.abc import Hashable

    from brails.imputers.observed_donor_imputer.donor_pool import DonorPool

# Prefix of the purpose labels of the keyed uniforms, put before the donor
# search's PURPOSE_DONOR and PURPOSE_FALLBACK. Part of the reproducibility
# contract; see keyed_uniforms.
NEIGHBOR_YEAR_PURPOSE_PREFIX = 'year_built_neighbor'


class NeighborYearSampler(InferenceEngine):
    """
    Copy to each building the construction year of a nearby residential one.

    The donors are the residential buildings of one world of a frozen stage
    that hold a year, whatever its source. The sampler works in five
    steps.

    1. :meth:`fit` reads the stage and takes as donors its assets whose
       ``occupancy_key`` is among
       :data:`~brails.inferers.acs_year_built_sampler.acs_year_built_sampler.RESIDENTIAL_CLASSES`,
       whose ``year_built_key`` is a whole number, an integer or an
       integral float, and whose coordinates enclose an area.
    2. Each call of :meth:`check`, :meth:`draw` or :meth:`picks` indexes
       those donors and the sites of the inventory's assets.
    3. Each asset draws one donor among the ``k`` nearest within ``cap_m``,
       by the great-circle distance between centroids, with weights one
       over the distance.
    4. An asset with no donor within ``cap_m`` draws one donor of its block
       group, read from :data:`~brails.constants.BLOCK_GROUP_KEY`, with
       equal weights. In a block group that covers a large area the donor
       may lie beyond the cap.
    5. The donor's year is copied.

    Steps 3 and 4 are the draw of
    :meth:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.ObservedDonorImputer.draw`,
    run with the residential status on a distance of location alone and
    with the labels of :data:`NEIGHBOR_YEAR_PURPOSE_PREFIX`. A draw depends
    on the asset's id and coordinates, its block group, the seed, the
    world, the iteration, ``k``, ``cap_m`` and the donors, and never on
    processing order or on another asset's draw. The README beside this module states the
    defaults and the scope.

    The sampler writes ``year_built_key`` whether or not one is already
    there; keeping an observed year is the caller's decision.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        world: int = DEFAULT_WORLD,
        # see the README beside this module, "The defaults"
        k: int = DEFAULT_DONOR_K,
        cap_m: float = DEFAULT_DONOR_CAP_M,
        occupancy_key: str = 'OccupancyClass',
        year_built_key: str = 'YearBuilt',
    ) -> None:
        """
        Configure the sampler and its donor search.

        Args:
            seed (int | None):
                Seed for the draw. If None, a seed is drawn from system
                entropy and recorded on ``self.seed``.
            world (int):
                Possible-world index, at least 0. A world other than the
                default requires an explicit seed.
            k (int):
                The number of nearest donors picked among. Defaults to
                :data:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.DEFAULT_DONOR_K`.
            cap_m (float):
                The donor distance cap in meters, the same for every asset.
                Defaults to
                :data:`~brails.imputers.observed_donor_imputer.observed_donor_imputer.DEFAULT_DONOR_CAP_M`.
            occupancy_key (str):
                Feature holding the occupancy class.
            year_built_key (str):
                Feature holding a donor's year, and the name the copied
                year is written to.

        Raises:
            ValueError:
                If ``world`` is negative, or not the default and no seed
                is given, ``k`` is below 1, or ``cap_m`` is at most 0.
        """
        if int(world) < 0:
            msg = f'world must be at least 0, got {world!r}.'
            raise ValueError(msg)
        require_run_seed(world, seed)
        self._imputer = ObservedDonorImputer(
            features=(year_built_key,),
            soft_filters=(),
            required=(occupancy_key, year_built_key),
            status_key=occupancy_key,
            status_values=RESIDENTIAL_CLASSES,
            k=k,
            cap_m=cap_m,
            location_scale_m=1.0,
            area_weight=0.0,
            stories_key=None,
            stories_weight=0.0,
            seed=seed,
        )
        self.seed = self._imputer.seed
        self.world = int(world)
        self.k = self._imputer.k
        self.cap_m = self._imputer.cap_m
        self.occupancy_key = occupancy_key
        self.year_built_key = year_built_key
        self._donors: AssetInventory | None = None
        self._donor_frame: pd.DataFrame | None = None

    # -----------------------------------------------------------------
    # Fit

    def fit(self, freeze: StageFreeze, stage: str) -> NeighborYearSampler:
        """
        Take the donors from one world of a frozen stage.

        A stage holding several worlds is read in :attr:`world`, and a
        stage holding one is read as it is. The fit changes nothing in the
        stage and reads nothing but it. A later call replaces the earlier
        fit.

        Args:
            freeze (StageFreeze):
                The freeze holding the stage.
            stage (str):
                The name of the frozen stage.

        Returns:
            NeighborYearSampler:
                This sampler, fitted.

        Raises:
            TypeError:
                If ``freeze`` is not a :class:`StageFreeze`.
            LookupError:
                If ``stage`` is not frozen.
            ValueError:
                If the stage holds several worlds and not :attr:`world`, or
                an asset holds one value per world in ``occupancy_key`` or
                ``year_built_key`` in the world read.
        """
        if not isinstance(freeze, StageFreeze):
            msg = (
                'fit reads a StageFreeze and the name of a frozen stage, not '
                f'{type(freeze).__name__}.'
            )
            raise TypeError(msg)
        output = freeze.read(stage)
        if output.n_pw > 1 and self.world >= output.n_pw:
            msg = (
                f'Stage {stage!r} holds {output.n_pw} worlds; the sampler '
                f'draws world {self.world}.'
            )
            raise ValueError(msg)
        realization = (
            output.get_world_realization(self.world) if output.n_pw > 1 else output
        )
        donors = AssetInventory(vector_features=realization.vector_features)
        for asset_id, asset in realization.inventory.items():
            occupancy = asset.features.get(self.occupancy_key)
            year = asset.features.get(self.year_built_key)
            for key, value in (
                (self.occupancy_key, occupancy),
                (self.year_built_key, year),
            ):
                if is_probabilistic_feature(key, value, realization.vector_features):
                    msg = (
                        f'Asset {asset_id}: {key} holds one value per world; '
                        'fit takes one world at a time.'
                    )
                    raise ValueError(msg)
            whole = (
                not isinstance(year, bool) and isinstance(year, (int, np.integer))
            ) or (
                isinstance(year, (float, np.floating))
                and math.isfinite(year)
                and float(year).is_integer()
            )
            # an asset without area is left to the donor search, which
            # never draws it
            if occupancy in RESIDENTIAL_CLASSES and whole:
                donors.add_asset(asset_id, asset)
        self._donors = donors
        self._donor_frame = None
        return self

    @property
    def donors(self) -> pd.DataFrame:
        """
        The fitted donors as the donor search sees them.

        Returns:
            pd.DataFrame:
                One row per donor, in ascending order of the id as a string,
                with the centroid's ``lon`` and ``lat``, the block group
                under :data:`~brails.constants.BLOCK_GROUP_KEY` (``None``
                where the donor holds none) and the year under
                ``year_built_key``. A copy, built on the first read after
                each fit.

        Raises:
            RuntimeError:
                If the sampler has not been fitted.
        """
        donors = self._require_fit()
        if self._donor_frame is not None:
            return self._donor_frame.copy()
        pool = self._imputer.pool(donors)
        self._donor_frame = pd.DataFrame(
            {
                'lon': pool.lon,
                'lat': pool.lat,
                BLOCK_GROUP_KEY: list(pool.block_group),
                self.year_built_key: [
                    int(donors.inventory[asset_id].features[self.year_built_key])
                    for asset_id in pool.donor_ids
                ],
            },
            index=pd.Index(list(pool.donor_ids), dtype=object),
        )
        return self._donor_frame.copy()

    def _require_fit(self) -> AssetInventory:
        """
        Return the fitted donors, or raise if there are none.

        Returns:
            AssetInventory:
                The donors, one world.

        Raises:
            RuntimeError:
                If :meth:`fit` has not been called.
        """
        if self._donors is None:
            raise RuntimeError(
                'NeighborYearSampler must be fitted with fit() first.'
            )
        return self._donors

    # -----------------------------------------------------------------
    # Check

    def check(self, input_inventory: AssetInventory) -> pd.DataFrame:
        """
        Diagnose every asset without drawing anything.

        A verdict reads the asset's features and coordinates and the fitted
        donors, the asset itself counted among them where it is one, so a
        one-asset inventory gets the verdict of the full one.

        Args:
            input_inventory (AssetInventory):
                One world's scalar inventory.

        Returns:
            pd.DataFrame:
                One row per asset id, with ``diagnosis``, ``detail`` and
                ``resolvable_by``. An occupancy or block group holding one
                value per world is unsupported. An absent occupancy,
                coordinates that enclose no area, or, for an asset with no
                donor within ``cap_m``, an absent block group, is invalid.
                A residential class, or a block group without a donor for
                an asset with none within ``cap_m``, is infeasible.
                ``resolvable_by`` names the failing feature where it is the
                only failure, and nothing where several fail or the
                coordinates do.

        Raises:
            RuntimeError:
                If the sampler has not been fitted.
        """
        pool = self._imputer.pool(self._require_fit(), input_inventory)
        rows = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            rows[asset_id] = self._resolve(
                pool, asset_id, features, input_inventory.vector_features
            )
        return diagnosis_frame(rows)

    def _resolve(  # noqa: C901
        self,
        pool: DonorPool,
        asset_id: Hashable,
        features: dict[str, Any],
        vector_features: list[str],
    ) -> tuple[str, str, tuple[str, ...]]:
        """
        Diagnose one asset from its features, its site and the donors.

        Args:
            pool (DonorPool):
                A pool of the donors that records the asset.
            asset_id (Hashable):
                The asset.
            features (dict[str, Any]):
                Its features.
            vector_features (list[str]):
                Names of the features whose values are vectors.

        Returns:
            tuple[str, str, tuple[str, ...]]:
                The diagnosis, the detail, and the features that would
                each, alone, admit the asset.
        """
        for key in (self.occupancy_key, BLOCK_GROUP_KEY):
            if is_probabilistic_feature(key, features.get(key), vector_features):
                return (
                    DIAGNOSIS_UNSUPPORTED,
                    f'{key} holds one value per world; the sampler takes '
                    'one world at a time',
                    (),
                )
        # (feature or None, message) per failure
        invalid: list[tuple[str | None, str]] = []
        uncovered: list[tuple[str | None, str]] = []
        occupancy = features.get(self.occupancy_key)
        if not isinstance(occupancy, str):
            invalid.append(
                (
                    self.occupancy_key,
                    f'{self.occupancy_key} is absent or not a string: {occupancy!r}',
                )
            )
        elif occupancy in RESIDENTIAL_CLASSES:
            uncovered.append(
                (
                    self.occupancy_key,
                    f'{self.occupancy_key} {occupancy!r} is a residential '
                    'class, which ACSYearBuiltSampler covers',
                )
            )
        if asset_id in pool.no_area:
            invalid.append((None, 'its coordinates enclose no area'))
        else:
            site = pool.site(asset_id)
            if pool.neighborhood(site, self.cap_m).index.size == 0:
                if site.block_group is None:
                    invalid.append(
                        (
                            BLOCK_GROUP_KEY,
                            f'no residential asset with a year within '
                            f'{self.cap_m:g} m, and {BLOCK_GROUP_KEY} is '
                            'absent or not a string: '
                            f'{features.get(BLOCK_GROUP_KEY)!r}',
                        )
                    )
                elif site.block_group not in pool.block_group_donors:
                    uncovered.append(
                        (
                            BLOCK_GROUP_KEY,
                            'no residential asset with a year within '
                            f'{self.cap_m:g} m or in block group '
                            f'{site.block_group}',
                        )
                    )
        failing = invalid + uncovered
        if not failing:
            return DIAGNOSIS_OK, '', ()
        details = '; '.join(message for _, message in failing)
        resolvable = (
            (failing[0][0],)
            if len(failing) == 1 and failing[0][0] is not None
            else ()
        )
        if invalid:
            return DIAGNOSIS_INVALID, details, resolvable
        return DIAGNOSIS_INFEASIBLE, details, resolvable

    # -----------------------------------------------------------------
    # Draw

    def draw(
        self,
        input_inventory: AssetInventory,
        *,
        iteration: int = DEFAULT_ITERATION,
    ) -> pd.Series:
        """
        Copy a year to every asset. Every asset must pass check.

        Args:
            input_inventory (AssetInventory):
                An inventory whose assets all pass :meth:`check`.
            iteration (int):
                Resolution-loop attempt number. Distinct iterations draw
                independently for the same asset.

        Returns:
            pd.Series:
                Integer years indexed by asset id, named
                ``year_built_key``.

        Raises:
            RuntimeError:
                If the sampler has not been fitted.
            ValueError:
                If an asset does not pass :meth:`check`, naming the
                assets, or draws no donor, which happens only when the
                inventory is not a realization of the fitted stage.
        """
        return self.picks(input_inventory, iteration=iteration)[self.year_built_key]

    def picks(
        self,
        input_inventory: AssetInventory,
        *,
        iteration: int = DEFAULT_ITERATION,
    ) -> pd.DataFrame:
        """
        Report the donor each asset draws. Every asset must pass check.

        :meth:`draw` returns the year column of this frame.

        Args:
            input_inventory (AssetInventory):
                An inventory whose assets all pass :meth:`check`.
            iteration (int):
                Resolution-loop attempt number, as :meth:`draw` takes it.

        Returns:
            pd.DataFrame:
                One row per asset id, with ``donor_id``, ``outcome``,
                ``n_candidates``, ``distance_m`` and the copied integer
                year under ``year_built_key``. ``outcome`` is the donor
                search's:
                :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_DONOR`
                or
                :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_FEWER_THAN_K`
                within the cap, and
                :data:`~brails.imputers.observed_donor_imputer.donor_records.OUTCOME_BLOCK_GROUP`
                from the block group. ``n_candidates`` counts the donors
                the pick was made among.

        Raises:
            RuntimeError:
                If the sampler has not been fitted.
            ValueError:
                If an asset does not pass :meth:`check`, naming the
                assets, or draws no donor, which happens only when the
                inventory is not a realization of the fitted stage.
        """
        pool = self._imputer.pool(self._require_fit(), input_inventory)
        assets = {}
        failures = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            diagnosis, detail, _ = self._resolve(
                pool, asset_id, features, input_inventory.vector_features
            )
            if diagnosis == DIAGNOSIS_OK:
                assets[asset_id] = features
            else:
                failures[asset_id] = detail
        if failures:
            raise ValueError(
                f'draw needs assets that pass check(); {len(failures)} '
                f'fail: {preview_failures(pd.Series(failures, dtype=object))}.'
            )
        rows = []
        for asset_id, features in assets.items():
            # status=True keeps the residential donors in place of the
            # status round, since the asset's own class lies outside it
            pick = self._imputer.draw(
                pool,
                asset_id,
                dict(features),
                (self.year_built_key,),
                world=self.world,
                iteration=iteration,
                status=True,
                purpose_prefix=NEIGHBOR_YEAR_PURPOSE_PREFIX,
            )
            if pick.outcome not in (
                OUTCOME_DONOR,
                OUTCOME_FEWER_THAN_K,
                OUTCOME_BLOCK_GROUP,
            ):
                msg = (
                    f'Asset {asset_id} passed check but drew no donor '
                    f'({pick.outcome}); pass a realization of the stage the '
                    'sampler was fitted on.'
                )
                raise ValueError(msg)
            copied = dict(pick.values)
            rows.append(
                (
                    pick.donor_id,
                    pick.outcome,
                    pick.n_candidates,
                    pick.distance_m,
                    int(copied[self.year_built_key]),
                )
            )
        frame = pd.DataFrame(
            rows,
            index=pd.Index(list(assets), dtype=object),
            columns=[
                'donor_id',
                'outcome',
                'n_candidates',
                'distance_m',
                self.year_built_key,
            ],
        )
        return frame.astype({self.year_built_key: 'int64'})

    def infer(self, input_inventory: AssetInventory) -> AssetInventory:
        """
        Copy a construction year to every asset in the inventory.

        Args:
            input_inventory (AssetInventory):
                The inventory to process. Every asset must pass
                :meth:`check`; filter the inventory with it first if that
                is not guaranteed.

        Returns:
            AssetInventory:
                A copy of the input with the copied year written under
                ``year_built_key``, replacing any year already there.

        Raises:
            RuntimeError:
                If the sampler has not been fitted.
            ValueError:
                If any asset cannot be drawn for, naming the assets and
                the reason each fails.
        """
        verdicts = self.check(input_inventory)
        failed = verdicts.loc[verdicts['diagnosis'] != DIAGNOSIS_OK, 'detail']
        if not failed.empty:
            raise ValueError(
                f'Cannot copy {self.year_built_key} to {len(failed)} of '
                f'{len(verdicts)} assets: {preview_failures(failed)}. Remove '
                'them or correct their data, then call infer() again.'
            )
        years = self.draw(input_inventory)
        output_inventory = deepcopy(input_inventory)
        for asset_id, year in years.items():
            output_inventory.add_asset_features(
                asset_id, {self.year_built_key: int(year)}
            )
        return output_inventory
