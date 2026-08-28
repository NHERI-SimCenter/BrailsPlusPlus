# Copyright (c) 2025 The Regents of the University of California
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
Sample construction years from a tract median.

.. autosummary::

      YearBuiltSampler
      sample_years
"""

from __future__ import annotations

import secrets
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from brails.inferers.inference_engine import InferenceEngine
from brails.types.asset_inventory import is_probabilistic_feature
from brails.utils.deterministic_sampling import (
    DEFAULT_ITERATION,
    DEFAULT_WORLD,
    asset_key_digest,
    keyed_uniforms,
    truncated_normal,
)

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory

# The oldest year the sampler will draw. It exists to bound the tail.
DEFAULT_YEAR_FLOOR = 1800

# The newest year the sampler will draw, set to the vintage of the NSI
# 2026 data this was built against. It is a constant rather than the
# current year on purpose: a wall-clock cap would change the draws from
# one calendar year to the next and break the promise that a seed
# reproduces an inventory exactly. Set it to your inventory's vintage.
DEFAULT_YEAR_CAP = 2026

# Part of the reproducibility contract; see keyed_uniforms.
YEAR_PURPOSE = 'year_built'

# Width of a generated seed, matching the key width of the sampling
# helpers. A seed is drawn rather than fixed so that a probabilistic
# result is visibly probabilistic; anyone who wants the same years twice
# passes the seed they were given.
_SEED_BITS = 64

# The verdict check() returns for an asset the sampler can draw for.
DIAGNOSIS_OK = 'OK'

# Rounding a draw taken on the half-open-widened interval keeps every
# integer cell the same width, including the two at the bounds.
_HALF_YEAR = 0.5


def _preview(asset_ids: list) -> str:
    """
    Format up to ten asset ids for an error message.

    Args:
        asset_ids (list):
            The asset ids to show.

    Returns:
        str:
            A comma-separated preview with a count of any remainder.
    """
    shown = ', '.join(str(asset_id) for asset_id in asset_ids[:10])
    extra = len(asset_ids) - 10
    if extra > 0:
        shown += f' and {extra} more'
    return shown


def sample_years(
    medians: pd.Series,
    *,
    sigma: float,
    seed: int,
    asset_keys: np.ndarray | None = None,
    floor: int = DEFAULT_YEAR_FLOOR,
    cap: int = DEFAULT_YEAR_CAP,
    lower: pd.Series | None = None,
    upper: pd.Series | None = None,
    world: int = DEFAULT_WORLD,
    iteration: int = DEFAULT_ITERATION,
) -> pd.Series:
    """
    Draw one construction year per asset around its tract median.

    Each year is a truncated normal centered on the asset's median,
    discretized by rounding. The truncation runs from half a year below
    the floor to half a year above the cap, so every integer in
    [floor, cap] owns a full unit of probability and none of them
    accumulate the mass that clipping an untruncated draw would leave on
    the bounds.

    Args:
        medians (pd.Series):
            Tract median years, indexed by asset id. Every value must be
            usable: run :meth:`YearBuiltSampler.check` first.
        sigma (float):
            Standard deviation of the year deviation, in years.
        seed (int):
            Seed for the draw.
        asset_keys (np.ndarray | None):
            Keys from :func:`asset_key_digest`, aligned with the index.
            Digesting ids is the expensive part of a draw, so a caller
            that draws for the same assets repeatedly should digest once
            and pass the result. Left out, the ids are digested here.
        floor (int):
            Oldest year that can be drawn, unless overridden per asset.
        cap (int):
            Newest year that can be drawn, unless overridden per asset.
        lower (pd.Series | None):
            Per-asset lower bounds overriding the floor, for draws
            restricted to a legal era.
        upper (pd.Series | None):
            Per-asset upper bounds overriding the cap.
        world (int):
            Possible-world index.
        iteration (int):
            Resolution-loop attempt number.

    Returns:
        pd.Series:
            Integer years named ``YearBuilt``, indexed as the input.

    Raises:
        ValueError:
            If a median is missing or non-numeric, which means check()
            was skipped, or if the supplied keys do not align with the
            medians.
    """
    centers = pd.to_numeric(medians, errors='coerce').to_numpy(dtype=float)
    if not np.all(np.isfinite(centers)):
        unusable = list(medians.index[~np.isfinite(centers)])
        raise ValueError(
            'sample_years needs a usable median for every asset. Run '
            f'check() and drop the failures first. Unusable on: '
            f'{_preview(unusable)}.'
        )
    low = (
        np.full(centers.shape, float(floor))
        if lower is None
        else lower.to_numpy(dtype=float)
    )
    high = (
        np.full(centers.shape, float(cap))
        if upper is None
        else upper.to_numpy(dtype=float)
    )
    if asset_keys is None:
        keys = asset_key_digest(medians.index)
    else:
        keys = np.asarray(asset_keys, dtype=np.uint64)
        if keys.shape != centers.shape:
            raise ValueError(
                f'Got {keys.shape[0]} asset keys for {centers.shape[0]} '
                'medians. Digest the same index the medians carry.'
            )
    uniforms = keyed_uniforms(
        keys,
        seed=seed,
        purpose=YEAR_PURPOSE,
        world=world,
        iteration=iteration,
    )
    drawn = truncated_normal(
        uniforms,
        mean=centers,
        sigma=sigma,
        lower=low - _HALF_YEAR,
        upper=high + _HALF_YEAR,
    )
    # Both numpy and Python round half to even, so a half-year landing on
    # an odd bound rounds away from it: 1800.5 with a floor of 1801 gives
    # 1800. The clip is what actually holds the bounds.
    years = np.clip(np.rint(drawn), low, high).astype(np.int64)
    return pd.Series(years, index=medians.index, name='YearBuilt')


class YearBuiltSampler(InferenceEngine):
    """
    Add a sampled construction year to every asset in an inventory.

    The National Structure Inventory reports a median year built for the
    Census tract, not for the building. Treating that median as an
    observed year gives a city where every building in a tract was built
    the same year; this class replaces it with a draw around it.

    The class writes ``YearBuilt`` whether or not one is already there.
    Deciding when a sampled year should displace an observed one belongs
    to the caller.

    Each sampler draws its own seed unless given one, so two runs over
    the same inventory produce different years. Read ``self.seed`` to
    learn the value a run used, and pass it back to repeat that run.
    """

    def __init__(
        self,
        sigma: float,
        seed: int | None = None,
        floor: int = DEFAULT_YEAR_FLOOR,
        cap: int = DEFAULT_YEAR_CAP,
        median_key: str = 'YearBuiltTractMedian',
        year_built_key: str = 'YearBuilt',
        world: int = DEFAULT_WORLD,
    ) -> None:
        """
        Set up the sampler and validate its configuration.

        Args:
            sigma (float):
                Standard deviation of the year distribution, in years.
            seed (int | None):
                Seed for the draw. Left out, a seed is generated at
                random, so two runs of the same inventory give different
                years. The value used is recorded on ``self.seed``; pass
                it back to repeat a run exactly.
            floor (int):
                Oldest year that can be drawn.
            cap (int):
                Newest year that can be drawn.
            median_key (str):
                Feature name the tract median is read from.
            year_built_key (str):
                Feature name the sampled year is written to.
            world (int):
                Possible-world index. Only a single world is
                supported at present.

        Raises:
            ValueError:
                If sigma is not positive or the bounds are inverted.
        """
        if sigma <= 0:
            raise ValueError(f'sigma must be positive, got {sigma!r}.')
        if cap <= floor:
            raise ValueError(
                f'cap must sit above floor, got floor={floor!r} and '
                f'cap={cap!r}.'
            )
        self.sigma = float(sigma)
        self.seed = secrets.randbits(_SEED_BITS) if seed is None else int(seed)
        self.floor = int(floor)
        self.cap = int(cap)
        self.median_key = median_key
        self.year_built_key = year_built_key
        self.world = int(world)

    def check(self, input_inventory: AssetInventory) -> pd.Series:
        """
        Diagnose every asset without drawing anything.

        Args:
            input_inventory (AssetInventory):
                The inventory to diagnose.

        Returns:
            pd.Series:
                One verdict per asset id: ``'OK'`` where a year can be
                drawn, otherwise what stands in the way. A tract median
                of zero is a real NSI value, not a missing one, and
                sampling around it would date the building to the
                floor with full confidence, so it is diagnosed rather
                than drawn.
        """
        verdicts = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            median = features.get(self.median_key)
            if median is None:
                verdicts[asset_id] = f'no {self.median_key}'
            elif is_probabilistic_feature(
                self.median_key, median, input_inventory.vector_features
            ):
                verdicts[asset_id] = (
                    f'{self.median_key} is a per-world list; multi-world '
                    'sampling is planned and not supported yet'
                )
            else:
                verdicts[asset_id] = self._check_value(median)
        return pd.Series(verdicts, dtype=object, name='diagnosis')

    def _check_value(self, median: object) -> str:
        """
        Diagnose one median value.

        Args:
            median (object):
                The value read from the asset.

        Returns:
            str:
                ``'OK'`` or the reason it cannot be used.
        """
        try:
            year = float(median)
        except (TypeError, ValueError):
            return f'{self.median_key} is not a number: {median!r}'
        if not np.isfinite(year):
            return f'{self.median_key} is not a number: {median!r}'
        if year < self.floor or year > self.cap:
            return (
                f'{self.median_key} of {median!r} sits outside '
                f'[{self.floor}, {self.cap}]'
            )
        return DIAGNOSIS_OK

    def draw(
        self,
        input_inventory: AssetInventory,
        *,
        iteration: int = DEFAULT_ITERATION,
    ) -> pd.Series:
        """
        Draw a year for every asset, assuming all of them are usable.

        Args:
            input_inventory (AssetInventory):
                An inventory whose assets all passed :meth:`check`.
            iteration (int):
                Resolution-loop attempt number. Distinct iterations draw
                independently for the same asset.

        Returns:
            pd.Series:
                Integer years indexed by asset id.

        Raises:
            ValueError:
                If an asset carries no usable median, which means
                :meth:`check` was skipped.
        """
        medians = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            medians[asset_id] = features.get(self.median_key)
        if not medians:
            return pd.Series(dtype='int64', name=self.year_built_key)
        years = sample_years(
            pd.Series(medians, dtype=object),
            sigma=self.sigma,
            seed=self.seed,
            floor=self.floor,
            cap=self.cap,
            world=self.world,
            iteration=iteration,
        )
        return years.rename(self.year_built_key)

    def infer(self, input_inventory: AssetInventory) -> AssetInventory:
        """
        Sample a construction year for every asset in the inventory.

        Args:
            input_inventory (AssetInventory):
                The inventory to process. Every asset must carry a usable
                tract median; filter the inventory with :meth:`check`
                first if that is not guaranteed.

        Returns:
            AssetInventory:
                A copy of the input with the sampled year added under
                ``year_built_key``.

        Raises:
            ValueError:
                If any asset cannot be drawn for, naming the assets and
                their diagnoses.
        """
        verdicts = self.check(input_inventory)
        failed = verdicts[verdicts != DIAGNOSIS_OK]
        if not failed.empty:
            reasons = ', '.join(
                f'{asset_id} ({reason})'
                for asset_id, reason in list(failed.items())[:10]
            )
            extra = len(failed) - 10
            if extra > 0:
                reasons += f' and {extra} more'
            raise ValueError(
                f'Cannot sample {self.year_built_key} for {len(failed)} of '
                f'{len(verdicts)} assets: {reasons}. Remove them or supply '
                'the missing data, then call infer() again.'
            )
        years = self.draw(input_inventory)
        output_inventory = deepcopy(input_inventory)
        for asset_id, year in years.items():
            output_inventory.add_asset_features(
                asset_id, {self.year_built_key: int(year)}
            )
        return output_inventory
