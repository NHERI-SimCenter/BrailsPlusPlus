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

from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
    diagnosis_frame,
    preview_failures,
    preview_ids,
)
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

# Rounding a draw taken on the half-open-widened interval keeps every
# integer cell the same width, including the two at the bounds.
_HALF_YEAR = 0.5


def _as_center(value: object) -> float:
    """
    Read a tract median as the center a draw is taken around.

    Args:
        value (object):
            A raw median value, of any type.

    Returns:
        float:
            The value as a float. A value that reads as no finite number
            gives NaN.
    """
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float('nan')
    return number if np.isfinite(number) else float('nan')


def _aligned_bounds(
    bounds: pd.Series | None,
    index: pd.Index,
    default: int,
    name: str,
) -> np.ndarray:
    """
    Line per-asset bounds up with the assets they belong to.

    Args:
        bounds (pd.Series | None):
            Per-asset bounds, or None to use the shared default.
        index (pd.Index):
            The asset ids the draw is for.
        default (int):
            The bound to use for every asset when none are given.
        name (str):
            The parameter name, for the error message.

    Returns:
        np.ndarray:
            One bound per asset, ordered as the index.

    Raises:
        ValueError:
            If an asset has no bound.
    """
    if bounds is None:
        return np.full(len(index), float(default))
    aligned = bounds.reindex(index)
    if aligned.isna().any():
        unbounded = list(index[aligned.isna().to_numpy()])
        raise ValueError(
            f'{name} carries no bound for every asset being drawn for. '
            f'Bounds are matched by asset id, not by position. Missing '
            f'on: {preview_ids(unbounded)}.'
        )
    return aligned.to_numpy(dtype=float)


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
            Tract median years, indexed by asset id. Every value must
            be one :meth:`YearBuiltSampler.check` accepts; the same
            conditions are enforced here.
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
            restricted to a legal era. Matched to the medians by asset
            id, so the order they arrive in does not matter.
        upper (pd.Series | None):
            Per-asset upper bounds overriding the cap, matched the same
            way.
        world (int):
            Possible-world index.
        iteration (int):
            Resolution-loop attempt number.

    Returns:
        pd.Series:
            Integer years named ``YearBuilt``, indexed as the input.

    Raises:
        ValueError:
            If any median is one check() would refuse, which means
            check() was skipped, or if the supplied bounds or keys do
            not cover the assets being drawn for.
    """
    centers = medians.map(_as_center).to_numpy(dtype=float)
    usable = (
        np.isfinite(centers) & (centers >= floor) & (centers <= cap)
    )
    if not np.all(usable):
        unusable = list(medians.index[~usable])
        raise ValueError(
            'sample_years needs a usable median for every asset: a number '
            f'inside [{floor}, {cap}]. Run check() and drop the failures '
            f'first. Unusable on: {preview_ids(unusable)}.'
        )
    low = _aligned_bounds(lower, medians.index, floor, 'lower')
    high = _aligned_bounds(upper, medians.index, cap, 'upper')
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

    Each year is drawn around the tract median the asset carries under
    ``median_key``, through :func:`sample_years`. The class writes
    ``YearBuilt`` whether or not one is already there. Deciding when a
    sampled year should displace an observed one belongs to the caller.

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

    def check(self, input_inventory: AssetInventory) -> pd.DataFrame:
        """
        Diagnose every asset without drawing anything.

        Args:
            input_inventory (AssetInventory):
                The inventory to diagnose.

        Returns:
            pd.DataFrame:
                One row per asset id, with ``diagnosis``, ``detail`` and
                ``resolvable_by``. A median that is absent, unreadable
                or outside the sampler's bounds is invalid and names the
                median feature. A median holding one value per world is
                unsupported and names nothing.
        """
        rows = {}
        for asset_id in input_inventory.get_asset_ids():
            _, features = input_inventory.get_asset_features(asset_id)
            rows[asset_id] = self._resolve(
                features, input_inventory.vector_features
            )
        return diagnosis_frame(rows)

    def _resolve(
        self, features: dict, vector_features: set
    ) -> tuple[str, str, tuple[str, ...]]:
        """
        Diagnose one asset from its features.

        Args:
            features (dict):
                The asset's features.
            vector_features (set):
                Names of the features whose values are vectors.

        Returns:
            tuple[str, str, tuple[str, ...]]:
                The diagnosis, the detail, and the features that would
                each, alone, admit the asset.
        """
        median = features.get(self.median_key)
        if median is None:
            return (
                DIAGNOSIS_INVALID,
                f'no {self.median_key}',
                (self.median_key,),
            )
        if is_probabilistic_feature(
            self.median_key, median, vector_features
        ):
            return (
                DIAGNOSIS_UNSUPPORTED,
                f'{self.median_key} holds one value per world; '
                'multi-world sampling is planned and not supported yet',
                (),
            )
        failure = self._check_value(median)
        if failure is not None:
            return DIAGNOSIS_INVALID, failure, (self.median_key,)
        return DIAGNOSIS_OK, '', ()

    def _check_value(self, median: object) -> str | None:
        """
        Say why one median cannot be used.

        Args:
            median (object):
                The value read from the asset.

        Returns:
            str | None:
                The reason the value is unusable, or ``None`` where it is
                usable.
        """
        if isinstance(median, bool):
            return f'{self.median_key} is a boolean: {median!r}'
        try:
            year = float(median)
        except (TypeError, ValueError):
            return f'{self.median_key} is not a number: {median!r}'
        except OverflowError:
            return (
                f'{self.median_key} of {median!r} sits outside '
                f'[{self.floor}, {self.cap}]'
            )
        if not np.isfinite(year):
            return f'{self.median_key} is not a number: {median!r}'
        if year < self.floor or year > self.cap:
            return (
                f'{self.median_key} of {median!r} sits outside '
                f'[{self.floor}, {self.cap}]'
            )
        return None

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
        failed = verdicts.loc[
            verdicts['diagnosis'] != DIAGNOSIS_OK, 'detail'
        ]
        if not failed.empty:
            reasons = preview_failures(failed)
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
