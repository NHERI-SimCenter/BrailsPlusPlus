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

"""Tests for the NSI tract-median year sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr
from scipy.stats import truncnorm

import brails.inferers
from brails.inferers.year_built_sampler.year_built_sampler import (
    DIAGNOSIS_OK,
    YearBuiltSampler,
    sample_years,
)
from brails.utils import Importer
from brails.utils.deterministic_sampling import (
    asset_key_digest,
    keyed_uniforms,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from brails.types.asset_inventory import AssetInventory

# ---------------------------------------------------------------------
# check


def test_a_usable_median_passes(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """An ordinary tract median is drawable."""
    inventory = build_inventory({'a': {'YearBuiltTractMedian': 1962}})
    assert YearBuiltSampler(sigma=15).check(inventory)['a'] == DIAGNOSIS_OK


def test_a_missing_median_is_diagnosed(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """An asset outside NSI coverage carries no median and is named."""
    inventory = build_inventory({'a': {'OccupancyClass': 'RES1'}})
    assert 'no YearBuiltTractMedian' in YearBuiltSampler(sigma=15).check(
        inventory
    )['a']


def test_a_zero_median_is_diagnosed(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """NSI ships zeros, and sampling around one would invent an old building."""
    inventory = build_inventory({'a': {'YearBuiltTractMedian': 0}})
    assert 'sits outside' in YearBuiltSampler(sigma=15).check(inventory)['a']


def test_a_median_past_the_cap_is_diagnosed(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The same defect from the other end."""
    inventory = build_inventory({'a': {'YearBuiltTractMedian': 3000}})
    assert 'sits outside' in YearBuiltSampler(sigma=15).check(inventory)['a']


def test_a_non_numeric_median_is_diagnosed(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Text where a year belongs is reported, not coerced."""
    inventory = build_inventory({'a': {'YearBuiltTractMedian': 'unknown'}})
    assert 'not a number' in YearBuiltSampler(sigma=15).check(inventory)['a']


def test_a_per_world_median_is_diagnosed(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Multi-world sampling is not supported yet, and the sampler says so."""
    inventory = build_inventory(
        {'a': {'YearBuiltTractMedian': [1960, 1970]}}, n_pw=2
    )
    assert 'per-world' in YearBuiltSampler(sigma=15).check(inventory)['a']


def test_check_leaves_the_inventory_alone(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The screen is pure; the compiler runs it on every asset."""
    inventory = build_inventory({'a': {'YearBuiltTractMedian': 1962}})
    YearBuiltSampler(sigma=15).check(inventory)
    _, features = inventory.get_asset_features('a')
    assert 'YearBuilt' not in features


# ---------------------------------------------------------------------
# infer


def test_infer_names_every_asset_it_cannot_draw_for(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The plain caller gets a loud failure with the diagnoses attached."""
    inventory = build_inventory(
        {
            'a': {'YearBuiltTractMedian': 1962},
            'b': {'OccupancyClass': 'RES1'},
            'c': {'YearBuiltTractMedian': 0},
        }
    )
    with pytest.raises(ValueError, match=r'Cannot sample YearBuilt for 2 of 3'):
        YearBuiltSampler(sigma=15).infer(inventory)


def test_infer_writes_a_year_to_every_asset(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The happy path fills the inventory and leaves the input alone."""
    inventory = build_inventory(
        {
            'a': {'YearBuiltTractMedian': 1962},
            'b': {'YearBuiltTractMedian': 1938},
        }
    )
    result = YearBuiltSampler(sigma=15, seed=7).infer(inventory)
    for asset_id in ('a', 'b'):
        _, features = result.get_asset_features(asset_id)
        assert 1800 <= features['YearBuilt'] <= 2026
        _, original = inventory.get_asset_features(asset_id)
        assert 'YearBuilt' not in original


def test_years_are_python_integers(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Making sure we do not have a numpy scalar here."""
    inventory = build_inventory({'a': {'YearBuiltTractMedian': 1962}})
    _, features = (
        YearBuiltSampler(sigma=15).infer(inventory).get_asset_features('a')
    )
    assert type(features['YearBuilt']) is int


def test_an_observed_year_is_overwritten(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Pinned behavior, not an accident.

    Deciding when a sampled year may displace an observed one is the
    compiler's call; the sampler stays narrow and always writes.
    """
    inventory = build_inventory(
        {'a': {'YearBuiltTractMedian': 1962, 'YearBuilt': 1905}}
    )
    _, features = (
        YearBuiltSampler(sigma=15, seed=7).infer(inventory).get_asset_features('a')
    )
    assert features['YearBuilt'] != 1905


def test_custom_keys_are_honored(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The feature names are configuration, not constants."""
    inventory = build_inventory({'a': {'tract_median': 1962}})
    sampler = YearBuiltSampler(
        sigma=15, median_key='tract_median', year_built_key='erected'
    )
    _, features = sampler.infer(inventory).get_asset_features('a')
    assert 1800 <= features['erected'] <= 2026


# ---------------------------------------------------------------------
# determinism


def test_years_survive_reordering_and_subsetting(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """An asset keeps its year alone, reordered, and in company."""
    features = {
        'a': {'YearBuiltTractMedian': 1962},
        'b': {'YearBuiltTractMedian': 1938},
        'c': {'YearBuiltTractMedian': 2014},
    }
    sampler = YearBuiltSampler(sigma=15, seed=7)
    full = sampler.infer(build_inventory(features))
    reversed_order = sampler.infer(
        build_inventory(dict(reversed(list(features.items()))))
    )
    alone = sampler.infer(build_inventory({'b': features['b']}))
    _, from_full = full.get_asset_features('b')
    _, from_reversed = reversed_order.get_asset_features('b')
    _, from_alone = alone.get_asset_features('b')
    assert from_full['YearBuilt'] == from_reversed['YearBuilt']
    assert from_full['YearBuilt'] == from_alone['YearBuilt']


def test_each_sampler_draws_its_own_seed() -> None:
    """A probabilistic result must look probabilistic on a rerun."""
    first, second = YearBuiltSampler(sigma=15), YearBuiltSampler(sigma=15)
    assert first.seed != second.seed


def test_a_recorded_seed_repeats_the_run(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The seed a run used is readable, and replaying it gives those years."""
    features = {'a': {'YearBuiltTractMedian': 1962}}
    sampler = YearBuiltSampler(sigma=15)
    _, drawn = sampler.infer(build_inventory(features)).get_asset_features('a')
    replay = YearBuiltSampler(sigma=15, seed=sampler.seed)
    _, repeated = replay.infer(build_inventory(features)).get_asset_features('a')
    assert repeated['YearBuilt'] == drawn['YearBuilt']


def test_pinned_years() -> None:
    """Pin the draw itself, so a refactor cannot move it quietly."""
    years = sample_years(
        pd.Series({'a': 1962, 'b': 1938, 'c': 2014}), sigma=15.0, seed=7
    )
    assert years.tolist() == [1966, 1940, 2020]


def test_different_seeds_draw_differently() -> None:
    """The seed is not a decoration."""
    medians = pd.Series({str(i): 1962 for i in range(50)})
    first = sample_years(medians, sigma=15.0, seed=1)
    second = sample_years(medians, sigma=15.0, seed=2)
    assert not first.equals(second)


def test_iterations_draw_independently() -> None:
    """Multiple realizations for the same asset should not be identical."""
    medians = pd.Series({str(i): 1962 for i in range(50)})
    first = sample_years(medians, sigma=15.0, seed=1)
    second = sample_years(medians, sigma=15.0, seed=1, iteration=1)
    assert not first.equals(second)


# ---------------------------------------------------------------------
# the distribution itself


def _cell_probabilities(
    years: np.ndarray, mean: float, sigma: float, lower: float, upper: float
) -> np.ndarray:
    """
    Exact probability of each integer year under the truncated normal.

    Args:
        years (np.ndarray):
            The integer years to score.
        mean (float):
            Distribution mean.
        sigma (float):
            Standard deviation before truncation.
        lower (float):
            Lowest year the sampler may draw.
        upper (float):
            Highest year the sampler may draw.

    Returns:
        np.ndarray:
            One probability per year, normalized over the interval.
    """
    edge = ndtr((years + 0.5 - mean) / sigma) - ndtr(
        (years - 0.5 - mean) / sigma
    )
    total = ndtr((upper + 0.5 - mean) / sigma) - ndtr(
        (lower - 0.5 - mean) / sigma
    )
    return edge / total


def test_years_match_an_independently_computed_reference() -> None:
    """Pin the map from uniform to year against a separate scipy path.

    This is what holds the discretization. A rounding rule half a year
    out of step changes these years and passes every other test here,
    the frequency check below included. The truncation interval is the
    neighboring concern, covered by the nearby-bound test.
    """
    medians = pd.Series({str(i): 1962.0 for i in range(200)})
    sigma, seed, floor, cap = 15.0, 5, 1800, 2026
    drawn = sample_years(medians, sigma=sigma, seed=seed, floor=floor, cap=cap)
    uniforms = keyed_uniforms(
        asset_key_digest(medians.index), seed=seed, purpose='year_built'
    )
    lower, upper = (floor - 0.5 - 1962.0) / sigma, (cap + 0.5 - 1962.0) / sigma
    reference = np.rint(
        truncnorm.ppf(uniforms, lower, upper, loc=1962.0, scale=sigma)
    ).astype(int)
    assert drawn.tolist() == reference.tolist()


def test_the_sampled_years_match_the_exact_cell_probabilities() -> None:
    """The hash stream must actually be uniform once it becomes years.

    The reference test above compares against the same uniforms the
    sampler used. This one checks if the keyed stream is evenly distributed.

    Each year is a Bernoulli trial, so an observed frequency carries a
    standard error of sqrt(p(1-p)/N). Four standard deviations is a wide band,
    chosen to leave no room for a spurious failure; the seed is fixed,
    so the test either passes or fails every time. It is blind to a half-year
    shift, which moves no cell by even one standard error at this sample size.
    """
    count = 20000
    mean, sigma = 1962.0, 15.0
    drawn = sample_years(
        pd.Series({str(i): mean for i in range(count)}), sigma=sigma, seed=11
    )
    years = np.arange(1930, 1996)
    expected = _cell_probabilities(years, mean, sigma, 1800.0, 2026.0)
    observed = (
        drawn.value_counts().reindex(years, fill_value=0).to_numpy() / count
    )
    standard_error = np.sqrt(expected * (1.0 - expected) / count)
    tolerance = 4.0 * standard_error
    assert np.all(np.abs(observed - expected) <= tolerance)


def test_no_mass_accumulates_on_a_nearby_bound() -> None:
    """The bound gets one year's worth of probability, not a whole tail.

    With a median two years above the floor, clipping an untruncated
    draw would leave roughly 45% of the assets sitting on 1800.
    Truncation leaves about 3%.
    """
    count = 20000
    mean, sigma, floor = 1802.0, 15.0, 1800
    drawn = sample_years(
        pd.Series({str(i): mean for i in range(count)}),
        sigma=sigma,
        seed=13,
        floor=floor,
    )
    observed = float((drawn == floor).mean())
    expected = float(
        _cell_probabilities(np.array([floor]), mean, sigma, 1800.0, 2026.0)[0]
    )
    assert abs(observed - expected) < 0.01


def test_odd_bounds_are_respected() -> None:
    """Rounding half to even reaches past an odd bound; the clip catches it."""
    drawn = sample_years(
        pd.Series({str(i): 1801.0 for i in range(2000)}),
        sigma=15.0,
        seed=17,
        floor=1801,
        cap=2025,
    )
    assert drawn.min() >= 1801
    assert drawn.max() <= 2025


def test_per_asset_bounds_restrict_the_draw() -> None:
    """What the compiler's force-consistent draw needs from this function."""
    medians = pd.Series({'a': 1962, 'b': 1962})
    drawn = sample_years(
        medians,
        sigma=15.0,
        seed=7,
        lower=pd.Series({'a': 1900, 'b': 1950}),
        upper=pd.Series({'a': 1940, 'b': 2000}),
    )
    assert 1900 <= drawn['a'] <= 1940
    assert 1950 <= drawn['b'] <= 2000


def test_pre_digested_keys_give_the_same_years() -> None:
    """Hoisting the digest out of a redraw loop must not move a draw."""
    medians = pd.Series({'a': 1962, 'b': 1938, 'c': 2014})
    internal = sample_years(medians, sigma=15.0, seed=7)
    hoisted = sample_years(
        medians,
        sigma=15.0,
        seed=7,
        asset_keys=asset_key_digest(medians.index),
    )
    assert internal.equals(hoisted)


def test_misaligned_keys_are_refused() -> None:
    """Keys digested from the wrong index would silently swap draws."""
    medians = pd.Series({'a': 1962, 'b': 1938})
    with pytest.raises(ValueError, match='asset keys for'):
        sample_years(
            medians,
            sigma=15.0,
            seed=7,
            asset_keys=asset_key_digest(['a', 'b', 'c']),
        )


def test_sample_years_refuses_an_unusable_median() -> None:
    """Skipping check() is a caller error, and it is caught."""
    with pytest.raises(ValueError, match='usable median'):
        sample_years(pd.Series({'a': None}), sigma=15.0, seed=7)


# ---------------------------------------------------------------------
# configuration and wiring


def test_a_non_positive_sigma_is_refused() -> None:
    """Sigma is checked where the user sets it, not at the first draw."""
    with pytest.raises(ValueError, match='sigma must be positive'):
        YearBuiltSampler(sigma=0)


def test_inverted_bounds_are_refused() -> None:
    """A cap below the floor is caught at construction."""
    with pytest.raises(ValueError, match='cap must sit above floor'):
        YearBuiltSampler(sigma=15, floor=2000, cap=1900)


def test_the_importer_finds_the_sampler() -> None:
    """The class is discoverable by name like every other component."""
    assert Importer().get_class('YearBuiltSampler') is YearBuiltSampler
