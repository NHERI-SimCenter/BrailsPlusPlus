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

"""Tests for the keyed sampling helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from brails.utils.deterministic_sampling import (
    asset_key_digest,
    grouped_categorical,
    keyed_uniforms,
    require_run_seed,
    truncated_normal,
)

# ---------------------------------------------------------------------
# keys


def test_ids_of_either_type_share_a_key() -> None:
    """An integer id and its string form must not draw differently."""
    assert asset_key_digest([7])[0] == asset_key_digest(['7'])[0]


def test_distinct_ids_get_distinct_keys() -> None:
    """Keys separate the assets they stand for."""
    keys = asset_key_digest(['a', 'b', 'c', 1, 2])
    assert len(set(keys.tolist())) == len(keys)


# ---------------------------------------------------------------------
# uniforms


def test_uniforms_are_in_the_unit_interval() -> None:
    """Every draw lands in [0, 1)."""
    uniforms = keyed_uniforms(
        asset_key_digest(range(1000)), seed=1, purpose='year_built'
    )
    assert uniforms.min() >= 0.0
    assert uniforms.max() < 1.0


def test_draws_survive_reordering() -> None:
    """An asset draws the same number wherever it sits in the sequence."""
    forward = keyed_uniforms(
        asset_key_digest(['a', 'b', 'c']), seed=5, purpose='year_built'
    )
    reversed_draws = keyed_uniforms(
        asset_key_digest(['c', 'b', 'a']), seed=5, purpose='year_built'
    )
    assert reversed_draws.tolist() == forward[::-1].tolist()


def test_draws_survive_subsetting() -> None:
    """An asset draws the same number alone as in company."""
    full = keyed_uniforms(
        asset_key_digest(['a', 'b', 'c']), seed=5, purpose='year_built'
    )
    alone = keyed_uniforms(asset_key_digest(['b']), seed=5, purpose='year_built')
    assert alone[0] == full[1]


def test_the_seed_reaches_the_draw() -> None:
    """Two seeds must disagree."""
    keys = asset_key_digest(['a', 'b', 'c'])
    first = keyed_uniforms(keys, seed=1, purpose='year_built')
    second = keyed_uniforms(keys, seed=2, purpose='year_built')
    assert not np.array_equal(first, second)


def test_purpose_world_and_iteration_reach_the_draw() -> None:
    """Each coordinate of the key gives an independent stream."""
    keys = asset_key_digest(['a', 'b', 'c'])
    base = keyed_uniforms(keys, seed=1, purpose='year_built')
    for other in (
        keyed_uniforms(keys, seed=1, purpose='structure_type'),
        keyed_uniforms(keys, seed=1, purpose='year_built', world=1),
        keyed_uniforms(keys, seed=1, purpose='year_built', iteration=1),
    ):
        assert not np.array_equal(base, other)


def test_pinned_uniforms() -> None:
    """Pin the stream so a refactor cannot quietly move every draw."""
    uniforms = keyed_uniforms(
        asset_key_digest(['a', 'b', 'c']), seed=42, purpose='year_built'
    )
    assert uniforms.tolist() == [
        0.821191294533576,
        0.12624415651006138,
        0.9172472761615413,
    ]


# ---------------------------------------------------------------------
# truncated normal


def test_inversion_is_exact() -> None:
    """The draw sits at the quantile the uniform asked for."""
    uniforms = np.linspace(0.0, 1.0, 101)[:-1]
    mean, sigma, lower, upper = 1962.0, 15.0, 1899.5, 2026.5
    drawn = truncated_normal(
        uniforms, mean=mean, sigma=sigma, lower=lower, upper=upper
    )
    lower_cdf = ndtr((lower - mean) / sigma)
    upper_cdf = ndtr((upper - mean) / sigma)
    expected = lower_cdf + uniforms * (upper_cdf - lower_cdf)
    assert np.allclose(ndtr((drawn - mean) / sigma), expected)


def test_draws_stay_inside_the_bounds() -> None:
    """The ends of the uniform range land on the bounds, not past them."""
    uniforms = np.array([0.0, 0.5, np.nextafter(1.0, 0.0)])
    drawn = truncated_normal(
        uniforms, mean=1962.0, sigma=25.0, lower=1799.5, upper=2026.5
    )
    assert drawn.min() >= 1799.5
    assert drawn.max() <= 2026.5


def test_a_bound_whose_probability_underflows_still_draws() -> None:
    """A window so wide that its lower CDF is exactly zero.

    Here the inversion really would reach negative infinity, which is
    what the probability floor exists for. The draw has to come back
    finite and inside the bounds.
    """
    drawn = truncated_normal(
        np.array([0.0]), mean=1962.0, sigma=1.0, lower=1800.0, upper=2026.0
    )
    assert np.isfinite(drawn).all()
    assert 1800.0 <= drawn[0] <= 2026.0


def test_per_asset_bounds_apply() -> None:
    """Bounds may vary asset by asset, as era-restricted draws need."""
    uniforms = np.array([0.5, 0.5])
    drawn = truncated_normal(
        uniforms,
        mean=np.array([1962.0, 1962.0]),
        sigma=25.0,
        lower=np.array([1799.5, 1950.5]),
        upper=np.array([1940.5, 2026.5]),
    )
    assert drawn[0] <= 1940.5
    assert drawn[1] >= 1950.5


def test_a_non_positive_sigma_is_refused() -> None:
    """Sigma must be positive."""
    with pytest.raises(ValueError, match='sigma must be positive'):
        truncated_normal(
            np.array([0.5]), mean=1962.0, sigma=0.0, lower=1800.0, upper=2026.0
        )


def test_inverted_bounds_are_refused() -> None:
    """An upper bound below its lower bound is a caller error."""
    with pytest.raises(ValueError, match='above its lower bound'):
        truncated_normal(
            np.array([0.5]), mean=1962.0, sigma=15.0, lower=2000.0, upper=1900.0
        )


def test_an_interval_with_no_mass_is_refused() -> None:
    """A window far into the tail cannot be sampled, and says so."""
    with pytest.raises(ValueError, match='no representable probability'):
        truncated_normal(
            np.array([0.5]),
            mean=0.0,
            sigma=1.0,
            lower=100.0,
            upper=200.0,
        )


# ---------------------------------------------------------------------
# categorical draws


def test_weights_need_not_sum_to_one_hundred() -> None:
    """A printed row summing to 101 draws like one summing to 100."""
    uniforms = np.linspace(0.0, 1.0, 101, endpoint=False)
    printed = grouped_categorical(
        uniforms, weights=[[50.0, 51.0]], groups=np.zeros(101, dtype=int)
    )
    exact = grouped_categorical(
        uniforms,
        weights=[[50.0 / 101.0, 51.0 / 101.0]],
        groups=np.zeros(101, dtype=int),
    )
    assert printed.tolist() == exact.tolist()


def test_a_zero_weight_category_is_never_drawn() -> None:
    """A category with no share cannot come out of the draw."""
    uniforms = np.linspace(0.0, 1.0, 1000, endpoint=False)
    positions = grouped_categorical(
        uniforms,
        weights=[[30.0, 0.0, 70.0]],
        groups=np.zeros(1000, dtype=int),
    )
    assert set(positions.tolist()) == {0, 2}


def test_a_uniform_on_a_cumulative_edge_takes_the_upper_category() -> None:
    """Cells are half-open, so an edge belongs to the category above it."""
    below = np.nextafter(0.25, 0.0)
    positions = grouped_categorical(
        np.array([0.0, below, 0.25, 0.75]),
        weights=[[25.0, 50.0, 25.0]],
        groups=np.zeros(4, dtype=int),
    )
    assert positions.tolist() == [0, 0, 1, 2]


def test_the_largest_uniform_stays_inside_its_own_categories() -> None:
    """
    The top of a row is where a draw would spill past its last category.

    Rows of different widths sit in one padded array, so the widest one
    decides how much padding the others carry. The largest uniform a
    keyed draw can produce has to land on the last real category of each
    of them, but never on the padding.
    """
    largest = np.nextafter(1.0, 0.0)
    weights = [[1.0], [1.0, 1.0], [3.0, 0.0, 1.0], [1.0] * 16]
    positions = grouped_categorical(
        np.full(len(weights), largest),
        weights=weights,
        groups=np.arange(len(weights)),
    )
    assert positions.tolist() == [len(row) - 1 for row in weights]


def test_a_row_that_cannot_sum_to_one_still_draws_inside_itself() -> None:
    """Ten tenths normalize to just under one, and the top cell absorbs it."""
    row = [0.1] * 10
    assert np.cumsum(row)[-1] / np.sum(row) < 1.0
    positions = grouped_categorical(
        np.array([np.nextafter(1.0, 0.0)]),
        weights=[row],
        groups=np.array([0]),
    )
    assert positions.tolist() == [9]


def test_a_trailing_zero_weight_category_is_never_drawn() -> None:
    """
    A row that normalizes to just under one may end in zero weights.

    The uniforms between the last positive cumulative value and one have
    to land on the last category of positive weight, not on the
    zero-weight categories after it.
    """
    row = [0.1] * 10 + [0.0, 0.0]
    assert np.cumsum(row)[9] / np.sum(row) < 1.0
    positions = grouped_categorical(
        np.array([np.nextafter(1.0, 0.0)]),
        weights=[row],
        groups=np.array([0]),
    )
    assert positions.tolist() == [9]


def test_frequencies_follow_the_weights() -> None:
    """Draws land in each category about as often as its weight says."""
    sample = 20000
    expected = np.array([0.2, 0.3, 0.5])
    tolerance = 4 * np.sqrt(0.25 / sample)
    keys = asset_key_digest([f'asset_{index}' for index in range(sample)])
    uniforms = keyed_uniforms(keys, seed=11, purpose='structure_type')
    positions = grouped_categorical(
        uniforms,
        weights=[[20.0, 30.0, 50.0]],
        groups=np.zeros(uniforms.size, dtype=int),
    )
    shares = np.bincount(positions, minlength=3) / positions.size
    assert np.allclose(shares, expected, atol=tolerance)


def test_group_codes_must_have_weight_vectors() -> None:
    """A code with no distribution behind it is refused."""
    with pytest.raises(ValueError, match='Group codes must index'):
        grouped_categorical(np.array([0.5]), weights=[[1.0]], groups=np.array([1]))


def test_uniforms_and_codes_must_line_up() -> None:
    """One group code per uniform, or the draw is not defined."""
    with pytest.raises(ValueError, match='one group code per uniform'):
        grouped_categorical(
            np.array([0.5, 0.5]), weights=[[1.0]], groups=np.array([0])
        )


def test_an_empty_weight_vector_is_refused() -> None:
    """Empty weights cannot be used to draw from."""
    with pytest.raises(ValueError, match='is empty'):
        grouped_categorical(np.array([0.5]), weights=[[]], groups=np.array([0]))


def test_a_vector_of_zeros_is_refused() -> None:
    """Restricting a cell down to nothing must not draw silently."""
    with pytest.raises(ValueError, match='sums to'):
        grouped_categorical(
            np.array([0.5]), weights=[[0.0, 0.0]], groups=np.array([0])
        )


def test_a_negative_weight_is_refused() -> None:
    """Negative shares are a transcription error."""
    with pytest.raises(ValueError, match='non-negative'):
        grouped_categorical(
            np.array([0.5]), weights=[[1.0, -1.0]], groups=np.array([0])
        )


def test_an_empty_draw_returns_an_empty_result() -> None:
    """No assets is a valid input and not an error."""
    positions = grouped_categorical(
        np.zeros(0), weights=[[1.0]], groups=np.zeros(0, dtype=int)
    )
    assert positions.size == 0


def test_a_draw_with_no_distributions_returns_an_empty_result() -> None:
    """A caller that filtered every asset out has nothing to draw from."""
    positions = grouped_categorical(
        np.zeros(0), weights=[], groups=np.zeros(0, dtype=int)
    )
    assert positions.size == 0


def test_each_asset_is_compared_against_its_own_row() -> None:
    """
    A draw must read the distribution its group code names.

    Neighboring rows here are opposites, so a code that lands one row off
    draws the other category every time.
    """
    group_count = 50
    weights = [
        [100.0, 0.0] if code % 2 else [0.0, 100.0] for code in range(group_count)
    ]
    positions = grouped_categorical(
        np.full(group_count, 0.5),
        weights=weights,
        groups=np.arange(group_count),
    )
    assert positions.tolist() == [
        0 if code % 2 else 1 for code in range(group_count)
    ]


# ---------------------------------------------------------------------
# worlds of one run


def test_a_world_other_than_the_default_needs_a_seed() -> None:
    """World 0 may draw its own seed; any other world must be given one."""
    require_run_seed(0, None)
    require_run_seed(3, 11)
    with pytest.raises(ValueError, match='seed'):
        require_run_seed(3, None)
