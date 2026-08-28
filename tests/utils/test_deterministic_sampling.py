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
    keyed_uniforms,
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
    alone = keyed_uniforms(
        asset_key_digest(['b']), seed=5, purpose='year_built'
    )
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
