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

"""Tests for the residential probability of the observed-donor imputer."""

from __future__ import annotations

import math

import pytest

from brails.constants import BLOCK_GROUP_KEY
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    ROUND_STATUS,
)
from tests.imputers.donor_fixtures import BG_A, BG_B, asset, imputer, inventory

OCC = 'OccupancyClass'
STORIES = 'NumberOfStories'


def donor(east: float, north: float, occupancy: str, bg: str = BG_A) -> tuple:
    """
    Return a one-story donor of a given class.

    Args:
        east (float):
            Meters east.
        north (float):
            Meters north.
        occupancy (str):
            Its class.
        bg (str, optional):
            Its block group.

    Returns:
        tuple:
            Coordinates and features.
    """
    return asset(east, north, **{OCC: occupancy, STORIES: 1, BLOCK_GROUP_KEY: bg})


def target(east: float, north: float, bg: str = BG_A) -> tuple:
    """
    Return a target whose only feature is its block group.

    Args:
        east (float):
            Meters east.
        north (float):
            Meters north.
        bg (str, optional):
            Its block group.

    Returns:
        tuple:
            Coordinates and features.
    """
    return asset(east, north, **{BLOCK_GROUP_KEY: bg})


class TestResidentialProbability:
    """The chance of the residential round, before its uniform."""

    def test_inverse_distance_share(self) -> None:
        """The chance is the residential share of the k nearest weights."""
        donors = inventory(
            {
                'h1': donor(100, 0, 'RES1'),
                'h2': donor(200, 0, 'RES1'),
                'c1': donor(400, 0, 'COM1'),
                'c2': donor(-1500, 0, 'COM1'),
            }
        )
        targets = inventory({'t': target(0, 0)})
        model = imputer(k=3)
        chance = model.status_probability(model.pool(donors, targets), ['t'])
        # With equal areas and no story count, the donor distance is the
        # location term alone, in units of 100 m.
        weights = {'h1': 1 / 1.0, 'h2': 1 / 2.0, 'c1': 1 / 4.0}
        expected = (weights['h1'] + weights['h2']) / sum(weights.values())
        assert chance['t'] == pytest.approx(expected, rel=1e-6)

    def test_configured_status(self) -> None:
        """Under a material status the chance is the share of that material."""
        donors = inventory(
            {
                'w1': asset(100, 0, **{OCC: 'COM1', STORIES: 1, 'Material': 'W'}),
                'w2': asset(200, 0, **{OCC: 'COM1', STORIES: 1, 'Material': 'W'}),
                'c1': asset(400, 0, **{OCC: 'RES1', STORIES: 1, 'Material': 'C'}),
            }
        )
        targets = inventory({'t': target(0, 0)})
        model = imputer(
            k=3,
            required=(OCC, STORIES, 'Material'),
            status_key='Material',
            status_values=('W',),
            status_name='wood',
        )
        chance = model.status_probability(model.pool(donors, targets), ['t'])
        weights = {'w1': 1 / 1.0, 'w2': 1 / 2.0, 'c1': 1 / 4.0}
        expected = (weights['w1'] + weights['w2']) / sum(weights.values())
        assert chance['t'] == pytest.approx(expected, rel=1e-6)
        with pytest.raises(ValueError, match='pool was built'):
            model.status_probability(imputer().pool(donors, targets), ['t'])

    def test_all_residential(self) -> None:
        """Residential neighbors alone give one."""
        donors = inventory({f'h{i}': donor(50 * i, 0, 'RES3A') for i in range(1, 7)})
        targets = inventory({'t': target(0, 0)})
        model = imputer()
        assert model.status_probability(model.pool(donors, targets), ['t']) == {
            't': 1.0
        }

    def test_block_group_fallback(self) -> None:
        """Without a donor in the cap, the block group's share decides."""
        donors = inventory(
            {
                'h': donor(5000, 0, 'RES1'),
                'c1': donor(5000, 100, 'COM1'),
                'c2': donor(5000, 200, 'COM1'),
                'x': donor(5000, 300, 'RES1', bg=BG_B),
            }
        )
        targets = inventory({'t': target(0, 0), 'u': target(0, 50, bg=BG_B)})
        model = imputer()
        chance = model.status_probability(model.pool(donors, targets), ['t', 'u'])
        assert chance['t'] == pytest.approx(1 / 3)
        assert chance['u'] == 1.0

    def test_no_donor_or_area(self) -> None:
        """No donor anywhere, or no area, gives NaN."""
        targets = inventory({'t': target(0, 0), 'p': ([[-122.0, 38.0]] * 4, {})})
        model = imputer()
        chance = model.status_probability(
            model.pool(inventory({}), targets), ['t', 'p']
        )
        assert math.isnan(chance['t'])
        assert math.isnan(chance['p'])

    def test_unknown_asset(self) -> None:
        """An asset the pool does not record raises."""
        model = imputer()
        with pytest.raises(LookupError):
            model.status_probability(model.pool(inventory({})), ['missing'])

    @pytest.mark.parametrize('far', [False, True])
    def test_matches_the_round(self, *, far: bool) -> None:
        """Over many worlds the round's residential share meets the chance."""
        offset = 5000 if far else 0
        donors = inventory(
            {
                'h1': asset(
                    offset + 100,
                    0,
                    12,
                    **{OCC: 'RES1', STORIES: 1, BLOCK_GROUP_KEY: BG_A},
                ),
                'h2': donor(offset + 150, 80, 'RES1'),
                'c1': donor(offset + 60, -40, 'COM1'),
                'c2': asset(
                    offset + 300,
                    0,
                    25,
                    **{OCC: 'COM1', STORIES: 1, BLOCK_GROUP_KEY: BG_A},
                ),
                'c3': donor(offset + 500, 200, 'COM1'),
            }
        )
        targets = inventory({'t': target(0, 0)})
        model = imputer(k=4)
        chance = model.status_probability(model.pool(donors, targets), ['t'])['t']
        worlds = 2000
        log = model.impute_stage(targets, donors, n_worlds=worlds).log
        picks = log.loc[log['round'] == ROUND_STATUS, 'donor_id']
        share = picks.isin(['h1', 'h2']).mean()
        assert len(picks) == worlds
        assert 0.0 < chance < 1.0
        assert abs(share - chance) < 4 * math.sqrt(chance * (1 - chance) / worlds)
