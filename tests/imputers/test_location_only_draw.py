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

"""Tests for the observed-donor draw configured for location alone."""

from __future__ import annotations

from brails.imputers import ObservedDonorImputer
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    DEFAULT_DONOR_CAP_M,
    DEFAULT_DONOR_K,
    OUTCOME_BLOCK_GROUP,
    OUTCOME_FEWER_THAN_K,
    PURPOSE_DONOR,
    PURPOSE_FALLBACK,
)
from brails.utils.deterministic_sampling import asset_key_digest, keyed_uniforms
from tests.imputers.donor_fixtures import BG_A, BG_B, SEED, asset, inventory

# The arguments under which the donor distance is the great-circle distance
# in meters, for a donor that supplies a year.
LOCATION_ONLY = {
    'features': ('YearBuilt',),
    'soft_filters': (),
    'required': ('OccupancyClass', 'YearBuilt'),
    'location_scale_m': 1.0,
    'area_weight': 0.0,
    'stories_key': None,
    'stories_weight': 0.0,
    'seed': SEED,
}

WORLDS = 50


def test_the_defaults_are_the_named_constants() -> None:
    """The imputer's k and cap default to the module's constants."""
    imputer = ObservedDonorImputer()
    assert (imputer.k, imputer.cap_m) == (DEFAULT_DONOR_K, DEFAULT_DONOR_CAP_M)


def test_location_alone_ignores_plan_area_and_stories() -> None:
    """The distance is in meters and ignores plan areas and story counts."""
    target = {'t': asset(0.0, 0.0, 10.0, OccupancyClass='COM1', NumberOfStories=2)}
    picks = []
    # five donors 5 to 25 m away, given other areas and stories, ranked
    # differently, the second time
    for sides, stories in (
        ((6, 8, 10, 12, 14), (1, 2, 3, 4, 5)),
        ((30, 20, 4, 9, 3), (9, 1, 6, 2, 8)),
    ):
        records = {
            f'r{i}': asset(
                5.0 * (i + 1),
                0.0,
                float(sides[i]),
                OccupancyClass='RES1',
                NumberOfStories=stories[i],
                YearBuilt=1950 + i,
            )
            for i in range(5)
        }
        imputer = ObservedDonorImputer(**LOCATION_ONLY)
        pool = imputer.pool(inventory(records), inventory(target))
        draws = [
            imputer.draw(
                pool, 't', target['t'][1], ('YearBuilt',), world=w, status=True
            )
            for w in range(WORLDS)
        ]
        assert all(d.distance == d.distance_m for d in draws)
        picks.append([d.donor_id for d in draws])
    assert picks[0] == picks[1]
    assert len(set(picks[0])) > 1


def test_a_given_status_keeps_only_its_donors() -> None:
    """A target drawn with the residential status copies only such donors."""
    # c0 and c1, non-residential, lie nearer than r0 and r1 to near
    records = {
        'c0': asset(100.0, 0.0, OccupancyClass='COM1', YearBuilt=1960),
        'c1': asset(0.0, 150.0, OccupancyClass='IND1', YearBuilt=1961),
        'r0': asset(300.0, 0.0, OccupancyClass='RES1', YearBuilt=1970),
        'r1': asset(0.0, -400.0, OccupancyClass='RES3A', YearBuilt=1980),
    }
    for features in records.values():
        features[1]['BLOCKGROUP_GEOID'] = BG_A
    # one target within the cap, one beyond it that falls back to block group A
    targets = {
        'near': asset(0.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_A),
        'far': asset(10000.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_A),
    }
    imputer = ObservedDonorImputer(**LOCATION_ONLY)
    pool = imputer.pool(inventory(records), inventory(targets))
    for target, outcome in (
        ('near', OUTCOME_FEWER_THAN_K),
        ('far', OUTCOME_BLOCK_GROUP),
    ):
        draws = [
            imputer.draw(
                pool,
                target,
                targets[target][1],
                ('YearBuilt',),
                world=w,
                status=True,
            )
            for w in range(WORLDS)
        ]
        assert {d.donor_id for d in draws} == {'r0', 'r1'}
        assert {d.outcome for d in draws} == {outcome}


def test_a_prefix_forms_the_labels_of_both_picks() -> None:
    """With a prefix, each pick is the one the prefixed label's uniform implies."""
    # donors at 100 m and 300 m for the donor round, and g0 to g2, of equal
    # weight and in id order, for the fallback
    records = {
        'a': asset(100.0, 0.0, OccupancyClass='RES1', YearBuilt=1950),
        'b': asset(300.0, 0.0, OccupancyClass='RES1', YearBuilt=1960),
        **{
            f'g{i}': asset(
                10000.0 + 100.0 * i,
                0.0,
                OccupancyClass='RES1',
                YearBuilt=1970 + i,
                BLOCKGROUP_GEOID=BG_A,
            )
            for i in range(3)
        },
    }
    targets = {
        'near': asset(0.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_B),
        'far': asset(20000.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_A),
    }
    imputer = ObservedDonorImputer(**LOCATION_ONLY)
    pool = imputer.pool(inventory(records), inventory(targets))
    for w in range(WORLDS):
        near, far = (
            imputer.draw(
                pool,
                target,
                targets[target][1],
                ('YearBuilt',),
                world=w,
                status=True,
                purpose_prefix='p',
            )
            for target in ('near', 'far')
        )
        u_near, u_far = (
            keyed_uniforms(
                asset_key_digest([target]),
                seed=SEED,
                purpose=f'p:{purpose}',
                world=w,
                iteration=0,
            )[0]
            for target, purpose in (
                ('near', PURPOSE_DONOR),
                ('far', PURPOSE_FALLBACK),
            )
        )
        # the donor at 100 m holds three quarters of the total weight
        # 1/100 + 1/300
        assert near.donor_id == ('a' if u_near < 0.75 else 'b')
        assert far.donor_id == ('g0', 'g1', 'g2')[min(int(u_far * 3), 2)]
        assert (near.outcome, far.outcome) == (
            OUTCOME_FEWER_THAN_K,
            OUTCOME_BLOCK_GROUP,
        )


def test_only_the_k_nearest_are_drawn() -> None:
    """With k 2 and three donors within the cap, the third is never drawn."""
    records = {
        f'r{i}': asset(100.0 * (i + 1), 0.0, OccupancyClass='RES1', YearBuilt=1950)
        for i in range(3)
    }
    target = {'t': asset(0.0, 0.0, OccupancyClass='COM1')}
    imputer = ObservedDonorImputer(**LOCATION_ONLY, k=2)
    pool = imputer.pool(inventory(records), inventory(target))
    drawn = {
        imputer.draw(
            pool, 't', target['t'][1], ('YearBuilt',), world=w, status=True
        ).donor_id
        for w in range(200)
    }
    assert drawn == {'r0', 'r1'}


def test_a_tie_in_distance_goes_to_the_lower_id() -> None:
    """Donors at one distance rank by their ids as strings."""
    # with k 1, d10 wins because it precedes d9 as a string
    records = {
        'd9': asset(100.0, 0.0, OccupancyClass='RES1', YearBuilt=1950),
        'd10': asset(100.0, 0.0, OccupancyClass='RES1', YearBuilt=1960),
    }
    target = {'t': asset(0.0, 0.0, OccupancyClass='COM1')}
    imputer = ObservedDonorImputer(**LOCATION_ONLY, k=1)
    pool = imputer.pool(inventory(records), inventory(target))
    drawn = {
        imputer.draw(
            pool, 't', target['t'][1], ('YearBuilt',), world=w, status=True
        ).donor_id
        for w in range(WORLDS)
    }
    assert drawn == {'d10'}
