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

"""Tests for the neighbor year sampler."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import brails.imputers
from brails.constants import BLOCK_GROUP_KEY
from brails.imputers.observed_donor_imputer.observed_donor_imputer import (
    DEFAULT_DONOR_CAP_M,
    DEFAULT_DONOR_K,
    OUTCOME_BLOCK_GROUP,
    OUTCOME_FEWER_THAN_K,
)
from brails.imputers.residential_neighbor_year_sampler.residential_neighbor_year_sampler import (
    ResidentialNeighborYearSampler,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INFEASIBLE,
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
)
from brails.utils import Importer
from brails.utils.stage_freeze import StageFreeze
from tests.imputers.donor_fixtures import (
    BG_A,
    BG_B,
    OTHER_SEED,
    SEED,
    asset,
    inventory,
    lonlat,
)

# A block group that holds no donor.
BG_C = '060010001003'

STAGE = 'residential'

# Golden picks under SEED: a target with donors within the cap, and one
# that falls back to its block group.
GOLDEN_PICKS = {'t0': ('r1', 1955), 'far': ('b0', 1990)}


def _frozen(
    records: dict[str, tuple[list, dict[str, Any]]],
    order: list[str] | None = None,
    n_pw: int = 1,
) -> tuple[StageFreeze, Any]:
    """
    Freeze an inventory of fixture assets as the residential stage.

    Args:
        records (dict[str, tuple[list, dict[str, Any]]]):
            Asset id to coordinates and features.
        order (list[str], optional):
            The insertion order. Defaults to the order of ``records``.
        n_pw (int, optional):
            The number of worlds the stage declares. Defaults to 1.

    Returns:
        tuple[StageFreeze, AssetInventory]:
            The freeze and the stage it holds.
    """
    stage = inventory(records, order)
    stage.n_pw = n_pw
    freeze = StageFreeze()
    freeze.freeze(STAGE, stage)
    return freeze, stage


def _donors() -> dict[str, tuple[list, dict[str, Any]]]:
    """
    Return residential donors near the origin and far from it.

    Returns:
        dict[str, tuple[list, dict[str, Any]]]:
            ``r0`` to ``r4``, 100 to 500 m east of the origin in block group
            A, with years 1950 to 1970, and ``b0`` to ``b2``, 10 km east in
            block group B, with years 1990 to 1992.
    """
    classes = ('RES1', 'RES2', 'RES3A', 'RES3C', 'RES3F')
    records = {
        f'r{i}': asset(
            100.0 * (i + 1),
            50.0 * (i % 2),
            OccupancyClass=classes[i],
            YearBuilt=1950 + 5 * i,
            BLOCKGROUP_GEOID=BG_A,
        )
        for i in range(5)
    }
    for i in range(3):
        records[f'b{i}'] = asset(
            10000.0 + 200.0 * i,
            0.0,
            OccupancyClass='RES1',
            YearBuilt=1990 + i,
            BLOCKGROUP_GEOID=BG_B,
        )
    return records


def _targets() -> dict[str, tuple[list, dict[str, Any]]]:
    """
    Return non-residential targets near the donors and one far from them.

    Returns:
        dict[str, tuple[list, dict[str, Any]]]:
            ``t0`` to ``t3`` near the origin in block group A, and ``far``,
            14 km east in block group B, beyond the cap of every donor.
    """
    classes = ('COM1', 'RES4', 'IND1', 'EDU1')
    records = {
        f't{i}': asset(
            40.0 * i, 80.0, OccupancyClass=classes[i], BLOCKGROUP_GEOID=BG_A
        )
        for i in range(4)
    }
    records['far'] = asset(
        14000.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_B
    )
    return records


def _point(east: float, north: float, **features: Any) -> tuple[list, dict]:
    """
    Return a fixture asset whose coordinates are one point.

    Args:
        east (float):
            Meters east of the origin.
        north (float):
            Meters north of the origin.
        **features (Any):
            The asset's features.

    Returns:
        tuple[list, dict]:
            Coordinates enclosing no area, and the features.
    """
    return [list(lonlat(east, north))], dict(features)


def _sampler(freeze: StageFreeze, **kwargs: Any) -> ResidentialNeighborYearSampler:
    """
    Build a sampler on the fixture seed and select its donors from the stage.

    Args:
        freeze (StageFreeze):
            The freeze holding :data:`STAGE`.
        **kwargs (Any):
            Overrides of the constructor arguments.

    Returns:
        ResidentialNeighborYearSampler:
            The sampler, with its donors selected.
    """
    return ResidentialNeighborYearSampler(**{'seed': SEED, **kwargs}).select_donors(
        freeze, STAGE
    )


# ---------------------------------------------------------------------
# What select_donors reads


def test_select_donors_reads_a_freeze_and_a_frozen_stage() -> None:
    """An inventory in place of the freeze, or an unfrozen stage, raises."""
    freeze, stage = _frozen(_donors())
    with pytest.raises(TypeError, match='StageFreeze'):
        ResidentialNeighborYearSampler(seed=SEED).select_donors(stage, STAGE)
    with pytest.raises(LookupError):
        ResidentialNeighborYearSampler(seed=SEED).select_donors(freeze, 'step4')


def test_select_donors_reads_the_held_stage_and_changes_nothing() -> None:
    """The freeze stays intact, and a second select_donors sees a changed stage."""
    records = {
        'r': asset(100.0, 0.0, OccupancyClass='RES1', YearBuilt=[1950, 1960, 1970]),
    }
    freeze, stage = _frozen(records, n_pw=3)
    targets = inventory({'t': asset(0.0, 0.0, OccupancyClass='COM1')})
    assert _sampler(freeze, world=1).draw(targets).loc['t'] == 1960
    freeze.manifest()
    # a change to the held stage after select_donors fails the manifest
    stage.inventory['r'].features['YearBuilt'][1] = 1999
    with pytest.raises(RuntimeError):
        freeze.manifest()
    assert _sampler(freeze, world=1).draw(targets).loc['t'] == 1999


def test_each_world_copies_its_own_residential_year() -> None:
    """Each world's draw copies its donor's year in that world."""
    # only r is a donor: the nearer assets are non-residential, lack a
    # year, hold a fractional year or enclose no area
    records = {
        'r': asset(300.0, 0.0, OccupancyClass='RES1', YearBuilt=[1950, 1960, 1970]),
        'c': asset(30.0, 0.0, OccupancyClass='COM1', YearBuilt=1900),
        'n': asset(40.0, 0.0, OccupancyClass='RES1'),
        'x': asset(50.0, 0.0, OccupancyClass='RES3A', YearBuilt=1955.5),
        'p': _point(60.0, 0.0, OccupancyClass='RES1', YearBuilt=1901),
    }
    freeze, _ = _frozen(records, n_pw=3)
    targets = inventory({'t': asset(0.0, 0.0, OccupancyClass='COM1')})
    for world, year in enumerate((1950, 1960, 1970)):
        sampler = _sampler(freeze, world=world)
        assert list(sampler.donors.index) == ['r']
        assert sampler.donors.loc['r', 'YearBuilt'] == year
        for iteration in range(20):
            picks = sampler.draw_details(targets, iteration=iteration)
            assert (picks.loc['t', 'donor_id'], picks.loc['t', 'YearBuilt']) == (
                'r',
                year,
            )


def test_a_world_matches_its_realization_frozen_alone() -> None:
    """World 2 of a three-world stage draws as that world's scalar stage."""
    records = {
        f'r{i}': asset(
            100.0 * (i + 1),
            0.0,
            OccupancyClass='RES1',
            YearBuilt=[1950 + i, 1960 + i, 1970 + i],
        )
        for i in range(5)
    }
    records['m'] = asset(
        20.0, 0.0, OccupancyClass=['COM1', 'COM1', 'RES1'], YearBuilt=2005
    )
    freeze, stage = _frozen(records, n_pw=3)
    alone = StageFreeze()
    alone.freeze(STAGE, stage.get_world_realization(2))
    targets = inventory({k: v for k, v in _targets().items() if k != 'far'})
    worlds = _sampler(freeze, world=2)
    assert 'm' in worlds.donors.index
    assert 'm' not in _sampler(freeze, world=0).donors.index
    pd.testing.assert_frame_equal(
        worlds.draw_details(targets), _sampler(alone, world=2).draw_details(targets)
    )


def test_select_donors_refuses_one_value_per_world() -> None:
    """A per-world list left in the world read raises, as does a missing world."""
    records = {'r': asset(0.0, 0.0, OccupancyClass='RES1', YearBuilt=[1950, 1960])}
    freeze, _ = _frozen(records)
    with pytest.raises(ValueError, match='one value per world'):
        _sampler(freeze)
    freeze, _ = _frozen(
        {'r': asset(0.0, 0.0, OccupancyClass='RES1', YearBuilt=[1950, 1960])},
        n_pw=2,
    )
    with pytest.raises(ValueError, match='holds 2 worlds'):
        _sampler(freeze, world=2)


# ---------------------------------------------------------------------
# The draws


def test_golden_picks() -> None:
    """The donor round's and the fallback's picks are pinned."""
    freeze, _ = _frozen(_donors())
    picks = _sampler(freeze).draw_details(inventory(_targets()))
    assert {
        asset_id: (picks.loc[asset_id, 'donor_id'], picks.loc[asset_id, 'YearBuilt'])
        for asset_id in GOLDEN_PICKS
    } == GOLDEN_PICKS
    assert picks.loc['t0', 'outcome'] == OUTCOME_FEWER_THAN_K
    assert picks.loc['far', 'outcome'] == OUTCOME_BLOCK_GROUP


def test_seed_world_and_iteration_change_picks() -> None:
    """A second seed, world or iteration changes at least one pick of each label."""
    freeze, _ = _frozen(_donors())
    # ten targets with donors within the cap, ten that fall back to block group B
    near = {f'n{i}': asset(20.0 * i, 0.0, OccupancyClass='COM1') for i in range(10)}
    far = {
        f'f{i}': asset(
            14000.0, 20.0 * i, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_B
        )
        for i in range(10)
    }
    for targets in (inventory(near), inventory(far)):
        first = _sampler(freeze).draw_details(targets)['donor_id']
        for other in (
            _sampler(freeze, seed=OTHER_SEED).draw_details(targets),
            _sampler(freeze, world=1).draw_details(targets),
            _sampler(freeze).draw_details(targets, iteration=1),
        ):
            assert not first.equals(other['donor_id'])


def test_reordering_changes_nothing() -> None:
    """Donors and targets inserted in another order give the same picks."""
    donors, targets = _donors(), _targets()
    freeze, _ = _frozen(donors)
    reversed_freeze, _ = _frozen(donors, order=list(reversed(donors)))
    expected = _sampler(freeze).draw_details(inventory(targets))
    picks = _sampler(reversed_freeze).draw_details(
        inventory(targets, order=list(reversed(targets)))
    )
    pd.testing.assert_frame_equal(picks.loc[expected.index], expected)


def test_the_distance_is_location_alone() -> None:
    """With k 1 the nearest donor is drawn, whatever its area and stories."""
    records = {
        'near': asset(
            50.0, 0.0, 40.0, OccupancyClass='RES1', NumberOfStories=5, YearBuilt=1950
        ),
        'same': asset(
            300.0,
            0.0,
            10.0,
            OccupancyClass='RES1',
            NumberOfStories=1,
            YearBuilt=1960,
        ),
    }
    targets = inventory(
        {'t': asset(0.0, 0.0, 10.0, OccupancyClass='COM1', NumberOfStories=1)}
    )
    picks = _sampler(_frozen(records)[0], k=1).draw_details(targets)
    assert picks.loc['t', 'donor_id'] == 'near'


def test_draws_are_local_to_the_cap_and_the_block_group() -> None:
    """Donor changes beyond the cap and the fallback's block group change no draw."""
    base = {
        **_donors(),
        'z1': asset(
            7000.0,
            3000.0,
            OccupancyClass='RES1',
            YearBuilt=2001,
            BLOCKGROUP_GEOID=BG_A,
        ),
    }
    added = {
        **base,
        'z0': asset(
            6000.0, 0.0, OccupancyClass='RES1', YearBuilt=2002, BLOCKGROUP_GEOID=BG_A
        ),
    }
    removed = {k: v for k, v in base.items() if k != 'z1'}
    changed = {
        **base,
        'z1': asset(
            7000.0,
            3000.0,
            OccupancyClass='RES1',
            YearBuilt=1801,
            BLOCKGROUP_GEOID=BG_A,
        ),
    }
    targets = inventory(_targets())
    expected = _sampler(_frozen(base)[0]).draw_details(targets)
    for records in (added, removed, changed):
        pd.testing.assert_frame_equal(
            _sampler(_frozen(records)[0]).draw_details(targets), expected
        )


def test_a_donor_just_past_the_cap_is_only_a_fallback() -> None:
    """A donor just past the cap is drawn only as its block group's fallback."""
    target = inventory(
        {'t': asset(0.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_A)}
    )
    edge = {'OccupancyClass': 'RES1', 'YearBuilt': 1977}
    records = {
        'edge': asset(2001.0, 0.0, **edge, BLOCKGROUP_GEOID=BG_B),
        'home': asset(
            5000.0, 0.0, OccupancyClass='RES1', YearBuilt=1988, BLOCKGROUP_GEOID=BG_A
        ),
    }
    sampler = _sampler(_frozen(records)[0])
    assert {
        sampler.draw_details(target, iteration=i).loc['t', 'donor_id']
        for i in range(20)
    } == {'home'}
    own = _sampler(
        _frozen({'edge': asset(2001.0, 0.0, **edge, BLOCKGROUP_GEOID=BG_A)})[0]
    ).draw_details(target)
    assert (own.loc['t', 'donor_id'], own.loc['t', 'outcome']) == (
        'edge',
        OUTCOME_BLOCK_GROUP,
    )
    assert own.loc['t', 'distance_m'] > DEFAULT_DONOR_CAP_M


def test_one_asset_calls_match_the_full_inventory() -> None:
    """One-asset draws match the full inventory's, and checks ignore the asset id."""
    freeze, _ = _frozen(_donors())
    sampler = _sampler(freeze)
    targets = _targets()
    years = sampler.draw(inventory(targets))
    refused = {
        'res': asset(0.0, 0.0, OccupancyClass='RES1', BLOCKGROUP_GEOID=BG_A),
        'alone': asset(30000.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_C),
        'r0': _donors()['r0'],
    }
    for asset_id, record in {**targets, **refused}.items():
        pd.testing.assert_frame_equal(
            sampler.check(inventory({'screened_asset': record})).reset_index(
                drop=True
            ),
            sampler.check(inventory({asset_id: record})).reset_index(drop=True),
        )
    for asset_id, record in targets.items():
        assert (
            sampler.draw(inventory({asset_id: record})).loc[asset_id]
            == (years.loc[asset_id])
        )


# ---------------------------------------------------------------------
# Check and infer


# The block group is read only for an asset with no donor within the cap, and
# coordinates without area, or two failures at once, name nothing.
@pytest.mark.parametrize(
    ('record', 'diagnosis', 'resolvable_by'),
    [
        (
            asset(0.0, 0.0, OccupancyClass='RES1', BLOCKGROUP_GEOID=BG_A),
            DIAGNOSIS_INFEASIBLE,
            ('OccupancyClass',),
        ),
        (
            asset(0.0, 0.0, OccupancyClass='RES4', BLOCKGROUP_GEOID=BG_A),
            DIAGNOSIS_OK,
            (),
        ),
        (asset(0.0, 0.0, OccupancyClass='COM1'), DIAGNOSIS_OK, ()),
        (
            asset(0.0, 0.0, BLOCKGROUP_GEOID=BG_A),
            DIAGNOSIS_INVALID,
            ('OccupancyClass',),
        ),
        (asset(0.0, 0.0, OccupancyClass=3), DIAGNOSIS_INVALID, ('OccupancyClass',)),
        (_point(0.0, 0.0, OccupancyClass='COM1'), DIAGNOSIS_INVALID, ()),
        (
            asset(30000.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_A),
            DIAGNOSIS_OK,
            (),
        ),
        (
            asset(30000.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_C),
            DIAGNOSIS_INFEASIBLE,
            (BLOCK_GROUP_KEY,),
        ),
        (
            asset(30000.0, 0.0, OccupancyClass='COM1'),
            DIAGNOSIS_INVALID,
            (BLOCK_GROUP_KEY,),
        ),
        (
            asset(30000.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=60010001001),
            DIAGNOSIS_INVALID,
            (BLOCK_GROUP_KEY,),
        ),
        (_point(0.0, 0.0, OccupancyClass='RES1'), DIAGNOSIS_INVALID, ()),
        (
            asset(30000.0, 0.0, OccupancyClass='RES1', BLOCKGROUP_GEOID=BG_C),
            DIAGNOSIS_INFEASIBLE,
            (),
        ),
    ],
)
def test_each_diagnosis_names_its_feature(
    record: tuple[list, dict],
    diagnosis: str,
    resolvable_by: tuple[str, ...],
) -> None:
    """Each verdict has its category and names what alone would admit the asset."""
    sampler = _sampler(_frozen(_donors())[0])
    verdicts = sampler.check(inventory({'x': record}))
    assert verdicts.loc['x', 'diagnosis'] == diagnosis
    assert verdicts.loc['x', 'resolvable_by'] == resolvable_by


def test_the_refusals_say_why() -> None:
    """The residential and the block-group refusals name their cause."""
    sampler = _sampler(_frozen(_donors())[0])
    verdicts = sampler.check(
        inventory(
            {
                'res': asset(0.0, 0.0, OccupancyClass='RES3A'),
                'alone': asset(
                    30000.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_C
                ),
            }
        )
    )
    assert 'ACSYearBuiltSampler' in verdicts.loc['res', 'detail']
    assert f'block group {BG_C}' in verdicts.loc['alone', 'detail']


def test_a_non_residential_year_makes_no_donor() -> None:
    """A target whose only nearby years are non-residential ones is refused."""
    records = {
        'c': asset(
            100.0, 0.0, OccupancyClass='COM1', YearBuilt=1960, BLOCKGROUP_GEOID=BG_C
        ),
        'r': asset(
            30000.0,
            0.0,
            OccupancyClass='RES1',
            YearBuilt=1950,
            BLOCKGROUP_GEOID=BG_A,
        ),
    }
    targets = inventory(
        {'t': asset(0.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_C)}
    )
    verdicts = _sampler(_frozen(records)[0]).check(targets)
    assert verdicts.loc['t', 'diagnosis'] == DIAGNOSIS_INFEASIBLE
    assert verdicts.loc['t', 'resolvable_by'] == (BLOCK_GROUP_KEY,)


def test_an_inventory_from_another_stage_raises_in_the_draw() -> None:
    """A donor passed back as a target passes check, then draw_details raises."""
    records = {
        'r0': asset(
            0.0, 0.0, OccupancyClass='RES1', YearBuilt=1950, BLOCKGROUP_GEOID=BG_A
        ),
    }
    sampler = _sampler(_frozen(records)[0])
    # check counts the target among the donors, and the draw excludes it
    targets = inventory(
        {'r0': asset(0.0, 0.0, OccupancyClass='COM1', BLOCKGROUP_GEOID=BG_A)}
    )
    assert sampler.check(targets).loc['r0', 'diagnosis'] == DIAGNOSIS_OK
    with pytest.raises(ValueError, match='passed check but drew no donor'):
        sampler.draw_details(targets)


def test_an_integral_float_year_is_copied_as_an_integer() -> None:
    """A year held as a whole float makes a donor, and the copy is an int."""
    records = {'r': asset(100.0, 0.0, OccupancyClass='RES1', YearBuilt=1980.0)}
    targets = inventory({'t': asset(0.0, 0.0, OccupancyClass='COM1')})
    years = _sampler(_frozen(records)[0]).draw(targets)
    assert years.loc['t'] == 1980
    assert years.dtype == 'int64'


@pytest.mark.parametrize('key', ['OccupancyClass', BLOCK_GROUP_KEY])
def test_one_value_per_world_is_unsupported(key: str) -> None:
    """A per-world list in the occupancy or block group names nothing."""
    features = {'OccupancyClass': 'COM1', BLOCK_GROUP_KEY: BG_A}
    features[key] = [features[key], features[key]]
    targets = inventory({'x': asset(0.0, 0.0, **features)})
    targets.n_pw = 2
    verdicts = _sampler(_frozen(_donors())[0]).check(targets)
    assert verdicts.loc['x', 'diagnosis'] == DIAGNOSIS_UNSUPPORTED
    assert verdicts.loc['x', 'resolvable_by'] == ()


def test_check_is_pure() -> None:
    """Checking twice gives the same verdicts and changes nothing."""
    sampler = _sampler(_frozen(_donors())[0])
    targets = inventory(_targets())
    donors = sampler.donors
    first = sampler.check(targets)
    pd.testing.assert_frame_equal(first, sampler.check(targets))
    pd.testing.assert_frame_equal(donors, sampler.donors)
    assert 'YearBuilt' not in targets.get_asset_features('t0')[1]


def test_calls_before_select_donors_raise() -> None:
    """Checking or drawing before select_donors is a programming error."""
    sampler = ResidentialNeighborYearSampler(seed=SEED)
    targets = inventory(_targets())
    for call in (sampler.check, sampler.draw, sampler.draw_details, sampler.infer):
        with pytest.raises(RuntimeError, match='select_donors'):
            call(targets)
    with pytest.raises(RuntimeError, match='select_donors'):
        _ = sampler.donors


def test_draw_refuses_assets_that_fail_check() -> None:
    """Draw and draw_details name the assets check would refuse."""
    sampler = _sampler(_frozen(_donors())[0])
    bad = inventory({'bad': asset(0.0, 0.0, OccupancyClass='RES1')})
    for call in (sampler.draw, sampler.draw_details):
        with pytest.raises(ValueError, match='bad'):
            call(bad)


def test_infer_writes_years_and_raises_with_a_preview() -> None:
    """Infer writes the copied year on a copy, or raises with a preview."""
    sampler = _sampler(_frozen(_donors())[0])
    records = _targets()
    records['t0'][1]['YearBuilt'] = 1900
    targets = inventory(records)
    output = sampler.infer(targets)
    for asset_id, year in sampler.draw(targets).items():
        assert output.get_asset_features(asset_id)[1]['YearBuilt'] == year
    assert targets.get_asset_features('t0')[1]['YearBuilt'] == 1900

    failing = inventory(
        {f'c{k}': asset(0.0, 10.0 * k, OccupancyClass='RES1') for k in range(12)}
    )
    with pytest.raises(ValueError, match='and 2 more'):
        sampler.infer(failing)


# ---------------------------------------------------------------------
# Construction and discovery


def test_the_arguments() -> None:
    """The k and cap default to the donor imputer's, and bad arguments raise."""
    sampler = ResidentialNeighborYearSampler()
    assert (sampler.k, sampler.cap_m) == (DEFAULT_DONOR_K, DEFAULT_DONOR_CAP_M)
    assert isinstance(sampler.seed, int)
    with pytest.raises(ValueError, match='k'):
        ResidentialNeighborYearSampler(k=0)
    with pytest.raises(ValueError, match='cap_m'):
        ResidentialNeighborYearSampler(cap_m=0)
    with pytest.raises(ValueError, match='seed'):
        ResidentialNeighborYearSampler(world=1)
    with pytest.raises(ValueError, match='at least 0'):
        ResidentialNeighborYearSampler(seed=SEED, world=-1)
    passed = ResidentialNeighborYearSampler(k=2, cap_m=500.0)
    assert (passed.k, passed.cap_m) == (2, 500.0)


def test_the_sampler_is_exported_and_discoverable() -> None:
    """The class is exported from the package and found by Importer."""
    assert (
        brails.imputers.ResidentialNeighborYearSampler
        is ResidentialNeighborYearSampler
    )
    assert (
        Importer().get_class('ResidentialNeighborYearSampler')
        is ResidentialNeighborYearSampler
    )
