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

"""Tests for the block-group ACS year sampler."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import brails.inferers
from brails.inferers.acs_year_built_sampler.acs_year_built_sampler import (
    ACS_YEAR_PURPOSE,
    B25034_BINS,
    B25127_CELLS,
    COARSE_YEARS,
    DEFAULT_SMOOTHING,
    NEWEST_YEAR,
    OLDEST_YEAR,
    SIZE_CLASSES,
    ACSYearBuiltSampler,
    size_class,
)
from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_INFEASIBLE,
    DIAGNOSIS_INVALID,
    DIAGNOSIS_OK,
    DIAGNOSIS_UNSUPPORTED,
)
from brails.inferers.year_built_sampler.year_built_sampler import YEAR_PURPOSE
from brails.utils import Importer

if TYPE_CHECKING:
    from collections.abc import Callable

    from brails.types.asset_inventory import AssetInventory

SEED = 20261102

# The year the golden test's asset draws under SEED.
GOLDEN_YEAR = 1975

# Two tracts of one county in every table, a third absent from B25127, a fourth
# absent from every table, and a block group of a second county.
TRACT_1 = '06001000100'
TRACT_2 = '06001000200'
TRACT_3 = '06001000300'
TRACT_4 = '06001000400'
BG_A = '060010001001'
BG_B = '060010001002'
BG_C = '060010002001'
BG_D = '060010002002'
BG_E = '060010003001'
BG_F = '060010004001'
BG_G = '060750001001'
BG_H = '060010001003'
BG_I = '060010001004'

# B25034 units per bin, oldest first. BG_A has no units in the 1940s.
B25034_BG = {
    BG_A: [10, 0, 30, 20, 40, 25, 15, 5, 8, 2],
    BG_B: [5, 10, 20, 20, 10, 10, 5, 5, 5, 0],
    BG_C: [0] * 10,
    BG_E: [4, 6, 12, 10, 8, 6, 4, 3, 2, 1],
    BG_G: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    BG_H: [10, 5, 20, 20, 10, 0, 0, 0, 0, 0],
    BG_I: [50, 0, 0, 0, 0, 0, 0, 0, 0, 50],
}
B25034_TRACT = {
    TRACT_1: [15, 10, 50, 40, 50, 35, 20, 10, 13, 2],
    TRACT_2: [3, 5, 10, 10, 10, 8, 6, 4, 3, 1],
    TRACT_3: [4, 6, 12, 10, 8, 6, 4, 3, 2, 1],
}

# B25127 units, one row per class of SIZE_CLASSES, one column per year bin of
# COARSE_YEARS.
B25127_TRACT_1 = [
    [30, 50, 80, 40, 20, 2],
    [10, 5, 20, 15, 5, 0],
    [2, 8, 25, 10, 6, 1],
    [0, 3, 6, 9, 4, 0],
    [0, 0, 2, 3, 5, 1],
    [0, 1, 4, 2, 0, 0],
]
B25127_TRACT_2 = [
    [12, 30, 40, 22, 9, 1],
    [4, 6, 9, 7, 3, 1],
    [1, 3, 8, 6, 2, 0],
    [0, 1, 2, 4, 3, 0],
    [0, 0, 1, 1, 2, 1],
    [1, 2, 5, 1, 0, 0],
]


def _b25034(rows: dict[str, list[float]]) -> pd.DataFrame:
    """
    Build a B25034 table in the Census API's form.

    Args:
        rows (dict[str, list[float]]):
            GEOID to units per bin, oldest first.

    Returns:
        pd.DataFrame:
            The table indexed by GEOID, with the total and an unrelated column beside
            the bin variables.
    """
    variables = [row[3] for row in B25034_BINS]
    frame = pd.DataFrame(
        [list(map(float, values)) for values in rows.values()],
        index=pd.Index(list(rows), dtype=object),
        columns=variables,
    )
    frame['B25034_001E'] = frame[variables].sum(axis=1)
    frame['NAME'] = 'somewhere'
    return frame


def _b25127(rows: dict[str, list[list[float]]]) -> pd.DataFrame:
    """
    Build a B25127 table in the Census API's form.

    Each cell's units are split between the owner and renter variables.

    Args:
        rows (dict[str, list[list[float]]]):
            Tract GEOID to units by class and year bin.

    Returns:
        pd.DataFrame:
            The table indexed by GEOID.
    """
    records = {}
    for tract, joint in rows.items():
        record = {}
        for year_bin, size, (owner, renter) in B25127_CELLS:
            value = joint[SIZE_CLASSES.index(size)][
                [label for label, _ in COARSE_YEARS].index(year_bin)
            ]
            record[owner] = float(value - value // 3)
            record[renter] = float(value // 3)
        records[tract] = record
    frame = pd.DataFrame.from_dict(records, orient='index')
    frame.index = pd.Index(frame.index, dtype=object)
    return frame


def _tables(
    b25034_bg: dict | None = None,
    b25034_tract: dict | None = None,
    b25127: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the three tables, defaulting to the module's fixture rows.

    Args:
        b25034_bg (dict | None):
            Block-group rows, or None for the fixture's.
        b25034_tract (dict | None):
            Tract rows, or None for the fixture's.
        b25127 (dict | None):
            B25127 rows, or None for the fixture's two tracts.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            The block-group B25034, tract B25034 and tract B25127 tables.
    """
    return (
        _b25034(B25034_BG if b25034_bg is None else b25034_bg),
        _b25034(B25034_TRACT if b25034_tract is None else b25034_tract),
        _b25127(
            {TRACT_1: B25127_TRACT_1, TRACT_2: B25127_TRACT_2}
            if b25127 is None
            else b25127
        ),
    )


def _sampler(
    tables: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    **kwargs: object,
) -> ACSYearBuiltSampler:
    """
    Build a sampler on the fixture tables with the module's seed.

    Args:
        tables (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None):
            The three tables, or None for the fixture's.
        **kwargs (object):
            Further keyword arguments for the sampler.

    Returns:
        ACSYearBuiltSampler:
            The unfitted sampler.
    """
    kwargs.setdefault('seed', SEED)
    return ACSYearBuiltSampler(*(tables or _tables()), **kwargs)


def _asset(occupancy: object, units: object, block_group: object) -> dict:
    """
    Build the driver features of one asset.

    Args:
        occupancy (object):
            The occupancy class.
        units (object):
            The unit count.
        block_group (object):
            The block-group GEOID.

    Returns:
        dict:
            The features.
    """
    return {
        'OccupancyClass': occupancy,
        'NumberOfUnits': units,
        'BLOCKGROUP_GEOID': block_group,
    }


def _neighborhood() -> dict[str, dict]:
    """
    Build residential assets in two block groups of one tract.

    Returns:
        dict[str, dict]:
            Asset id to features. BG_A holds one-unit, two-to-four and
            five-to-nineteen assets with fractional counts; BG_B holds one-unit
            assets and a manufactured home.
    """
    assets = {}
    for k in range(12):
        assets[f'a1-{k}'] = _asset('RES1', 1, BG_A)
    for k, units in enumerate((2, 3.5, 4, 1.1, 2.3)):
        assets[f'a3-{k}'] = _asset('RES3A' if units <= 2 else 'RES3B', units, BG_A)
    for k, units in enumerate((7, 14.5, 12)):
        assets[f'a5-{k}'] = _asset('RES3C', units, BG_A)
    for k in range(6):
        assets[f'b1-{k}'] = _asset('RES1', 1, BG_B)
    assets['b-mh'] = _asset('RES2', 1, BG_B)
    return assets


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize rows to one, as the hand computation does.

    Args:
        matrix (np.ndarray):
            Non-negative rows, each with some mass.

    Returns:
        np.ndarray:
            The rows divided by their sums.
    """
    return matrix / matrix.sum(axis=1, keepdims=True)


def _hand_vectors(class_units: np.ndarray) -> np.ndarray:
    """
    Compute BG_A's raked rows independently of the sampler.

    The computation uses the default smoothing and rakes in the linear domain.

    Args:
        class_units (np.ndarray):
            The fitted units per class of SIZE_CLASSES.

    Returns:
        np.ndarray:
            Six rows over the ten bins.
    """
    t1 = np.array(B25127_TRACT_1, dtype=float)
    t2 = np.array(B25127_TRACT_2, dtype=float)
    pooled = _normalize(t1 + t2)
    conditional = _normalize(t1 + DEFAULT_SMOOTHING * pooled)
    shares = np.array(B25034_BG[BG_A], dtype=float)
    shares = shares / shares.sum()
    coarse_of = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
    within = np.zeros(10)
    for j in range(6):
        members = coarse_of == j
        within[members] = shares[members] / shares[members].sum()
    seed = conditional[:, coarse_of] * within
    seed[:, shares == 0] = 0.0
    target = shares * class_units.sum()
    multiplier = np.ones(10)
    for _ in range(100000):
        rows = _normalize(seed * multiplier)
        column = class_units @ rows
        if np.abs(column - target).max() / class_units.sum() < 1e-14:
            break
        live = target > 0
        multiplier[live] *= target[live] / column[live]
        multiplier /= multiplier[live].max()
    return rows


# ---------------------------------------------------------------------
# Tables and classes


def test_b25127_cells_cover_each_class_and_bin_once() -> None:
    """Every class and year bin has one cell, renter id owner id plus 43."""
    pairs = {(year_bin, size) for year_bin, size, _ in B25127_CELLS}
    assert len(pairs) == len(B25127_CELLS) == 36
    variables = [v for _, _, pair in B25127_CELLS for v in pair]
    assert len(set(variables)) == 72
    for _, _, (owner, renter) in B25127_CELLS:
        assert int(renter[7:10]) == int(owner[7:10]) + 43


def test_bins_tile_the_year_range() -> None:
    """The B25034 bins run without gaps and fill the B25127 bins in order."""
    for (_, _, last, _), (_, first, _, _) in zip(B25034_BINS[:-1], B25034_BINS[1:]):
        assert first == last + 1
    members = [m for _, group in COARSE_YEARS for m in group]
    assert members == [row[0] for row in B25034_BINS]


@pytest.mark.parametrize(
    ('occupancy', 'units', 'expected'),
    [
        ('RES1', 1, '1 unit'),
        ('RES3A', 1.5, '2 to 4'),
        ('RES3B', 3.5, '2 to 4'),
        ('RES3B', 4, '2 to 4'),
        ('RES3C', 4.5, '5 to 19'),
        ('RES3D', 19, '5 to 19'),
        ('RES3E', 49, '20 to 49'),
        ('RES3F', 50, '50 or more'),
        ('RES1', 7, '5 to 19'),
        ('RES2', 30, 'Manufactured home'),
    ],
)
def test_size_class_upper_bounds_are_inclusive(
    occupancy: str, units: float, expected: str
) -> None:
    """Counts at a bound fall in the lower class, and RES2 is its own."""
    assert SIZE_CLASSES[size_class(occupancy, units)] == expected


def test_purpose_differs_from_the_tract_median_sampler() -> None:
    """The two year samplers draw under different labels."""
    assert ACS_YEAR_PURPOSE != YEAR_PURPOSE


# ---------------------------------------------------------------------
# Fit


def test_raked_vector_meets_targets_and_matches_hand_computation(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Rows sum to one, units meet B25034, and the hand result agrees."""
    assets = {f'one-{k}': _asset('RES1', 1, BG_A) for k in range(8)}
    assets.update({f'mid-{k}': _asset('RES3C', 6, BG_A) for k in range(3)})
    sampler = _sampler().fit(build_inventory(assets))
    rows = sampler.vectors.loc[BG_A].to_numpy()
    class_units = np.array([8.0, 0.0, 18.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(rows.sum(axis=1), 1.0, atol=1e-12)
    shares = np.array(B25034_BG[BG_A], dtype=float) / sum(B25034_BG[BG_A])
    np.testing.assert_allclose(
        class_units @ rows, shares * class_units.sum(), atol=1e-10
    )
    assert (rows[:, 1] == 0).all()
    np.testing.assert_allclose(rows, _hand_vectors(class_units), atol=1e-9)
    report = sampler.fit_report.loc[BG_A]
    assert report['shares_origin'] == 'block_group'
    assert report['conditional_origin'] == 'tract'
    assert report['fitted_assets'] == 11
    assert report['fitted_units'] == 26.0
    assert bool(report['converged'])


def test_tract_shares_stand_in_for_a_block_group_without_units(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A zero-unit or absent block group takes its tract's shares."""
    inventory = build_inventory(
        {'c': _asset('RES1', 1, BG_C), 'd': _asset('RES1', 1, BG_D)}
    )
    sampler = _sampler().fit(inventory)
    tract = np.array(B25034_TRACT[TRACT_2], dtype=float)
    for block_group in (BG_C, BG_D):
        assert sampler.fit_report.loc[block_group, 'shares_origin'] == 'tract'
        row = sampler.vectors.loc[(block_group, '1 unit')].to_numpy()
        np.testing.assert_allclose(row, tract / tract.sum(), atol=1e-10)


def test_county_pool_stands_in_for_a_tract_missing_from_b25127(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A tract absent from B25127 takes its county's pooled conditional."""
    inventory = build_inventory(
        {'e1': _asset('RES1', 1, BG_E), 'e2': _asset('RES3C', 8, BG_E)}
    )
    sampler = _sampler().fit(inventory)
    assert sampler.fit_report.loc[BG_E, 'conditional_origin'] == 'county_pool'
    verdicts = sampler.check(inventory)
    assert (verdicts['diagnosis'] == DIAGNOSIS_OK).all()


def test_block_groups_without_a_table_are_not_fitted(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Missing B25034 units or a missing county leave the asset refused."""
    inventory = build_inventory(
        {
            'f': _asset('RES1', 1, BG_F),
            'g': _asset('RES1', 1, BG_G),
            'a': _asset('RES1', 1, BG_A),
        }
    )
    sampler = _sampler().fit(inventory)
    verdicts = sampler.check(inventory)
    assert verdicts.loc['f', 'diagnosis'] == DIAGNOSIS_INFEASIBLE
    assert 'B25034' in verdicts.loc['f', 'detail']
    assert verdicts.loc['g', 'diagnosis'] == DIAGNOSIS_INFEASIBLE
    assert 'B25127' in verdicts.loc['g', 'detail']
    assert verdicts.loc['f', 'resolvable_by'] == ('BLOCKGROUP_GEOID',)
    assert verdicts.loc['g', 'resolvable_by'] == ('BLOCKGROUP_GEOID',)
    assert verdicts.loc['a', 'diagnosis'] == DIAGNOSIS_OK
    report = sampler.fit_report
    assert pd.isna(report.loc[BG_F, 'shares_origin'])
    assert pd.isna(report.loc[BG_G, 'conditional_origin'])
    assert not report.loc[BG_F, 'converged']


def test_a_negative_value_reads_as_zero_and_is_counted(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A Census placeholder below zero acts as zero in the fit."""
    inventory = build_inventory({'a': _asset('RES1', 1, BG_A)})
    flagged = dict(B25034_BG)
    flagged[BG_A] = list(B25034_BG[BG_A])
    flagged[BG_A][1] = -666666666
    marked = _sampler(_tables(b25034_bg=flagged)).fit(inventory)
    plain = _sampler().fit(inventory)
    pd.testing.assert_frame_equal(marked.vectors, plain.vectors, check_exact=True)
    assert marked.fit_report.loc[BG_A, 'negative_values'] == 1
    assert plain.fit_report.loc[BG_A, 'negative_values'] == 0


def test_fit_leaves_the_inventory_unchanged(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The fit reads the inventory and writes nothing into it."""
    inventory = build_inventory(_neighborhood())
    before = copy.deepcopy(inventory.inventory)
    _sampler().fit(inventory)
    assert {k: v.features for k, v in inventory.inventory.items()} == {
        k: v.features for k, v in before.items()
    }


def test_fit_refuses_one_value_per_world(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A per-world list in a driver is the caller's error in fit."""
    inventory = build_inventory({'a': _asset(['RES1', 'RES3A'], 1, BG_A)}, n_pw=2)
    with pytest.raises(ValueError, match='one value per world'):
        _sampler().fit(inventory)


# ---------------------------------------------------------------------
# Locality of the tables


def test_another_county_in_b25127_leaves_draws_unchanged(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Tracts of a second county do not enter the first county's pool."""
    inventory = build_inventory(_neighborhood())
    extended = _tables(
        b25127={
            TRACT_1: B25127_TRACT_1,
            TRACT_2: B25127_TRACT_2,
            '06013000100': B25127_TRACT_2,
            '06013000200': B25127_TRACT_1,
        }
    )
    base = _sampler().fit(inventory)
    wider = _sampler(extended).fit(inventory)
    pd.testing.assert_series_equal(base.draw(inventory), wider.draw(inventory))
    pd.testing.assert_frame_equal(base.vectors, wider.vectors, check_exact=True)


def test_another_tract_of_the_county_changes_the_pool(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The pooled conditional reads every tract of the county."""
    inventory = build_inventory(_neighborhood())
    changed = [list(row) for row in B25127_TRACT_2]
    changed[0][0] += 200
    base = _sampler().fit(inventory)
    other = _sampler(
        _tables(b25127={TRACT_1: B25127_TRACT_1, TRACT_2: changed})
    ).fit(inventory)
    assert not np.allclose(
        base.vectors.loc[BG_A].to_numpy(), other.vectors.loc[BG_A].to_numpy()
    )


# ---------------------------------------------------------------------
# Draw


def test_golden_year(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """One asset, fixed seed, world and tables, gives one known year."""
    inventory = build_inventory({'golden': _asset('RES1', 1, BG_A)})
    years = _sampler().fit(inventory).draw(inventory)
    assert years.loc['golden'] == GOLDEN_YEAR


def test_seed_changes_draws(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Another seed gives other years."""
    inventory = build_inventory(_neighborhood())
    first = _sampler().fit(inventory).draw(inventory)
    second = _sampler(seed=SEED + 1).fit(inventory).draw(inventory)
    assert not first.equals(second)


def test_iterations_draw_independently(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Another iteration gives other years for the same assets."""
    inventory = build_inventory(_neighborhood())
    sampler = _sampler().fit(inventory)
    assert not sampler.draw(inventory).equals(sampler.draw(inventory, iteration=1))
    pd.testing.assert_series_equal(
        sampler.draw(inventory, iteration=3),
        sampler.draw(inventory, iteration=3),
    )


def test_reordering_changes_nothing(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A permuted inventory gives identical sums, vectors and years."""
    assets = _neighborhood()
    # Summed left to right these give 1e18; summed right to left, 1e18 + 128.
    assets['big'] = _asset('RES3F', 1e18, BG_B)
    assets['tower-1'] = _asset('RES3F', 60, BG_B)
    assets['tower-2'] = _asset('RES3F', 60, BG_B)
    ids = list(assets)
    forward = build_inventory(assets)
    backward = build_inventory({k: assets[k] for k in reversed(ids)})
    first = _sampler().fit(forward)
    second = _sampler().fit(backward)
    pd.testing.assert_frame_equal(
        first.fit_report, second.fit_report, check_exact=True
    )
    pd.testing.assert_frame_equal(first.vectors, second.vectors, check_exact=True)
    pd.testing.assert_series_equal(
        first.draw(forward).sort_index(), second.draw(backward).sort_index()
    )


def test_draws_are_local_to_the_block_group(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """
    Assets elsewhere leave draws unchanged; a block group's own change its rows.

    Changing another block group's assets leaves the draws unchanged, and changing
    the block group's own assets changes its fitted rows.
    """
    assets = _neighborhood()
    in_a = [k for k, v in assets.items() if v['BLOCKGROUP_GEOID'] == BG_A]
    base = _sampler().fit(build_inventory(assets))
    base_years = base.draw(build_inventory({k: assets[k] for k in in_a}))

    changed = dict(assets)
    changed['b1-0'] = _asset('RES3E', 30, BG_B)
    changed['b-new'] = _asset('RES3F', 80, BG_B)
    del changed['b1-1']
    other = _sampler().fit(build_inventory(changed))
    pd.testing.assert_series_equal(
        base_years, other.draw(build_inventory({k: assets[k] for k in in_a}))
    )
    pd.testing.assert_frame_equal(
        base.vectors.loc[BG_A], other.vectors.loc[BG_A], check_exact=True
    )

    same = dict(assets)
    same['a5-0'] = _asset('RES3C', 18, BG_A)
    moved = _sampler().fit(build_inventory(same))
    assert not np.allclose(
        base.vectors.loc[BG_A].to_numpy(), moved.vectors.loc[BG_A].to_numpy()
    )


def test_worlds_draw_differently_with_the_fitted_frequencies(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Twenty worlds give different years and the vector's frequencies."""
    inventory = build_inventory(
        {f'w{k}': _asset('RES1', 1, BG_B) for k in range(2000)}
    )
    counts = np.zeros(10)
    previous = None
    vector = None
    first_years = np.array([row[1] for row in B25034_BINS])
    for world in range(20):
        sampler = _sampler(world=world).fit(inventory)
        vector = sampler.vectors.loc[(BG_B, '1 unit')].to_numpy()
        years = sampler.draw(inventory)
        if previous is not None:
            assert not years.equals(previous)
        previous = years
        bins = np.searchsorted(first_years, years.to_numpy(), side='right') - 1
        counts += np.bincount(bins, minlength=10)
    draws = counts.sum()
    bound = 4.0 * np.sqrt(vector * (1 - vector) / draws) + 1e-12
    assert (np.abs(counts / draws - vector) <= bound).all()


def test_a_non_default_world_needs_a_seed() -> None:
    """Per-world samplers of one run must share an explicit seed."""
    with pytest.raises(ValueError, match='seed'):
        ACSYearBuiltSampler(*_tables(), world=1)
    sampler = ACSYearBuiltSampler(*_tables())
    assert isinstance(sampler.seed, int)


def test_one_asset_calls_match_the_full_inventory(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """
    One-asset calls see the distribution of the fitted inventory.

    A one-asset check under a placeholder id matches the check under the real id, and
    a one-asset draw matches the full-inventory draw.
    """
    assets = _neighborhood()
    full = build_inventory(assets)
    sampler = _sampler().fit(full)
    all_years = sampler.draw(full)
    for asset_id in ('a1-0', 'a5-1', 'b-mh'):
        screened = build_inventory({'screened_asset': assets[asset_id]})
        real = build_inventory({asset_id: assets[asset_id]})
        pd.testing.assert_frame_equal(
            sampler.check(screened).reset_index(drop=True),
            sampler.check(real).reset_index(drop=True),
        )
        assert sampler.draw(real).loc[asset_id] == all_years.loc[asset_id]


def test_draws_stay_in_bins_with_units(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A block group with no units after the 1970s draws no later year."""
    inventory = build_inventory(
        {f'h{k}': _asset('RES1', 1, BG_H) for k in range(2000)}
    )
    years = _sampler().fit(inventory).draw(inventory)
    assert years.max() <= B25034_BINS[4][2]
    assert years.max() >= B25034_BINS[4][1]


def test_years_spread_over_the_whole_of_each_open_bin(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The open end bins are drawn from their first to their last year."""
    inventory = build_inventory(
        {f'i{k}': _asset('RES1', 1, BG_I) for k in range(2000)}
    )
    years = _sampler().fit(inventory).draw(inventory)
    oldest = years[years <= B25034_BINS[0][2]]
    newest = years[years >= B25034_BINS[-1][1]]
    assert len(oldest) + len(newest) == len(years)
    assert oldest.min() == OLDEST_YEAR
    assert oldest.max() == B25034_BINS[0][2]
    assert set(newest) == set(range(B25034_BINS[-1][1], NEWEST_YEAR + 1))


def test_inner_decades_are_drawn_in_every_year(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A block group with units in the 1940s and 1970s draws all their years."""
    units = [0, 50, 0, 0, 50, 0, 0, 0, 0, 0]
    tables = _tables(
        b25034_bg={BG_A: units},
        b25034_tract={TRACT_1: units},
        b25127={TRACT_1: [[3] * 6] * 6},
    )
    inventory = build_inventory(
        {f'j{k}': _asset('RES1', 1, BG_A) for k in range(2000)}
    )
    years = _sampler(tables).fit(inventory).draw(inventory)
    assert set(years) == set(range(1940, 1950)) | set(range(1970, 1980))


# ---------------------------------------------------------------------
# Check and infer


@pytest.mark.parametrize(
    ('features', 'diagnosis', 'resolvable_by'),
    [
        (_asset(None, 1, BG_A), 'invalid', 'OccupancyClass'),
        (_asset(3, 1, BG_A), 'invalid', 'OccupancyClass'),
        (_asset('COM1', 1, BG_A), 'infeasible', 'OccupancyClass'),
        (_asset('COM1', 1, BG_F), 'infeasible', None),
        (_asset(None, None, BG_A), 'invalid', None),
        (_asset(None, 1, BG_F), 'invalid', None),
        (_asset('RES1', None, BG_A), 'invalid', 'NumberOfUnits'),
        (_asset('RES1', 'many', BG_A), 'invalid', 'NumberOfUnits'),
        (_asset('RES1', float('nan'), BG_A), 'invalid', 'NumberOfUnits'),
        (_asset('RES1', units=True, block_group=BG_A), 'invalid', 'NumberOfUnits'),
        (_asset('RES1', 0.5, BG_A), 'invalid', 'NumberOfUnits'),
        (_asset('RES1', 1, None), 'invalid', 'BLOCKGROUP_GEOID'),
        (_asset('RES1', 1, 60010001001), 'invalid', 'BLOCKGROUP_GEOID'),
        (_asset('RES1', 1, '06001000100'), 'invalid', 'BLOCKGROUP_GEOID'),
        (_asset('RES1', 1, BG_F), 'infeasible', 'BLOCKGROUP_GEOID'),
        (_asset('RES1', 1, BG_B), 'infeasible', 'BLOCKGROUP_GEOID'),
        (_asset('RES1', 1, BG_A), 'OK', None),
    ],
)
def test_each_diagnosis_names_its_feature(
    build_inventory: Callable[..., AssetInventory],
    features: dict,
    diagnosis: str,
    resolvable_by: str | None,
) -> None:
    """
    Each refusal has its category and names what alone would admit it.

    A driver that fails alone is named, whether it is unusable, a class outside the
    sampler's coverage or an unfitted block group. Two failing drivers name nothing,
    and a usable asset is OK and names none.
    """
    sampler = _sampler().fit(build_inventory({'a': _asset('RES1', 1, BG_A)}))
    verdicts = sampler.check(build_inventory({'x': features}))
    assert verdicts.loc['x', 'diagnosis'] == diagnosis
    expected = () if resolvable_by is None else (resolvable_by,)
    assert verdicts.loc['x', 'resolvable_by'] == expected


def test_the_residential_refusal_says_so(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A non-residential class is outside the classes the sampler covers."""
    sampler = _sampler().fit(build_inventory({'a': _asset('RES1', 1, BG_A)}))
    verdicts = sampler.check(build_inventory({'x': _asset('COM1', 1, BG_A)}))
    assert 'outside the residential classes' in verdicts.loc['x', 'detail']


def test_a_block_group_outside_the_fit_says_so(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A block group both tables cover but the fit never saw is refused."""
    sampler = _sampler().fit(build_inventory({'a': _asset('RES1', 1, BG_A)}))
    verdicts = sampler.check(build_inventory({'x': _asset('RES1', 1, BG_B)}))
    assert 'had no asset in the fit' in verdicts.loc['x', 'detail']


def test_one_value_per_world_is_unsupported(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """A per-world list in check is refused and names nothing."""
    sampler = _sampler().fit(build_inventory({'a': _asset('RES1', 1, BG_A)}))
    inventory = build_inventory({'x': _asset('RES1', [1, 2], BG_A)}, n_pw=2)
    verdicts = sampler.check(inventory)
    assert verdicts.loc['x', 'diagnosis'] == DIAGNOSIS_UNSUPPORTED
    assert verdicts.loc['x', 'resolvable_by'] == ()


def test_configured_keys_are_read_and_named(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """The keys given to the sampler are the ones read and diagnosed."""
    sampler = _sampler(
        occupancy_key='occ',
        units_key='units',
        block_group_key='bg',
        year_built_key='year',
    )
    inventory = build_inventory(
        {'a': {'occ': 'RES1', 'units': 1, 'bg': BG_A}, 'b': {'occ': 'RES1'}}
    )
    sampler.fit(inventory)
    verdicts = sampler.check(inventory)
    assert verdicts.loc['b', 'resolvable_by'] == ()
    lone = sampler.check(build_inventory({'c': {'occ': 'RES1', 'bg': BG_A}}))
    assert lone.loc['c', 'resolvable_by'] == ('units',)
    output = sampler.infer(
        build_inventory({'a': {'occ': 'RES1', 'units': 1, 'bg': BG_A}})
    )
    assert 'year' in output.get_asset_features('a')[1]


def test_check_is_pure(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Checking twice gives the same verdicts and changes nothing."""
    inventory = build_inventory(_neighborhood())
    sampler = _sampler().fit(inventory)
    vectors = sampler.vectors
    first = sampler.check(inventory)
    pd.testing.assert_frame_equal(first, sampler.check(inventory))
    pd.testing.assert_frame_equal(vectors, sampler.vectors)
    assert 'YearBuilt' not in inventory.get_asset_features('a1-0')[1]


def test_calls_before_fit_raise(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Checking or drawing an unfitted sampler is a programming error."""
    inventory = build_inventory({'a': _asset('RES1', 1, BG_A)})
    sampler = _sampler()
    for call in (sampler.check, sampler.draw, sampler.infer):
        with pytest.raises(RuntimeError, match='fit'):
            call(inventory)
    with pytest.raises(RuntimeError, match='fit'):
        _ = sampler.vectors


def test_draw_refuses_assets_that_fail_check(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Draw names the assets check would refuse."""
    sampler = _sampler().fit(build_inventory({'a': _asset('RES1', 1, BG_A)}))
    with pytest.raises(ValueError, match='bad'):
        sampler.draw(build_inventory({'bad': _asset('COM1', 1, BG_A)}))


def test_infer_writes_years_and_raises_with_a_preview(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """Infer writes the drawn year on a copy, or raises with a preview."""
    inventory = build_inventory(
        {**_neighborhood(), 'a1-0': {**_asset('RES1', 1, BG_A), 'YearBuilt': 1900}}
    )
    sampler = _sampler().fit(inventory)
    output = sampler.infer(inventory)
    years = sampler.draw(inventory)
    for asset_id, year in years.items():
        assert output.get_asset_features(asset_id)[1]['YearBuilt'] == year
    assert inventory.get_asset_features('a1-0')[1]['YearBuilt'] == 1900

    failing = build_inventory({f'c{k}': _asset('COM1', 1, BG_A) for k in range(12)})
    with pytest.raises(ValueError, match='and 2 more'):
        sampler.infer(failing)


# ---------------------------------------------------------------------
# Construction


def test_a_missing_column_raises() -> None:
    """Each table must carry every variable the sampler reads."""
    bg, tract, b127 = _tables()
    with pytest.raises(ValueError, match='b25127_tracts'):
        ACSYearBuiltSampler(bg, tract, b127.drop(columns='B25127_039E'))
    with pytest.raises(ValueError, match='b25034_block_groups'):
        ACSYearBuiltSampler(bg.drop(columns='B25034_002E'), tract, b127)


def test_a_non_finite_value_raises() -> None:
    """A missing estimate is not read as zero."""
    bg, tract, b127 = _tables()
    tract.loc[TRACT_1, 'B25034_005E'] = np.nan
    with pytest.raises(ValueError, match='not finite'):
        ACSYearBuiltSampler(bg, tract, b127)


@pytest.mark.parametrize('bad', [6001000100, '6001000100', '0600100010A'])
def test_an_index_that_is_not_a_geoid_string_raises(bad: object) -> None:
    """An index entry that is not a GEOID digit string is refused."""
    bg, tract, b127 = _tables()
    b127 = b127.rename(index={TRACT_2: bad})
    with pytest.raises(ValueError, match='GEOID'):
        ACSYearBuiltSampler(bg, tract, b127)


def test_negative_smoothing_raises() -> None:
    """Smoothing is a non-negative count of pseudo-units."""
    with pytest.raises(ValueError, match='smoothing'):
        ACSYearBuiltSampler(*_tables(), smoothing=-1.0)


# ---------------------------------------------------------------------
# Discovery


def test_the_sampler_is_exported_and_discoverable() -> None:
    """The class is exported from the package and found by Importer."""
    assert brails.inferers.ACSYearBuiltSampler is ACSYearBuiltSampler
    assert Importer().get_class('ACSYearBuiltSampler') is ACSYearBuiltSampler
