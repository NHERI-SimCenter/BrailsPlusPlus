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

"""Tests for the completed West Coast structure type tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from brails.inferers.hazus_inferer_earthquake import hazus_table_completion
from brails.inferers.hazus_inferer_earthquake.hazus_eq_structure_type_inferer import (
    HazusEQStructureTypeInferer,
)
from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    WEST_COAST,
    get_hazus_occ_type_mapping,
)
from brails.inferers.hazus_inferer_earthquake.hazus_table_completion import (
    HALF_MASS,
    IMPLAUSIBLE_PAIRS,
    ROUTE_ADJACENT_ERA,
    ROUTE_BAND_BELOW,
    ROUTE_FALLBACK,
    CellFill,
    get_completed_west_coast_weights,
    get_west_coast_fill_routes,
    is_implausible,
)
from brails.inferers.inference_diagnosis import DIAGNOSIS_OK

if TYPE_CHECKING:
    from collections.abc import Callable

    from brails.types.asset_inventory import AssetInventory

TYPE_LISTS = get_hazus_occ_type_mapping()[0][WEST_COAST]
PRINTED = get_hazus_occ_type_mapping()[1][WEST_COAST]
ROUTES = get_west_coast_fill_routes()
COMPLETED = get_completed_west_coast_weights()
BANDS = ('low_rise', 'mid_rise', 'high_rise')
ERAS = ('pre_1950', '1950_to_1970', 'post_1970')

# Four of the six design levels the inferer reads. Very High-Code and
# Severe-Code bar URM, C3 and S5 as Moderate-Code does, so they add no case.
HAZUS_DESIGN_LEVELS = ('Pre-Code', 'Low-Code', 'Moderate-Code', 'High-Code')

# A story count inside each story band and a year inside each era.
STORIES_IN_BAND = {'low_rise': 2, 'mid_rise': 5, 'high_rise': 10}
YEAR_IN_ERA = {'pre_1950': 1930, '1950_to_1970': 1960, 'post_1970': 1990}


def _cells() -> list[tuple[str, str, str]]:
    """
    List every West Coast cell of occupancy class, story band and era.

    Returns:
        list[tuple[str, str, str]]:
            ``(occupancy, band, era)`` for each non-RES1 cell.
    """
    return [
        (occupancy, band, era)
        for occupancy in PRINTED
        if occupancy != 'RES1'
        for band in BANDS
        for era in ERAS
    ]


def _split_routes() -> tuple[dict, dict]:
    """
    Split the routes into those of plausible and implausible cells.

    Returns:
        tuple[dict, dict]:
            The routes of the cells off the implausible list, then those
            of the cells on it.
    """
    plausible = {
        cell: fill
        for cell, fill in ROUTES.items()
        if cell[:2] not in IMPLAUSIBLE_PAIRS
    }
    implausible = {
        cell: fill for cell, fill in ROUTES.items() if cell not in plausible
    }
    return plausible, implausible


def _kept(cell: tuple[str, str, str]) -> dict[str, float]:
    """
    Return the kept mass each of the first two routes offers a cell.

    Args:
        cell (tuple[str, str, str]):
            The empty cell.

    Returns:
        dict[str, float]:
            Route to the kept mass of its source, rounded to two places.
    """
    return {
        route: round(kept_mass, 2)
        for route, _, kept_mass in hazus_table_completion._route_candidates(
            cell, TYPE_LISTS, PRINTED
        )
    }


# ---------------------------------------------------------------------
# the implausible list


def test_the_list_holds_twelve_pairs() -> None:
    """Two classes are implausible above low-rise and eight at high-rise."""
    assert len(IMPLAUSIBLE_PAIRS) == 12
    assert ('RES2', 'mid_rise') in IMPLAUSIBLE_PAIRS
    assert ('AGR1', 'high_rise') in IMPLAUSIBLE_PAIRS
    assert ('REL1', 'high_rise') in IMPLAUSIBLE_PAIRS
    assert ('REL1', 'mid_rise') not in IMPLAUSIBLE_PAIRS


def test_a_pair_is_judged_by_its_story_band() -> None:
    """Four and eight stories enter mid-rise and high-rise, three and seven not."""
    assert is_implausible('RES2', 4)
    assert is_implausible('EDU1', 8)
    assert not is_implausible('RES2', 3)
    assert not is_implausible('EDU1', 7)
    assert not is_implausible('COM7', 9)


def test_thirty_six_cells_are_marked_implausible() -> None:
    """The 35 empty cells and printed AGR1 mid-rise before 1950 are marked."""
    marked = [cell for cell in _cells() if cell[:2] in IMPLAUSIBLE_PAIRS]
    printed = [cell for cell in marked if PRINTED[cell[0]][cell[1]][cell[2]]]
    assert len(marked) == 36
    assert printed == [('AGR1', 'mid_rise', 'pre_1950')]


# ---------------------------------------------------------------------
# the routes


def test_every_empty_cell_has_a_route() -> None:
    """Each of the 71 empty cells has a route, 36 plausible and 35 not."""
    plausible, implausible = _split_routes()
    empty = {cell for cell in _cells() if not PRINTED[cell[0]][cell[1]][cell[2]]}
    assert set(ROUTES) == empty
    assert len(plausible) == 36
    assert len(implausible) == 35
    assert all(isinstance(fill, CellFill) for fill in ROUTES.values())


def test_the_route_counts_of_plausible_and_implausible_cells() -> None:
    """
    The plausible cells split 4, 25 and 7 over the three routes.

    The routes are taken in the order adjacent era, band below and
    fallback. The implausible cells split 0, 11 and 24.
    """
    plausible, implausible = _split_routes()
    for routes, expected in (
        (plausible, (4, 25, 7)),
        (implausible, (0, 11, 24)),
    ):
        counts = tuple(
            sum(fill.route == route for fill in routes.values())
            for route in (ROUTE_ADJACENT_ERA, ROUTE_BAND_BELOW, ROUTE_FALLBACK)
        )
        assert counts == expected


def test_the_seven_plausible_fallbacks_have_no_passing_source() -> None:
    """For each, every route's source keeps less than ``HALF_MASS``."""
    plausible, _ = _split_routes()
    fallbacks = {
        cell: _kept(cell)
        for cell, fill in plausible.items()
        if fill.route == ROUTE_FALLBACK
    }
    assert fallbacks == {
        ('COM1', 'high_rise', 'pre_1950'): {ROUTE_BAND_BELOW: 0.39},
        ('COM2', 'high_rise', 'pre_1950'): {ROUTE_BAND_BELOW: 0.35},
        ('COM3', 'high_rise', 'pre_1950'): {ROUTE_BAND_BELOW: 0.40},
        ('COM3', 'high_rise', '1950_to_1970'): {ROUTE_BAND_BELOW: 0.46},
        ('EDU1', 'mid_rise', 'pre_1950'): {ROUTE_BAND_BELOW: 0.39},
        ('IND6', 'mid_rise', '1950_to_1970'): {
            ROUTE_ADJACENT_ERA: 0.20,
            ROUTE_BAND_BELOW: 0.31,
        },
        ('IND6', 'mid_rise', 'post_1970'): {ROUTE_BAND_BELOW: 0.38},
    }


def test_edu1_mid_rise_and_rel1_high_rise_kept_masses() -> None:
    """
    EDU1 mid-rise and REL1 high-rise pass the band-below route from 1950.

    Before 1950 both keep less than ``HALF_MASS`` and fall back.
    """
    assert _kept(('EDU1', 'mid_rise', '1950_to_1970')) == {ROUTE_BAND_BELOW: 0.66}
    assert _kept(('EDU1', 'mid_rise', 'post_1970')) == {ROUTE_BAND_BELOW: 0.68}
    assert _kept(('REL1', 'high_rise', 'pre_1950')) == {ROUTE_BAND_BELOW: 0.10}
    assert _kept(('REL1', 'high_rise', '1950_to_1970')) == {ROUTE_BAND_BELOW: 0.90}
    assert _kept(('REL1', 'high_rise', 'post_1970')) == {ROUTE_BAND_BELOW: 1.0}
    assert ROUTES[('EDU1', 'mid_rise', 'post_1970')].route == ROUTE_BAND_BELOW
    assert ROUTES[('REL1', 'high_rise', 'pre_1950')].route == ROUTE_FALLBACK


def test_the_implausible_fallbacks_split_nine_and_fifteen() -> None:
    """Of the 24, nine have a source below ``HALF_MASS`` and fifteen none."""
    _, implausible = _split_routes()
    fallbacks = [
        cell for cell, fill in implausible.items() if fill.route == ROUTE_FALLBACK
    ]
    with_source = [cell for cell in fallbacks if _kept(cell)]
    assert len(with_source) == 9
    assert len(fallbacks) - len(with_source) == 15
    assert not _kept(('AGR1', 'high_rise', 'pre_1950'))


def test_the_printed_implausible_agr1_row_is_never_a_source() -> None:
    """
    Printed AGR1 mid-rise before 1950 is on the implausible list.

    No route uses it and no fallback combines it.
    """
    excluded = ('AGR1', 'mid_rise', 'pre_1950')
    assert PRINTED['AGR1']['mid_rise']['pre_1950']
    inputs = hazus_table_completion._fallback_inputs(
        'mid_rise', 'pre_1950', PRINTED
    )
    assert 'AGR1' not in inputs
    assert 'COM1' in inputs
    for fill in ROUTES.values():
        assert fill.source != excluded
    # Were it a source, it would keep 0.25 for AGR1 mid-rise 1950 to 1970
    # and for AGR1 high-rise before 1950, under HALF_MASS, so the exclusion
    # changes neither route. It acts on the fallback inputs above.
    kept = hazus_table_completion._kept_mass(
        excluded, 'mid_rise', TYPE_LISTS, PRINTED
    )
    assert round(kept, 2) == 0.25


def test_no_source_is_filled_or_implausible() -> None:
    """Every source is a printed row off the list."""
    for fill in ROUTES.values():
        if fill.route == ROUTE_FALLBACK:
            assert fill.kept_mass is None
            assert len(fill.source) == 2
            continue
        occupancy, band, era = fill.source
        assert PRINTED[occupancy][band][era]
        assert (occupancy, band) not in IMPLAUSIBLE_PAIRS
        assert fill.kept_mass >= HALF_MASS


def test_no_empty_cell_has_two_printed_era_neighbors() -> None:
    """The later-neighbor rule never has to choose."""
    for occupancy, band, era in ROUTES:
        position = ERAS.index(era)
        neighbors = [
            ERAS[index]
            for index in (position - 1, position + 1)
            if 0 <= index < len(ERAS)
        ]
        printed = [
            other for other in neighbors if PRINTED[occupancy][band][other]
        ]
        assert len(printed) < 2


def _toy_tables(low_row: list[int]) -> tuple[dict, dict]:
    """
    Build a two-type table set whose COM1 mid-rise and high-rise are empty.

    Args:
        low_row (list[int]):
            The COM1 low-rise weights of every era, over W1 and URM.

    Returns:
        tuple[dict, dict]:
            Type lists and weights in the West Coast nesting.
    """
    type_lists = {band: ['W1', 'URM'] for band in BANDS}
    weights = {
        'COM1': {
            'low_rise': dict.fromkeys(ERAS, low_row),
            'mid_rise': {era: [] for era in ERAS},
            'high_rise': {era: [] for era in ERAS},
        }
    }
    return type_lists, weights


@pytest.mark.parametrize(
    ('low_row', 'route'),
    [([50, 50], ROUTE_BAND_BELOW), ([49, 51], ROUTE_FALLBACK)],
)
def test_a_source_at_half_mass_passes(low_row: list[int], route: str) -> None:
    """URM is set aside, so W1's share is the kept mass."""
    type_lists, weights = _toy_tables(low_row)
    fill = hazus_table_completion._fill_route(
        ('COM1', 'mid_rise', 'pre_1950'), type_lists, weights
    )
    assert fill.route == route


# ---------------------------------------------------------------------
# the completed weights


def test_every_cell_draws_at_four_design_levels(
    build_inventory: Callable[..., AssetInventory],
) -> None:
    """
    The inferer finds something to draw in every cell at each of four levels.

    URM, C3 and S5 are set aside by the inferer where the West Coast bars
    them, so the check runs the region's own legality rule.
    """
    assets = {
        f'{occupancy}|{band}|{era}|{level}': {
            'OccupancyClass': occupancy,
            'NumberOfStories': STORIES_IN_BAND[band],
            'YearBuilt': YEAR_IN_ERA[era],
            'DesignLevel': level,
        }
        for occupancy, band, era in _cells()
        for level in HAZUS_DESIGN_LEVELS
    }
    inferer = HazusEQStructureTypeInferer(
        state='California', seed=20260828, complete_tables=True
    )
    verdicts = inferer.check(build_inventory(assets))
    refused = verdicts[verdicts['diagnosis'] != DIAGNOSIS_OK]
    assert len(verdicts) == len(_cells()) * len(HAZUS_DESIGN_LEVELS)
    assert refused.empty, list(refused.index)


def test_printed_cells_are_unchanged() -> None:
    """Every printed row, RES1 included, equals the transcription."""
    for occupancy, bands in PRINTED.items():
        for band, rows in bands.items():
            for era, row in rows.items():
                if row:
                    assert COMPLETED[occupancy][band][era] == row


def test_filled_rows_sum_to_one_hundred() -> None:
    """A filled row is a distribution over its story band's type list."""
    for occupancy, band, era in ROUTES:
        row = COMPLETED[occupancy][band][era]
        assert len(row) == len(TYPE_LISTS[band])
        assert sum(row) == pytest.approx(100.0)


def test_a_band_below_fill_carries_its_source() -> None:
    """The fill is the source row on the target list, renormalized."""
    cell = ('COM7', 'high_rise', 'post_1970')
    fill = ROUTES[cell]
    assert fill.route == ROUTE_BAND_BELOW
    occupancy, band, era = fill.source
    source = dict(zip(TYPE_LISTS[band], PRINTED[occupancy][band][era]))
    expected = np.array([source.get(t, 0) for t in TYPE_LISTS['high_rise']])
    expected = 100.0 * expected / expected.sum()
    assert np.allclose(COMPLETED[cell[0]][cell[1]][cell[2]], expected)


def test_the_fallback_matches_values_worked_by_hand() -> None:
    """
    EDU1 high-rise from 1970 draws the combined high-rise rows of 1970 on.

    The expected values are worked from the rows of Table A-10 (p. A-8)
    of the Hazus Inventory Technical Manual 6.0.
    """
    filled = COMPLETED['EDU1']['high_rise']['post_1970']
    row = dict(zip(TYPE_LISTS['high_rise'], filled))
    assert ROUTES[('EDU1', 'high_rise', 'post_1970')].route == ROUTE_FALLBACK
    assert sum(row[t] for t in ('S1', 'S2', 'S4', 'S5')) == pytest.approx(68.625)
    assert sum(row[t] for t in ('C1', 'C2', 'C3', 'PC2')) == pytest.approx(31.0)
    assert row['RM2'] == pytest.approx(0.375)
    assert row['S1'] == pytest.approx(68.625 * 0.65638, abs=1e-3)
    assert row['PC2'] == pytest.approx(31.0 * 0.035996, abs=1e-3)
    assert row['S5'] == 0
    assert row['C3'] == 0
