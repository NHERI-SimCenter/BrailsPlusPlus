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

"""
Complete the West Coast Appendix A structure type tables.

The section "The completed West Coast tables" of the README beside this module states
what the completion rests on.

.. autosummary::

      CellFill
      is_implausible
      get_west_coast_fill_routes
      get_completed_west_coast_weights
"""

from __future__ import annotations

from copy import deepcopy
from typing import NamedTuple

from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    UNREINFORCED_MASONRY_TYPES,
    WEST_COAST,
    get_hazus_general_type_membership,
    get_hazus_occ_type_mapping,
    get_hazus_sampling_height_band,
    get_hazus_sampling_height_bands,
    get_hazus_year_classes,
)

# The least share of its weight a source row must keep to pass a route.
HALF_MASS = 0.5

ROUTE_ADJACENT_ERA = 'adjacent_era'
ROUTE_BAND_BELOW = 'band_below'
ROUTE_FALLBACK = 'fallback'

_BANDS = tuple(get_hazus_sampling_height_bands())
_ERAS = tuple(get_hazus_year_classes())
_LOWEST_BAND = _BANDS[0]
_HIGHEST_BAND = _BANDS[-1]

# Occupancy classes treated as implausible above low-rise, and those treated as
# implausible at high-rise.
_IMPLAUSIBLE_ABOVE_LOW_RISE = ('RES2', 'AGR1')
_IMPLAUSIBLE_AT_HIGH_RISE = (
    'IND1',
    'IND2',
    'IND3',
    'IND4',
    'IND6',
    'EDU1',
    'GOV2',
    'REL1',
)

# The occupancy class and story band pairs treated as implausible for the region.
IMPLAUSIBLE_PAIRS = frozenset(
    {
        (occupancy, band)
        for occupancy in _IMPLAUSIBLE_ABOVE_LOW_RISE
        for band in _BANDS
        if band != _LOWEST_BAND
    }
    | {(occupancy, _HIGHEST_BAND) for occupancy in _IMPLAUSIBLE_AT_HIGH_RISE}
)


class CellFill(NamedTuple):
    """
    How one empty West Coast cell is filled.

    Attributes:
        route (str):
            ``ROUTE_ADJACENT_ERA``, ``ROUTE_BAND_BELOW`` or ``ROUTE_FALLBACK``.
        source (tuple):
            The source cell as ``(occupancy, band, era)``, or for the fallback the
            ``(band, era)`` whose printed rows it combines.
        kept_mass (float | None):
            The share of the source row's weight left on the target story band's type
            list without ``UNREINFORCED_MASONRY_TYPES``, or None for the fallback.
    """

    route: str
    source: tuple
    kept_mass: float | None


def is_implausible(occupancy: str, stories: int) -> bool:
    """
    Say whether an occupancy class and story count form an implausible pair.

    Args:
        occupancy (str):
            Occupancy class as the Appendix A tables key it.
        stories (int):
            Story count, one or more.

    Returns:
        bool:
            True where ``occupancy`` and the story band of ``stories`` form a pair in
            ``IMPLAUSIBLE_PAIRS``.

    Raises:
        ValueError:
            If no story band covers the count.
    """
    return (
        occupancy,
        get_hazus_sampling_height_band(stories),
    ) in IMPLAUSIBLE_PAIRS


def get_west_coast_fill_routes() -> dict[tuple[str, str, str], CellFill]:
    """
    Return the route that fills each empty West Coast cell.

    A cell takes the first of the two routes below whose source keeps at least
    ``HALF_MASS`` of its weight on the target story band's type list without
    ``UNREINFORCED_MASONRY_TYPES``, and otherwise the fallback. Cells of
    ``IMPLAUSIBLE_PAIRS`` are filled but never serve as sources.

    1. ``ROUTE_ADJACENT_ERA``: the same occupancy and story band in the neighboring
       era, the later one where both are sources.
    2. ``ROUTE_BAND_BELOW``: the same occupancy and era one story band down.
    3. ``ROUTE_FALLBACK``: the printed rows of the story band and era, combined
       within each general building type (``_fallback_row``).

    Returns:
        dict[tuple[str, str, str], CellFill]:
            One entry per cell of occupancy class, story band and era the tables
            print no row for, keyed ``(occupancy, band, era)``.
    """
    type_lists, weights = _west_coast_tables()
    # blank cells of the printed tables, RES1 excluded
    empty_cells = [
        (occupancy, band, era)
        for occupancy, bands in weights.items()
        if occupancy != 'RES1'
        for band in _BANDS
        for era in _ERAS
        # an empty row is a cell the table leaves blank
        if not bands[band][era]
    ]
    return {cell: _fill_route(cell, type_lists, weights) for cell in empty_cells}


def get_completed_west_coast_weights() -> dict:
    """
    Return the West Coast weights with every empty cell filled.

    The nesting is that of the West Coast entry of the weights
    ``get_hazus_occ_type_mapping`` returns, and each filled row aligns with its story
    band's type list there. Printed rows, RES1 included, are returned unchanged.

    Returns:
        dict:
            Occupancy class to story band to era to a weight list, which sums to 100
            for a filled cell.
    """
    type_lists, weights = _west_coast_tables()
    # The fills are written into a copy, so that no filled row becomes a source or a
    # fallback input for another cell.
    completed = deepcopy(weights)
    for cell, fill in get_west_coast_fill_routes().items():
        occupancy, band, era = cell
        if fill.route == ROUTE_FALLBACK:
            row = _fallback_row(band, era, type_lists, weights)
        else:
            # the source cell's weights on this band's types
            source_occupancy, source_band, source_era = fill.source
            by_type = dict(
                zip(
                    type_lists[source_band],
                    weights[source_occupancy][source_band][source_era],
                )
            )
            row = [float(by_type.get(t, 0)) for t in type_lists[band]]
            total = sum(row)
            row = [100.0 * weight / total for weight in row]
        completed[occupancy][band][era] = row
    return completed


def _west_coast_tables() -> tuple[dict, dict]:
    """
    Return the West Coast type lists and weights of the transcription.

    Returns:
        tuple[dict, dict]:
            Story band to structure type list, and occupancy class to story band to
            era to the printed weights.
    """
    type_lists, weights = get_hazus_occ_type_mapping()
    return type_lists[WEST_COAST], weights[WEST_COAST]


def _is_source(weights: dict, cell: tuple[str, str, str]) -> bool:
    """
    Say whether a cell may serve as a source for a fill.

    Args:
        weights (dict):
            The West Coast weights.
        cell (tuple[str, str, str]):
            ``(occupancy, band, era)``.

    Returns:
        bool:
            True where the cell is printed and its occupancy and story band are not
            in ``IMPLAUSIBLE_PAIRS``.
    """
    occupancy, band, era = cell
    return (occupancy, band) not in IMPLAUSIBLE_PAIRS and bool(
        weights[occupancy][band][era]
    )


def _kept_mass(
    source: tuple[str, str, str],
    target_band: str,
    type_lists: dict,
    weights: dict,
) -> float:
    """
    Measure the share of a source row a target story band can use.

    Args:
        source (tuple[str, str, str]):
            The source cell.
        target_band (str):
            The story band of the cell being filled.
        type_lists (dict):
            The West Coast type lists.
        weights (dict):
            The West Coast weights.

    Returns:
        float:
            The source row's weight on the target story band's type list without
            ``UNREINFORCED_MASONRY_TYPES``, over its total weight.
    """
    occupancy, band, era = source
    row = weights[occupancy][band][era]
    usable = set(type_lists[target_band]) - set(UNREINFORCED_MASONRY_TYPES)
    kept = sum(
        weight
        for structure_type, weight in zip(type_lists[band], row)
        if structure_type in usable
    )
    return kept / sum(row)


def _route_candidates(
    cell: tuple[str, str, str], type_lists: dict, weights: dict
) -> list[tuple[str, tuple[str, str, str], float]]:
    """
    List the source each of the first two routes offers a cell.

    Args:
        cell (tuple[str, str, str]):
            The empty cell.
        type_lists (dict):
            The West Coast type lists.
        weights (dict):
            The West Coast weights.

    Returns:
        list[tuple[str, tuple[str, str, str], float]]:
            ``(route, source, kept_mass)`` for each route that has a source, in route
            order.
    """
    occupancy, band, era = cell
    position = _ERAS.index(era)
    band_position = _BANDS.index(band)
    # The later era neighbor comes first, so it wins where both are sources.
    offered = [
        (ROUTE_ADJACENT_ERA, (occupancy, band, _ERAS[index]))
        for index in (position + 1, position - 1)
        if 0 <= index < len(_ERAS)
    ]
    if band_position > 0:
        offered.append(
            (ROUTE_BAND_BELOW, (occupancy, _BANDS[band_position - 1], era))
        )
    chosen: dict[str, tuple[str, str, str]] = {}
    for route, source in offered:
        if route not in chosen and _is_source(weights, source):
            chosen[route] = source
    return [
        (route, source, _kept_mass(source, band, type_lists, weights))
        for route, source in chosen.items()
    ]


def _fill_route(
    cell: tuple[str, str, str], type_lists: dict, weights: dict
) -> CellFill:
    """
    Choose the route that fills one empty cell.

    Args:
        cell (tuple[str, str, str]):
            The empty cell.
        type_lists (dict):
            The West Coast type lists.
        weights (dict):
            The West Coast weights.

    Returns:
        CellFill:
            The first route whose source keeps at least ``HALF_MASS``, or the
            fallback.
    """
    _, band, era = cell
    return next(
        (
            CellFill(route, source, kept_mass)
            for route, source, kept_mass in _route_candidates(
                cell, type_lists, weights
            )
            if kept_mass >= HALF_MASS
        ),
        CellFill(ROUTE_FALLBACK, (band, era), None),
    )


def _fallback_inputs(band: str, era: str, weights: dict) -> dict[str, list]:
    """
    Return the printed rows the fallback of a story band and era combines.

    Args:
        band (str):
            The story band of the cell being filled.
        era (str):
            The era of the cell being filled.
        weights (dict):
            The West Coast weights.

    Returns:
        dict[str, list]:
            Occupancy class to its printed row, for every non-RES1 class whose row of
            the story band and era is a source.
    """
    return {
        occupancy: bands[band][era]
        for occupancy, bands in sorted(weights.items())
        if occupancy != 'RES1' and _is_source(weights, (occupancy, band, era))
    }


def _fallback_row(
    band: str, era: str, type_lists: dict, weights: dict
) -> list[float]:
    """
    Combine the printed rows of a story band and era within each general type.

    The inputs are the rows of ``_fallback_inputs``. Each general type of
    ``get_hazus_general_type_membership`` receives the mean share of the input rows'
    weight on its members of the story band's type list, split among those members by
    their mean share within the general type over the rows that hold weight in it.

    Args:
        band (str):
            The story band of the cell being filled.
        era (str):
            The era of the cell being filled.
        type_lists (dict):
            The West Coast type lists.
        weights (dict):
            The West Coast weights.

    Returns:
        list[float]:
            Weights aligned with the story band's type list, summing to 100.
    """
    types = type_lists[band]
    rows = list(_fallback_inputs(band, era, weights).values())
    stored = dict.fromkeys(types, 0.0)
    for members in get_hazus_general_type_membership().values():
        positions = [i for i, t in enumerate(types) if t in members]
        if not positions:
            continue
        holding = [row for row in rows if sum(row[i] for i in positions) > 0]
        if not holding:
            continue
        share = sum(sum(row[i] for i in positions) / sum(row) for row in rows)
        share /= len(rows)
        for i in positions:
            conditional = sum(
                row[i] / sum(row[j] for j in positions) for row in holding
            ) / len(holding)
            stored[types[i]] = 100.0 * share * conditional
    return [stored[t] for t in types]
