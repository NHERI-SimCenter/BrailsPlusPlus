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

"""Tests for the extended Hazus EQ rulesets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from brails.inferers.hazus_inferer_earthquake.hazus_rulesets import (
    get_hazus_general_type_membership,
    get_hazus_height_classes_by_type,
    get_hazus_occ_type_mapping,
    get_hazus_sampling_height_band,
    get_hazus_sampling_height_bands,
    get_hazus_state_region_mapping,
    get_hazus_story_count,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

STRUCTURE_TYPES = [
    'W1',
    'W2',
    'S1',
    'S2',
    'S3',
    'S4',
    'S5',
    'C1',
    'C2',
    'C3',
    'PC1',
    'PC2',
    'RM1',
    'RM2',
    'URM',
    'MH',
]

FULL_GRID = {'Low-Rise': (1, 3), 'Mid-Rise': (4, 7), 'High-Rise': (8, None)}

BANDS = ('low_rise', 'mid_rise', 'high_rise')


def _walk_vectors(
    node: dict | list, band: str | None
) -> Iterator[tuple[str | None, list]]:
    """Yield (band, weight vector) leaves of a nested weights entry."""
    if isinstance(node, list):
        yield band, node
        return
    for key, child in node.items():
        yield from _walk_vectors(child, key if key in BANDS else band)


# ---------------------------------------------------------------------
# get_hazus_occ_type_mapping


def test_weight_vectors_align_with_their_type_lists() -> None:
    """Every weight vector is empty or as long as its band's type list."""
    type_lists, type_weights = get_hazus_occ_type_mapping()
    for region, occupancies in type_weights.items():
        for occupancy, entry in occupancies.items():
            start = 'RES1' if occupancy == 'RES1' else None
            for band, vector in _walk_vectors(entry, start):
                assert band is not None, (region, occupancy)
                expected = len(type_lists[region][band])
                assert len(vector) in (0, expected), (
                    region,
                    occupancy,
                    band,
                )


# ---------------------------------------------------------------------
# get_hazus_sampling_height_bands


def test_sampling_bands_split_at_three_and_seven() -> None:
    """Pin the Appendix A bands: 1-3, 4-7, and 8 with no upper bound."""
    assert get_hazus_sampling_height_bands() == {
        'low_rise': (1, 3),
        'mid_rise': (4, 7),
        'high_rise': (8, None),
    }


def test_a_story_count_selects_its_story_band() -> None:
    """Three and four stories, and seven and eight, fall in adjacent bands."""
    assert get_hazus_sampling_height_band(3) == 'low_rise'
    assert get_hazus_sampling_height_band(4) == 'mid_rise'
    assert get_hazus_sampling_height_band(7) == 'mid_rise'
    assert get_hazus_sampling_height_band(8) == 'high_rise'
    with pytest.raises(ValueError, match='No story band'):
        get_hazus_sampling_height_band(0)


@pytest.mark.parametrize(('value', 'count'), [(10, 10), (10.0, 10), ('10', 10)])
def test_a_story_count_is_read_as_float_reads_it(value: object, count: int) -> None:
    """Numbers and numeric strings read as whole story counts."""
    assert get_hazus_story_count(value) == count


@pytest.mark.parametrize(
    ('value', 'error'),
    [
        ('ten', TypeError),
        (None, TypeError),
        (0, ValueError),
        (9.5, ValueError),
        (float('nan'), ValueError),
    ],
)
def test_an_unusable_story_count_raises(value: object, error: type) -> None:
    """A value float cannot read, or no whole count of one or more, raises."""
    with pytest.raises(error):
        get_hazus_story_count(value)


def test_sampling_band_names_key_the_weight_tables() -> None:
    """Every band name indexes the type lists of every region group."""
    type_lists, _ = get_hazus_occ_type_mapping()
    for region_lists in type_lists.values():
        for band in get_hazus_sampling_height_bands():
            assert band in region_lists


# ---------------------------------------------------------------------
# get_hazus_height_classes_by_type


def test_height_class_map_covers_every_structure_type() -> None:
    """The Table 5-1 map has exactly the 16 Hazus structure types."""
    assert set(get_hazus_height_classes_by_type()) == set(STRUCTURE_TYPES)


def test_height_free_types_have_no_height_class() -> None:
    """W1, W2, S3, PC1, and MH carry no height class."""
    height_free = {
        name
        for name, classes in get_hazus_height_classes_by_type().items()
        if classes is None
    }
    assert height_free == {'W1', 'W2', 'S3', 'PC1', 'MH'}


def test_urm_splits_at_two_stories() -> None:
    """URM is Low-Rise through 2 stories and Mid-Rise from 3 up."""
    assert get_hazus_height_classes_by_type()['URM'] == {
        'Low-Rise': (1, 2),
        'Mid-Rise': (3, None),
    }


def test_rm1_mid_rise_absorbs_taller_buildings() -> None:
    """RM1 has no High-Rise class; Mid-Rise runs from 4 stories up."""
    assert get_hazus_height_classes_by_type()['RM1'] == {
        'Low-Rise': (1, 3),
        'Mid-Rise': (4, None),
    }


def test_remaining_types_use_the_full_grid() -> None:
    """All other classed types split at 1-3, 4-7, and 8 and up."""
    height_classes = get_hazus_height_classes_by_type()
    for name in ['S1', 'S2', 'S4', 'S5', 'C1', 'C2', 'C3', 'PC2', 'RM2']:
        assert height_classes[name] == FULL_GRID


def test_height_class_bands_are_contiguous_from_one() -> None:
    """Each classed type covers every story count from 1 with no gaps."""
    for classes in get_hazus_height_classes_by_type().values():
        if classes is None:
            continue
        bands = list(classes.values())
        assert bands[0][0] == 1
        for lower, upper in zip(bands, bands[1:]):
            assert upper[0] == lower[1] + 1
        assert bands[-1][1] is None


def test_full_grid_entries_do_not_alias() -> None:
    """Mutating one type's class dict leaves the other types intact."""
    height_classes = get_hazus_height_classes_by_type()
    height_classes['S1']['Low-Rise'] = (0, 0)
    assert height_classes['S2']['Low-Rise'] == (1, 3)


# ---------------------------------------------------------------------
# get_hazus_general_type_membership


def test_membership_matches_the_contract() -> None:
    """Pin the general class members, with precast under C."""
    assert get_hazus_general_type_membership() == {
        'W': ['W1', 'W2'],
        'S': ['S1', 'S2', 'S3', 'S4', 'S5'],
        'C': ['C1', 'C2', 'C3', 'PC1', 'PC2'],
        'M': ['RM1', 'RM2', 'URM'],
        'MH': ['MH'],
    }


def test_membership_partitions_the_structure_types() -> None:
    """The general classes cover all 16 types with no overlap."""
    membership = get_hazus_general_type_membership()
    members = [name for group in membership.values() for name in group]
    assert sorted(members) == sorted(STRUCTURE_TYPES)
    assert len(members) == len(set(members))


def test_type_lists_stay_within_the_structure_types() -> None:
    """Every type in the Appendix A row sets is a known structure type."""
    type_lists, _ = get_hazus_occ_type_mapping()
    for region_lists in type_lists.values():
        for row in region_lists.values():
            assert set(row) <= set(STRUCTURE_TYPES)


# ---------------------------------------------------------------------
# get_hazus_state_region_mapping


def test_res1_weight_states_have_region_mappings() -> None:
    """Every state in the RES1 weight tables has a region mapping."""
    _, type_weights = get_hazus_occ_type_mapping()
    mapped = set(get_hazus_state_region_mapping())
    for region, occupancies in type_weights.items():
        res1 = occupancies['RES1']
        if region == 'West Coast':
            states = {name for era in res1.values() for name in era}
        else:
            states = set(res1)
        missing = states - mapped
        assert not missing, (region, sorted(missing))
