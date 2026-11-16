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


"""Tests for the shared inferer diagnosis vocabulary."""

from __future__ import annotations

import pandas as pd

from brails.inferers.inference_diagnosis import (
    DIAGNOSIS_COLUMNS,
    PREVIEW_LIMIT,
    diagnosis_frame,
    preview_failures,
    preview_ids,
)


def test_a_short_list_of_ids_is_shown_whole() -> None:
    """Nothing is counted away while everything fits."""
    assert preview_ids(['a', 'b', 'c']) == 'a, b, c'


def test_a_long_list_of_ids_is_cut_and_counted() -> None:
    """An inventory-wide failure must not print an inventory-wide message."""
    shown = preview_ids(range(PREVIEW_LIMIT + 5))
    assert shown.startswith('0, 1, 2')
    assert shown.endswith(f'{PREVIEW_LIMIT - 1} and 5 more')


def test_ids_of_any_type_are_shown() -> None:
    """Asset ids arrive as integers as readily as strings."""
    assert preview_ids([1, 2]) == '1, 2'


def test_no_ids_produce_an_empty_preview() -> None:
    """A caller with nothing to report gets nothing, not a stray count."""
    assert preview_ids([]) == ''


def test_a_short_set_of_failures_is_shown_whole() -> None:
    """Each asset is named with what stands in its way."""
    failures = pd.Series({'a': 'no YearBuilt', 'b': 'no OccupancyClass'})
    assert preview_failures(failures) == ('a (no YearBuilt), b (no OccupancyClass)')


def test_a_long_set_of_failures_is_cut_and_counted() -> None:
    """The count covers the assets the message leaves out."""
    failures = pd.Series(
        {f'asset_{index}': 'no YearBuilt' for index in range(PREVIEW_LIMIT + 3)}
    )
    shown = preview_failures(failures)
    assert shown.count('asset_') == PREVIEW_LIMIT
    assert shown.endswith('and 3 more')


def test_a_frame_carries_the_shared_columns() -> None:
    """Every inferer reports in one shape, whatever it infers."""
    frame = diagnosis_frame({'a': ('OK', '', ()), 'b': ('invalid', 'no X', ())})
    assert list(frame.columns) == list(DIAGNOSIS_COLUMNS)
    assert frame.loc['b', 'detail'] == 'no X'
    assert frame.loc['a', 'resolvable_by'] == ()


def test_an_empty_frame_keeps_its_columns() -> None:
    """A caller that filtered everything out still gets the shape."""
    frame = diagnosis_frame({})
    assert list(frame.columns) == list(DIAGNOSIS_COLUMNS)
    assert frame.empty


def test_a_frame_holds_a_tuple_of_features() -> None:
    """The resolvable features arrive as data, not as text to parse."""
    frame = diagnosis_frame(
        {'a': ('infeasible', 'nothing to draw', ('OccupancyClass', 'YearBuilt'))}
    )
    assert frame.loc['a', 'resolvable_by'] == ('OccupancyClass', 'YearBuilt')
