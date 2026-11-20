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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.

"""Tests for brails.validators.validation_report."""

from __future__ import annotations

from brails.validators import ValidationReport
from brails.validators.validation_report import REPORT_COLUMNS

DATASET = 'seismic/building/portfolio/Hazus v6.1'


def test_an_empty_report_is_valid() -> None:
    """A report with no violations reports the inventory as valid."""
    report = ValidationReport(DATASET, n_pw=1, n_assets=3)

    assert report.is_valid is True
    assert len(report) == 0
    assert bool(report) is False


def test_a_report_with_violations_is_not_valid() -> None:
    """Recording a violation makes the report invalid and truthy."""
    report = ValidationReport(DATASET, n_pw=1, n_assets=3)
    report.add('A1', 0, ['$.DesignLevel: bad'])

    assert report.is_valid is False
    assert len(report) == 1
    assert bool(report) is True
    assert report.assets_with_violations() == ['A1']


def test_empty_message_lists_are_ignored() -> None:
    """Only offending assets appear in the report."""
    report = ValidationReport(DATASET)
    report.add('A1', 0, [])

    assert report.is_valid is True
    assert report.assets_with_violations() == []


def test_by_asset_collapses_worlds_and_deduplicates() -> None:
    """A message repeated across worlds is reported once per asset."""
    report = ValidationReport(DATASET, n_pw=2, n_assets=1)
    report.add('A1', 0, ['$.DesignLevel: bad', '$.HeightClass: bad'])
    report.add('A1', 1, ['$.DesignLevel: bad'])

    assert report.by_asset() == {'A1': ['$.DesignLevel: bad', '$.HeightClass: bad']}


def test_by_world_groups_by_world() -> None:
    """Violations are retrievable per possible world."""
    report = ValidationReport(DATASET, n_pw=2, n_assets=2)
    report.add('A1', 1, ['$.DesignLevel: bad'])
    report.add('A2', 0, ['$.HeightClass: bad'])

    assert report.by_world() == {
        1: {'A1': ['$.DesignLevel: bad']},
        0: {'A2': ['$.HeightClass: bad']},
    }


def test_to_dataframe_splits_path_from_message() -> None:
    """The long form separates where a violation was reported from what."""
    report = ValidationReport(DATASET, n_pw=1, n_assets=1)
    report.add(
        'A1',
        0,
        [
            "$.DesignLevel: 'High-Code' is not one of ['Pre-Code']",
            "$: 'HeightClass' is a required property",
        ],
    )

    frame = report.to_dataframe()

    assert list(frame.columns) == list(REPORT_COLUMNS)
    assert len(frame) == 2
    assert frame.iloc[0]['path'] == '$.DesignLevel'
    assert frame.iloc[0]['message'].startswith("'High-Code' is not one of")
    assert frame.iloc[1]['path'] == '$'
    assert frame.iloc[1]['message'] == "'HeightClass' is a required property"
    assert set(frame['asset_id']) == {'A1'}
    assert set(frame['world_index']) == {0}


def test_to_dataframe_is_empty_but_typed_when_valid() -> None:
    """A valid report still yields a frame with the documented columns."""
    frame = ValidationReport(DATASET).to_dataframe()

    assert list(frame.columns) == list(REPORT_COLUMNS)
    assert frame.empty


def test_to_dataframe_handles_a_message_without_a_path() -> None:
    """A message lacking the path prefix is kept whole."""
    report = ValidationReport(DATASET)
    report.add('A1', 0, ['something unexpected'])

    frame = report.to_dataframe()

    assert frame.iloc[0]['path'] == ''
    assert frame.iloc[0]['message'] == 'something unexpected'


def test_repr_reports_the_outcome() -> None:
    """The repr summarizes the dataset and the counts."""
    report = ValidationReport(DATASET, n_pw=2, n_assets=5)
    report.add('A1', 0, ['$.DesignLevel: bad'])

    text = repr(report)

    assert DATASET in text
    assert '1 of 5 assets invalid' in text
    assert '2 possible world' in text
