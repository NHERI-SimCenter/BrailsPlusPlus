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

"""Tests for the compilation report."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import brails.compilers
from brails.compilers.compilation_report import (
    EXCLUSION_COLUMNS,
    EXCLUSION_DTYPES,
    RECORD_COLUMNS,
    RECORD_DTYPES,
    STAGES,
    CompilationReport,
)
from brails.constants import (
    BLOCK_GROUP_KEY,
    BUILDING_TYPE_KEY,
    DESIGN_LEVEL_KEY,
    STORIES_KEY,
    STRUCTURE_TYPE_KEY,
    YEAR_BUILT_KEY,
)
from brails.types.asset_inventory import SOURCE_CODES
from brails.utils import Importer

if TYPE_CHECKING:
    from pathlib import Path

# A block group identifier, carried through the records as text.
BLOCK_GROUP = '060750101001'


def test_a_recorded_action_appears_in_the_records() -> None:
    """A recorded action is one row carrying every field it was given."""
    report = CompilationReport(11)
    report.record_action(
        'a',
        BUILDING_TYPE_KEY,
        stage='resolution',
        action='withheld',
        world=2,
        old='M',
        source='nsi',
        reason='the structure type cell holds no masonry type',
        block_group=BLOCK_GROUP,
    )

    frame = report.records

    assert list(frame.columns) == list(RECORD_COLUMNS)
    assert frame.iloc[0].to_dict() == {
        'asset_id': 'a',
        'world': 2,
        'stage': 'resolution',
        'action': 'withheld',
        'feature': BUILDING_TYPE_KEY,
        'old': 'M',
        'new': None,
        'source': 'nsi',
        'reason': 'the structure type cell holds no masonry type',
        'block_group': BLOCK_GROUP,
    }


def test_values_are_recorded_as_text() -> None:
    """Old and new values and the block group are kept as text, None as None."""
    report = CompilationReport(11)
    report.record_action(
        'a',
        STORIES_KEY,
        stage='pair_redraw',
        action='redrawn',
        world=0,
        old=7,
        new=2,
        block_group=60750101001,
    )
    report.record_action(
        'b', BUILDING_TYPE_KEY, stage='resolution', action='withheld', world=0
    )

    frame = report.records

    assert list(frame['old']) == ['7', None]
    assert list(frame['new']) == ['2', None]
    assert list(frame['block_group']) == ['60750101001', None]


@pytest.mark.parametrize('stage', STAGES)
def test_every_stage_is_accepted(stage: str) -> None:
    """The record's stage takes each stage of the run."""
    report = CompilationReport(11)
    report.record_action('a', STORIES_KEY, stage=stage, action='withheld', world=0)

    assert list(report.records['stage']) == [stage]
    assert 'stories' in STAGES


@pytest.mark.parametrize('source', list(SOURCE_CODES))
def test_every_source_label_is_accepted(source: str) -> None:
    """The record's source takes each label of the source string's codes."""
    report = CompilationReport(11)
    report.record_action(
        'a', YEAR_BUILT_KEY, stage='year', action='derived', world=0, source=source
    )

    assert list(report.records['source']) == [source]


@pytest.mark.parametrize(
    ('arguments', 'message', 'named', 'listed'),
    [
        ({'stage': 'guessing', 'action': 'withheld'}, 'stage', 'guessing', 'units'),
        ({'stage': 'year', 'action': 'held'}, 'action', 'held', 'restored'),
        (
            {'stage': 'year', 'action': 'withheld', 'source': 'N'},
            'source',
            'N',
            'imputed',
        ),
    ],
)
def test_a_value_outside_its_set_is_refused(
    arguments: dict[str, str], message: str, named: str, listed: str
) -> None:
    """An unknown stage, action or source raises and names the valid values."""
    report = CompilationReport(11)

    with pytest.raises(ValueError, match=f'not a recognized {message}') as raised:
        report.record_action('a', YEAR_BUILT_KEY, world=0, **arguments)

    assert repr(named) in str(raised.value)
    assert listed in str(raised.value)
    assert report.records.empty


def test_an_unknown_exclusion_category_is_refused() -> None:
    """An unknown category raises and the error names the valid ones."""
    report = CompilationReport(11)

    with pytest.raises(
        ValueError, match='not a recognized exclusion category'
    ) as raised:
        report.record_exclusion('a', world=0, stage='screen', category='bad_data')

    assert 'bad_data' in str(raised.value)
    assert 'dropped' in str(raised.value)
    assert report.exclusions.empty


def test_a_record_needs_a_world() -> None:
    """An action or an exclusion recorded without a world is refused."""
    report = CompilationReport(11)

    with pytest.raises(TypeError, match='world'):
        report.record_action('a', YEAR_BUILT_KEY, stage='year', action='derived')
    with pytest.raises(TypeError, match='world'):
        report.record_exclusion('a', stage='screen', category='unsupported')


def test_worlds_outside_the_run_are_refused() -> None:
    """A world that is negative, not an integer or past the count raises."""
    report = CompilationReport(11)
    with pytest.raises(ValueError, match='cannot be negative'):
        report.record_action(
            'a', YEAR_BUILT_KEY, stage='year', action='derived', world=-1
        )
    with pytest.raises(TypeError):
        report.record_exclusion(
            'a', world=1.5, stage='screen', category='unsupported'
        )
    report.record_exclusion('a', world=4, stage='screen', category='unsupported')
    with pytest.raises(ValueError, match='World 4 is already recorded'):
        report.set_world_count(3)
    with pytest.raises(TypeError):
        report.set_world_count(5.5)
    with pytest.raises(ValueError, match='at least 1'):
        report.set_world_count(0)

    report.set_world_count(5)

    with pytest.raises(ValueError, match='World 5 is outside the 5 worlds'):
        report.record_exclusion('a', world=5, stage='screen', category='unsupported')
    with pytest.raises(ValueError, match='World 7 is outside'):
        report.summary(worlds=(7,))
    with pytest.raises(ValueError, match='World 7 is outside'):
        report.set_timing('step6', 1.0, world=7)
    assert len(report.exclusions) == 1


def test_a_numpy_seed_is_kept_as_an_integer() -> None:
    """A seed given as a numpy integer is summarized as a Python integer."""
    report = CompilationReport(np.int64(5))

    assert json.dumps(report.summary()['seed']) == '5'


def test_every_exclusion_row_carries_its_world() -> None:
    """Each exclusion row holds the world it was recorded for, as an integer."""
    report = CompilationReport(11)
    report.record_exclusion('a', world=0, stage='units', category='dropped')
    report.record_exclusion('a', world=1, stage='units', category='dropped')
    report.record_exclusion('b', world=1, stage='resolution', category='conflict')

    frame = report.exclusions

    assert list(frame.columns) == list(EXCLUSION_COLUMNS)
    assert list(frame['world']) == [0, 1, 1]
    assert frame['world'].dtype == 'int64'


def test_an_empty_report_carries_the_columns_and_dtypes() -> None:
    """Frames of a report holding nothing keep their columns and dtypes."""
    report = CompilationReport(11)

    assert report.records.dtypes.to_dict() == RECORD_DTYPES
    assert report.exclusions.dtypes.to_dict() == EXCLUSION_DTYPES


def test_the_records_frame_is_a_copy() -> None:
    """Changing the returned records leaves the report alone."""
    report = CompilationReport(11)
    report.record_action(
        'a', YEAR_BUILT_KEY, stage='year', action='derived', world=0
    )

    frame = report.records
    frame.loc[0, 'feature'] = STRUCTURE_TYPE_KEY

    assert report.records.loc[0, 'feature'] == YEAR_BUILT_KEY


def test_the_summary_counts_actions_and_exclusions() -> None:
    """The summary counts actions, and exclusion rows per world and category."""
    report = CompilationReport(11)
    report.record_action(
        'a', BUILDING_TYPE_KEY, stage='resolution', action='withheld', world=0
    )
    report.record_action(
        'a', BUILDING_TYPE_KEY, stage='resolution', action='restored', world=0
    )
    report.record_action(
        'b', DESIGN_LEVEL_KEY, stage='carve_out', action='derived', world=1
    )
    report.record_exclusion(
        'c', world=1, stage='resolution', category='conflict', diagnosis='infeasible'
    )
    report.record_exclusion(
        'a', world=1, stage='screen', category='unsupported', diagnosis='unsupported'
    )
    report.record_exclusion(
        'a', world=0, stage='screen', category='unsupported', diagnosis='unsupported'
    )

    summary = report.summary()

    assert summary['actions'] == {
        'by_action': {'withheld': 1, 'restored': 1, 'derived': 1},
        'by_feature': {BUILDING_TYPE_KEY: 2, DESIGN_LEVEL_KEY: 1},
    }
    assert summary['exclusions'] == {
        'by_category': {'unsupported': 2, 'conflict': 1},
        'by_diagnosis': {'unsupported': 2, 'infeasible': 1},
        'by_world': {0: {'unsupported': 1}, 1: {'unsupported': 1, 'conflict': 1}},
        'total': 3,
    }
    assert summary['assets_excluded']['total'] == 2
    # The counts are keyed in sorted order, whatever the recording order.
    assert list(summary['actions']['by_action']) == [
        'derived',
        'restored',
        'withheld',
    ]
    assert list(summary['exclusions']['by_category']) == ['conflict', 'unsupported']
    assert list(summary['exclusions']['by_world']) == [0, 1]
    assert list(summary['exclusions']['by_world'][1]) == ['conflict', 'unsupported']


def test_assets_excluded_in_every_world_and_in_some_are_separated() -> None:
    """An asset excluded in every world is told apart from one excluded in some."""
    report = CompilationReport(11)
    report.set_world_count(3)
    for world in range(3):
        report.record_exclusion(
            'a', world=world, stage='certificate', category='certificate'
        )
        # A dropped footprint is in the group 'dropped', not 'every' or 'some'.
        report.record_exclusion('d', world=world, stage='units', category='dropped')
    report.record_exclusion('b', world=1, stage='screen', category='unsupported')
    # Asset 'c' has two rows in world 1, which count as one world.
    report.record_exclusion('c', world=1, stage='screen', category='missing_driver')
    report.record_exclusion('c', world=1, stage='resolution', category='conflict')
    report.record_exclusion(
        'c', world=2, stage='certificate', category='certificate'
    )

    groups = report.exclusion_groups()

    assert list(groups.columns) == [*EXCLUSION_COLUMNS, 'group']
    assert groups.groupby('asset_id')['group'].unique().map(list).to_dict() == {
        'a': ['every'],
        'b': ['some'],
        'c': ['some'],
        'd': ['dropped'],
    }
    assert list(groups.loc[groups['asset_id'] == 'c', 'world']) == [1, 1, 2]
    assert report.summary()['assets_excluded'] == {
        'total': 4,
        'dropped': 1,
        'every': 1,
        'some': 2,
    }

    # The groups are taken over the selected worlds, so a world written
    # alone puts every asset it excluded in 'every'.
    alone = report.exclusion_groups(worlds=(1,))
    assert set(alone.loc[alone['asset_id'] != 'd', 'group']) == {'every'}
    assert report.summary(worlds=(1,))['assets_excluded'] == {
        'total': 4,
        'dropped': 1,
        'every': 3,
        'some': 0,
    }


def test_the_groups_need_a_known_world_set() -> None:
    """Without a world count or a selection the groups are not counted."""
    report = CompilationReport(11)
    report.record_exclusion('a', world=0, stage='screen', category='unsupported')

    assert report.summary()['assets_excluded'] == {
        'total': 1,
        'dropped': 0,
        'every': None,
        'some': None,
    }
    assert report.summary()['worlds'] == {'computed': None, 'written': None}
    with pytest.raises(ValueError, match='worlds are unknown'):
        report.exclusion_groups()


def test_the_summary_holds_timings_worlds_and_stages() -> None:
    """Seconds per stage and world, the world numbers and the stage order."""
    report = CompilationReport(11)
    report.set_world_count(6)
    report.set_manifest_stages(['footprints', 'join_nsi', 'step4'])
    report.set_timing('footprints', 1.5)
    report.set_timing('step6', 3.0, world=0)
    report.set_timing('step6', 2.0, world=1)
    report.set_timing('step6', 2.5, world=0)
    report.set_timing('manifest', 0.25)

    summary = report.summary(worlds=(5, 2))

    assert summary['timings'] == [
        {'stage': 'footprints', 'world': None, 'seconds': 1.5},
        {'stage': 'step6', 'world': 0, 'seconds': 2.5},
        {'stage': 'step6', 'world': 1, 'seconds': 2.0},
        {'stage': 'manifest', 'world': None, 'seconds': 0.25},
    ]
    assert summary['worlds'] == {'computed': 6, 'written': [2, 5]}
    assert summary['manifest_stages'] == ['footprints', 'join_nsi', 'step4']
    assert report.summary()['worlds']['written'] == [0, 1, 2, 3, 4, 5]


def test_save_writes_the_files_it_documents(tmp_path: Path) -> None:
    """The records, the exclusions and the summary read back as recorded."""
    report = CompilationReport(11, settings={'n_worlds': 2})
    report.set_world_count(2)
    report.record_action(
        'a',
        STORIES_KEY,
        stage='pair_redraw',
        action='redrawn',
        world=1,
        old=7,
        new=2,
        source='imputed',
        block_group=BLOCK_GROUP,
    )
    report.record_exclusion(
        'b',
        world=0,
        stage='resolution',
        category='conflict',
        diagnosis='infeasible',
        detail='no structure type left to draw',
        resolvable_by=(YEAR_BUILT_KEY, BUILDING_TYPE_KEY),
    )
    report.record_exclusion('c', world=1, stage='screen', category='empty_cell')
    report.note('two observed features disagree on asset b')
    report.set_coverage({BLOCK_GROUP_KEY: 1.0})
    report.set_certificate('passed')
    report.set_timing('join', 4.0)

    report.save(tmp_path / 'run' / 'report.parquet')

    written = tmp_path / 'run'
    records = pd.read_parquet(written / 'report.parquet')
    pd.testing.assert_frame_equal(records, report.records, check_dtype=False)
    exclusions = pd.read_parquet(written / 'report_exclusions.parquet')
    assert list(exclusions.columns) == list(EXCLUSION_COLUMNS)
    assert list(exclusions['world']) == [0, 1]
    # Parquet gives the tuple back as an array.
    assert list(exclusions['resolvable_by'].iloc[0]) == [
        YEAR_BUILT_KEY,
        BUILDING_TYPE_KEY,
    ]
    assert list(exclusions['resolvable_by'].iloc[1]) == []
    summary = json.loads(
        (written / 'report_summary.json').read_text(encoding='utf-8')
    )
    assert summary == json.loads(json.dumps(report.summary()))
    assert summary['settings'] == {'n_worlds': 2}
    assert summary['notices'] == ['two observed features disagree on asset b']
    assert summary['coverage'] == {BLOCK_GROUP_KEY: 1.0}
    assert summary['certificate'] == 'passed'
    assert summary['exclusions']['by_world'] == {
        '0': {'conflict': 1},
        '1': {'empty_cell': 1},
    }


def test_save_keeps_the_rows_of_the_selected_worlds(tmp_path: Path) -> None:
    """A selection of worlds writes their rows only, and a summary of them."""
    report = CompilationReport(11)
    report.set_world_count(4)
    for world in range(4):
        report.record_action(
            'a',
            BUILDING_TYPE_KEY,
            stage='resolution',
            action='withheld',
            world=world,
        )
        report.record_exclusion('b', world=world, stage='units', category='dropped')
    report.record_exclusion('c', world=1, stage='screen', category='unsupported')

    report.save(tmp_path / 'report.parquet', worlds=(3, 1))

    records = pd.read_parquet(tmp_path / 'report.parquet')
    exclusions = pd.read_parquet(tmp_path / 'report_exclusions.parquet')
    summary = json.loads((tmp_path / 'report_summary.json').read_text('utf-8'))
    assert list(records['world']) == [1, 3]
    assert list(zip(exclusions['asset_id'], exclusions['world'])) == [
        ('b', 1),
        ('b', 3),
        ('c', 1),
    ]
    assert summary['worlds'] == {'computed': 4, 'written': [1, 3]}
    assert summary['actions']['by_action'] == {'withheld': 2}
    assert summary['exclusions'] == {
        'by_category': {'dropped': 2, 'unsupported': 1},
        'by_diagnosis': {'': 3},
        'by_world': {'1': {'dropped': 1, 'unsupported': 1}, '3': {'dropped': 1}},
        'total': 3,
    }
    assert summary['assets_excluded'] == {
        'total': 2,
        'dropped': 1,
        'every': 0,
        'some': 1,
    }


def test_an_empty_report_saves(tmp_path: Path) -> None:
    """A report holding nothing saves, with the columns of its frames."""
    report = CompilationReport(11)

    report.save(tmp_path / 'empty.parquet')

    assert list(pd.read_parquet(tmp_path / 'empty.parquet').columns) == list(
        RECORD_COLUMNS
    )
    assert pd.read_parquet(tmp_path / 'empty_exclusions.parquet').empty


def test_save_logs_writes_each_log_by_name(tmp_path: Path) -> None:
    """Each log is written under its name; a selection filters world columns."""
    report = CompilationReport(11)
    per_world = pd.DataFrame({'asset_id': ['a', 'a', 'b'], 'world': [0, 1, 2]})
    once = pd.DataFrame({'asset_id': ['a', 'b'], 'tie': [True, False]})

    report.save_logs(
        tmp_path / 'logs', 'run', {'pair_redraw': per_world, 'ties': once}
    )
    report.save_logs(
        tmp_path / 'some',
        'run',
        {'pair_redraw': per_world, 'ties': once},
        worlds=(0, 2),
    )

    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / 'logs' / 'run_pair_redraw.parquet'), per_world
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / 'some' / 'run_ties.parquet'), once
    )
    assert list(
        pd.read_parquet(tmp_path / 'some' / 'run_pair_redraw.parquet')['world']
    ) == [0, 2]


def test_save_logs_refuses_a_name_that_is_a_path(tmp_path: Path) -> None:
    """A log name holding a directory is refused before anything is written."""
    report = CompilationReport(11)

    with pytest.raises(ValueError, match='plain file name'):
        report.save_logs(tmp_path / 'logs', 'run', {'../join': pd.DataFrame()})

    assert not (tmp_path / 'logs').exists()


@pytest.mark.parametrize('name', ['report', 'report_exclusions'])
def test_save_logs_refuses_the_report_file_names(tmp_path: Path, name: str) -> None:
    """A log named as one of the report's own files is refused."""
    report = CompilationReport(11)

    with pytest.raises(ValueError, match='overwrite the files of a report'):
        report.save_logs(tmp_path, 'run', {name: pd.DataFrame({'world': [0]})})

    assert not (tmp_path / f'run_{name}.parquet').exists()


def test_discovered_and_exported() -> None:
    """The Importer finds the class, and the package exports it."""
    assert Importer().get_class('CompilationReport') is CompilationReport
    assert 'CompilationReport' in brails.compilers.__all__
    assert brails.compilers.CompilationReport is CompilationReport
