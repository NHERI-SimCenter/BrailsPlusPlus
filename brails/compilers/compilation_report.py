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


"""
Record what a Hazus earthquake compilation did to each asset in each world.

.. autosummary::

      CompilationReport
"""

from __future__ import annotations

import json
import operator
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from brails.types.asset_inventory import SOURCE_CODES

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

# The stages of the compilation. Every recorded action and exclusion names
# the stage it came from.
STAGES = (
    'footprints',
    'join',
    'stories',
    'units',
    'imputation',
    'year',
    'design_level',
    'pair_redraw',
    'screen',
    'resolution',
    'structure_type',
    'height_class',
    'carve_out',
    'nonresidential_year',
    'value',
    'certificate',
)

# What a stage did to a feature. ``withheld`` is a value kept out of a
# draw or removed from its source, ``restored`` a withheld value put back
# after a failed attempt, ``redrawn`` a value the pair re-draw replaced,
# and ``derived`` a value computed from another feature of the asset, such
# as a design level the W1 carve-out changed.
ACTIONS = ('withheld', 'restored', 'redrawn', 'derived')

# Why an asset left the inventory of a world, the vocabulary every
# exclusion is filed under.
# ``missing_driver``: a driver an inferer reads is absent or holds a value
# it cannot use, or no step supplied a year.
# ``unsupported``: a driver is present in a form the inferers do not read,
# such as a list of values per world.
# ``conflict``: the asset's own features empty a published or filled row
# of the structure type tables, or its refusal named a feature the caller
# acted on without admitting the asset.
# ``empty_cell``: the structure type cell has neither a published nor a
# filled row.
# ``undetermined``: an infeasible asset whose cause the caller could not
# separate.
# ``certificate``: the input schema refused the asset in the compiled
# world.
# ``dropped``: a footprint left out before any world was compiled, the
# same in every world.
EXCLUSION_CATEGORIES = (
    'missing_driver',
    'unsupported',
    'conflict',
    'empty_cell',
    'undetermined',
    'certificate',
    'dropped',
)

# The columns of the frame :attr:`CompilationReport.records` returns, in
# order, with the dtype each one carries. A report holding no rows
# carries these same dtypes.
RECORD_DTYPES = {
    'asset_id': 'object',
    'world': 'int64',
    'stage': 'object',
    'action': 'object',
    'feature': 'object',
    'old': 'object',
    'new': 'object',
    'source': 'object',
    'reason': 'object',
    'block_group': 'object',
}

# The columns of the record frame, in order.
RECORD_COLUMNS = tuple(RECORD_DTYPES)

# The columns of the frame :attr:`CompilationReport.exclusions` returns,
# in order, with the dtype each one carries.
EXCLUSION_DTYPES = {
    'asset_id': 'object',
    'world': 'int64',
    'stage': 'object',
    'category': 'object',
    'diagnosis': 'object',
    'detail': 'object',
    'resolvable_by': 'object',
}

# The columns of the exclusion frame, in order.
EXCLUSION_COLUMNS = tuple(EXCLUSION_DTYPES)

# What :meth:`CompilationReport.save` adds to the stem of its path for the
# file of exclusions.
EXCLUSIONS_SUFFIX = '_exclusions'


def _in_set(value: str, valid: Collection[str], kind: str) -> str:
    """
    Check a recorded value against the closed set it comes from.

    Args:
        value (str):
            The value to check.
        valid (Collection[str]):
            The values the set holds.
        kind (str):
            What the set names, for the error message.

    Returns:
        str:
            The value.

    Raises:
        ValueError:
            If the value is outside the set.
    """
    if value not in valid:
        raise ValueError(
            f'{value!r} is not a recognized {kind}. Valid values: {list(valid)}.'
        )
    return value


class CompilationReport:
    """
    What a Hazus earthquake compilation did, world by world.

    A report holds one long-form row per action taken on a feature and one
    row per asset excluded from a world, both carrying the world number,
    together with the seconds the run spent per stage and world, the
    stage order of the run's manifest, notices, driver coverage and the
    verdict of the input-schema check on the run. :meth:`summary` counts
    that state and :meth:`save` and :meth:`save_logs` write it.

    The stage, action, source and category of each row must come from
    :data:`STAGES`, :data:`ACTIONS`,
    :data:`~brails.types.asset_inventory.SOURCE_CODES` and
    :data:`EXCLUSION_CATEGORIES`, and a value outside its set is refused.
    ``old``, ``new`` and ``block_group`` are recorded as text. Asset ids
    must be all strings or all integers, or :meth:`save` fails.

    World numbers start at 0. After :meth:`set_world_count`, a world
    outside ``range(count)`` is refused when recorded or selected. The
    methods that take ``worlds`` read the rows of those worlds only.
    Without a selection, :meth:`summary` and :meth:`save` read every row
    and :meth:`exclusion_groups` the worlds counted.
    """

    def __init__(
        self,
        seed: int,
        *,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        """
        Initialize an empty report.

        Args:
            seed (int):
                The seed the run drew from. A run is repeated by passing
                this value back.
            settings (Mapping[str, object] or None):
                The settings the run resolved, echoed into
                :meth:`summary`. ``None`` records no settings.

        Raises:
            TypeError:
                If ``seed`` is not an integer.
        """
        self.seed = operator.index(seed)
        self.settings: dict[str, object] = dict(settings or {})
        self._rows: list[tuple] = []
        self._exclusion_rows: list[tuple] = []
        self._notices: list[str] = []
        self._coverage: dict[str, float] = {}
        self._certificate: object = None
        self._timings: dict[tuple[str, int | None], float] = {}
        self._computed: int | None = None
        self._stages: list[str] = []

    def record_action(
        self,
        asset_id: str | int,
        feature: str,
        *,
        stage: str,
        action: str,
        world: int,
        old: object = None,
        new: object = None,
        source: str | None = None,
        reason: str = '',
        block_group: str | int | None = None,
    ) -> None:
        """
        Record one action taken on one feature of one asset.

        Args:
            asset_id (str or int):
                The asset the action was taken on.
            feature (str):
                The feature the action was taken on.
            stage (str):
                The stage that took the action, one of :data:`STAGES`.
            action (str):
                What was done to the feature, one of :data:`ACTIONS`.
            world (int):
                The world the action belongs to.
            old (object):
                The value the feature held. ``None`` records no value;
                any other value, NaN included, is recorded as its text.
            new (object):
                The value the feature holds after the action, recorded
                as ``old`` is.
            source (str or None):
                The source the action concerns, a key of
                :data:`~brails.types.asset_inventory.SOURCE_CODES`: the
                source a withheld value was held by, or the one ``new``
                was written to. ``None`` records no source.
            reason (str):
                What prompted the action, in words.
            block_group (str or int or None):
                The asset's ``BLOCKGROUP_GEOID``, recorded as text.
                ``None`` records no block group.

        Raises:
            TypeError:
                If ``world`` is not an integer.
            ValueError:
                If the stage, the action or the source is outside its
                set, or the world is outside the run's worlds.
        """
        self._rows.append(
            (
                asset_id,
                self._checked_world(world),
                _in_set(stage, STAGES, 'stage'),
                _in_set(action, ACTIONS, 'action'),
                feature,
                None if old is None else str(old),
                None if new is None else str(new),
                None if source is None else _in_set(source, SOURCE_CODES, 'source'),
                reason,
                None if block_group is None else str(block_group),
            )
        )

    def record_exclusion(
        self,
        asset_id: str | int,
        *,
        world: int,
        stage: str,
        category: str,
        diagnosis: str = '',
        detail: str = '',
        resolvable_by: Iterable[str] = (),
    ) -> None:
        """
        Record that an asset left the inventory of one world.

        An asset that leaves several worlds has one row per world.

        Args:
            asset_id (str or int):
                The excluded asset.
            world (int):
                The world it was excluded from.
            stage (str):
                The stage that excluded it, one of :data:`STAGES`.
            category (str):
                Why it was excluded, one of
                :data:`EXCLUSION_CATEGORIES`.
            diagnosis (str):
                The verdict an inferer reached on the asset.
            detail (str):
                What happened, in words.
            resolvable_by (Iterable[str]):
                The ``resolvable_by`` of the refusal's diagnosis
                (:data:`~brails.inferers.inference_diagnosis.DIAGNOSIS_COLUMNS`).

        Raises:
            TypeError:
                If ``world`` is not an integer.
            ValueError:
                If the stage or the category is outside its set, or the
                world is outside the run's worlds.
        """
        self._exclusion_rows.append(
            (
                asset_id,
                self._checked_world(world),
                _in_set(stage, STAGES, 'stage'),
                _in_set(category, EXCLUSION_CATEGORIES, 'exclusion category'),
                diagnosis,
                detail,
                tuple(resolvable_by),
            )
        )

    def note(self, text: str) -> None:
        """
        Record a notice about the run, such as a contradiction it met.

        Args:
            text (str):
                The notice.
        """
        self._notices.append(text)

    def set_coverage(self, mapping: Mapping[str, float]) -> None:
        """
        Record how much of the inventory each driver covers.

        Args:
            mapping (Mapping[str, float]):
                Driver name to the fraction of assets that state it.
        """
        self._coverage = dict(mapping)

    def set_certificate(self, verdict: object) -> None:
        """
        Record the verdict of the input-schema check on the run.

        Args:
            verdict (object):
                The check's result, echoed into :meth:`summary`.
        """
        self._certificate = verdict

    def set_timing(
        self, stage: str, seconds: float, *, world: int | None = None
    ) -> None:
        """
        Record the seconds the run spent on one stage, in one world or once.

        Setting the same stage and world again replaces the earlier value.

        Args:
            stage (str):
                The caller's name for the timed part of the run, such as
                a frozen stage or the manifest. It is not checked against
                :data:`STAGES`.
            seconds (float):
                The wall-clock seconds spent.
            world (int or None):
                The world the seconds belong to. ``None`` for a stage run
                once for every world.

        Raises:
            TypeError:
                If ``world`` is neither ``None`` nor an integer.
            ValueError:
                If the world is outside the run's worlds.
        """
        key = (stage, None if world is None else self._checked_world(world))
        self._timings[key] = float(seconds)

    def set_world_count(self, count: int) -> None:
        """
        Record how many worlds the run computed.

        Args:
            count (int):
                The number of worlds, numbered 0 to ``count - 1``.

        Raises:
            TypeError:
                If ``count`` is not an integer.
            ValueError:
                If ``count`` is below one, or a world already recorded is
                not below it.
        """
        count = operator.index(count)
        if count < 1:
            raise ValueError(f'The world count must be at least 1, got {count}.')
        recorded = [row[1] for row in self._rows + self._exclusion_rows]
        recorded += [world for _, world in self._timings if world is not None]
        if recorded and max(recorded) >= count:
            raise ValueError(
                f'World {max(recorded)} is already recorded, outside a world '
                f'count of {count}.'
            )
        self._computed = count

    def set_manifest_stages(self, stages: Iterable[str]) -> None:
        """
        Record the run's stages in the order of its manifest.

        Args:
            stages (Iterable[str]):
                The stage names, echoed into :meth:`summary`.
        """
        self._stages = list(stages)

    @property
    def records(self) -> pd.DataFrame:
        """
        Return the recorded actions of every world, one row each.

        Returns:
            pd.DataFrame:
                A frame with the columns of :data:`RECORD_COLUMNS`, in
                that order. A copy, so changing it leaves the report
                alone.
        """
        # Built as objects first, so that pandas keeps None as None.
        return pd.DataFrame(
            self._rows, columns=list(RECORD_COLUMNS), dtype=object
        ).astype(RECORD_DTYPES)

    @property
    def exclusions(self) -> pd.DataFrame:
        """
        Return the exclusions of every world, one row each.

        Returns:
            pd.DataFrame:
                A frame with the columns of :data:`EXCLUSION_COLUMNS`,
                in that order. A copy, so changing it leaves the report
                alone.
        """
        return pd.DataFrame(
            self._exclusion_rows, columns=list(EXCLUSION_COLUMNS), dtype=object
        ).astype(EXCLUSION_DTYPES)

    def exclusion_groups(
        self, *, worlds: Iterable[int] | None = None
    ) -> pd.DataFrame:
        """
        Return the exclusion rows of the selected worlds, each marked by group.

        A row of category ``'dropped'`` is in the group ``'dropped'``. The
        other rows are grouped by asset: an asset is excluded in a world
        where it has at least one such row for that world, and its rows
        are in the group ``'every'`` when that holds in every selected
        world, and in ``'some'`` otherwise.

        Args:
            worlds (Iterable[int] or None):
                The worlds to read. ``None`` reads the worlds counted by
                :meth:`set_world_count`.

        Returns:
            pd.DataFrame:
                The columns of :data:`EXCLUSION_COLUMNS` followed by
                ``group``, ``'dropped'``, ``'every'`` or ``'some'``. Rows
                are in the order they were recorded, so each asset's
                reason in each world is its own row.

        Raises:
            ValueError:
                If ``worlds`` is ``None`` and :meth:`set_world_count` has
                not been called, or a selected world is outside the run's
                worlds.
        """
        selected = self._selected_worlds(worlds)
        if selected is None:
            raise ValueError(
                'The worlds are unknown: pass worlds or call set_world_count first.'
            )
        frame = self.exclusions
        frame = frame[frame['world'].isin(selected)].reset_index(drop=True)
        dropped = frame['category'] == 'dropped'
        worlds_per_asset = (
            frame[~dropped].groupby('asset_id', sort=False)['world'].nunique()
        )
        frame['group'] = (
            frame['asset_id']
            .map(worlds_per_asset == len(selected))
            .map({True: 'every', False: 'some'})
            .astype('object')
        )
        frame.loc[dropped, 'group'] = 'dropped'
        return frame

    def summary(self, *, worlds: Iterable[int] | None = None) -> dict[str, object]:
        """
        Summarize the run as a plain dictionary.

        Args:
            worlds (Iterable[int] or None):
                The worlds whose actions and exclusions are counted.
                ``None`` counts every row. The timings are those of the
                whole run either way.

        Returns:
            dict[str, object]:
                The summary, with these keys:

                - ``seed`` and ``settings``, as given to the constructor.
                - ``worlds``: ``computed``, the count given to
                  :meth:`set_world_count` or ``None``, and ``written``,
                  the selected worlds in ascending order, so position
                  ``i`` of a file holding them is world ``written[i]``
                  (every computed world without a selection, ``None``
                  when neither is known).
                - ``manifest_stages``: the manifest's order.
                - ``timings``: one ``{'stage', 'world', 'seconds'}`` per
                  stage and world, in the order first set.
                - ``actions``: rows counted ``by_action`` and
                  ``by_feature``.
                - ``exclusions``: rows counted ``by_category``,
                  ``by_diagnosis``, ``by_world`` (world to category to
                  count) and ``total``.
                - ``assets_excluded``: distinct assets counted as
                  ``total`` and per group of :meth:`exclusion_groups` as
                  ``dropped``, ``every`` and ``some``, the last two
                  ``None`` when the world set is unknown.
                - ``notices``, ``coverage`` and ``certificate``.

                Counts are keyed in sorted order. JSON can serialize every
                value except a setting or verdict the caller gave in
                another form, and it writes the world keys of
                ``by_world`` as text.

        Raises:
            ValueError:
                If a selected world is outside the run's worlds.
        """
        selected = self._selected_worlds(worlds)
        records, exclusions = self.records, self.exclusions
        if worlds is not None:
            records = records[records['world'].isin(selected)]
            exclusions = exclusions[exclusions['world'].isin(selected)]

        by_world: dict[int, dict[str, int]] = {}
        for (world, category), count in sorted(
            Counter(zip(exclusions['world'], exclusions['category'])).items()
        ):
            by_world.setdefault(int(world), {})[category] = count

        every = some = None
        if selected is not None:
            groups = self.exclusion_groups(worlds=selected)
            groups = groups.drop_duplicates(['asset_id', 'group'])['group']
            every = int((groups == 'every').sum())
            some = int((groups == 'some').sum())
        dropped = exclusions.loc[exclusions['category'] == 'dropped', 'asset_id']

        return {
            'seed': self.seed,
            'settings': dict(self.settings),
            'worlds': {'computed': self._computed, 'written': selected},
            'manifest_stages': list(self._stages),
            'timings': [
                {'stage': stage, 'world': world, 'seconds': seconds}
                for (stage, world), seconds in self._timings.items()
            ],
            'actions': {
                'by_action': dict(sorted(Counter(records['action']).items())),
                'by_feature': dict(sorted(Counter(records['feature']).items())),
            },
            'exclusions': {
                'by_category': dict(sorted(Counter(exclusions['category']).items())),
                'by_diagnosis': dict(
                    sorted(Counter(exclusions['diagnosis']).items())
                ),
                'by_world': by_world,
                'total': len(exclusions),
            },
            'assets_excluded': {
                'total': int(exclusions['asset_id'].nunique()),
                'dropped': int(dropped.nunique()),
                'every': every,
                'some': some,
            },
            'notices': list(self._notices),
            'coverage': dict(self._coverage),
            'certificate': self._certificate,
        }

    def save(self, path: str | Path, *, worlds: Iterable[int] | None = None) -> None:
        """
        Write the records, the exclusions and the summary from one path.

        The records go to ``path`` as Parquet, and beside it the exclusions
        to the stem, :data:`EXCLUSIONS_SUFFIX` and the suffix of ``path``,
        and the summary to the stem and ``_summary.json``: given
        ``run/report.parquet``, ``run/report_exclusions.parquet`` and
        ``run/report_summary.json``. A summary value JSON cannot serialize
        is written as its text form.

        Args:
            path (str or Path):
                Where the records go. Its parent directory is created if
                it is absent.
            worlds (Iterable[int] or None):
                The worlds whose rows are written, with the summary of
                :meth:`summary` for the same selection. ``None`` writes
                every row.

        Raises:
            ValueError:
                If a selected world is outside the run's worlds.
        """
        path = Path(path)
        selected = None if worlds is None else self._selected_worlds(worlds)
        path.parent.mkdir(parents=True, exist_ok=True)

        records = self.records
        exclusions = self.exclusions
        if selected is not None:
            records = records[records['world'].isin(selected)]
            exclusions = exclusions[exclusions['world'].isin(selected)]
        records.to_parquet(path, index=False)
        exclusions.to_parquet(
            path.with_name(f'{path.stem}{EXCLUSIONS_SUFFIX}{path.suffix}'),
            index=False,
        )
        path.with_name(f'{path.stem}_summary.json').write_text(
            json.dumps(self.summary(worlds=selected), indent=2, default=str),
            encoding='utf-8',
        )

    def save_logs(
        self,
        directory: str | Path,
        stem: str,
        logs: Mapping[str, pd.DataFrame],
        *,
        worlds: Iterable[int] | None = None,
    ) -> None:
        """
        Write each component log as Parquet, named after the log.

        Given the stem ``run`` and a log named ``join``, the file is
        ``<directory>/run_join.parquet``.

        Args:
            directory (str or Path):
                Where the logs go. It is created if it is absent.
            stem (str):
                The prefix of every file name.
            logs (Mapping[str, pd.DataFrame]):
                Log name to the log's frame.
            worlds (Iterable[int] or None):
                The worlds whose rows are written. A log with a ``world``
                column keeps the rows of those worlds. A log without one
                is written whole, so a log that aggregates over worlds
                carries every computed world into a selection. ``None``
                writes every row.

        Raises:
            ValueError:
                If a log name is empty, holds a path separator, or is
                ``report``, alone or followed by :data:`EXCLUSIONS_SUFFIX`,
                whose files would
                overwrite those of a report saved as
                ``<stem>_report.parquet`` in the same directory; or if a
                selected world is outside the run's worlds.
        """
        directory = Path(directory)
        bad = [name for name in logs if not name or Path(name).name != name]
        if bad:
            raise ValueError(f'Log names must be plain file name parts: {bad}.')
        taken = sorted({'report', f'report{EXCLUSIONS_SUFFIX}'}.intersection(logs))
        if taken:
            raise ValueError(
                f'Log names {taken} would overwrite the files of a report '
                f'saved as {stem}_report.parquet in the same directory; '
                'rename the logs.'
            )
        selected = None if worlds is None else self._selected_worlds(worlds)
        directory.mkdir(parents=True, exist_ok=True)
        for name, log in logs.items():
            kept = log
            if selected is not None and 'world' in log.columns:
                kept = log[log['world'].isin(selected)]
            kept.to_parquet(directory / f'{stem}_{name}.parquet', index=False)

    def _checked_world(self, world: int) -> int:
        """
        Check a world number against the run's worlds.

        Args:
            world (int):
                The world number.

        Returns:
            int:
                The world number as a Python integer.

        Raises:
            TypeError:
                If ``world`` is not an integer.
            ValueError:
                If ``world`` is negative, or not below the count given to
                :meth:`set_world_count`.
        """
        world = operator.index(world)
        if world < 0:
            raise ValueError(f'A world number cannot be negative, got {world}.')
        if self._computed is not None and world >= self._computed:
            raise ValueError(
                f'World {world} is outside the {self._computed} worlds of the run.'
            )
        return world

    def _selected_worlds(self, worlds: Iterable[int] | None) -> list[int] | None:
        """
        Resolve a selection of worlds against the run's worlds.

        Args:
            worlds (Iterable[int] or None):
                The caller's selection, or ``None``.

        Returns:
            list[int] or None:
                The distinct selected worlds in ascending order. Without
                a selection, every computed world, or ``None`` when
                :meth:`set_world_count` has not been called.

        Raises:
            TypeError:
                If a selected world is not an integer.
            ValueError:
                If a selected world is outside the run's worlds.
        """
        if worlds is not None:
            return sorted({self._checked_world(world) for world in worlds})
        if self._computed is None:
            return None
        return list(range(self._computed))

    def __repr__(self) -> str:
        """
        Summarize the report.

        Returns:
            str:
                The seed, the number of recorded actions, and the number
                of distinct excluded assets.
        """
        return (
            f'<CompilationReport seed={self.seed}: '
            f'{len(self._rows)} actions, '
            f'{len({row[0] for row in self._exclusion_rows})} assets excluded>'
        )
