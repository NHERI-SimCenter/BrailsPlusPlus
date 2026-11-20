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
Record what a compilation did to each asset in each world.

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

# What a step did to a feature: ``withheld``, a value kept out of a draw or
# removed from its source; ``restored``, a withheld value put back after a
# failed attempt; ``redrawn``, a value a new draw replaced; ``derived``, a
# value computed from other features of the asset. The README beside this
# module gives examples.
ACTIONS = ('withheld', 'restored', 'redrawn', 'derived')

# The exclusion category every report accepts besides its compiler's: a
# footprint a step removes in every world. The report counts these
# footprints in a group of their own, apart from the assets excluded in
# every world and from those excluded in some.
DROPPED_CATEGORY = 'dropped'

# The diagnosis of an exclusion that no component refused, where a step that
# keeps some assets and drops others did not keep the asset, as the unit
# budget of the Hazus earthquake compiler drops a footprint it makes no
# dwelling. No check returns it, so it is not one of the verdicts of
# :mod:`~brails.inferers.inference_diagnosis`.
DIAGNOSIS_NOT_SELECTED = 'not_selected'

# The exclusion categories the Hazus earthquake compiler and its resolution
# loop file under, besides DROPPED_CATEGORY. The section "Exclusion
# categories" of the compilers' README defines each.
MISSING_INPUT_CATEGORY = 'missing_input'
UNSUPPORTED_CATEGORY = 'unsupported'
CONFLICT_CATEGORY = 'conflict'
NO_PLAUSIBLE_VALUE_CATEGORY = 'no_plausible_value'
UNDETERMINED_CATEGORY = 'undetermined'
FAILS_SCHEMA_CATEGORY = 'fails_schema'

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
}

# The columns of the frame of action rows, in order.
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


def _check_in_set(value: str, valid: Collection[str], kind: str) -> str:
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
            ``value``, unchanged.

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
    Record what a compilation did to each asset, world by world.

    A report holds one row per action taken on a feature and one row per
    asset excluded from a world, with the run's timings, notices, input
    coverage and schema check. The section "The compilation report" of the
    README beside this module states what the rows hold and what
    :meth:`summary` counts.

    A row names one of the steps the compiler gave the constructor, and an
    exclusion one of its categories. Its action must be one of
    :data:`ACTIONS` and its source a key of
    :data:`~brails.types.asset_inventory.SOURCE_CODES`. A value outside its
    set is refused. Asset ids must be all strings or all integers, or
    :meth:`save` fails. After :meth:`set_world_count`, a world outside
    ``range(count)`` is refused when recorded or selected.

    Attributes:
        stages (tuple[str, ...]):
            The names of the compiler's steps, in the order given. They are
            not the names of frozen stages.
        exclusion_categories (tuple[str, ...]):
            The compiler's categories, in the order given, followed by
            :data:`DROPPED_CATEGORY`.
    """

    def __init__(
        self,
        seed: int,
        *,
        stages: Iterable[str],
        exclusion_categories: Iterable[str],
        settings: Mapping[str, object] | None = None,
    ) -> None:
        """
        Initialize an empty report for a compiler's steps and categories.

        Args:
            seed (int):
                The seed the run drew from. A run is repeated by passing
                this value back.
            stages (Iterable[str]):
                The names of the compiler's steps, one of which every action
                and exclusion names, each once.
            exclusion_categories (Iterable[str]):
                The reasons the compiler files exclusions under, each once,
                besides :data:`DROPPED_CATEGORY`, which every report accepts.
            settings (Mapping[str, object] or None):
                The settings the run resolved, echoed into
                :meth:`summary`. ``None`` records no settings.

        Raises:
            TypeError:
                If ``seed`` is not an integer, or ``stages`` or
                ``exclusion_categories`` is a string.
            ValueError:
                If ``stages`` or ``exclusion_categories`` names one value
                twice, or ``exclusion_categories`` holds
                :data:`DROPPED_CATEGORY`.
        """
        self.seed = operator.index(seed)
        named = {}
        for argument, given in (
            ('stages', stages),
            ('exclusion_categories', exclusion_categories),
        ):
            if isinstance(given, str):
                msg = f'{argument} must be a sequence of names, not a string.'
                raise TypeError(msg)
            names = tuple(given)
            if len(set(names)) < len(names):
                msg = f'{argument} names a value twice: {list(names)}.'
                raise ValueError(msg)
            named[argument] = names
        self.stages = named['stages']
        categories = named['exclusion_categories']
        if DROPPED_CATEGORY in categories:
            msg = (
                f'{DROPPED_CATEGORY!r} is accepted by every report and must not '
                'be listed in exclusion_categories.'
            )
            raise ValueError(msg)
        # Every category the report accepts, the compiler's and its own.
        self.exclusion_categories = (*categories, DROPPED_CATEGORY)
        self.settings: dict[str, object] = dict(settings or {})
        self._rows: list[tuple] = []
        self._exclusion_rows: list[tuple] = []
        self._notices: list[str] = []
        self._coverage: dict[str, float] = {}
        self._schema_check: object = None
        self._schema_refusals: pd.DataFrame | None = None
        self._timings: dict[tuple[str, int | None], float] = {}
        self._computed: int | None = None
        self._manifest_stages: list[str] = []
        self._inside: frozenset[str | int] | None = None
        self._footprints = 0

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
    ) -> None:
        """
        Record one action taken on one feature of one asset.

        Args:
            asset_id (str or int):
                The asset the action was taken on.
            feature (str):
                The feature the action was taken on.
            stage (str):
                The step that took the action, one of the names in
                :attr:`stages`.
            action (str):
                What was done to the feature, one of :data:`ACTIONS`.
            world (int):
                The index of the world the action was taken in.
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
                _check_in_set(stage, self.stages, 'stage'),
                _check_in_set(action, ACTIONS, 'action'),
                feature,
                None if old is None else str(old),
                None if new is None else str(new),
                None
                if source is None
                else _check_in_set(source, SOURCE_CODES, 'source'),
                reason,
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
                The id of the excluded asset.
            world (int):
                The index of the world it was excluded from.
            stage (str):
                The step that excluded it, one of the names in
                :attr:`stages`.
            category (str):
                Why it was excluded, one of :attr:`exclusion_categories`.
            diagnosis (str):
                What happened, one of the verdicts of
                :mod:`~brails.inferers.inference_diagnosis`, such as
                ``'infeasible'``, or :data:`DIAGNOSIS_NOT_SELECTED`. It is
                the verdict a check returned, where an inferer, a sampler or
                the composed screen refused the asset. Where a component
                refused it without a verdict, such as a schema validator or
                an imputer's outcome, the compiler names the verdict that
                describes the refusal. Where no component refused it, the
                compiler gives the verdict or label that describes what
                happened. Defaults to ``''``, for a compiler that gives none.
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
                _check_in_set(stage, self.stages, 'stage'),
                _check_in_set(
                    category, self.exclusion_categories, 'exclusion category'
                ),
                diagnosis,
                detail,
                tuple(resolvable_by),
            )
        )

    def note(self, text: str) -> None:
        """
        Record a notice about the run, such as a count of missing default values.

        Args:
            text (str):
                The notice, as :meth:`summary` lists it.
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

    def set_schema_check(
        self, verdict: object, *, refused: pd.DataFrame | None = None
    ) -> None:
        """
        Store the result of checking buildings against a loss model's input schema.

        An input schema states the features a damage and loss model reads
        and the values it accepts. :meth:`summary` returns the stored result
        as ``schema_check``.

        Args:
            verdict (object):
                The result of the check, stored as given, such as the
                schema's dataset and the number of buildings it refused in
                each world.
            refused (pandas.DataFrame, optional):
                The buildings the check refused, with an ``asset_id`` and a
                ``world`` column. Given, the verdict is a mapping, to which
                :meth:`summary` adds ``excluded_by_world``: the number of the
                buildings it reads that the check refused in each world.
        """
        self._schema_check = verdict
        self._schema_refusals = refused

    def set_timing(
        self, stage: str, seconds: float, *, world: int | None = None
    ) -> None:
        """
        Record the seconds spent on one named piece of work, in one world or once.

        Setting the same name and world again replaces the earlier value.

        Args:
            stage (str):
                The caller's name for the timed part of the run, such as
                a frozen stage or the manifest. It is not checked against
                the names in :attr:`stages`.
            seconds (float):
                The wall-clock seconds spent.
            world (int or None):
                The world the seconds belong to. ``None`` for work done
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

    def set_boundary(self, inside: Iterable[str | int], footprints: int) -> None:
        """
        Name the assets inside a boundary, so that the report reads only their rows.

        After this call, :meth:`summary`, :meth:`save` and :meth:`exclusion_groups`
        read the rows of the assets inside unless given ``keep_outside``, and the
        summary counts the assets inside and around the boundary.

        Args:
            inside (Iterable[str or int]):
                The ids of the assets inside the boundary.
            footprints (int):
                The number of assets compiled, inside the boundary and around it.

        Raises:
            ValueError:
                If ``footprints`` is less than the number of assets inside.
        """
        inside = frozenset(inside)
        footprints = operator.index(footprints)
        if footprints < len(inside):
            msg = (
                f'footprints is {footprints}, fewer than the {len(inside)} assets '
                'inside the boundary.'
            )
            raise ValueError(msg)
        self._inside = inside
        self._footprints = footprints

    def set_manifest_stages(self, stages: Iterable[str]) -> None:
        """
        Record the run's stages in the order of its manifest.

        Args:
            stages (Iterable[str]):
                The names of the frozen stages, echoed into :meth:`summary`.
        """
        self._manifest_stages = list(stages)

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
        self, *, worlds: Iterable[int] | None = None, keep_outside: bool = False
    ) -> pd.DataFrame:
        """
        Return the exclusion rows of the selected worlds, each with its group.

        A row of :data:`DROPPED_CATEGORY` is in the group ``'dropped'``. The
        other rows are grouped by asset: an asset is excluded in a world
        where it has at least one such row for that world, and its rows
        are in the group ``'every'`` when that holds in every selected
        world, and in ``'some'`` otherwise.

        Args:
            worlds (Iterable[int] or None):
                The worlds to read. ``None`` reads the worlds counted by
                :meth:`set_world_count`.
            keep_outside (bool, optional):
                Whether to read the rows of the assets outside the boundary of
                :meth:`set_boundary` too. Defaults to ``False``.

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
        frame = self._select_inside(self.exclusions, keep_outside=keep_outside)
        frame = frame[frame['world'].isin(selected)].reset_index(drop=True)
        dropped = frame['category'] == DROPPED_CATEGORY
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

    def summary(
        self, *, worlds: Iterable[int] | None = None, keep_outside: bool = False
    ) -> dict[str, object]:
        """
        Summarize the run as a plain dictionary.

        Args:
            worlds (Iterable[int] or None):
                The worlds whose actions and exclusions are counted.
                ``None`` counts every row. The timings are those of the
                whole run either way.
            keep_outside (bool, optional):
                Whether to count the actions, exclusions and schema refusals of
                the assets outside the boundary of :meth:`set_boundary` too.
                Defaults to ``False``.

        Returns:
            dict[str, object]:
                The summary. The section "The summary" of the README beside
                this module lists its keys. Counts are keyed in sorted
                order.

        Raises:
            ValueError:
                If a selected world is outside the run's worlds.
        """
        selected = self._selected_worlds(worlds)
        records, exclusions = (
            self._select_inside(frame, keep_outside=keep_outside)
            for frame in (self.records, self.exclusions)
        )
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
            groups = self.exclusion_groups(
                worlds=selected, keep_outside=keep_outside
            )
            groups = groups.drop_duplicates(['asset_id', 'group'])['group']
            every = int((groups == 'every').sum())
            some = int((groups == 'some').sum())
        dropped = exclusions.loc[
            exclusions['category'] == DROPPED_CATEGORY, 'asset_id'
        ]

        notices = list(self._notices)
        if self._inside is not None and not keep_outside:
            notices.append(
                'The actions, exclusions and schema check count the '
                f'{len(self._inside)} assets inside the boundary. The counts in the '
                'notices cover the whole area compiled unless a notice names the '
                'inside.'
            )
        schema_check = self._schema_check
        if self._schema_refusals is not None:
            refused = self._select_inside(
                self._schema_refusals, keep_outside=keep_outside
            )
            schema_check = {
                **schema_check,
                'excluded_by_world': {
                    world: int(
                        refused.loc[refused['world'] == world, 'asset_id'].nunique()
                    )
                    for world in range(self._computed or 0)
                },
            }
        summary = {
            'seed': self.seed,
            'settings': dict(self.settings),
            'worlds': {'computed': self._computed, 'written': selected},
            'manifest_stages': list(self._manifest_stages),
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
            'notices': notices,
            'coverage': dict(self._coverage),
            'schema_check': schema_check,
        }
        if self._inside is not None:
            summary['boundary'] = {
                'footprints': self._footprints,
                'inside': len(self._inside),
                'outside': self._footprints - len(self._inside),
            }
        return summary

    def save(
        self,
        path: str | Path,
        *,
        worlds: Iterable[int] | None = None,
        actions: bool = True,
        keep_outside: bool = False,
    ) -> None:
        """
        Write the report's rows and summary from one path.

        The action rows go to ``path`` as Parquet, and beside it the exclusions
        to the stem, :data:`EXCLUSIONS_SUFFIX` and the suffix of ``path``,
        and the summary to the stem and ``_summary.json``: given
        ``run/report.parquet``, ``run/report_exclusions.parquet`` and
        ``run/report_summary.json``. A summary value JSON cannot serialize
        is written as its text form. Without ``actions``, the file at
        ``path`` is not written, and the other two are written as named
        from it.

        Args:
            path (str or Path):
                Where the action rows go. Its parent directory is created if
                it is absent.
            worlds (Iterable[int] or None):
                The worlds whose rows are written, with the summary of
                :meth:`summary` for the same selection. ``None`` writes
                every row.
            actions (bool, optional):
                Whether to write the action rows. Defaults to ``True``.
            keep_outside (bool, optional):
                Whether to write and count the rows of the assets outside the
                boundary of :meth:`set_boundary` too. Defaults to ``False``.

        Raises:
            ValueError:
                If a selected world is outside the run's worlds.
        """
        path = Path(path)
        selected = None if worlds is None else self._selected_worlds(worlds)
        path.parent.mkdir(parents=True, exist_ok=True)

        exclusions = self._select_inside(self.exclusions, keep_outside=keep_outside)
        if selected is not None:
            exclusions = exclusions[exclusions['world'].isin(selected)]
        if actions:
            records = self._select_inside(self.records, keep_outside=keep_outside)
            if selected is not None:
                records = records[records['world'].isin(selected)]
            records.to_parquet(path, index=False)
        exclusions.to_parquet(
            path.with_name(f'{path.stem}{EXCLUSIONS_SUFFIX}{path.suffix}'),
            index=False,
        )
        path.with_name(f'{path.stem}_summary.json').write_text(
            json.dumps(
                self.summary(worlds=selected, keep_outside=keep_outside),
                indent=2,
                default=str,
            ),
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

    def _select_inside(
        self, frame: pd.DataFrame, *, keep_outside: bool
    ) -> pd.DataFrame:
        """
        Keep the rows of the assets inside the boundary of :meth:`set_boundary`.

        Args:
            frame (pd.DataFrame):
                Action or exclusion rows, with an ``asset_id`` column.
            keep_outside (bool):
                Whether to keep the rows of the assets outside the boundary too.

        Returns:
            pd.DataFrame:
                ``frame`` itself without a boundary or with ``keep_outside``, and
                otherwise its rows of the assets inside.
        """
        if self._inside is None or keep_outside:
            return frame
        return frame[frame['asset_id'].isin(self._inside)]

    def _checked_world(self, world: int) -> int:
        """
        Check a world number against the run's worlds.

        Args:
            world (int):
                A world index to check.

        Returns:
            int:
                The world number as a Python integer.

        Raises:
            TypeError:
                If ``world`` is a bool or not an integer.
            ValueError:
                If ``world`` is negative, or not below the count given to
                :meth:`set_world_count`.
        """
        if isinstance(world, bool) or not hasattr(world, '__index__'):
            msg = f'A world number must be an integer, not {world!r}.'
            raise TypeError(msg)
        world = operator.index(world)
        if self._computed is not None and not 0 <= world < self._computed:
            raise ValueError(
                f'World {world} is outside the {self._computed} worlds of the '
                f'run, 0 to {self._computed - 1}.'
            )
        if world < 0:
            raise ValueError(f'A world number cannot be negative, got {world}.')
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
