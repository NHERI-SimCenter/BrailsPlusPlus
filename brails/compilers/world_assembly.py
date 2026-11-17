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
Compose one world's sources from frozen stages, and assemble worlds.

.. autosummary::

      compose_source
      WorldAssembler
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from copy import copy
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

from brails.types.asset_inventory import (
    MAX_REPORTED_WORLD_PROBLEMS,
    AssetInventory,
    _merge_id_order,
    _report_first_problems,
    is_missing_value,
    is_probabilistic_feature,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from brails.types.asset_inventory import Asset
    from brails.utils.stage_freeze import StageFreeze


def compose_source(  # noqa: C901
    freeze: StageFreeze,
    parts: Sequence[str],
    *,
    world: int,
    withheld: Sequence[str] = (),
    label: str,
    dropped: str | None = None,
    excluded: AbstractSet[str] = frozenset(),
    excluded_parts: Sequence[str] = (),
) -> AssetInventory:
    """
    Compose one source of one world from frozen stages.

    The source is the union of ``parts`` in ``world``, less the assets of
    the ``dropped`` record, the values ``excluded`` assets take from
    ``excluded_parts``, and the values a ``withheld`` record withholds from
    ``label``. Every stage is read from ``freeze`` by name. At most one part
    may hold a value for a feature of an asset. An asset takes the
    coordinates of the first part that holds it, and a feature no part
    holds a value for keeps the first part's missing value. The result may
    share records and shares values with the frozen stages, so a caller
    must not change it:
    :meth:`~brails.utils.stage_freeze.StageFreeze.manifest` reports a
    frozen stage changed that way. The section "Composition and assembly"
    of the README beside this module states the procedure.

    Args:
        freeze (StageFreeze):
            The freeze holding the stages.
        parts (Sequence[str]):
            Names of the stages whose values the source unites, each once.
        world (int):
            The world composed, from 0.
        withheld (Sequence[str], optional):
            Names of records of withheld values. A record holds, per asset,
            one feature per withheld value, whose value is the label of the
            source the value is withheld from. Defaults to no records.
        label (str):
            The label of the source composed, which a withheld record
            names.
        dropped (str, optional):
            Name of a record holding the assets to leave out. Defaults to
            ``None``, no record.
        excluded (AbstractSet[str], optional):
            Ids of the assets that take no value from ``excluded_parts``,
            each held by a part or by the ``dropped`` record. Defaults to
            the empty set.
        excluded_parts (Sequence[str], optional):
            Names, among ``parts``, of the parts whose values the assets in
            ``excluded`` do not take. Defaults to none.

    Returns:
        AssetInventory:
            The source in ``world``, with one possible world. It declares
            the vector features of every part.

    Raises:
        TypeError:
            If ``parts``, ``withheld`` or ``excluded_parts`` is a string or
            not a sequence, a stage name is not a string, ``label`` is not
            a string, ``world`` is not an integer, or ``excluded`` is not a
            set.
        LookupError:
            If a named stage is not frozen.
        ValueError:
            If ``parts`` is empty or names a stage twice; if
            ``excluded_parts`` names a stage that is not among ``parts``, or
            ``excluded`` holds ids while ``excluded_parts`` is empty; if
            ``world`` is negative, or not below the number of possible
            worlds of a part or withheld record that holds more than one; if
            ``excluded`` holds an id that no part and no ``dropped`` record
            holds; or if two parts hold a value for one feature of one asset
            in ``world``. A message names up to
            :data:`~brails.types.asset_inventory.MAX_REPORTED_WORLD_PROBLEMS`
            assets or features.
    """
    for argument, names in (
        ('parts', parts),
        ('withheld', withheld),
        ('excluded_parts', excluded_parts),
    ):
        if isinstance(names, str) or not isinstance(names, Sequence):
            msg = (
                f'{argument} must be a sequence of stage names, not '
                f'{type(names).__name__}.'
            )
            raise TypeError(msg)
    named = [*parts, *withheld, *([] if dropped is None else [dropped])]
    for name in (*named, *excluded_parts):
        if not isinstance(name, str):
            msg = (
                'compose_source reads frozen stages by name only, and a stage '
                f'name must be a string, not {type(name).__name__}.'
            )
            raise TypeError(msg)
    if not isinstance(label, str):
        msg = f'label must be a string, not {type(label).__name__}.'
        raise TypeError(msg)
    if isinstance(world, bool) or not isinstance(world, Integral):
        msg = f'world must be an integer, not {type(world).__name__}.'
        raise TypeError(msg)
    world = int(world)
    if not isinstance(excluded, AbstractSet):
        msg = f'excluded must be a set of asset ids, not {type(excluded).__name__}.'
        raise TypeError(msg)
    if not parts or len(set(parts)) < len(parts):
        msg = f'parts must name at least one stage, each once, not {list(parts)}.'
        raise ValueError(msg)
    if not set(excluded_parts) <= set(parts):
        msg = (
            f'excluded_parts {list(excluded_parts)} must be among the parts '
            f'{list(parts)}.'
        )
        raise ValueError(msg)
    if excluded and not excluded_parts:
        msg = 'excluded holds assets, but excluded_parts names no part.'
        raise ValueError(msg)
    if world < 0:
        msg = f'world must not be negative, not {world}.'
        raise ValueError(msg)
    stages = {name: freeze.read(name) for name in named}
    for name in (*parts, *withheld):
        if world >= stages[name].n_pw > 1:
            msg = (
                f'Stage {name!r} holds {stages[name].n_pw} worlds, so it has '
                f'no world {world}.'
            )
            raise ValueError(msg)
    dropped_ids = () if dropped is None else stages[dropped].inventory.keys()

    vector_features: dict[str, None] = {}
    records: dict[Hashable, Asset] = {}
    # The feature dicts this composition made, which it may change.
    owned: dict[Hashable, dict[str, Any]] = {}

    def writable(asset_id: Hashable) -> dict[str, Any]:
        """
        Return an asset's features in the result, copied on the first write.

        Args:
            asset_id (Hashable):
                An asset of the result.

        Returns:
            dict[str, Any]:
                The features, held by this composition alone.
        """
        if asset_id not in owned:
            record = copy(records[asset_id])
            record.features = dict(record.features)
            records[asset_id] = record
            owned[asset_id] = record.features
        return owned[asset_id]

    def realized(stage: AssetInventory, asset: Asset) -> dict[str, Any]:
        """
        Return an asset's features in ``world``.

        Args:
            stage (AssetInventory):
                The stage holding the asset.
            asset (Asset):
                The asset.

        Returns:
            dict[str, Any]:
                The asset's own features where no value is a list of
                worlds, and otherwise a new dict of their values in
                ``world``.
        """
        if stage.n_pw == 1 or not any(
            isinstance(value, list) for value in asset.features.values()
        ):
            return asset.features
        return {
            key: value[world]
            if is_probabilistic_feature(key, value, stage.vector_features)
            else value
            for key, value in asset.features.items()
        }

    conflicts: list[str] = []
    n_conflicts = 0
    for name in parts:
        stage = stages[name]
        leaving = excluded if name in excluded_parts else ()
        vector_features.update(dict.fromkeys(stage.vector_features))
        for asset_id, asset in stage.inventory.items():
            if asset_id in dropped_ids:
                continue
            features = {} if asset_id in leaving else realized(stage, asset)
            if asset_id not in records:
                if features is asset.features:
                    records[asset_id] = asset
                else:
                    records[asset_id] = copy(asset)
                    records[asset_id].features = owned[asset_id] = features
                continue
            current = records[asset_id].features
            for key, value in features.items():
                if key not in current or (
                    is_missing_value(current[key]) and not is_missing_value(value)
                ):
                    current = writable(asset_id)
                    current[key] = value
                elif not is_missing_value(value):
                    n_conflicts += 1
                    if len(conflicts) < MAX_REPORTED_WORLD_PROBLEMS:
                        conflicts.append(
                            f'asset {asset_id!r}, feature {key!r} ({name!r} and '
                            'an earlier part)'
                        )
    if n_conflicts:
        more = n_conflicts - len(conflicts)
        tail = f'; ... and {more} more' if more else ''
        msg = (
            f'{n_conflicts} feature(s) hold a value in more than one of the '
            f'parts {list(parts)} in world {world}: {"; ".join(conflicts)}{tail}.'
        )
        raise ValueError(msg)
    unknown = sorted(
        (
            asset_id
            for asset_id in excluded
            if asset_id not in records and asset_id not in dropped_ids
        ),
        key=_merge_id_order,
    )
    if unknown:
        msg = (
            f'{len(unknown)} excluded asset id(s) are held by no part and no '
            f'dropped record: {_report_first_problems(list(map(repr, unknown)), ", ")}.'
        )
        raise ValueError(msg)

    for name in withheld:
        stage = stages[name]
        for asset_id, asset in stage.inventory.items():
            if asset_id not in records:
                continue
            removed = [
                key
                for key, value in realized(stage, asset).items()
                if isinstance(value, str) and value == label
            ]
            if removed and any(key in records[asset_id].features for key in removed):
                target = writable(asset_id)
                for key in removed:
                    target.pop(key, None)

    result = AssetInventory(vector_features=list(vector_features))
    result.inventory = records
    return result


def _same_value(first: object, second: object) -> bool:
    """
    Check whether two feature values are equal under the assembler's rule.

    Args:
        first (object):
            One value.
        second (object):
            The other value.

    Returns:
        bool:
            ``True`` if the values are equal.
    """
    if first is second:
        return True
    if isinstance(first, float) and isinstance(second, float):
        return first == second or (math.isnan(first) and math.isnan(second))
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        try:
            return bool(np.array_equal(first, second))
        except (TypeError, ValueError):
            return False
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


class WorldAssembler:
    """
    Assemble one-world inventories into one inventory over the worlds.

    Realizations are added at positions in order from 0, each with the
    asset ids and coordinates of position 0, so a caller pads each
    realization to one asset set before adding it. A key absent from a
    realization counts as ``None`` there. The result holds a feature as
    position 0's object where every position agrees and as a list of one
    value per position where they differ, so a realization may hold a list
    only for a vector feature. The result shares coordinates and feature
    values with the realizations added, so a caller must not change a
    realization after adding it. The section "Composition and assembly" of
    the README beside this module states how values are compared.

    Example:
        >>> from brails.types.asset_inventory import Asset, AssetInventory
        >>> assembler = WorldAssembler(2)
        >>> for world, year in enumerate([1950, 1960]):
        ...     realization = AssetInventory()
        ...     features = {'YearBuilt': year, 'OccupancyClass': 'RES1'}
        ...     asset = Asset('a', [[-122.4, 37.7]], features)
        ...     _ = realization.add_asset('a', asset)
        ...     assembler.add(world, realization)
        >>> assembler.result().get_asset_features('a')[1]
        {'YearBuilt': [1950, 1960], 'OccupancyClass': 'RES1'}
    """

    def __init__(self, n_worlds: int) -> None:
        """
        Create an assembler for a number of positions.

        Args:
            n_worlds (int):
                The number of positions the result holds.

        Raises:
            TypeError:
                If ``n_worlds`` is not an integer.
            ValueError:
                If ``n_worlds`` is below 1.
        """
        if isinstance(n_worlds, bool) or not isinstance(n_worlds, Integral):
            msg = f'n_worlds must be an integer, not {type(n_worlds).__name__}.'
            raise TypeError(msg)
        if n_worlds < 1:
            msg = f'n_worlds must be at least 1, not {n_worlds}.'
            raise ValueError(msg)
        self._n_worlds = int(n_worlds)
        self._added = 0
        self._assets: dict[Hashable, Asset] = {}
        # Per asset, the features held as one value per position.
        self._lists: dict[Hashable, set[str]] = {}
        self._vector_features: dict[str, None] = {}
        self._result: AssetInventory | None = None

    def add(self, world: int, realization: AssetInventory) -> None:  # noqa: C901
        """
        Add the realization at the next position.

        Args:
            world (int):
                The world, which is the next position: the number of
                realizations added so far.
            realization (AssetInventory):
                An inventory with one possible world.

        Raises:
            TypeError:
                If ``world`` is not an integer or ``realization`` is not an
                :class:`AssetInventory`.
            ValueError:
                If ``world`` is not the next position; if the realization
                has more than one possible world; if its asset ids or
                coordinates differ from those of position 0; or if a vector
                feature, declared by any realization, whose position 0 value
                is not a list takes a different value at some position. The
                message names up to
                :data:`~brails.types.asset_inventory.MAX_REPORTED_WORLD_PROBLEMS`
                assets. A refused realization changes nothing.
        """
        if isinstance(world, bool) or not isinstance(world, Integral):
            msg = f'world must be an integer, not {type(world).__name__}.'
            raise TypeError(msg)
        if not isinstance(realization, AssetInventory):
            msg = (
                'A realization must be an AssetInventory, not '
                f'{type(realization).__name__}.'
            )
            raise TypeError(msg)
        position = self._added
        if position == self._n_worlds:
            msg = f'All {self._n_worlds} positions have been added.'
            raise ValueError(msg)
        if world != position:
            msg = f'Position {world} was given where position {position} is next.'
            raise ValueError(msg)
        if realization.n_pw != 1:
            msg = f'A realization must have one world, not {realization.n_pw}.'
            raise ValueError(msg)

        if position > 0:
            held, given = self._assets.keys(), realization.inventory.keys()
            missing = sorted(held - given, key=_merge_id_order)
            extra = sorted(given - held, key=_merge_id_order)
            if missing or extra:
                problems = [f'{asset_id!r} missing' for asset_id in missing] + [
                    f'{asset_id!r} not at position 0' for asset_id in extra
                ]
                msg = (
                    f'Position {position} holds assets other than those of '
                    f'position 0: {_report_first_problems(problems, ", ")}.'
                )
                raise ValueError(msg)
            moved = [
                asset_id
                for asset_id in held
                if realization.inventory[asset_id].coordinates
                != self._assets[asset_id].coordinates
            ]
            if moved:
                moved = [repr(i) for i in sorted(moved, key=_merge_id_order)]
                msg = (
                    f'Position {position} holds {len(moved)} asset(s) with '
                    'coordinates other than at position 0: '
                    f'{_report_first_problems(moved, ", ")}.'
                )
                raise ValueError(msg)
            # A vector feature listed by position must hold a list at
            # position 0, since a list of positions is told from a vector
            # value by its first item. A feature listed before a realization
            # declared it a vector is checked through its stored list.
            vector = {*self._vector_features, *realization.vector_features}
            ambiguous = []
            for asset_id in given:
                held = self._assets[asset_id].features
                listed = self._lists.get(asset_id, ())
                for key in vector:
                    first = held[key][0] if key in listed else held.get(key)
                    if not isinstance(first, list) and (
                        key in listed
                        or not _same_value(
                            first, realization.inventory[asset_id].features.get(key)
                        )
                    ):
                        ambiguous.append((asset_id, key))
            if ambiguous:
                ambiguous = [
                    f'asset {asset_id!r}, feature {key!r}'
                    for asset_id, key in sorted(
                        ambiguous,
                        key=lambda pair: (_merge_id_order(pair[0]), pair[1]),
                    )
                ]
                msg = (
                    f'Position {position} gives {len(ambiguous)} vector '
                    'feature(s) whose position 0 value is not a list a '
                    'value that differs between positions: '
                    f'{_report_first_problems(ambiguous, "; ")}.'
                )
                raise ValueError(msg)

        if position == 0:
            for asset_id, asset in realization.inventory.items():
                record = copy(asset)
                record.features = dict(asset.features)
                self._assets[asset_id] = record
        else:
            for asset_id, asset in realization.inventory.items():
                features = asset.features
                held = self._assets[asset_id].features
                lists = self._lists.get(asset_id, set())
                for key in [*held, *(key for key in features if key not in held)]:
                    value = features.get(key)
                    if key in lists:
                        held[key].append(value)
                    elif not _same_value(held.get(key), value):
                        held[key] = [held.get(key)] * position + [value]
                        lists.add(key)
                if lists:
                    self._lists[asset_id] = lists
        self._vector_features.update(dict.fromkeys(realization.vector_features))
        self._added += 1

    def result(self) -> AssetInventory:
        """
        Return the assembled inventory.

        Returns:
            AssetInventory:
                An inventory whose number of possible worlds is the number
                of positions, with assets in ascending order of the asset id
                as a string. It declares the vector features of every
                realization. Later calls return the same inventory.

        Raises:
            ValueError:
                If fewer positions were added than the assembler was created
                for.
        """
        if self._result is None:
            if self._added < self._n_worlds:
                msg = (
                    f'{self._added} of {self._n_worlds} positions have been '
                    'added; add the rest before taking the result.'
                )
                raise ValueError(msg)
            result = AssetInventory(vector_features=list(self._vector_features))
            result.n_pw = self._n_worlds
            for asset_id in sorted(self._assets, key=_merge_id_order):
                result.inventory[asset_id] = self._assets[asset_id]
            self._result = result
            self._assets, self._lists = {}, {}
        return self._result
