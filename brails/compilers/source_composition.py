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
Compose a source from frozen stages, and select or build inventories of assets.

.. autosummary::

      compose_source_parts
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from numbers import Integral
from typing import TYPE_CHECKING, Any

from brails.types.asset_inventory import (
    MAX_REPORTED_WORLD_PROBLEMS,
    Asset,
    AssetInventory,
    is_missing_value,
    is_probabilistic_feature,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

    from brails.utils.stage_freeze import StageFreeze


def _select_assets(
    source: AssetInventory,
    ids: Iterable[Hashable],
    *,
    vector_features: Iterable[str] | None = None,
    n_pw: int = 1,
) -> AssetInventory:
    """
    Return a new inventory holding the given assets of ``source`` themselves.

    The result holds the source's own :class:`Asset` objects, not copies, so a change
    to an asset's coordinates or features shows in both. It declares ``n_pw`` worlds
    whatever ``source`` holds, so a source of several worlds needs ``n_pw`` given.

    Args:
        source (AssetInventory):
            The inventory the assets are selected from.
        ids (Iterable[Hashable]):
            The ids of the assets to select, each held by ``source``, in the order
            the result holds them.
        vector_features (Iterable[str], optional):
            The features whose values are vectors in the result. ``None`` declares
            those of ``source``.
        n_pw (int, optional):
            The number of possible worlds the result declares. Defaults to 1.

    Returns:
        AssetInventory:
            The selected assets, with one possible world.
    """
    selected = AssetInventory(
        vector_features=list(
            source.vector_features if vector_features is None else vector_features
        )
    )
    selected.inventory = {asset_id: source.inventory[asset_id] for asset_id in ids}
    selected.n_pw = n_pw
    return selected


def _build_inventory(
    ids: Iterable[Hashable],
    coordinates: Mapping[Hashable, Any],
    features: Mapping[Hashable, dict[str, Any]],
    *,
    n_pw: int = 1,
    vector_features: Iterable[str] | None = None,
) -> AssetInventory:
    """
    Build an inventory of new assets from their coordinates and features.

    Each asset holds the coordinates and the feature dict given for it, not copies,
    so a caller that must not share a feature dict passes a copy. An id with no entry
    in ``features`` gets a new, empty feature dict.

    Args:
        ids (Iterable[Hashable]):
            The ids of the assets to build, in the order the result holds them.
        coordinates (Mapping[Hashable, Any]):
            The coordinates of each asset, by id.
        features (Mapping[Hashable, dict[str, Any]]):
            The feature dict of each asset that has features, by id.
        n_pw (int, optional):
            The number of possible worlds the features hold. Defaults to 1.
        vector_features (Iterable[str], optional):
            The features whose values are vectors. ``None`` declares the default of
            :class:`AssetInventory`.

    Returns:
        AssetInventory:
            One new asset per id.
    """
    built = AssetInventory(
        vector_features=None if vector_features is None else list(vector_features)
    )
    for asset_id in ids:
        built.inventory[asset_id] = Asset(
            asset_id, coordinates[asset_id], features.get(asset_id, {})
        )
    built.n_pw = n_pw
    return built


def compose_source_parts(  # noqa: C901
    freeze: StageFreeze,
    parts: Sequence[str],
    *,
    world: int,
    withheld: Sequence[str] = (),
    label: str,
    dropped: str | None = None,
) -> AssetInventory:
    """
    Build the inventory of one source in one world from frozen stages.

    The result holds every asset that one of ``parts`` holds, and each of its
    features takes the value of the one part that holds a value for it in ``world``.
    A part holds no value for a feature where its value is one that
    ``AssetInventory.merge`` counts as missing, such as ``None``, NaN, the empty
    string or the missing marker. Two parts that hold a value for the same feature of
    one asset raise ``ValueError``. The result shares assets and values with the
    frozen stages, so a caller must not change it. The section "Composition and
    assembly" of the README beside this module states the procedure and what the
    optional records do.

    Args:
        freeze (StageFreeze):
            Holds the frozen stages that the other arguments name. Every stage is
            read from it by its name.
        parts (Sequence[str]):
            The names of the stages that hold the source's values, each named once.
            Their order matters: an asset takes the coordinates of the first part
            that holds it.
        world (int):
            The world to compose, numbered from 0. A stage that holds several worlds
            is read through its values in this world.
        withheld (Sequence[str], optional):
            The names of records of withheld values, which remove values of this
            source from the result. Defaults to an empty sequence, which withholds
            nothing.
        label (str):
            The label of the source being composed, such as ``'imputed'``. It is used
            only to find, in the ``withheld`` records, the values withheld from this
            source.
        dropped (str, optional):
            The name of a record of dropped footprints, whose assets the result
            leaves out. Defaults to ``None``, which leaves out no asset.

    Returns:
        AssetInventory:
            The inventory of the source in ``world``, holding one possible world. It
            declares every vector feature of every part.

    Raises:
        TypeError:
            If ``parts`` or ``withheld`` is a string or not a sequence, a stage name
            is not a string, ``label`` is not a string, or ``world`` is not an
            integer.
        LookupError:
            If a named stage is not frozen.
        ValueError:
            If ``parts`` is empty or names a stage twice; if ``world`` is negative,
            or not below the number of possible worlds of a part or withheld record
            that holds more than one; or if two parts hold a value for one feature of
            one asset in ``world``. A message names up to
            :data:`~brails.types.asset_inventory.MAX_REPORTED_WORLD_PROBLEMS` assets
            or features.
    """
    for argument, names in (('parts', parts), ('withheld', withheld)):
        if isinstance(names, str) or not isinstance(names, Sequence):
            msg = (
                f'{argument} must be a sequence of stage names, not '
                f'{type(names).__name__}.'
            )
            raise TypeError(msg)
    named = [*parts, *withheld, *([] if dropped is None else [dropped])]
    for name in named:
        if not isinstance(name, str):
            msg = (
                'compose_source_parts reads frozen stages by name only, and a stage '
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
    if not parts or len(set(parts)) < len(parts):
        msg = f'parts must name at least one stage, each once, not {list(parts)}.'
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

    def owned_features(asset_id: Hashable) -> dict[str, Any]:
        """
        Return an asset's features in the result, copied on the first write.

        Args:
            asset_id (Hashable):
                The id of the asset of the result whose features are about to be
                written.

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

    def features_in_world(stage: AssetInventory, asset: Asset) -> dict[str, Any]:
        """
        Return an asset's features in ``world``.

        Args:
            stage (AssetInventory):
                The stage holding the asset.
            asset (Asset):
                The asset whose features are read.

        Returns:
            dict[str, Any]:
                The asset's own features where no value is a list of worlds, and
                otherwise a new dict of their values in ``world``.
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
        vector_features.update(dict.fromkeys(stage.vector_features))
        for asset_id, asset in stage.inventory.items():
            if asset_id in dropped_ids:
                continue
            features = features_in_world(stage, asset)
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
                    current = owned_features(asset_id)
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

    for name in withheld:
        stage = stages[name]
        for asset_id, asset in stage.inventory.items():
            if asset_id not in records:
                continue
            removed = [
                key
                for key, value in features_in_world(stage, asset).items()
                if isinstance(value, str) and value == label
            ]
            if removed and any(key in records[asset_id].features for key in removed):
                target = owned_features(asset_id)
                for key in removed:
                    target.pop(key, None)

    result = AssetInventory(vector_features=list(vector_features))
    result.inventory = records
    return result
