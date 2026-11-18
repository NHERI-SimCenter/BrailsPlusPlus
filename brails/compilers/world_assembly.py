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
) -> AssetInventory:
    """
    Return a new inventory holding the given assets of ``source`` themselves.

    The result holds the source's own :class:`Asset` objects, not copies, so
    a change to an asset's coordinates or features shows in both.
    ``source`` must hold one possible world: the result declares one world
    whatever ``source`` holds.

    Args:
        source (AssetInventory):
            The inventory the assets are selected from.
        ids (Iterable[Hashable]):
            The ids of the assets to select, each held by ``source``, in the
            order the result holds them.
        vector_features (Iterable[str], optional):
            The features whose values are vectors in the result. ``None``
            declares those of ``source``.

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

    Each asset holds the coordinates and the feature dict given for it, not
    copies, so a caller that must not share a feature dict passes a copy. An
    id with no entry in ``features`` gets a new, empty feature dict.

    Args:
        ids (Iterable[Hashable]):
            The ids of the assets to build, in the order the result holds
            them.
        coordinates (Mapping[Hashable, Any]):
            The coordinates of each asset, by id.
        features (Mapping[Hashable, dict[str, Any]]):
            The feature dict of each asset that has features, by id.
        n_pw (int, optional):
            The number of possible worlds the features hold. Defaults to 1.
        vector_features (Iterable[str], optional):
            The features whose values are vectors. ``None`` declares the
            default of :class:`AssetInventory`.

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
    Compose one source of one world from frozen stages.

    The source is the union of ``parts`` in ``world``, less the assets of
    the ``dropped`` record and the values a ``withheld`` record withholds
    from ``label``. Every stage is read from ``freeze`` by name. At most one
    part may hold a value for a feature of an asset. An asset takes the
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

    Returns:
        AssetInventory:
            The source in ``world``, with one possible world. It declares
            the vector features of every part.

    Raises:
        TypeError:
            If ``parts`` or ``withheld`` is a string or not a sequence, a
            stage name is not a string, ``label`` is not a string, or
            ``world`` is not an integer.
        LookupError:
            If a named stage is not frozen.
        ValueError:
            If ``parts`` is empty or names a stage twice; if ``world`` is
            negative, or not below the number of possible worlds of a part
            or withheld record that holds more than one; or if two parts
            hold a value for one feature of one asset in ``world``. A
            message names up to
            :data:`~brails.types.asset_inventory.MAX_REPORTED_WORLD_PROBLEMS`
            assets or features.
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

    def features_in_world(stage: AssetInventory, asset: Asset) -> dict[str, Any]:
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
