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
Replace values in one world's per-source inventories.

.. autosummary::

      apply_to_sources
"""

from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any

from brails.types.asset_inventory import Asset, AssetInventory, is_missing_value

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    import pandas as pd

# Source labels whose values are never withheld.
FIXED_SOURCES = ('local', 'footprint')

# Columns of a table of withheld values. ``source`` holds the label of the
# source the value came from.
WITHHELD_COLUMNS = ('asset_id', 'feature', 'source', 'value')


def apply_to_sources(
    sources: Mapping[str, AssetInventory],
    updates: Mapping[Hashable, Mapping[str, Any]],
    withheld: pd.DataFrame,
    *,
    imputed_label: str = 'imputed',
    precedence: Sequence[str] | None = None,
) -> dict[str, AssetInventory]:
    """
    Withhold replaced values and write their replacements as imputed.

    Each row of ``withheld`` removes one value from the source it names,
    whether or not ``updates`` replaces it, and each entry of ``updates``
    is written as given into the ``imputed_label`` source, which is
    created when absent. The :data:`FIXED_SOURCES` are never changed. A
    source that changes is returned as a new inventory that shares its
    unchanged assets with the input. The other sources are returned as
    they are. Merging the result (:meth:`AssetInventory.merge`) codes the
    written values as imputed and, where a withheld value has no
    replacement, takes the feature from the next source that holds it.

    Args:
        sources (Mapping[str, AssetInventory]):
            The per-source inventories of one possible world, keyed by
            their merge labels, the spine among them. None of them is
            modified.
        updates (Mapping[Hashable, Mapping[str, Any]]):
            The replacement values per asset id.
        withheld (pandas.DataFrame):
            The replaced values, with columns :data:`WITHHELD_COLUMNS`.
        imputed_label (str, optional):
            The label of the imputed source. Defaults to ``'imputed'``.
        precedence (Sequence[str], optional):
            The merge precedence, highest first. When given, a source
            ranked above ``imputed_label`` that still holds a written
            feature of the asset raises, since the merge would take its
            value over the replacement.

    Returns:
        dict[str, AssetInventory]:
            The sources with the changes made.

    Raises:
        ValueError:
            If a withheld row names a fixed source, the imputed source or
            a label the sources do not hold, or a value its source does
            not hold; if the imputed source already holds a written value;
            if no source holds an asset new to the imputed source; if a
            changed source holds more than one possible world; or if,
            under ``precedence``, a higher-ranked source holds a written
            feature.
    """
    out = dict(sources)
    copied: dict[str, set] = {}
    for row in withheld.to_dict('records'):
        label = row['source']
        if label not in out:
            msg = (
                f'A withheld value names the source {label!r}, which the '
                'sources do not hold.'
            )
            raise ValueError(msg)
        if label in FIXED_SOURCES or label == imputed_label:
            msg = (
                f'A withheld value names the source {label!r}, which may not change.'
            )
            raise ValueError(msg)
        asset = _editable_asset(out, copied, label, row['asset_id'])
        feature = row['feature']
        if (
            asset is None
            or feature not in asset.features
            or asset.features[feature] != row['value']
        ):
            msg = (
                f'The {label!r} source does not hold {feature} = '
                f'{row["value"]!r} for asset {row["asset_id"]!r}.'
            )
            raise ValueError(msg)
        del asset.features[feature]
    if updates and imputed_label not in out:
        out[imputed_label] = AssetInventory()
    for asset_id, values in updates.items():
        asset = _editable_asset(out, copied, imputed_label, asset_id)
        if asset is None:
            coordinates = _coordinates_of(asset_id, sources)
            asset = Asset(asset_id, coordinates, {})
            out[imputed_label].inventory[asset_id] = asset
        held = [
            feature
            for feature in values
            if not is_missing_value(asset.features.get(feature))
        ]
        if held:
            msg = (
                f'The {imputed_label!r} source already holds {held} for asset '
                f'{asset_id!r}.'
            )
            raise ValueError(msg)
        asset.features.update(values)
    if precedence is not None:
        _check_precedence(out, updates, imputed_label, precedence)
    return out


def _editable_asset(
    out: dict[str, AssetInventory],
    copied: dict[str, set],
    label: str,
    asset_id: Hashable,
) -> Asset | None:
    """
    Return an asset of a source that may be changed, copying on first use.

    The first call for a source replaces it in ``out`` with a shallow copy
    whose asset table is its own, and the first call for an asset replaces
    the asset with a copy whose features are its own. The input's
    inventories and assets are never changed.

    Args:
        out (dict[str, AssetInventory]):
            The sources being written, updated in place.
        copied (dict[str, set]):
            The asset ids already copied per copied source, updated in
            place.
        label (str):
            The source.
        asset_id (Hashable):
            The asset.

    Returns:
        Asset | None:
            The editable asset, or None where the source lacks it.

    Raises:
        ValueError:
            If the source holds more than one possible world.
    """
    if label not in copied:
        source = out[label]
        if source.n_pw != 1:
            msg = (
                f'The {label!r} source holds {source.n_pw} possible worlds. '
                'Values are replaced one world at a time.'
            )
            raise ValueError(msg)
        source = copy(source)
        source.inventory = dict(source.inventory)
        out[label] = source
        copied[label] = set()
    inventory = out[label].inventory
    asset = inventory.get(asset_id)
    if asset is None or asset_id in copied[label]:
        return asset
    asset = copy(asset)
    asset.features = dict(asset.features)
    inventory[asset_id] = asset
    copied[label].add(asset_id)
    return asset


def _coordinates_of(
    asset_id: Hashable, sources: Mapping[str, AssetInventory]
) -> list:
    """
    Copy an asset's coordinates from the first source holding it.

    Args:
        asset_id (Hashable):
            The asset.
        sources (Mapping[str, AssetInventory]):
            The sources, searched in ascending order of their labels.

    Returns:
        list:
            A deep copy of the coordinates.

    Raises:
        ValueError:
            If no source holds the asset.
    """
    for label in sorted(sources):
        asset = sources[label].inventory.get(asset_id)
        if asset is not None:
            return deepcopy(asset.coordinates)
    msg = f'No source holds asset {asset_id!r}.'
    raise ValueError(msg)


def _check_precedence(
    out: Mapping[str, AssetInventory],
    updates: Mapping[Hashable, Mapping[str, Any]],
    imputed_label: str,
    precedence: Sequence[str],
) -> None:
    """
    Check that no source ranked above the imputed one holds a written value.

    Args:
        out (Mapping[str, AssetInventory]):
            The sources after the changes.
        updates (Mapping[Hashable, Mapping[str, Any]]):
            The written values per asset id.
        imputed_label (str):
            The imputed source's label.
        precedence (Sequence[str]):
            The merge precedence, highest first.

    Raises:
        ValueError:
            If the imputed label is not in ``precedence``, or a higher
            source holds a written feature of the asset.
    """
    if imputed_label not in precedence:
        msg = f'The precedence {list(precedence)} lacks {imputed_label!r}.'
        raise ValueError(msg)
    above = [
        label
        for label in precedence[: list(precedence).index(imputed_label)]
        if label in out
    ]
    for asset_id, values in updates.items():
        for label in above:
            asset = out[label].inventory.get(asset_id)
            if asset is None:
                continue
            held = [
                feature
                for feature in values
                if not is_missing_value(asset.features.get(feature))
            ]
            if held:
                msg = (
                    f'The {label!r} source, ranked above {imputed_label!r}, '
                    f'still holds {held} for asset {asset_id!r}.'
                )
                raise ValueError(msg)
