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
Join inventories that hold different features of their assets.

.. autosummary::

      union_disjoint
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from brails.types.asset_inventory import Asset, AssetInventory, is_missing_value

if TYPE_CHECKING:
    from collections.abc import Hashable


def union_disjoint(*inventories: AssetInventory) -> AssetInventory:
    """
    Join inventories that hold disjoint features of their assets.

    The result holds every asset of any input with every value the inputs hold for
    it, and raises where two inputs hold a value of the same feature of one asset.
    Missing values follow :func:`~brails.types.asset_inventory.is_missing_value` and
    are not values. Unlike :meth:`AssetInventory.merge`, the join takes no spine:
    its asset set is the union of the inputs' asset sets. It resolves no feature by
    precedence, since the features must be disjoint per asset, writes no source
    string, and drops a feature no input holds a value for, where ``merge`` keeps the
    missing marker of the first source.

    Args:
        *inventories (AssetInventory):
            The inventories, each with one possible world or a common number of them.
            The inputs are not modified.

    Returns:
        AssetInventory:
            One asset per asset of any input, with deep copies of every value the
            inputs hold for it and of the coordinates of the first input holding it,
            in ascending order of the id as a string, with the inputs' vector
            features and the common number of worlds.

    Raises:
        ValueError:
            If two inputs hold a value of the same feature of one asset, naming up to
            ten such assets, or if the inputs hold different numbers of worlds above
            one.
    """
    worlds = {inventory.n_pw for inventory in inventories} - {1}
    if len(worlds) > 1:
        msg = f'The inventories hold different numbers of worlds: {sorted(worlds)}.'
        raise ValueError(msg)
    vector_features = sorted(
        {f for inventory in inventories for f in inventory.vector_features}
    )
    coordinates: dict[Hashable, Any] = {}
    features: dict[Hashable, dict[str, Any]] = {}
    overlaps: dict[Hashable, list[str]] = {}
    for inventory in inventories:
        for asset_id, asset in inventory.inventory.items():
            coordinates.setdefault(asset_id, asset.coordinates)
            held = features.setdefault(asset_id, {})
            for feature, value in asset.features.items():
                if is_missing_value(value):
                    continue
                if feature in held:
                    overlaps.setdefault(asset_id, []).append(feature)
                    continue
                held[feature] = value
    if overlaps:
        named = sorted(overlaps, key=str)
        preview = {a: overlaps[a] for a in named[:10]}
        msg = (
            f'Assets holding the same feature in two inventories: {len(named)}, '
            f'among them {preview}.'
        )
        raise ValueError(msg)
    union = AssetInventory(vector_features=vector_features)
    for asset_id in sorted(features, key=str):
        asset = Asset(
            asset_id, deepcopy(coordinates[asset_id]), deepcopy(features[asset_id])
        )
        union.add_asset(asset_id, asset)
    union.n_pw = worlds.pop() if worlds else 1
    return union
