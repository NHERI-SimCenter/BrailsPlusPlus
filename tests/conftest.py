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

"""Fixtures shared across the BRAILS test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from brails.types.asset_inventory import Asset, AssetInventory

if TYPE_CHECKING:
    from collections.abc import Callable

# Coordinates for test assets whose location does not matter.
PLACEHOLDER_COORDINATES = [[-122.4, 37.8]]


@pytest.fixture
def build_inventory() -> Callable[..., AssetInventory]:
    """
    Return a factory that builds an AssetInventory from feature dicts.

    Assets are added directly so the number of possible worlds can be set
    explicitly. Going through ``add_asset_features`` would infer it from
    whichever feature happened to be added last.

    Returns:
        Callable[..., AssetInventory]:
            A callable taking ``features_by_asset``, and optionally
            ``n_pw`` and ``vector_features``.

    Examples:
        >>> def test_something(build_inventory):
        ...     inventory = build_inventory(
        ...         {'A1': {'StructureType': ['W1', 'URM']}}, n_pw=2
        ...     )
    """

    def _build(
        features_by_asset: dict[str, dict[str, Any]],
        n_pw: int = 1,
        vector_features: list[str] | None = None,
    ) -> AssetInventory:
        """
        Build an inventory with the given features and number of worlds.

        Args:
            features_by_asset (dict[str, dict[str, Any]]):
                Features keyed by asset id.
            n_pw (int):
                The number of possible worlds to declare.
            vector_features (list[str], optional):
                Vector feature names to pass to the inventory.

        Returns:
            AssetInventory:
                The populated inventory.
        """
        inventory = AssetInventory(vector_features=vector_features)

        for asset_id, features in features_by_asset.items():
            inventory.add_asset(
                asset_id,
                Asset(asset_id, PLACEHOLDER_COORDINATES, dict(features)),
            )

        inventory.n_pw = n_pw
        return inventory

    return _build
