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

"""Fixtures shared by the compiler tests of merged and per-source inventories."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from brails.compilers.source_composition import _build_inventory
from brails.types.asset_inventory import (
    HAZUS_EQ_SOURCE_FEATURES,
    SOURCE_CODE_NONE,
    Asset,
    AssetInventory,
)
from tests.imputers.donor_fixtures import square

# Merge precedence of the per-source fixtures, highest first.
PRECEDENCE = ['local', 'footprint', 'nsi', 'derived', 'imputed']


def codes(occupancy: str = 'N', units: str = '-', stories: str = 'N') -> str:
    """
    Return a source string over the default source features.

    Args:
        occupancy (str, optional):
            The occupancy class code. Defaults to ``'N'``.
        units (str, optional):
            The unit count code. Defaults to ``'-'``.
        stories (str, optional):
            The story count code. Defaults to ``'N'``.

    Returns:
        str:
            The string, with the missing code for the remaining source features.
    """
    given = {
        'OccupancyClass': occupancy,
        'NumberOfUnits': units,
        'NumberOfStories': stories,
    }
    return ''.join(
        given.get(feature, SOURCE_CODE_NONE) for feature in HAZUS_EQ_SOURCE_FEATURES
    )


def merged(
    records: dict[str, dict[str, Any]], order: list[str] | None = None
) -> AssetInventory:
    """
    Build a merged one-world inventory of square footprints.

    Args:
        records (dict[str, dict[str, Any]]):
            Features per asset id. Asset ``i`` of the sorted ids stands ``20 * i``
            meters east of the origin.
        order (list[str], optional):
            The insertion order. Defaults to the sorted ids.

    Returns:
        AssetInventory:
            The inventory, its assets inserted in ``order``.
    """
    positions = {asset_id: i for i, asset_id in enumerate(sorted(records))}
    order = order or sorted(records)
    return _build_inventory(
        order,
        {asset_id: square(20.0 * positions[asset_id], 0.0) for asset_id in order},
        {asset_id: dict(records[asset_id]) for asset_id in order},
    )


def nsi_tall_industry(**features: Any) -> dict[str, Any]:
    """
    Return the features of a ten-story IND1 building from NSI.

    Its class and story count form an implausible pair.

    Args:
        **features (Any):
            Features that replace or extend the defaults.

    Returns:
        dict[str, Any]:
            The default features, updated with ``features``.
    """
    base = {
        'OccupancyClass': 'IND1',
        'NumberOfStories': 10,
        'SourceString': codes(),
    }
    base.update(features)
    return base


def sources_for_one_asset() -> dict[str, AssetInventory]:
    """
    Return per-source inventories of one ten-story IND1 building from NSI.

    Returns:
        dict[str, AssetInventory]:
            ``footprint`` (the spine, with a height of unknown source), ``local`` (a
            year) and ``nsi`` (the class and story count).
    """
    coordinates = square(0.0, 0.0)
    footprint = AssetInventory()
    footprint.add_asset('a', Asset('a', coordinates, {'height': 12.0}))
    local = AssetInventory()
    local.add_asset('a', Asset('a', deepcopy(coordinates), {'YearBuilt': 1975}))
    nsi = AssetInventory()
    nsi.add_asset(
        'a',
        Asset(
            'a',
            deepcopy(coordinates),
            {'OccupancyClass': 'IND1', 'NumberOfStories': 10},
        ),
    )
    return {'footprint': footprint, 'local': local, 'nsi': nsi}


def merge_sources(sources: dict[str, AssetInventory]) -> AssetInventory:
    """
    Merge per-source inventories on the footprint spine.

    Args:
        sources (dict[str, AssetInventory]):
            The sources, ``footprint`` among them.

    Returns:
        AssetInventory:
            The merged inventory.
    """
    others = {label: inv for label, inv in sources.items() if label != 'footprint'}
    precedence = [label for label in PRECEDENCE if label in sources]
    return sources['footprint'].merge(others, precedence, name='footprint')


def source_features(sources: dict[str, AssetInventory]) -> dict[str, dict]:
    """
    Return a deep copy of every source's features, for comparison.

    Args:
        sources (dict[str, AssetInventory]):
            The per-source inventories, by source label.

    Returns:
        dict[str, dict]:
            Features per asset id per source label.
    """
    return {
        label: {k: deepcopy(a.features) for k, a in inv.inventory.items()}
        for label, inv in sources.items()
    }
