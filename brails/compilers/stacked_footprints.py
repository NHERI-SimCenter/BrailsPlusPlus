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
Merge the stacked Overture footprints of one building into one asset.

.. autosummary::

      StackedFootprintMerge
      StackedMergeResult
"""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon

from brails.constants import (
    DEFAULT_UNITS,
    FOOTPRINT_PROVIDER_FEATURE,
    PROVIDER_ID_FEATURES,
)
from brails.inferers.stories_from_height_inferer.stories_from_height_inferer import (
    HEIGHT_PROPERTY,
    OVERTURE_PROVIDER,
    SOURCES_KEY,
)
from brails.types.asset_inventory import Asset, AssetInventory, is_missing_value
from brails.utils.asset_id import assign_asset_ids
from brails.utils.geo_tools import GeoTools
from brails.utils.unit_converter import UNIT_TO_METER, UnitConverter

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# The thresholds and the stacking signal: README.rst beside this module,
# "Stacked footprints".
NEST_MIN = 0.9  # share of the smaller footprint inside the larger
COPY_IOU = 0.75  # intersection over union of two drawings of one building
PART_TALLER_M = 3.0  # meters, compared after conversion from the length unit
ROOF_CLASS = 'roof'

OVERTURE_ID_FEATURE = PROVIDER_ID_FEATURES[OVERTURE_PROVIDER][0]
HEIGHT_FEATURE = 'height'
FLOORS_FEATURE = 'num_floors'
CLASS_FEATURE = 'class'
SOURCES_FEATURE = SOURCES_KEY
PART_IDS_FEATURE = 'overture_part_ids'
PART_IDS_SEPARATOR = ';'
# The separator of the entries of the Overture scraper's sources string.
SOURCES_SEPARATOR = '; '
# An entry of that string whose property is the height.
_HEIGHT_ENTRY = re.compile(
    rf'(?:^|, )property: {re.escape(HEIGHT_PROPERTY)}(?:, |$)'
)

RULE_STACKED = 'stacked'
RULE_COPY = 'copy'

PARTS_COLUMNS = (
    'asset_id',
    'rule',
    'head_overture_id',
    'overture_id',
    'input_asset_id',
    'is_head',
    'plan_area_m2',
    'height',
    'num_floors',
    'class',
    'merged_height',
    'merged_num_floors',
)


class StackedMergeResult(NamedTuple):
    """
    The merged inventory and the record of its merges.

    Attributes:
        inventory (AssetInventory):
            A new inventory of the footprints that remain, keyed by
            :func:`~brails.utils.asset_id.assign_asset_ids` over them.
        parts (pandas.DataFrame):
            One row per footprint of a merged group, its head included,
            with columns :data:`PARTS_COLUMNS`: the key in ``inventory`` of
            the asset the footprint is part of, the rule that merged the
            group (:data:`RULE_STACKED` or :data:`RULE_COPY`), the
            ``overture_id`` of the group's head and of the footprint, the
            footprint's key in the input inventory, whether it is the head,
            its plan area in square meters, its own height, floor count and
            class, and the height and floor count of the asset it is part
            of. Heights are in the merge's length unit, and a missing value
            is NaN. The rows are ordered by asset key, rule, head first,
            then ``overture_id``.
        id_log (list[dict]):
            The collision log of ``assign_asset_ids`` over the footprints
            that remain, with ``position`` counting the assets of the input
            inventory in its order.
    """

    inventory: AssetInventory
    parts: pd.DataFrame
    id_log: list[dict[str, Any]]


class StackedFootprintMerge:
    """
    Merge the stacked Overture footprints of one building into one asset.

    Overture draws some buildings as several outlines, such as a podium
    with the towers on it, each outline a building of its own. The merge
    joins them and re-keys the inventory:

    1. *Pairs.* Two footprints form a pair when their intersection covers
       at least :data:`NEST_MIN` of the smaller one's area (nested) or their
       intersection over union is at least :data:`COPY_IOU` (copies). The
       areas are planar in EPSG:4326, of the exterior rings a BRAILS asset
       stores. The larger footprint of a pair is its container, ties of
       area going to the lower ``overture_id``.
    2. *Copies.* Each connected set of copies keeps one footprint: one
       with a height before one without, then the larger, then the lower
       ``overture_id``. The others are dropped.
    3. *Nests.* Of the nested pairs between the footprints that remain,
       each footprint goes to its largest container, and from there up to
       the top of the nest, the container that lies in no other.
    4. *Signal.* A footprint merges into the top of its nest when it is
       taller than the top by more than :data:`PART_TALLER_M`, has a height
       where the top has none, or when either of the two has class
       :data:`ROOF_CLASS`. Otherwise it stays an asset of its own.
    5. *Merge.* The top of a nest, its head, and the footprints merging
       into it, its members, become one asset. Its coordinates are the
       exterior rings of the union of their polygons, and its ``height``
       and ``num_floors`` the largest over the parts, absent where no part
       has one. :data:`PART_IDS_FEATURE` holds the members' ``overture_id``
       values, sorted and joined by :data:`PART_IDS_SEPARATOR`. ``sources``
       holds the entries of the part whose height was taken, then the
       head's, then the other members', less the entries of the other
       parts whose property is the height, so that
       :class:`~brails.inferers.stories_from_height_inferer.stories_from_height_inferer.StoriesFromHeightInferer`
       reads the height's source from the tallest part's entries wherever
       those name one. Every other feature is the head's.
    6. *Keys.* The footprints that remain are keyed by
       :func:`~brails.utils.asset_id.assign_asset_ids` with the
       ``overture_id`` as tiebreak, so a footprint in no collision set of
       either keying keeps the key the Overture scraper gave it.

    A union is at least as large as its head, so the merge leaves no
    footprint below an area floor the scraper applied. The README beside
    this module states where the thresholds come from and what the rule
    leaves unmerged.
    """

    def __init__(self, *, length_unit: str = DEFAULT_UNITS['length']) -> None:
        """
        Configure the merge.

        Args:
            length_unit (str, optional):
                The unit of the inventory's heights, which the Overture
                scraper writes in its length unit. Defaults to the length
                unit of :data:`brails.constants.DEFAULT_UNITS`.

        Raises:
            ValueError:
                If ``length_unit`` is not a length unit.
        """
        unit = UnitConverter.normalize_unit(length_unit)
        if unit not in UNIT_TO_METER:
            msg = (
                f'{length_unit!r} is not a length unit; use one of '
                f'{sorted(UNIT_TO_METER)}.'
            )
            raise ValueError(msg)
        self.length_unit = length_unit
        self._taller = PART_TALLER_M / UNIT_TO_METER[unit]

    def merge(self, inventory: AssetInventory) -> StackedMergeResult:
        """
        Merge the stacked footprints of an inventory and re-key it.

        Args:
            inventory (AssetInventory):
                Overture footprints of one possible world keyed by string,
                as the Overture footprint scraper builds them under the
                ``'geohash'`` id scheme. It is not modified.

        Returns:
            StackedMergeResult:
                The merged inventory, its parts and its id log.

        Raises:
            ValueError:
                If the inventory holds more than one possible world, an
                asset key is not a string, an asset lacks
                ``footprint_provider`` :data:`OVERTURE_PROVIDER` or an
                ``overture_id``, or its footprint is not a polygon with
                area.
        """
        keys, features, polygons = _footprints(inventory)
        ids = [
            str(asset_features[OVERTURE_ID_FEATURE]) for asset_features in features
        ]
        geometries = np.array(polygons, dtype=object)
        area = shapely.area(geometries)

        pairs = _pairs(geometries, area, ids)
        copies = _copy_keepers(pairs, features, area, ids)
        groups = self._stacked_groups(pairs, copies, features, area, ids)

        # The position of the asset each merged-away footprint is part of.
        home = {
            member: head for head, members in groups.items() for member in members
        }
        for dropped, keeper in copies.items():
            home[dropped] = home.get(keeper, keeper)
        kept = [position for position in range(len(keys)) if position not in home]
        coordinates = []
        kept_features = []
        for position in kept:
            if position in groups:
                parts = [position, *groups[position]]
                union = shapely.union_all(geometries[parts])
                coordinates.append(GeoTools.geometry_to_list_of_lists(union))
                kept_features.append(self._merged_features(parts, features, ids))
            else:
                coordinates.append(
                    copy.deepcopy(inventory.inventory[keys[position]].coordinates)
                )
                kept_features.append(copy.deepcopy(features[position]))

        result = assign_asset_ids(
            coordinates, tiebreak=[ids[position] for position in kept]
        )
        merged_inventory = AssetInventory(vector_features=inventory.vector_features)
        for key, asset_coordinates, asset_features in zip(
            result.ids, coordinates, kept_features
        ):
            merged_inventory.add_asset(
                key, Asset(key, asset_coordinates, asset_features)
            )
        new_key = dict(zip(kept, result.ids))
        id_log = [
            {**entry, 'position': kept[entry['position']]} for entry in result.log
        ]

        # The groups of the parts table: each head, its rule and its parts.
        tables = [
            (head, RULE_STACKED, [head, *members])
            for head, members in groups.items()
        ]
        copy_sets: dict[int, list[int]] = {}
        for dropped, keeper in copies.items():
            copy_sets.setdefault(keeper, [keeper]).append(dropped)
        tables.extend(
            (keeper, RULE_COPY, parts) for keeper, parts in copy_sets.items()
        )
        rows = []
        for head, rule, parts in tables:
            for part in parts:
                asset_id = new_key[home.get(part, part)]
                asset_features = merged_inventory.inventory[asset_id].features
                rows.append(
                    {
                        'asset_id': asset_id,
                        'rule': rule,
                        'head_overture_id': ids[head],
                        'overture_id': ids[part],
                        'input_asset_id': keys[part],
                        'is_head': part == head,
                        'plan_area_m2': AssetInventory.centroid_and_plan_area(
                            inventory.inventory[keys[part]].coordinates
                        )[2],
                        'height': features[part].get(HEIGHT_FEATURE, np.nan),
                        'num_floors': features[part].get(FLOORS_FEATURE, np.nan),
                        'class': features[part].get(CLASS_FEATURE),
                        'merged_height': asset_features.get(HEIGHT_FEATURE, np.nan),
                        'merged_num_floors': asset_features.get(
                            FLOORS_FEATURE, np.nan
                        ),
                    }
                )
        parts_table = pd.DataFrame(rows, columns=list(PARTS_COLUMNS))
        parts_table = parts_table.sort_values(
            ['asset_id', 'rule', 'is_head', 'overture_id'],
            ascending=[True, True, False, True],
            ignore_index=True,
        )
        return StackedMergeResult(merged_inventory, parts_table, id_log)

    def _stacked_groups(
        self,
        pairs: list[tuple[int, int, float, float]],
        copies: Mapping[int, int],
        features: Sequence[Mapping[str, Any]],
        area: np.ndarray,
        ids: Sequence[str],
    ) -> dict[int, list[int]]:
        """
        Find each nest's head and the members that merge into it.

        Args:
            pairs (list[tuple[int, int, float, float]]):
                The pairs :func:`_pairs` returns.
            copies (Mapping[int, int]):
                The dropped copies, which take no part.
            features (Sequence[Mapping[str, Any]]):
                The features of each footprint.
            area (numpy.ndarray):
                The planar area of each footprint.
            ids (Sequence[str]):
                The ``overture_id`` of each footprint.

        Returns:
            dict[int, list[int]]:
                Each head's position to the positions of its members, in
                ascending order of ``overture_id``. A head with no member
                that shows the signal is absent.
        """
        container: dict[int, int] = {}
        for big, small, iou, share in pairs:
            if (
                iou >= COPY_IOU
                or share < NEST_MIN
                or big in copies
                or small in copies
            ):
                continue
            current = container.get(small)
            if current is None or (-area[big], ids[big]) < (
                -area[current],
                ids[current],
            ):
                container[small] = big

        groups: dict[int, list[int]] = {}
        for small in sorted(container, key=ids.__getitem__):
            top = container[small]
            # A container is larger than the footprint it holds, so the
            # climb ends at the top of the nest.
            while top in container:
                top = container[top]
            small_height = features[small].get(HEIGHT_FEATURE)
            top_height = features[top].get(HEIGHT_FEATURE)
            taller = not is_missing_value(small_height) and (
                is_missing_value(top_height)
                or small_height > top_height + self._taller
            )
            roof = ROOF_CLASS in (
                features[small].get(CLASS_FEATURE),
                features[top].get(CLASS_FEATURE),
            )
            if taller or roof:
                groups.setdefault(top, []).append(small)
        return groups

    @staticmethod
    def _merged_features(
        parts: Sequence[int],
        features: Sequence[Mapping[str, Any]],
        ids: Sequence[str],
    ) -> dict[str, Any]:
        """
        Combine the features of a head and its members.

        Args:
            parts (Sequence[int]):
                The head's position, then its members' in ascending order of
                ``overture_id``.
            features (Sequence[Mapping[str, Any]]):
                The features of each footprint.
            ids (Sequence[str]):
                The ``overture_id`` of each footprint.

        Returns:
            dict[str, Any]:
                The merged asset's features.
        """
        head, *members = parts
        merged = copy.deepcopy(features[head])
        merged[PART_IDS_FEATURE] = PART_IDS_SEPARATOR.join(
            sorted(ids[member] for member in members)
        )
        for key in (HEIGHT_FEATURE, FLOORS_FEATURE, SOURCES_FEATURE):
            merged.pop(key, None)

        # The first of the tallest parts, the head before its members.
        tallest = max(
            (
                part
                for part in parts
                if not is_missing_value(features[part].get(HEIGHT_FEATURE))
            ),
            key=lambda part: features[part][HEIGHT_FEATURE],
            default=None,
        )
        if tallest is not None:
            merged[HEIGHT_FEATURE] = features[tallest][HEIGHT_FEATURE]
        floor_counts = [
            features[part][FLOORS_FEATURE]
            for part in parts
            if not is_missing_value(features[part].get(FLOORS_FEATURE))
        ]
        if floor_counts:
            merged[FLOORS_FEATURE] = max(floor_counts)

        # The stories inferer takes the height's source from the first entry
        # whose property is the height, wherever it stands, so only the
        # tallest part keeps such entries.
        entries = []
        for part in dict.fromkeys(p for p in (tallest, *parts) if p is not None):
            value = features[part].get(SOURCES_FEATURE)
            if isinstance(value, str) and not is_missing_value(value):
                entries.extend(
                    entry
                    for entry in value.split(SOURCES_SEPARATOR)
                    if part == tallest or not _HEIGHT_ENTRY.search(entry)
                )
        if entries:
            merged[SOURCES_FEATURE] = SOURCES_SEPARATOR.join(entries)
        return merged


def _pairs(
    geometries: np.ndarray, area: np.ndarray, ids: Sequence[str]
) -> list[tuple[int, int, float, float]]:
    """
    Find the footprint pairs that are copies or nested.

    Args:
        geometries (numpy.ndarray):
            The footprints as shapely polygons in EPSG:4326.
        area (numpy.ndarray):
            Their planar areas, in square degrees.
        ids (Sequence[str]):
            Their ``overture_id`` values.

    Returns:
        list[tuple[int, int, float, float]]:
            For each pair whose intersection over union is at least
            :data:`COPY_IOU` or whose intersection covers at least
            :data:`NEST_MIN` of the smaller footprint: the position of the
            larger and of the smaller footprint, ties of area going to the
            lower id as the larger, the intersection over union, and the
            share of the smaller footprint inside the larger.
    """
    first, second = shapely.STRtree(geometries).query(
        geometries, predicate='intersects'
    )
    ordered = first < second
    first, second = first[ordered], second[ordered]
    # Each pair in overture_id order, so that its ratios, which can differ
    # in the last bits with the order of the intersection, do not depend on
    # the input order.
    names = np.array(ids, dtype=object)
    swap = names[first] > names[second]
    first, second = np.where(swap, second, first), np.where(swap, first, second)
    # Either kind of pair needs an intersection of at least the lower
    # threshold times the smaller area. The overlap of the bounding boxes
    # bounds the intersection, so footprints that only touch are not
    # intersected.
    bounds = shapely.bounds(geometries)
    width = np.minimum(bounds[first, 2], bounds[second, 2]) - np.maximum(
        bounds[first, 0], bounds[second, 0]
    )
    height = np.minimum(bounds[first, 3], bounds[second, 3]) - np.maximum(
        bounds[first, 1], bounds[second, 1]
    )
    smaller = np.minimum(area[first], area[second])
    candidate = np.clip(width, 0, None) * np.clip(height, 0, None) >= (
        min(COPY_IOU, NEST_MIN) * smaller
    )
    first, second, smaller = first[candidate], second[candidate], smaller[candidate]
    shared = shapely.area(
        shapely.intersection(geometries[first], geometries[second])
    )
    iou = shared / (area[first] + area[second] - shared)
    share = shared / smaller
    paired = (iou >= COPY_IOU) | (share >= NEST_MIN)

    pairs = []
    for one, other, ratio, inside in zip(
        first[paired], second[paired], iou[paired], share[paired]
    ):
        one_is_larger = area[one] > area[other] or (
            area[one] == area[other] and ids[one] <= ids[other]
        )
        big, small = (one, other) if one_is_larger else (other, one)
        pairs.append((int(big), int(small), float(ratio), float(inside)))
    return pairs


def _copy_keepers(
    pairs: list[tuple[int, int, float, float]],
    features: Sequence[Mapping[str, Any]],
    area: np.ndarray,
    ids: Sequence[str],
) -> dict[int, int]:
    """
    Choose the footprint each connected set of copies keeps.

    Args:
        pairs (list[tuple[int, int, float, float]]):
            The pairs :func:`_pairs` returns.
        features (Sequence[Mapping[str, Any]]):
            The features of each footprint.
        area (numpy.ndarray):
            The planar area of each footprint.
        ids (Sequence[str]):
            The ``overture_id`` of each footprint.

    Returns:
        dict[int, int]:
            The position of each dropped copy to that of its set's keeper.
    """
    sets: dict[int, frozenset[int]] = {}
    for big, small, iou, _ in pairs:
        if iou >= COPY_IOU:
            joined = sets.get(big, frozenset({big})) | sets.get(
                small, frozenset({small})
            )
            sets.update(dict.fromkeys(joined, joined))
    dropped = {}
    for members in set(sets.values()):
        keeper, *others = sorted(
            members,
            key=lambda position: (
                is_missing_value(features[position].get(HEIGHT_FEATURE)),
                -area[position],
                ids[position],
            ),
        )
        dropped.update(dict.fromkeys(others, keeper))
    return dropped


def _footprints(
    inventory: AssetInventory,
) -> tuple[list[str], list[Mapping[str, Any]], list[Polygon | MultiPolygon]]:
    """
    Check an inventory the merge takes and read its footprints.

    Args:
        inventory (AssetInventory):
            The inventory passed to :meth:`StackedFootprintMerge.merge`.

    Returns:
        tuple[list[str], list[Mapping[str, Any]], list[Polygon | MultiPolygon]]:
            The asset keys in the inventory's order, and each asset's
            features and footprint polygon.

    Raises:
        ValueError:
            On any of the conditions :meth:`StackedFootprintMerge.merge`
            lists.
    """
    if inventory.n_pw != 1:
        msg = (
            f'The inventory holds {inventory.n_pw} possible worlds; the '
            'merge takes footprints of one.'
        )
        raise ValueError(msg)
    keys = list(inventory.inventory)
    features = []
    polygons = []
    for key in keys:
        asset = inventory.inventory[key]
        if not isinstance(key, str):
            msg = (
                f'The asset key {key!r} is not a string. The merge re-keys '
                'by geohash and takes string keys, as the geohash id scheme '
                'writes them.'
            )
            raise ValueError(msg)  # noqa: TRY004
        provider = asset.features.get(FOOTPRINT_PROVIDER_FEATURE)
        if provider != OVERTURE_PROVIDER or is_missing_value(
            asset.features.get(OVERTURE_ID_FEATURE)
        ):
            msg = (
                f'The asset {key!r} is not an Overture footprint: the merge '
                f'needs {FOOTPRINT_PROVIDER_FEATURE} {OVERTURE_PROVIDER!r} '
                f'and an {OVERTURE_ID_FEATURE}.'
            )
            raise ValueError(msg)
        geometry = asset.get_geometry()
        if not isinstance(geometry, (Polygon, MultiPolygon)) or not geometry.area:
            msg = f'The footprint of asset {key!r} is not a polygon with area.'
            raise ValueError(msg)
        features.append(asset.features)
        polygons.append(geometry)
    return keys, features, polygons
