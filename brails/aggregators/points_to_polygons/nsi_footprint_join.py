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
Join NSI points to building footprints.

Every NSI point is either linked to one footprint or returned as unlinked,
never dropped.

.. autosummary::

      NSIFootprintJoin
      NSIJoinResult
"""

from __future__ import annotations

import copy
import math
import secrets
from typing import TYPE_CHECKING, Any, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from brails.scrapers.nsi_parser.nsi_conventions import to_simcenter
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils.deterministic_sampling import asset_key_digest, keyed_uniforms
from brails.utils.geo_tools import GeoTools
from brails.utils.metric_crs import meter_crs

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pyproj import CRS

# Purpose label of the walk order's keyed uniform. Renaming it moves every
# walk order and therefore every link the walk makes.
WALK_PURPOSE = 'nsi_footprint_walk'

# World of the walk order's keyed uniform.
WALK_WORLD = 0

# Default distance cap of the walk in meters.
DEFAULT_CAP_M = 15.0

# The housing classes, the classes whose records carry a unit count. RES4 to
# RES6 are residential classes in Hazus but carry no unit count here.
RES3_CLASSES = ('RES3A', 'RES3B', 'RES3C', 'RES3D', 'RES3E', 'RES3F')
UNIT_CLASSES = ('RES1', 'RES2', *RES3_CLASSES)

# Lower bound of each count band, the unit ranges of the occupancy classes in
# the Hazus Inventory Technical Manual, Hazus 7.0 (FEMA, June 2025), Table 4-1,
# p. 4-2. A unit count falls in the band whose lower bound it has reached.
COUNT_BANDS = (
    (50, 'RES3F'),
    (20, 'RES3E'),
    (10, 'RES3D'),
    (5, 'RES3C'),
    (3, 'RES3B'),
    (2, 'RES3A'),
    (0, 'RES1'),
)

# Families whose class is the count band of the unit count. A RES2 record
# keeps RES2 whatever its unit count.
COUNTED_FAMILIES = ('RES1', 'RES3')

# Units a housing record contributes when its resunits is zero, negative or
# missing: the midpoint of its class's unit band, one dwelling for RES1 and
# RES2, and a fixed count for the open RES3F band.
UNIT_MIDPOINTS = {
    'RES1': 1.0,
    'RES2': 1.0,
    'RES3A': 2.0,
    'RES3B': 3.5,
    'RES3C': 7.0,
    'RES3D': 14.5,
    'RES3E': 34.5,
    'RES3F': 100.0,
}

# Walk groups (see walk_group).
GROUP_SINGLE = 'single'
GROUP_MULTI_FAMILY = 'multi_family'
GROUP_NON_RESIDENTIAL = 'non_residential'
GROUPS = (GROUP_SINGLE, GROUP_MULTI_FAMILY, GROUP_NON_RESIDENTIAL)
SINGLE_CLASSES = ('RES1', 'RES2')

BRANCH_INSIDE = 'inside'
BRANCH_WALK_TO_EMPTY = 'walk_to_empty'
BRANCH_WALK_TO_SHARED = 'walk_to_shared'
REASON_NO_FOOTPRINT = 'no_footprint_within_cap'
REASON_REFUSED = 'refused'
STACKED_OUTSIDE = 'stacked_outside_footprints'
BRANCH_COUNT_ROWS = (
    BRANCH_INSIDE,
    BRANCH_WALK_TO_EMPTY,
    BRANCH_WALK_TO_SHARED,
    f'unlinked_{REASON_NO_FOOTPRINT}',
    f'unlinked_{REASON_REFUSED}',
    STACKED_OUTSIDE,
)
ALL_GROUPS = 'all'

REACH_INSIDE = 'inside'
REACH_WALK = 'walk'

STATUS_COUNTED = 'counted'
STATUS_MIDPOINT_FILLED = 'midpoint_filled'
STATUS_NONE = 'none'

LINK_COLUMNS = ('fd_id', 'asset_id', 'branch', 'distance_m', 'walk_position')
UNLINKED_COLUMNS = (
    'fd_id',
    'occtype',
    'point_class',
    'resunits',
    'units',
    'cbfips',
    'x',
    'y',
    'reason',
)

# NSI fields the join reads, under their NSI names.
OCCUPANCY_FIELD = 'occtype'
UNITS_FIELD = 'resunits'
ID_FIELD = 'fd_id'
AREA_FIELD = 'sqft'
VALUE_FIELD = 'fullrep'
BLOCK_FIELD = 'cbfips'

# SimCenter names of the summed fields, as to_simcenter writes them.
GROSS_AREA = 'GrossArea'
REPLACEMENT_COST = 'ReplacementCost'

# Values _preview lists before counting the rest.
_PREVIEW = 10


class NSIJoinResult(NamedTuple):
    """
    The footprints' NSI values and the record of every point's link.

    Attributes:
        observed (AssetInventory):
            One asset per footprint that received a point, keyed by its
            asset id in ascending order, with the footprint's geometry and
            the values NSI states. These are the governing record's fields
            through ``to_simcenter``, without ``Longitude``, ``Latitude``,
            ``GrossArea``, ``ReplacementCost``, ``resunits`` and ``fd_id``,
            with its class renamed ``NSIOccupancyClass``. Every other field
            the governing record carries is that one record's value and not
            a sum over the footprint's records. Beside them are
            ``OccupancyClass`` where the governing record's class is taken
            as it is; ``NumberOfUnits`` where every unit was counted;
            ``GrossArea`` and ``ReplacementCost`` where one received record
            states the value; and ``NSIGoverningRecord``, ``NSIRecords``,
            ``NSIReach`` and ``NumberOfUnitsStatus``.
        derived (AssetInventory):
            One asset per footprint for which the join derived a value,
            keyed by its asset id in ascending order, with the footprint's
            geometry: ``OccupancyClass`` where the count band set it,
            ``NumberOfUnits`` where any record's units were filled by a
            class midpoint, and ``GrossArea`` and ``ReplacementCost`` where
            two or more received records state a value, as their sums.
        links (pandas.DataFrame):
            One row per linked point, in ascending ``fd_id``, with the
            columns in :data:`LINK_COLUMNS`. ``walk_position`` is the
            point's place in the walk order, from 0, and is missing for a
            point linked inside its footprint.
        unlinked (pandas.DataFrame):
            One row per point no footprint took, in ascending ``fd_id``,
            with the columns in :data:`UNLINKED_COLUMNS`. ``units`` is what
            the point would contribute to a unit count, missing for a
            non-housing point. ``reason`` is
            ``'no_footprint_within_cap'`` or ``'refused'``.
        branch_counts (pandas.DataFrame):
            Points by branch (rows :data:`BRANCH_COUNT_ROWS`) and by walk
            group (the columns :data:`GROUPS` and ``'all'``). The last row
            counts the records outside every footprint that share their
            coordinate with another record.
        seed (int):
            The seed of the walk order.
        crs (pyproj.CRS):
            The projected coordinate reference system of the distances.
    """

    observed: AssetInventory
    derived: AssetInventory
    links: pd.DataFrame
    unlinked: pd.DataFrame
    branch_counts: pd.DataFrame
    seed: int
    crs: CRS


def _ordering_key(value: object) -> tuple[int, float, str, str]:
    """
    Build a sort key that orders identifiers of mixed types.

    Args:
        value (object):
            An identifier value.

    Returns:
        tuple[int, float, str, str]:
            A key comparable against every other key this function returns.
    """
    text = str(value)
    type_name = type(value).__name__
    if value is None:
        return (2, 0.0, text, type_name)
    try:
        number = float(value)
    except OverflowError:
        return (2, 0.0, text, type_name)
    except (TypeError, ValueError):
        return (1, 0.0, text, type_name)
    if not math.isfinite(number):
        return (2, 0.0, text, type_name)
    return (0, number, text, type_name)


def _stated_number(value: object) -> float | int | None:
    """
    Return a feature value that states a finite number, or None.

    Args:
        value (object):
            A raw feature value.

    Returns:
        float or int or None:
            The value itself where it is a finite int or float (a bool is
            not), otherwise None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    number = value.item() if isinstance(value, np.number) else value
    return number if math.isfinite(number) else None


def parse_occupancy_class(occtype: object) -> str | None:
    """
    Parse an NSI occupancy code to its class.

    Args:
        occtype (object):
            An NSI ``occtype`` such as ``'RES1-1SNB'``, ``'RES3A'`` or
            ``'COM1'``.

    Returns:
        str or None:
            The part before the first hyphen in upper case, with a bare
            ``RES3`` read as ``RES3A``; None where the code is not a
            string or is blank.
    """
    if not isinstance(occtype, str):
        return None
    head = occtype.split('-')[0].strip().upper()
    if head == 'RES3':
        return 'RES3A'
    return head or None


def count_band(units: float) -> str:
    """
    Return the class of the count band a unit count falls in.

    Args:
        units (float):
            A unit count, summed or per record.

    Returns:
        str:
            The class of its band in :data:`COUNT_BANDS`.
    """
    for lower_bound, name in COUNT_BANDS:
        if units >= lower_bound:
            return name
    return 'RES1'


def unit_contribution(
    occupancy_class: str, resunits: object
) -> tuple[float | int | None, bool]:
    """
    Return the units one NSI record contributes to a count.

    Args:
        occupancy_class (str):
            The record's parsed class.
        resunits (object):
            The record's ``resunits``, or None where it has none.

    Returns:
        tuple[float or int or None, bool]:
            For a housing class (:data:`UNIT_CLASSES`), its
            ``resunits`` where that is a positive number, otherwise the
            class midpoint, and whether the midpoint was used. For any
            other class, None and False.
    """
    if occupancy_class not in UNIT_CLASSES:
        return None, False
    stated = _stated_number(resunits)
    if stated is not None and stated > 0:
        return stated, False
    return UNIT_MIDPOINTS[occupancy_class], True


def point_class(occupancy_class: str, units: float | None) -> str:
    """
    Return a record's class after the count rule.

    Args:
        occupancy_class (str):
            The record's parsed class.
        units (float or None):
            The units it contributes (:func:`unit_contribution`).

    Returns:
        str:
            The count band of ``units`` (:func:`count_band`) for a record
            of :data:`COUNTED_FAMILIES`, otherwise the class itself.
    """
    # RES3A to RES3F count as RES3
    family = 'RES3' if occupancy_class in RES3_CLASSES else occupancy_class
    if family in COUNTED_FAMILIES and units is not None:
        return count_band(units)
    return occupancy_class


def walk_group(classified: str) -> str:
    """
    Return the walk group of a class after the count rule.

    Args:
        classified (str):
            A class from :func:`point_class`.

    Returns:
        str:
            ``'single'`` for RES1 and RES2, ``'multi_family'`` for RES3A to
            RES3F, and ``'non_residential'`` for every other class, RES4 to
            RES6 included.
    """
    if classified in SINGLE_CLASSES:
        return GROUP_SINGLE
    if classified in RES3_CLASSES:
        return GROUP_MULTI_FAMILY
    return GROUP_NON_RESIDENTIAL


def _preview(values: Iterable[object]) -> str:
    """
    List the leading values and count the rest.

    Args:
        values (Iterable[object]):
            The values to name.

    Returns:
        str:
            The leading values, comma separated, and the number not shown.
    """
    values = list(values)
    shown = ', '.join(str(value) for value in values[:_PREVIEW])
    more = len(values) - _PREVIEW
    return f'{shown} and {more} more' if more > 0 else shown


class _Point(NamedTuple):
    """One NSI record as the join reads it."""

    fd_id: Any
    features: dict[str, Any]
    x: float
    y: float
    nsi_class: str
    point_class: str
    group: str
    units: float | int | None
    filled: bool


class NSIFootprintJoin:
    """
    Join NSI points to footprints, one footprint per point at most.

    A point that a footprint covers, its boundary included, belongs to it. A
    point inside several footprints goes to the smallest, then to the one
    whose centroid is nearest, then to the lowest asset id. The remaining
    points are taken in the order of a keyed uniform of their ``fd_id`` (see
    :meth:`walk_uniforms`), ties by ``fd_id``, and each takes the nearest
    footprint within ``cap_m`` that accepts it, ties by asset id. The result
    is the same in every possible world. A footprint holding no point
    accepts any point. A footprint holding points refuses a point of the
    single walk group, and accepts any other point only while the class it
    takes from the points it already holds (below) is outside
    :data:`SINGLE_CLASSES`. A point that no footprint within the cap accepts
    is returned as unlinked.

    A point's class is its parsed ``occtype`` after the count rule (see
    :func:`point_class`), and its walk group follows from that class (see
    :func:`walk_group`). The join reads footprint geometry only, so
    footprint features held per possible world do not change it.

    Per footprint, the units are the sum of the housing records'
    contributions (see :func:`unit_contribution`). The governing record is
    chosen among every record the footprint receives, inside or by the
    walk: a housing class (:data:`UNIT_CLASSES`) first, then the largest
    stated ``sqft``, a record stating none ranking last, then the lowest
    ``fd_id``. The footprint's class is the count band of the units where
    the governing record's family is in :data:`COUNTED_FAMILIES`, and
    otherwise the governing record's class.

    To import the :class:`NSIFootprintJoin` class, use:

    .. code-block:: python

        from brails.aggregators import NSIFootprintJoin
    """

    def __init__(
        self,
        cap_m: float = DEFAULT_CAP_M,
        seed: int | None = None,
        projected_crs: object = None,
    ) -> None:
        """
        Initialize the join with its cap, seed and distance CRS.

        Args:
            cap_m (float):
                The walk's distance cap in meters, from a point to a
                footprint's edge, inclusive. One value for every point.
            seed (int, optional):
                The seed of the walk order. When None, a seed is drawn from
                system entropy and recorded on :attr:`seed`.
            projected_crs (object, optional):
                A projected CRS in meters, in any form ``pyproj`` accepts,
                for areas and distances. Web Mercator (EPSG:3857) is
                refused. When None, a UTM zone is estimated from the
                footprints on each call and returned in the result.

        Raises:
            ValueError:
                If ``cap_m`` is negative or not a finite number, or
                ``projected_crs`` is not a projected CRS in meters or is
                Web Mercator.
            TypeError:
                If ``seed`` is neither None nor an integer.
        """
        cap = _stated_number(cap_m)
        if cap is None or cap < 0:
            raise ValueError(
                f'cap_m must be a finite number of meters >= 0: {cap_m!r}'
            )
        if seed is None:
            seed = secrets.randbits(64)
        elif isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f'seed must be an integer or None: {seed!r}')
        self.cap_m = float(cap)
        self.seed = int(seed)
        self.projected_crs = (
            None if projected_crs is None else meter_crs(projected_crs)
        )

    def walk_uniforms(self, fd_ids: Sequence[Any]) -> np.ndarray:
        """
        Return the walk-order uniform of each point.

        Args:
            fd_ids (Sequence[Any]):
                The points' ``fd_id`` values.

        Returns:
            numpy.ndarray:
                One uniform per id, in the order given. A point's uniform
                depends on its id and the seed only.
        """
        return keyed_uniforms(
            asset_key_digest(fd_ids),
            seed=self.seed,
            purpose=WALK_PURPOSE,
            world=WALK_WORLD,
        )

    def join(
        self, footprints: AssetInventory, points: AssetInventory
    ) -> NSIJoinResult:
        """
        Join NSI points to footprints.

        Args:
            footprints (AssetInventory):
                Polygon assets in EPSG:4326, keyed by asset id.
            points (AssetInventory):
                NSI point assets keyed by ``fd_id``, with features under NSI
                names, as :class:`~brails.scrapers.NSIFileReader` reads them.

        Returns:
            NSIJoinResult:
                The footprints' NSI values and the record of every link.

        Raises:
            TypeError:
                If either input is not an ``AssetInventory``.
            ValueError:
                If either inventory is empty, a footprint is not a polygon,
                a point is not a single point, a point has no parseable
                ``occtype``, or the points hold values per possible world.
        """
        footprint_ids, polygons = self._read_footprints(footprints)
        records = self._read_points(points)

        crs = (
            self.projected_crs
            if self.projected_crs is not None
            else gpd.GeoSeries(polygons, crs='EPSG:4326').estimate_utm_crs()
        )
        polygons_m = np.asarray(
            gpd.GeoSeries(polygons, crs='EPSG:4326').to_crs(crs).array,
            dtype=object,
        )
        point_geoms = shapely.points(
            [record.x for record in records], [record.y for record in records]
        )
        points_m = np.asarray(
            gpd.GeoSeries(point_geoms, crs='EPSG:4326').to_crs(crs).array,
            dtype=object,
        )
        # rank of each footprint id in canonical id order
        order = sorted(
            range(len(footprint_ids)),
            key=lambda index: _ordering_key(footprint_ids[index]),
        )
        footprint_rank = np.empty(len(footprint_ids), dtype=np.int64)
        footprint_rank[order] = np.arange(len(footprint_ids))

        inside = self._inside_pass(
            polygons, polygons_m, point_geoms, points_m, footprint_rank
        )
        received: dict[int, list[tuple[int, str]]] = {}
        links: list[dict[str, Any]] = []
        for point_index, footprint_index in sorted(inside.items()):
            received.setdefault(footprint_index, []).append(
                (point_index, BRANCH_INSIDE)
            )
            links.append(
                {
                    'fd_id': records[point_index].fd_id,
                    'asset_id': footprint_ids[footprint_index],
                    'branch': BRANCH_INSIDE,
                    'distance_m': 0.0,
                    'walk_position': None,
                }
            )

        held_single = np.zeros(len(footprint_ids), dtype=bool)
        for footprint_index, arrivals in received.items():
            held_single[footprint_index] = _holds_single(arrivals, records)

        walkers = [index for index in range(len(records)) if index not in inside]
        candidates = self._candidates(walkers, points_m, polygons_m, footprint_rank)
        uniforms = self.walk_uniforms([records[index].fd_id for index in walkers])
        order = sorted(
            range(len(walkers)),
            key=lambda k: (uniforms[k], _ordering_key(records[walkers[k]].fd_id)),
        )

        unlinked: list[tuple[int, str]] = []
        for position, k in enumerate(order):
            point_index = walkers[k]
            group = records[point_index].group
            options = candidates.get(point_index, [])
            for footprint_index, distance in options:
                held = footprint_index in received
                if held and (held_single[footprint_index] or group == GROUP_SINGLE):
                    continue
                branch = BRANCH_WALK_TO_SHARED if held else BRANCH_WALK_TO_EMPTY
                arrivals = received.setdefault(footprint_index, [])
                arrivals.append((point_index, branch))
                held_single[footprint_index] = _holds_single(arrivals, records)
                links.append(
                    {
                        'fd_id': records[point_index].fd_id,
                        'asset_id': footprint_ids[footprint_index],
                        'branch': branch,
                        'distance_m': distance,
                        'walk_position': position,
                    }
                )
                break
            else:
                reason = REASON_REFUSED if options else REASON_NO_FOOTPRINT
                unlinked.append((point_index, reason))

        if len(links) + len(unlinked) != len(records):
            raise RuntimeError(
                f'The join accounted for {len(links) + len(unlinked)} of '
                f'{len(records)} points.'
            )

        observed, derived = self._per_footprint(
            footprints, footprint_ids, received, records
        )
        # links table, sorted by fd_id
        rows = sorted(links, key=lambda row: _ordering_key(row['fd_id']))
        links_frame = pd.DataFrame(rows, columns=list(LINK_COLUMNS))
        links_frame['distance_m'] = links_frame['distance_m'].astype(float)
        links_frame['walk_position'] = pd.array(
            [row['walk_position'] for row in rows], dtype='Int64'
        )
        return NSIJoinResult(
            observed=observed,
            derived=derived,
            links=links_frame,
            unlinked=_unlinked_frame(unlinked, records),
            branch_counts=_branch_counts(links, unlinked, records, inside),
            seed=self.seed,
            crs=crs,
        )

    @staticmethod
    def _read_footprints(
        footprints: AssetInventory,
    ) -> tuple[list[Any], np.ndarray]:
        """
        Read the footprints' ids and polygons.

        Args:
            footprints (AssetInventory):
                Polygon assets in EPSG:4326.

        Returns:
            tuple[list, numpy.ndarray]:
                The asset ids in inventory order and one shapely polygon or
                multipolygon per id.

        Raises:
            TypeError:
                If the input is not an ``AssetInventory``.
            ValueError:
                If it is empty or an asset is not a polygon.
        """
        if not isinstance(footprints, AssetInventory):
            raise TypeError(
                'footprints must be an AssetInventory, not '
                f'{type(footprints).__name__}.'
            )
        if not footprints.inventory:
            raise ValueError('The footprint inventory is empty.')
        ids = list(footprints.inventory)
        geometries = []
        bad = []
        for asset_id in ids:
            coordinates = footprints.inventory[asset_id].coordinates
            try:
                geometry = GeoTools.list_of_lists_to_geometry(coordinates)
            except (ValueError, TypeError):
                geometry = None
            if geometry is None or geometry.geom_type not in (
                'Polygon',
                'MultiPolygon',
            ):
                bad.append(asset_id)
            geometries.append(geometry)
        if bad:
            raise ValueError(
                f'{len(bad)} footprint(s) are not polygons: {_preview(bad)}.'
            )
        return ids, np.asarray(geometries, dtype=object)

    @staticmethod
    def _read_points(points: AssetInventory) -> list[_Point]:
        """
        Read and classify the NSI points.

        Args:
            points (AssetInventory):
                NSI point assets keyed by ``fd_id``.

        Returns:
            list[_Point]:
                One record per point, in inventory order.

        Raises:
            TypeError:
                If the input is not an ``AssetInventory``.
            ValueError:
                If it is empty, holds values per possible world, or has a
                point that is not a single coordinate or has no parseable
                ``occtype``.
        """
        if not isinstance(points, AssetInventory):
            raise TypeError(
                f'points must be an AssetInventory, not {type(points).__name__}.'
            )
        if not points.inventory:
            raise ValueError('The point inventory is empty.')
        if points.n_pw != 1:
            raise ValueError(
                f'The point inventory holds {points.n_pw} possible worlds. The '
                'join reads one NSI record per point and does not support '
                'values per world.'
            )
        records = []
        not_points = []
        no_class = []
        for fd_id, asset in points.inventory.items():
            coordinates = asset.coordinates
            if (
                len(coordinates) != 1
                or len(coordinates[0]) != 2  # noqa: PLR2004
                or _stated_number(coordinates[0][0]) is None
                or _stated_number(coordinates[0][1]) is None
            ):
                not_points.append(fd_id)
                continue
            features = asset.features
            nsi_class = parse_occupancy_class(features.get(OCCUPANCY_FIELD))
            if nsi_class is None:
                no_class.append(fd_id)
                continue
            units, filled = unit_contribution(nsi_class, features.get(UNITS_FIELD))
            classified = point_class(nsi_class, units)
            records.append(
                _Point(
                    fd_id=fd_id,
                    features=features,
                    x=float(coordinates[0][0]),
                    y=float(coordinates[0][1]),
                    nsi_class=nsi_class,
                    point_class=classified,
                    group=walk_group(classified),
                    units=units,
                    filled=filled,
                )
            )
        if not_points:
            raise ValueError(
                f'{len(not_points)} NSI asset(s) are not single points: '
                f'{_preview(not_points)}.'
            )
        if no_class:
            raise ValueError(
                f'{len(no_class)} NSI point(s) carry no parseable '
                f'{OCCUPANCY_FIELD}: {_preview(no_class)}.'
            )
        return records

    @staticmethod
    def _inside_pass(
        polygons: np.ndarray,
        polygons_m: np.ndarray,
        point_geoms: np.ndarray,
        points_m: np.ndarray,
        footprint_rank: np.ndarray,
    ) -> dict[int, int]:
        """
        Assign every point that a footprint covers.

        Args:
            polygons (numpy.ndarray):
                The footprints in EPSG:4326.
            polygons_m (numpy.ndarray):
                The footprints in the projected CRS.
            point_geoms (numpy.ndarray):
                The points in EPSG:4326.
            points_m (numpy.ndarray):
                The points in the projected CRS.
            footprint_rank (numpy.ndarray):
                Each footprint's rank in ascending asset id.

        Returns:
            dict[int, int]:
                The footprint index of each point inside one, by point
                index.
        """
        tree = shapely.STRtree(polygons)
        point_index, footprint_index = tree.query(
            point_geoms, predicate='intersects'
        )
        if point_index.size == 0:
            return {}
        hits = pd.DataFrame(
            {
                'point': point_index,
                'footprint': footprint_index,
                'area': shapely.area(polygons_m[footprint_index]),
                'centroid_distance': shapely.distance(
                    shapely.centroid(polygons_m[footprint_index]),
                    points_m[point_index],
                ),
                'rank': footprint_rank[footprint_index],
            }
        )
        hits = hits.sort_values(
            ['point', 'area', 'centroid_distance', 'rank'], kind='mergesort'
        ).drop_duplicates('point')
        return dict(zip(hits['point'].tolist(), hits['footprint'].tolist()))

    def _candidates(
        self,
        walkers: list[int],
        points_m: np.ndarray,
        polygons_m: np.ndarray,
        footprint_rank: np.ndarray,
    ) -> dict[int, list[tuple[int, float]]]:
        """
        List the footprints within the cap of each walking point.

        Args:
            walkers (list[int]):
                Indices of the points outside every footprint.
            points_m (numpy.ndarray):
                The points in the projected CRS.
            polygons_m (numpy.ndarray):
                The footprints in the projected CRS.
            footprint_rank (numpy.ndarray):
                Each footprint's rank in ascending asset id.

        Returns:
            dict[int, list[tuple[int, float]]]:
                By point index, the footprints within the cap as
                ``(footprint index, distance in meters)``, nearest first,
                ties by asset id. A point with none is absent.
        """
        if not walkers:
            return {}
        walker_array = np.asarray(walkers)
        tree = shapely.STRtree(polygons_m)
        query_index, footprint_index = tree.query(
            points_m[walker_array], predicate='dwithin', distance=self.cap_m
        )
        if query_index.size == 0:
            return {}
        pairs = pd.DataFrame(
            {
                'point': walker_array[query_index],
                'footprint': footprint_index,
                'distance': shapely.distance(
                    points_m[walker_array[query_index]], polygons_m[footprint_index]
                ),
                'rank': footprint_rank[footprint_index],
            }
        )
        pairs = pairs[pairs['distance'] <= self.cap_m].sort_values(
            ['point', 'distance', 'rank'], kind='mergesort'
        )
        candidates: dict[int, list[tuple[int, float]]] = {}
        for point, footprint, distance in zip(
            pairs['point'].tolist(),
            pairs['footprint'].tolist(),
            pairs['distance'].tolist(),
        ):
            candidates.setdefault(point, []).append((footprint, distance))
        return candidates

    @staticmethod
    def _per_footprint(
        footprints: AssetInventory,
        footprint_ids: list[Any],
        received: dict[int, list[tuple[int, str]]],
        records: list[_Point],
    ) -> tuple[AssetInventory, AssetInventory]:
        """
        Build the observed and derived values of each receiving footprint.

        Args:
            footprints (AssetInventory):
                The footprint inventory, for its geometry.
            footprint_ids (list):
                The footprint ids by index.
            received (dict):
                By footprint index, its points as ``(point index,
                branch)``.
            records (list[_Point]):
                The points.

        Returns:
            tuple[AssetInventory, AssetInventory]:
                The observed and the derived inventory, in ascending asset
                id.
        """
        observed = AssetInventory()
        derived = AssetInventory()
        for footprint_index in sorted(
            received, key=lambda index: _ordering_key(footprint_ids[index])
        ):
            asset_id = footprint_ids[footprint_index]
            arrivals = received[footprint_index]
            governing = _governing(arrivals, records)
            features, extra = _footprint_values(arrivals, governing, records)
            coordinates = footprints.inventory[asset_id].coordinates
            observed.add_asset(
                asset_id, Asset(asset_id, copy.deepcopy(coordinates), features)
            )
            if extra:
                derived.add_asset(
                    asset_id, Asset(asset_id, copy.deepcopy(coordinates), extra)
                )
        return observed, derived


def _governing(arrivals: list[tuple[int, str]], records: list[_Point]) -> int:
    """
    Choose the governing record of a footprint by the governing key.

    Args:
        arrivals (list):
            The footprint's points as ``(point index, branch)``.
        records (list[_Point]):
            The points.

    Returns:
        int:
            The index of the point that ranks first by
            :func:`_governing_key`. The order of arrival plays no part.
    """
    return min(
        (index for index, _ in arrivals),
        key=lambda index: _governing_key(records[index]),
    )


def _governing_key(record: _Point) -> tuple[bool, bool, float, tuple]:
    """
    Build the key that ranks a footprint's records for governance.

    Args:
        record (_Point):
            One received point.

    Returns:
        tuple:
            A key that sorts a housing class first, then the largest stated
            ``sqft`` (a record stating none after every record that states
            one), then the lowest ``fd_id``.
    """
    area = _stated_number(record.features.get(AREA_FIELD))
    return (
        record.nsi_class not in UNIT_CLASSES,
        area is None,
        0.0 if area is None else -float(area),
        _ordering_key(record.fd_id),
    )


def _unit_total(members: list[_Point]) -> float | int | None:
    """
    Sum the units of a footprint's housing records.

    Args:
        members (list[_Point]):
            The footprint's points.

    Returns:
        float or int or None:
            The sum of their contributions, or None where none of them is a
            housing record.
    """
    housing = [member.units for member in members if member.units is not None]
    return sum(housing) if housing else None


def _footprint_class(lead: _Point, units: float | None) -> tuple[str, bool]:
    """
    Return a footprint's class and whether the count band set it.

    Args:
        lead (_Point):
            The footprint's governing record.
        units (float or None):
            The footprint's units (:func:`_unit_total`).

    Returns:
        tuple[str, bool]:
            The count band of ``units`` and True where ``units`` is not None
            and the governing record's family is in
            :data:`COUNTED_FAMILIES`, otherwise the governing record's class
            and False.
    """
    # RES3A to RES3F count as RES3
    family = 'RES3' if lead.nsi_class in RES3_CLASSES else lead.nsi_class
    if family in COUNTED_FAMILIES and units is not None:
        return count_band(units), True
    return lead.nsi_class, False


def _holds_single(arrivals: list[tuple[int, str]], records: list[_Point]) -> bool:
    """
    Tell whether the class a footprint takes from its points is single.

    Args:
        arrivals (list):
            The footprint's points as ``(point index, branch)``.
        records (list[_Point]):
            The points.

    Returns:
        bool:
            Whether the footprint's class is in :data:`SINGLE_CLASSES`.
    """
    lead = records[_governing(arrivals, records)]
    units = _unit_total([records[index] for index, _ in arrivals])
    return _footprint_class(lead, units)[0] in SINGLE_CLASSES


def _footprint_values(
    arrivals: list[tuple[int, str]],
    governing: int,
    records: list[_Point],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Compute a footprint's observed and derived features.

    Args:
        arrivals (list):
            The footprint's points as ``(point index, branch)``.
        governing (int):
            The index of the governing point.
        records (list[_Point]):
            The points.

    Returns:
        tuple[dict, dict]:
            The observed and the derived features.
    """
    members = [records[index] for index, _ in arrivals]
    lead = records[governing]

    observed = to_simcenter(lead.features)
    for name in (
        'Longitude',
        'Latitude',
        GROSS_AREA,
        REPLACEMENT_COST,
        UNITS_FIELD,
        ID_FIELD,
    ):
        observed.pop(name, None)
    nsi_label = observed.pop('OccupancyClass', None)
    if nsi_label is not None:
        observed['NSIOccupancyClass'] = nsi_label
    derived: dict[str, Any] = {}

    units = _unit_total(members)
    if units is None:
        observed['NumberOfUnitsStatus'] = STATUS_NONE
    elif any(member.filled for member in members):
        observed['NumberOfUnitsStatus'] = STATUS_MIDPOINT_FILLED
        derived['NumberOfUnits'] = float(units)
    else:
        observed['NumberOfUnitsStatus'] = STATUS_COUNTED
        observed['NumberOfUnits'] = units

    footprint_class, banded = _footprint_class(lead, units)
    if banded:
        derived['OccupancyClass'] = footprint_class
    else:
        observed['OccupancyClass'] = footprint_class

    for field, name in ((AREA_FIELD, GROSS_AREA), (VALUE_FIELD, REPLACEMENT_COST)):
        stated = [
            value
            for value in (
                _stated_number(member.features.get(field)) for member in members
            )
            if value is not None
        ]
        if len(stated) == 1:
            observed[name] = stated[0]
        elif len(stated) > 1:
            derived[name] = float(sum(stated))

    observed['NSIGoverningRecord'] = lead.fd_id
    observed['NSIRecords'] = ';'.join(
        str(fd_id)
        for fd_id in sorted((member.fd_id for member in members), key=_ordering_key)
    )
    observed['NSIReach'] = (
        REACH_INSIDE
        if any(branch == BRANCH_INSIDE for _, branch in arrivals)
        else REACH_WALK
    )
    return observed, derived


def _unlinked_frame(
    unlinked: list[tuple[int, str]], records: list[_Point]
) -> pd.DataFrame:
    """
    Tabulate the unlinked points in ascending ``fd_id``.

    Args:
        unlinked (list[tuple[int, str]]):
            ``(point index, reason)`` per unlinked point.
        records (list[_Point]):
            The points.

    Returns:
        pandas.DataFrame:
            The columns of :data:`UNLINKED_COLUMNS`.
    """
    rows = []
    for index, reason in unlinked:
        record = records[index]
        rows.append(
            {
                'fd_id': record.fd_id,
                'occtype': record.features.get(OCCUPANCY_FIELD),
                'point_class': record.point_class,
                'resunits': record.features.get(UNITS_FIELD),
                'units': np.nan if record.units is None else float(record.units),
                'cbfips': record.features.get(BLOCK_FIELD),
                'x': record.x,
                'y': record.y,
                'reason': reason,
            }
        )
    rows.sort(key=lambda row: _ordering_key(row['fd_id']))
    return pd.DataFrame(rows, columns=list(UNLINKED_COLUMNS))


def _branch_counts(
    links: list[dict[str, Any]],
    unlinked: list[tuple[int, str]],
    records: list[_Point],
    inside: dict[int, int],
) -> pd.DataFrame:
    """
    Count the points by branch and walk group.

    Args:
        links (list[dict]):
            One entry per linked point.
        unlinked (list[tuple[int, str]]):
            ``(point index, reason)`` per unlinked point.
        records (list[_Point]):
            The points.
        inside (dict[int, int]):
            The footprint of each point inside one.

    Returns:
        pandas.DataFrame:
            Rows :data:`BRANCH_COUNT_ROWS`, columns :data:`GROUPS` and
            ``'all'``.
    """
    counts = pd.DataFrame(
        0, index=list(BRANCH_COUNT_ROWS), columns=[*GROUPS, ALL_GROUPS]
    )
    group_of = {record.fd_id: record.group for record in records}
    for row in links:
        counts.loc[row['branch'], group_of[row['fd_id']]] += 1
    for index, reason in unlinked:
        counts.loc[f'unlinked_{reason}', records[index].group] += 1

    at_coordinate: dict[tuple[float, float], int] = {}
    for record in records:
        key = (record.x, record.y)
        at_coordinate[key] = at_coordinate.get(key, 0) + 1
    for index, record in enumerate(records):
        if index not in inside and at_coordinate[(record.x, record.y)] > 1:
            counts.loc[STACKED_OUTSIDE, record.group] += 1

    counts[ALL_GROUPS] = counts[list(GROUPS)].sum(axis=1)
    counts.index.name = 'branch'
    return counts
