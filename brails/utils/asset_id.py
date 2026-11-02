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
Assign building footprints an asset id computed from their geometry.

The id of a footprint is the geohash of the planar centroid of its exterior
rings in EPSG:4326, with collisions resolved as :func:`assign_asset_ids`
describes. A change to any routine here changes ids. The module imports the
standard library only, so a copy of it used outside BRAILS gives the same ids.

.. autosummary::

      AssetIdResult
      assign_asset_ids
      centroid
      centroid_geohash
      geohash
      has_area
      plan_area_m2
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

GEOHASH_PRECISION = 10
SUFFIX_SEPARATOR = '-'
EARTH_RADIUS_M = 6371008.8

GEOHASH_ALPHABET = '0123456789bcdefghjkmnpqrstuvwxyz'
BITS_PER_CHARACTER = 5
LONGITUDE_RANGE = (-180.0, 180.0)
LATITUDE_RANGE = (-90.0, 90.0)

REASON_COLLISION = 'collision'  # the member's area separates it
REASON_EQUAL_AREA = 'equal_area'  # the member's rank separates it
REASON_IDENTICAL = 'identical'  # equal to another in geometry and tiebreak

Point = tuple[float, float]
Ring = list[Point]


class AssetIdResult(NamedTuple):
    """
    The ids of a footprint set and the record of its collisions.

    Attributes:
        ids (list[str]):
            One id per input geometry, in input order.
        log (list[dict]):
            One entry per member of a collision set, in input order, with the
            keys ``position``, ``geohash``, ``area_m2``, ``rank``, ``id`` and
            ``reason``. ``rank`` is ``None`` unless the member shares its
            rounded area with another member of its set. ``reason`` is one of
            the ``REASON_*`` constants.
    """

    ids: list[str]
    log: list[dict[str, Any]]


def _is_number(value: object) -> bool:
    """Return whether a value is a real number and not a boolean."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_point(value: object) -> bool:
    """Return whether a value looks like one coordinate pair."""
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and len(value) >= 2  # noqa: PLR2004
        and _is_number(value[0])
        and _is_number(value[1])
    )


def _to_ring(points: Sequence[Any]) -> Ring:
    """
    Convert ``[lon, lat]`` pairs into a validated open ring of floats.

    Args:
        points (Sequence):
            Coordinate pairs ``[lon, lat]``, with any further values, such as
            an altitude, ignored.

    Returns:
        list[tuple[float, float]]:
            The vertices as floats, without a closing vertex equal to the
            first.

    Raises:
        ValueError:
            If a vertex is not a coordinate pair, is not finite, or lies
            outside the longitude and latitude ranges of EPSG:4326.
    """
    ring: Ring = []
    for point in points:
        if not _is_point(point):
            raise ValueError(f'A footprint vertex is not a coordinate pair: {point!r}.')
        lon, lat = float(point[0]), float(point[1])
        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise ValueError(f'A footprint vertex is not finite: {point!r}.')
        if not (
            LONGITUDE_RANGE[0] <= lon <= LONGITUDE_RANGE[1]
            and LATITUDE_RANGE[0] <= lat <= LATITUDE_RANGE[1]
        ):
            raise ValueError(
                f'A footprint vertex lies outside EPSG:4326 longitude and '
                f'latitude ranges: {point!r}. Footprints must be passed in '
                'geographic coordinates, not after reprojection.'
            )
        ring.append((lon, lat))
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring.pop()
    return ring


def _exterior_rings(geometry: object) -> list[Ring]:
    """
    Return the open exterior rings of a footprint :func:`centroid` accepts.

    Args:
        geometry (object):
            A BRAILS coordinate list (a list of ``[lon, lat]`` for a polygon,
            a list of such rings for a multipolygon), a GeoJSON mapping of
            type ``Polygon`` or ``MultiPolygon``, or an object with
            ``__geo_interface__``, such as a shapely geometry.

    Returns:
        list[list[tuple[float, float]]]:
            One open ring per polygon part. Interior rings are dropped.

    Raises:
        ValueError:
            If the geometry is not one of the accepted forms or a vertex is
            invalid.
    """
    if hasattr(geometry, '__geo_interface__'):
        geometry = geometry.__geo_interface__

    if isinstance(geometry, Mapping):
        geometry_type = geometry.get('type')
        coordinates = geometry.get('coordinates')
        if geometry_type in ('Polygon', 'MultiPolygon') and not coordinates:
            raise ValueError(f'A {geometry_type} footprint has no coordinates.')
        if geometry_type == 'Polygon':
            return [_to_ring(coordinates[0])]
        if geometry_type == 'MultiPolygon':
            return [_to_ring(polygon[0]) for polygon in coordinates if polygon]
        raise ValueError(
            f'A footprint must be a Polygon or a MultiPolygon, not {geometry_type!r}.'
        )

    if (
        isinstance(geometry, Sequence)
        and not isinstance(geometry, str)
        and len(geometry) > 0
    ):
        if _is_point(geometry[0]):
            return [_to_ring(geometry)]
        return [_to_ring(ring) for ring in geometry]

    raise ValueError(f'A footprint geometry is empty or not recognized: {geometry!r}.')


def _origin(rings: list[Ring]) -> Point:
    """Return the lexicographically smallest vertex over all rings."""
    vertices = [vertex for ring in rings for vertex in ring]
    if not vertices:
        raise ValueError('A footprint has no vertices.')
    return min(vertices)


def _ring_moments(ring: Ring, origin: Point) -> tuple[float, float, float]:
    """
    Compute a ring's signed area and first moments about an origin.

    Args:
        ring (list[tuple[float, float]]):
            An open ring.
        origin (tuple[float, float]):
            The point the ring is translated by before the sums.

    Returns:
        tuple[float, float, float]:
            Twice the signed area, and six times the signed area multiplied
            by the centroid's translated longitude and latitude. Each is an
            exactly rounded sum over the ring's edges, so it does not depend
            on the starting vertex.
    """
    x0, y0 = origin
    points = [(lon - x0, lat - y0) for lon, lat in ring]
    count = len(points)
    crosses = []
    moments_x = []
    moments_y = []
    for index in range(count):
        xa, ya = points[index]
        xb, yb = points[(index + 1) % count]
        cross = xa * yb - xb * ya
        crosses.append(cross)
        moments_x.append((xa + xb) * cross)
        moments_y.append((ya + yb) * cross)
    return math.fsum(crosses), math.fsum(moments_x), math.fsum(moments_y)


def _parts(geometry: object) -> tuple[list[Ring], Point, list[tuple[float, float, float]]]:
    """Return the rings, the translation origin and each ring's moments."""
    rings = _exterior_rings(geometry)
    origin = _origin(rings)
    return rings, origin, [_ring_moments(ring, origin) for ring in rings]


def has_area(geometry: object) -> bool:
    """
    Return whether the exterior rings of a footprint enclose any area.

    Callers drop footprints without area with this test before calling
    :func:`assign_asset_ids`.

    Args:
        geometry (object):
            A footprint in any form :func:`centroid` accepts.

    Returns:
        bool:
            True when the absolute areas of the exterior rings sum to more than
            zero.

    Raises:
        ValueError:
            If the geometry is not an accepted form or a vertex is invalid.
    """
    _, _, moments = _parts(geometry)
    return math.fsum(abs(double_area) for double_area, _, _ in moments) > 0.0


def centroid(geometry: object) -> tuple[float, float]:
    """
    Compute the planar centroid of a footprint's exterior rings.

    The result does not depend on a ring's starting vertex, its orientation,
    a closing vertex, or the order of the parts. Interior rings are ignored.

    Args:
        geometry (object):
            A BRAILS coordinate list (a list of ``[lon, lat]`` for a polygon,
            a list of such rings for a multipolygon), a GeoJSON mapping of
            type ``Polygon`` or ``MultiPolygon``, or an object with
            ``__geo_interface__``. Coordinates are longitude and latitude in
            EPSG:4326.

    Returns:
        tuple[float, float]:
            The centroid's longitude and latitude.

    Raises:
        ValueError:
            If the geometry is not an accepted form, a vertex is invalid, or
            the exterior rings enclose no area.
    """
    _, origin, moments = _parts(geometry)
    return _centroid_of(origin, moments)


def _centroid_of(
    origin: Point, moments: list[tuple[float, float, float]]
) -> tuple[float, float]:
    """Combine the moments of a footprint's rings into its centroid."""
    weights = []
    weighted_x = []
    weighted_y = []
    for double_area, moment_x, moment_y in moments:
        if double_area == 0.0:
            continue
        weight = abs(double_area)
        weights.append(weight)
        weighted_x.append(weight * (moment_x / (3.0 * double_area)))
        weighted_y.append(weight * (moment_y / (3.0 * double_area)))
    total = math.fsum(weights)
    if total == 0.0:
        raise ValueError(
            'A footprint whose exterior rings enclose no area has no centroid. '
            'Drop it with has_area before computing ids.'
        )
    return (
        origin[0] + math.fsum(weighted_x) / total,
        origin[1] + math.fsum(weighted_y) / total,
    )


def plan_area_m2(geometry: object) -> float:
    """
    Compute the plan area of a footprint's exterior rings in square meters.

    The shoelace area in square degrees is scaled to a sphere of radius
    ``EARTH_RADIUS_M`` at the centroid's latitude. The result is unrounded.

    Args:
        geometry (object):
            A footprint in any form :func:`centroid` accepts.

    Returns:
        float:
            The plan area in square meters, unrounded.

    Raises:
        ValueError:
            In the cases :func:`centroid` raises it.
    """
    _, origin, moments = _parts(geometry)
    _, latitude = _centroid_of(origin, moments)
    square_degrees = math.fsum(abs(double_area) for double_area, _, _ in moments) / 2.0
    meters_per_degree = math.pi * EARTH_RADIUS_M / 180.0
    return square_degrees * meters_per_degree**2 * math.cos(math.radians(latitude))


def geohash(lon: float, lat: float, precision: int = GEOHASH_PRECISION) -> str:
    """
    Encode a longitude and latitude as a geohash.

    The standard bisection: longitude first, a bit of 1 when the value is at
    or above the midpoint of its interval. No rounding is applied before
    hashing.

    Args:
        lon (float):
            Longitude in degrees, within ``LONGITUDE_RANGE``.
        lat (float):
            Latitude in degrees, within ``LATITUDE_RANGE``.
        precision (int):
            Number of characters. Defaults to ``GEOHASH_PRECISION``.

    Returns:
        str:
            The geohash.

    Raises:
        ValueError:
            If a coordinate is outside its range or the precision is not a
            positive integer.
    """
    if not isinstance(precision, int) or isinstance(precision, bool) or precision < 1:
        raise ValueError(f'Geohash precision must be a positive integer, not {precision!r}.')
    try:
        _to_ring([(lon, lat)])
    except ValueError as error:
        raise ValueError(
            'A geohash needs a finite longitude in [-180, 180] and latitude in '
            f'[-90, 90], not ({lon!r}, {lat!r}).'
        ) from error

    lon_interval = list(LONGITUDE_RANGE)
    lat_interval = list(LATITUDE_RANGE)
    characters = []
    bits = 0
    bit_count = 0
    is_longitude = True
    while len(characters) < precision:
        interval, value = (lon_interval, lon) if is_longitude else (lat_interval, lat)
        midpoint = (interval[0] + interval[1]) / 2.0
        if value >= midpoint:
            bits = (bits << 1) | 1
            interval[0] = midpoint
        else:
            bits <<= 1
            interval[1] = midpoint
        is_longitude = not is_longitude
        bit_count += 1
        if bit_count == BITS_PER_CHARACTER:
            characters.append(GEOHASH_ALPHABET[bits])
            bits = 0
            bit_count = 0
    return ''.join(characters)


def centroid_geohash(geometry: object) -> str:
    """
    Return the geohash of a footprint's centroid.

    This is the id of a footprint outside every collision set.

    Args:
        geometry (object):
            A footprint in any form :func:`centroid` accepts.

    Returns:
        str:
            The geohash of the centroid.
    """
    return geohash(*centroid(geometry))


def _normalized_ring(ring: Ring) -> tuple[Point, ...]:
    """Orient a ring counterclockwise and start it at its smallest vertex."""
    double_area, _, _ = _ring_moments(ring, _origin([ring]))
    oriented = list(reversed(ring)) if double_area < 0.0 else list(ring)
    smallest = min(oriented)
    rotations = [
        tuple(oriented[index:] + oriented[:index])
        for index, vertex in enumerate(oriented)
        if vertex == smallest
    ]
    return min(rotations)


def _normalized_geometry(geometry: object) -> tuple[tuple[Point, ...], ...]:
    """Return a footprint's rings in a form free of vertex order and part order."""
    return tuple(sorted(_normalized_ring(ring) for ring in _exterior_rings(geometry)))


def _round_half_up(value: float) -> int:
    """Round a non-negative value to the nearest integer, halves up."""
    return math.floor(value + 0.5)


def assign_asset_ids(
    geometries: Sequence[object],
    tiebreak: Sequence[str | None] | None = None,
) -> AssetIdResult:
    """
    Assign each footprint of a set its asset id.

    A footprint's id is the geohash of its centroid (:func:`centroid_geohash`).
    Footprints of this call that share a geohash form a collision set, and
    each member is suffixed with its plan area in whole square meters,
    rounded half up, as ``<geohash>-<area>``. Members whose rounded areas are
    also equal are suffixed ``<geohash>-<area>-<rank>``, ranked from 1 by their
    normalized coordinates and then by ``tiebreak``. Input order decides a rank
    only between members equal in both. A geohash contains no
    ``SUFFIX_SEPARATOR``, so a suffixed id never equals a bare one.

    The id of a footprint outside every collision set does not depend on the
    other footprints passed. The id of a colliding footprint does, so a
    caller removes the footprints it does not want before the call.

    Args:
        geometries (Sequence[object]):
            Footprints in any form :func:`centroid` accepts.
        tiebreak (Sequence[str | None], optional):
            One string per footprint that orders members equal in normalized
            geometry, such as a provider's id. Defaults to None, so identical
            members rank by input order.

    Raises:
        ValueError:
            If ``tiebreak`` does not match ``geometries`` in length, a
            geometry is not an accepted form, a vertex is invalid, or a
            footprint encloses no area (see :func:`has_area`).

    Returns:
        AssetIdResult:
            The ids in input order and one log entry per member of a collision
            set.
    """
    count = len(geometries)
    if tiebreak is None:
        tiebreak = [None] * count
    elif len(tiebreak) != count:
        raise ValueError(
            f'tiebreak holds {len(tiebreak)} values for {count} geometries.'
        )

    hashes = []
    for position, geometry in enumerate(geometries):
        if not has_area(geometry):
            raise ValueError(
                f'The footprint at position {position} encloses no area and '
                'cannot receive an id. Drop it with has_area before the call.'
            )
        hashes.append(centroid_geohash(geometry))

    ids = list(hashes)
    members_by_hash: dict[str, list[int]] = {}
    for position, cell in enumerate(hashes):
        members_by_hash.setdefault(cell, []).append(position)

    log: list[dict[str, Any]] = []
    for cell, members in members_by_hash.items():
        if len(members) > 1:
            _resolve_collision_set(cell, members, geometries, tiebreak, ids, log)

    log.sort(key=lambda entry: entry['position'])
    return AssetIdResult(ids=ids, log=log)


def _resolve_collision_set(
    cell: str,
    members: list[int],
    geometries: Sequence[object],
    tiebreak: Sequence[str | None],
    ids: list[str],
    log: list[dict[str, Any]],
) -> None:
    """
    Suffix the members of one collision set and log each of them.

    Args:
        cell (str):
            The geohash the members share.
        members (list[int]):
            Positions of the members in the call's input.
        geometries (Sequence[object]):
            The call's footprints.
        tiebreak (Sequence[str | None]):
            The call's tiebreak, one value per footprint.
        ids (list[str]):
            The call's ids, updated in place.
        log (list[dict]):
            The call's collision log, extended in place.
    """
    members_by_area: dict[int, list[int]] = {}
    for position in members:
        area = _round_half_up(plan_area_m2(geometries[position]))
        members_by_area.setdefault(area, []).append(position)

    for area, same_area in members_by_area.items():
        if len(same_area) == 1:
            position = same_area[0]
            ids[position] = f'{cell}{SUFFIX_SEPARATOR}{area}'
            log.append(_log_entry(position, cell, area, None, ids[position],
                                  REASON_COLLISION))
            continue

        keys = {
            position: (
                _normalized_geometry(geometries[position]),
                tiebreak[position] is not None,
                '' if tiebreak[position] is None else str(tiebreak[position]),
            )
            for position in same_area
        }
        key_counts = Counter(keys.values())
        ranked = sorted(same_area, key=lambda position: (keys[position], position))
        for rank, position in enumerate(ranked, start=1):
            ids[position] = f'{cell}{SUFFIX_SEPARATOR}{area}{SUFFIX_SEPARATOR}{rank}'
            reason = (
                REASON_IDENTICAL if key_counts[keys[position]] > 1
                else REASON_EQUAL_AREA
            )
            log.append(_log_entry(position, cell, area, rank, ids[position], reason))


def _log_entry(
    position: int,
    cell: str,
    area: int,
    rank: int | None,
    asset_id: str,
    reason: str,
) -> dict[str, Any]:
    """Build one collision log entry."""
    return {
        'position': position,
        'geohash': cell,
        'area_m2': area,
        'rank': rank,
        'id': asset_id,
        'reason': reason,
    }
