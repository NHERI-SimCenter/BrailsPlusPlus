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
Filter an inventory to the assets whose reference point lies inside a region.

The module is imported by its full path,
``from brails.utils.region_filter import filter_to_region``.

.. autosummary::

      filter_to_region
      reference_point
"""

from __future__ import annotations

import json
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

from brails.types.asset_inventory import (
    Asset,
    AssetInventory,
    is_probabilistic_feature,
)
from brails.types.region_boundary import RegionBoundary
from brails.utils.asset_id import LATITUDE_RANGE, LONGITUDE_RANGE
from brails.utils.geo_tools import GeoTools

if TYPE_CHECKING:
    import os
    from collections.abc import Hashable, Iterator

READ_CHUNK_CHARS = 1 << 24
"""Characters :func:`filter_to_region` reads from a GeoJSON file at a time."""

GEOGRAPHIC_CRS_NAMES = (
    'urn:ogc:def:crs:OGC:1.3:CRS84',
    'urn:ogc:def:crs:OGC::CRS84',
    'OGC:CRS84',
    'CRS84',
    'urn:ogc:def:crs:EPSG::4326',
    'EPSG:4326',
)
"""Names a GeoJSON ``crs`` member may give for longitude and latitude in WGS84."""

_WHITESPACE = ' \t\n\r'

# The characters that can continue a JSON number.
_NUMBER_TAIL = '0123456789.eE+-'


def reference_point(asset: Asset, asset_id: Hashable) -> tuple[float, float]:
    """
    Return the point an asset is located by.

    :func:`filter_to_region` and the census tract scraper's
    ``CensusTractScraper.get_census_geographies`` locate an asset by this
    point, so an asset that the filter keeps inside a census polygon lies in
    that polygon's tract or block group.

    Args:
        asset (Asset):
            The asset.
        asset_id (Hashable):
            Its id, for the error messages.

    Returns:
        tuple[float, float]:
            The longitude and latitude of the centroid of the asset's
            exterior rings, of the point for a point asset, or of the
            Shapely centroid for a geometry enclosing no area.

    Raises:
        ValueError:
            If the asset has no coordinates or its coordinates form no
            geometry.
    """
    coordinates = asset.coordinates
    if not coordinates:
        msg = f'Asset {asset_id!r} has no coordinates.'
        raise ValueError(msg)
    if len(coordinates) == 1 and not isinstance(coordinates[0][0], list):
        return float(coordinates[0][0]), float(coordinates[0][1])
    if AssetInventory.has_area(coordinates):
        return AssetInventory.centroid_and_plan_area(coordinates)[:2]
    geometry = asset.get_geometry()
    if geometry is None or geometry.is_empty:
        msg = f'Asset {asset_id!r} has coordinates that form no geometry.'
        raise ValueError(msg)
    point = geometry.centroid
    return float(point.x), float(point.y)


def filter_to_region(  # noqa: C901
    inventory_or_geojson_path: AssetInventory | str | os.PathLike[str],
    region: str | RegionBoundary | Polygon | MultiPolygon,
) -> AssetInventory:
    """
    Return the assets of an inventory whose reference point lies inside a region.

    An asset is inside when its :func:`reference_point` lies strictly inside
    the region's boundary, as :func:`shapely.contains_xy` decides. A point on
    the boundary is outside, so an asset on the line two neighboring places
    share belongs to neither, and no asset belongs to both. An asset whose
    outline crosses the boundary is kept or left out whole, by its point.

    A boundary looked up by name is built by :class:`RegionBoundary` from the
    outer members of the place's OpenStreetMap relation only, so it has no
    holes, and the assets of an enclave, such as Piedmont inside Oakland, are
    kept. Where holes matter, pass the boundary as a Shapely polygon, whose
    holes the filter keeps.

    The kept assets are deep copies, so a change to the returned inventory
    leaves the source as it was.

    Given an :class:`AssetInventory`, the function returns a new inventory
    with the source's ``n_pw`` and ``vector_features``. Pass a GeoJSON file by
    its path rather than as an inventory read with
    :meth:`AssetInventory.read_from_geojson`, which leaves ``n_pw`` at 1 for
    a file of several worlds, adds ``type`` and turns digit-string ids into
    integers.

    Given the path of a GeoJSON FeatureCollection, such as a file
    :meth:`AssetInventory.write_to_geojson` wrote, it reads the file as
    UTF-8 one feature at a time, so for a valid file its memory grows with
    the assets kept and not with the file. Each kept feature becomes an asset
    keyed by the feature's ``id``, with its properties as written and its
    geometry as :meth:`AssetInventory.read_from_geojson` reads it, exterior
    rings only. No ``type`` is added. ``n_pw`` is the number of values of the
    kept assets' list-valued features, those that
    :func:`~brails.types.asset_inventory.is_probabilistic_feature` counts as
    holding worlds under the default ``vector_features``, and 1 without
    such lists. Of the file's other top-level members, ``type`` must be
    ``FeatureCollection`` and ``crs``, if present, must name WGS84 longitude
    and latitude, each checked when it is read. None of them is kept, and
    ``write_to_geojson`` writes a ``generated``, ``brails_version`` and
    ``crs`` of its own. For a file that ``write_to_geojson`` wrote without
    ``integer_ids``, the kept features written back with ``write_to_geojson``
    equal the source's features.

    A file already in the integer-id form cannot be written in that form again
    after filtering, since its assets carry the ``unique_id`` the export
    reserves, and one with several types repeats ids across them, which this
    function refuses.

    The function prints the number of assets kept.

    Args:
        inventory_or_geojson_path (AssetInventory | str | os.PathLike[str]):
            The inventory, or the path of a GeoJSON FeatureCollection in
            longitude and latitude (WGS84).
        region (str | RegionBoundary | Polygon | MultiPolygon):
            A place name, looked up once with :class:`RegionBoundary`, which
            prints the name it found; a :class:`RegionBoundary`, whose
            boundary is fetched; or a Shapely polygon or multipolygon in
            longitude and latitude (WGS84), taken as given and repaired with
            :func:`shapely.make_valid` if invalid.

    Returns:
        AssetInventory:
            A new inventory holding copies of the assets inside the region.

    Raises:
        FileNotFoundError:
            If a path is given and is not a file, before any lookup.
        TypeError:
            If ``region`` is none of the accepted types.
        ValueError:
            If the region's bounds are not longitude and latitude, or the
            source inventory holds a housing-unit inventory. For a file, if
            it is not a GeoJSON FeatureCollection, names a ``crs`` other than
            WGS84 longitude and latitude, holds a feature without an
            ``id``, with an id that is not a string or a number, with
            properties that are not an object, or with a geometry or
            coordinates BRAILS cannot read, or holds kept features with the
            same ``id``, with an empty list, or with lists of different
            lengths. Also if an asset of an inventory has no coordinates or
            its coordinates form no geometry (:func:`reference_point`).

    Examples:
        >>> from shapely.geometry import box
        >>> inventory = AssetInventory()
        >>> inventory.add_asset_coordinates('A', [[-122.27, 37.87]])
        True
        >>> inventory.add_asset_coordinates('B', [[-122.10, 37.87]])
        True
        >>> region = box(-122.37, 37.83, -122.23, 37.91)
        >>> berkeley = filter_to_region(inventory, region)
        Kept 1 of 2 assets inside the region.
        >>> berkeley.get_asset_ids()
        ['A']
    """
    if not isinstance(inventory_or_geojson_path, AssetInventory):
        path = Path(inventory_or_geojson_path)
        if not path.is_file():
            msg = f'{path} is not a file. Pass the path of a GeoJSON inventory.'
            raise FileNotFoundError(msg)

    if isinstance(region, str):
        region = RegionBoundary({'type': 'locationName', 'data': region})
    if isinstance(region, RegionBoundary):
        boundary, name, _ = region.get_boundary()
    elif isinstance(region, (Polygon, MultiPolygon)):
        boundary, name = region, 'the region'
    else:
        msg = (
            'region must be a place name, a RegionBoundary, or a Shapely '
            f'Polygon or MultiPolygon, not {type(region).__name__}.'
        )
        raise TypeError(msg)
    west, south, east, north = boundary.bounds
    if not (
        LONGITUDE_RANGE[0] <= west <= east <= LONGITUDE_RANGE[1]
        and LATITUDE_RANGE[0] <= south <= north <= LATITUDE_RANGE[1]
    ):
        msg = (
            f'The region has bounds {boundary.bounds}, which are not longitude '
            'and latitude in WGS84. Reproject it to EPSG:4326 first, for '
            'example with shapely.ops.transform and a pyproj Transformer.'
        )
        raise ValueError(msg)
    if not boundary.is_valid:
        boundary = shapely.make_valid(boundary)
    shapely.prepare(boundary)

    if isinstance(inventory_or_geojson_path, AssetInventory):
        source = inventory_or_geojson_path
        if source.housing_unit_inventory is not None:
            msg = (
                'The inventory holds a housing-unit inventory, which '
                'filter_to_region does not divide. Filter the inventory '
                'before assigning housing units, or remove them first with '
                'remove_housing_unit_inventory().'
            )
            raise ValueError(msg)
        asset_ids = list(source.inventory)
        points = np.array(
            [reference_point(source.inventory[a], a) for a in asset_ids],
            dtype=float,
        ).reshape(-1, 2)
        inside = shapely.contains_xy(boundary, points[:, 0], points[:, 1])
        result = AssetInventory(vector_features=source.vector_features)
        result.n_pw = source.n_pw
        for asset_id, is_inside in zip(asset_ids, inside):
            if is_inside:
                result.inventory[asset_id] = deepcopy(source.inventory[asset_id])
        total = len(asset_ids)
    else:
        result = AssetInventory()
        lengths: set[int] = set()
        total = 0
        for feature in _stream_features(path):
            total += 1
            if not isinstance(feature, dict) or 'id' not in feature:
                msg = (
                    f'Feature {total - 1} of {path} has no id. Give every '
                    'feature a unique id, as write_to_geojson does.'
                )
                raise ValueError(msg)
            asset_id = feature['id']
            if not isinstance(asset_id, (str, int, float)) or isinstance(
                asset_id, bool
            ):
                msg = (
                    f'Feature {total - 1} of {path} has the id {asset_id!r}. '
                    'Give every feature a string or a number as its id.'
                )
                raise ValueError(msg)  # noqa: TRY004, a malformed file
            properties = feature.get('properties') or {}
            if not isinstance(properties, dict):
                msg = (
                    f'Feature {asset_id!r} of {path} has properties that are '
                    'not an object. Write its properties as a JSON object.'
                )
                raise ValueError(msg)  # noqa: TRY004, a malformed file
            geometry = feature.get('geometry')
            remedy = (
                'Remove the feature from the file, or give it a Point, '
                'LineString, Polygon or MultiPolygon in longitude and latitude '
                '(EPSG:4326).'
            )
            try:
                coordinates = GeoTools.parse_geojson_geometry(geometry)
            except (TypeError, ValueError, NotImplementedError, IndexError) as error:
                kind = geometry.get('type') if isinstance(geometry, dict) else None
                msg = (
                    f'Feature {asset_id!r} of {path} has a geometry BRAILS '
                    f'cannot read ({kind}). {remedy}'
                )
                raise ValueError(msg) from error
            asset = Asset(asset_id, coordinates, properties)
            if not asset.coordinates:
                msg = (
                    f'Feature {asset_id!r} of {path} has coordinates BRAILS '
                    f'cannot read as longitude and latitude. {remedy}'
                )
                raise ValueError(msg)
            lon, lat = reference_point(asset, asset_id)
            if not shapely.contains_xy(boundary, lon, lat):
                continue
            if asset_id in result.inventory:
                msg = (
                    f'Two features of {path} inside the region have the id '
                    f'{asset_id!r}. Feature ids must be unique. A file that '
                    'write_to_geojson wrote with integer_ids and several types '
                    'numbers each type from 1, and filter_to_region cannot '
                    'read it. Filter the inventory it was written from instead.'
                )
                raise ValueError(msg)
            for key, value in properties.items():
                if not is_probabilistic_feature(key, value, result.vector_features):
                    continue
                if not value:
                    msg = (
                        f'Feature {asset_id!r} of {path} holds an empty list '
                        f'under {key!r}, which gives no possible world. Write '
                        'a single value or one value per world in its place.'
                    )
                    raise ValueError(msg)
                lengths.add(len(value))
            result.inventory[asset_id] = asset
        if len(lengths) > 1:
            msg = (
                f'The features of {path} inside the region hold lists of '
                f'{sorted(lengths)} values, so they share no number of possible '
                'worlds. A file written from one inventory holds lists of one '
                'length. Write the file again from a single inventory whose '
                'features pass validate_possible_worlds.'
            )
            raise ValueError(msg)
        result.n_pw = lengths.pop() if lengths else 1

    noun = 'asset' if total == 1 else 'assets'
    message = f'Kept {len(result.inventory)} of {total} {noun} inside {name}.'
    try:
        print(message)
    except UnicodeEncodeError:
        print(
            unicodedata.normalize('NFKD', message)
            .encode('ascii', 'ignore')
            .decode('ascii')
        )
    return result


def _stream_features(path: Path) -> Iterator[dict[str, Any]]:  # noqa: C901
    """
    Yield the features of a GeoJSON FeatureCollection one at a time.

    The file is read as UTF-8, with or without a byte order mark,
    :data:`READ_CHUNK_CHARS` characters at a time, and each value is parsed
    with :meth:`json.JSONDecoder.raw_decode`, so the top-level members may
    come in any order. A value that ends at the end of what has been read, or
    before a character that could continue a number, is parsed again with
    more of the file, so a number cut at a chunk's edge is not taken short.
    The members ``type`` and ``crs`` are checked when they are read, so a
    file that writes them before ``features``, as ``write_to_geojson`` and
    GDAL do, is refused before its first feature is yielded.

    Args:
        path (Path):
            The file.

    Yields:
        dict[str, Any]:
            Each element of the ``features`` array, as parsed.

    Raises:
        ValueError:
            If the file is not a JSON object, holds invalid JSON (a trailing
            comma or a member name that is not a string among it), has no
            ``features`` member or two, has a ``type`` other than
            ``FeatureCollection``, or has a ``crs`` other than one of
            :data:`GEOGRAPHIC_CRS_NAMES`.
    """
    decoder = json.JSONDecoder()
    buffer = ''
    position = 0
    at_end = False
    with path.open(encoding='utf-8-sig') as file:

        def read_more() -> bool:
            nonlocal buffer, position, at_end
            data = file.read(READ_CHUNK_CHARS)
            if not data:
                at_end = True
                return False
            buffer = buffer[position:] + data
            position = 0
            return True

        def next_char() -> str:
            # The next character that is not whitespace, or '' at the end.
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position] in _WHITESPACE:
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if not read_more():
                    return ''

        def refuse(problem: str) -> ValueError:
            return ValueError(
                f'{path} is not valid GeoJSON: {problem}. Write the file with '
                'AssetInventory.write_to_geojson or another GeoJSON writer.'
            )

        def expect(allowed: str) -> str:
            nonlocal position
            char = next_char()
            if not char or char not in allowed:
                found = repr(char) if char else 'the end of the file'
                raise refuse(f'expected one of {allowed!r}, found {found}')
            position += 1
            return char

        def value() -> Any:  # noqa: ANN401
            nonlocal position
            next_char()
            while True:
                try:
                    parsed, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError as error:
                    if read_more():
                        continue
                    raise refuse(str(error)) from error
                # A number that ends where the text read so far ends, or
                # before a character that could continue it, may be cut short
                # at a chunk's edge, so it is parsed again with more text.
                cut = end == len(buffer) or buffer[end] in _NUMBER_TAIL
                if cut and not at_end and read_more():
                    continue
                position = end
                return parsed

        seen: set[str] = set()
        expect('{')
        if next_char() == '}':
            expect('}')
        else:
            while True:
                key = value()
                if not isinstance(key, str):
                    raise refuse(f'the member name {key!r} is not a string')
                if key in seen:
                    raise refuse(f'the member {key!r} occurs twice')
                seen.add(key)
                expect(':')
                if key == 'features':
                    expect('[')
                    if next_char() == ']':
                        expect(']')
                    else:
                        while True:
                            yield value()
                            if expect(',]') == ']':
                                break
                            if next_char() == ']':
                                raise refuse('a comma ends the features array')
                else:
                    member = value()
                    if key == 'type' and member != 'FeatureCollection':
                        msg = (
                            f'{path} has the type {member!r}, not '
                            'FeatureCollection. Pass a file that holds a '
                            'FeatureCollection of the inventory.'
                        )
                        raise ValueError(msg)
                    if key == 'crs':
                        properties = (
                            member.get('properties')
                            if isinstance(member, dict)
                            and member.get('type') == 'name'
                            else None
                        )
                        name = (
                            properties.get('name')
                            if isinstance(properties, dict)
                            else None
                        )
                        if name not in GEOGRAPHIC_CRS_NAMES:
                            msg = (
                                f'{path} gives the crs '
                                f'{json.dumps(member)[:200]}. filter_to_region '
                                'reads longitude and latitude in WGS84 '
                                '(EPSG:4326) only. Write the file in EPSG:4326, '
                                'as write_to_geojson does, or remove the crs '
                                'member if its coordinates are already '
                                'longitude and latitude.'
                            )
                            raise ValueError(msg)
                if expect(',}') == '}':
                    break
                if next_char() == '}':
                    raise refuse('a comma ends the top-level object')
        if 'features' not in seen:
            raise refuse('it has no features member')
        if 'type' not in seen:
            msg = (
                f'{path} has no type member, so it is not a FeatureCollection. '
                'Pass a file that holds a FeatureCollection of the inventory.'
            )
            raise ValueError(msg)
        if next_char():
            raise refuse('data follows the top-level object')
