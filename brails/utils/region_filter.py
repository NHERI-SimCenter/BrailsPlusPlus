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
      find_inside
      read_boundary
      reference_point
      resolve_region
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

from brails.types.asset_inventory import (
    Asset,
    AssetInventory,
    check_n_worlds,
    is_probabilistic_feature,
)
from brails.types.census_boundary import CensusBoundary
from brails.types.region_boundary import RegionBoundary
from brails.utils.asset_id import LATITUDE_RANGE, LONGITUDE_RANGE
from brails.utils.console import print_ascii_fallback
from brails.utils.geo_tools import GeoTools
from brails.utils.tiger_files import DEFAULT_TIGER_VINTAGE

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Mapping

    from shapely.geometry.base import BaseGeometry

# Characters :func:`filter_to_region` reads from a GeoJSON file at a time.
READ_CHUNK_CHARS = 1 << 24

# Names a GeoJSON ``crs`` member may give for longitude and latitude in WGS84.
GEOGRAPHIC_CRS_NAMES = (
    'urn:ogc:def:crs:OGC:1.3:CRS84',
    'urn:ogc:def:crs:OGC::CRS84',
    'OGC:CRS84',
    'CRS84',
    'urn:ogc:def:crs:EPSG::4326',
    'EPSG:4326',
)

# Names a GeoJSON ``crs`` member may give for longitude and latitude in NAD83.
NAD83_CRS_NAMES = (
    'urn:ogc:def:crs:EPSG::4269',
    'EPSG:4269',
    'urn:ogc:def:crs:OGC:1.3:CRS83',
    'urn:ogc:def:crs:OGC::CRS83',
    'OGC:CRS83',
    'CRS83',
)

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
    inventory_or_geojson_path: AssetInventory
    | str
    | os.PathLike[str]
    | list[AssetInventory | str | os.PathLike[str]]
    | tuple[AssetInventory | str | os.PathLike[str], ...],
    region: str | RegionBoundary | Polygon | MultiPolygon,
    *,
    source: str = 'census',
) -> AssetInventory:
    """
    Return the assets of an inventory whose reference point lies inside a region.

    An asset is inside when its :func:`reference_point` lies strictly inside
    the region's boundary, as :func:`shapely.contains_xy` decides. A point on
    the boundary is outside, so an asset on the line two neighboring places
    share belongs to neither, and no asset belongs to both. An asset whose
    outline crosses the boundary is kept or left out whole, by its point.

    A place name is looked up among the census places and counties of its
    state by :class:`~brails.types.census_boundary.CensusBoundary`, whose
    boundary keeps its holes, so the assets of an enclave, such as Piedmont
    inside Oakland, are left out. With ``source='osm'`` the name is looked up
    in OpenStreetMap by :class:`RegionBoundary`, which builds the boundary
    from the outer members of the place's relation only, so it has no holes
    and the assets of an enclave are kept. A Shapely polygon keeps its holes.

    The kept assets are deep copies, so a change to the returned inventory
    leaves the source as it was.

    Given an :class:`AssetInventory`, the function returns a new inventory
    with the source's ``n_pw`` and ``vector_features``. Pass a GeoJSON file by
    its path rather than as an inventory read with
    :meth:`AssetInventory.read_from_geojson`, which leaves ``n_pw`` at 1 for
    a file of several worlds without ``n_worlds``, adds ``type`` and turns
    digit-string ids into integers.

    Given the path of a GeoJSON FeatureCollection, such as a file
    :meth:`AssetInventory.write_to_geojson` wrote, it reads the file as
    UTF-8 one feature at a time, so for a valid file its memory grows with
    the assets kept and not with the file. Each kept feature becomes an asset
    keyed by the feature's ``id``, with its properties as written and its
    geometry as :meth:`AssetInventory.read_from_geojson` reads it, exterior
    rings only. No ``type`` is added.

    A feature of the integer-id form, which ``write_to_geojson(...,
    integer_ids=True)`` writes as an integer ``id`` beside a string
    ``unique_id`` property, is keyed by its ``unique_id`` string instead. The
    ``unique_id`` property is dropped, the feature's number is not kept, and
    its ``type`` stays a property, so such a file can be filtered and written
    in that form again with the same features. Any other ``unique_id`` is an
    ordinary property: the default form of ``write_to_geojson`` writes every
    ``id`` as a string, so its files never take this form. The keys stay
    strings when every ``unique_id`` is digits, where
    ``read_from_geojson(key_by_unique_id=True)`` makes them integers, and the
    export writes the same file from either. The checks on ``unique_id``, an
    empty one, a repeated one, or kept features of both forms, cover the kept
    features.

    ``n_pw`` is the file's ``n_worlds``, an integer of at least 1, which
    every kept list-valued feature must hold as its number of values. Those
    features are the ones
    :func:`~brails.types.asset_inventory.is_probabilistic_feature` counts as
    holding worlds under the default ``vector_features``. In a file without
    the member, ``n_pw`` is the number of values of the kept lists, and 1
    without them. Of the file's other top-level members, ``type`` must be
    ``FeatureCollection`` and ``crs``, if present, must name WGS84 or NAD83
    longitude and latitude. ``n_worlds``, ``type`` and ``crs`` are each
    checked when read. NAD83 coordinates are used as they are, as WGS84 ones,
    and a line is printed to say so. No other member is kept, and
    ``write_to_geojson`` writes a ``generated``, ``brails_version``,
    ``n_worlds`` and ``crs`` of its own. Written back with
    ``write_to_geojson`` in the form the file has, the kept features equal
    the source's features, for a file that ``write_to_geojson`` wrote.

    The integer-id form that R2D reads requires a ``type`` on every asset.
    ``write_to_geojson(..., integer_ids=True, asset_type='building')`` writes
    that type for the assets without one, such as the buildings of a county
    file of the testbed as compiled.

    Given a list or tuple of inventories and paths, such as the county files
    of a city that spans several counties, the function filters them one
    after another into one inventory, in the order given, and looks the
    boundary up once. The sources must hold the same number of worlds and
    the same vector features. An inventory states its ``n_pw`` and a file its
    ``n_worlds``. For a file without the member, the count is the length of
    its kept lists, so one whose kept assets hold no lists takes the others'
    number unchecked. A file that states 1 world, or whose kept lists hold
    one value, is refused beside a source of more worlds, since it may be one
    world of a run. :meth:`AssetInventory.read_from_geojson` adds a file of 1
    world without lists to an inventory of more worlds. An asset id that
    lies inside the region in two sources is refused, since the sources then
    overlap, and so are two kept ids with the same string form, which
    ``write_to_geojson`` would write as one id. A one-item list gives what
    its item gives. Another sequence, such as a generator, is not taken as a
    list. Memory is that of the kept assets and of one file's stream at a
    time.

    The function prints the number of assets kept, and for several sources
    the number kept from each.

    Args:
        inventory_or_geojson_path (AssetInventory | str | PathLike | list | tuple):
            The inventory, the path of a GeoJSON FeatureCollection in
            longitude and latitude (WGS84), or a list or tuple of them.
        region (str | RegionBoundary | Polygon | MultiPolygon):
            A place name with its state, as in ``'Berkeley, CA'``, looked up
            once as ``source`` selects, and the lookup prints the name it
            found; a :class:`RegionBoundary` of any kind, whose boundary is
            fetched; or a Shapely polygon or multipolygon in longitude and
            latitude (WGS84), taken as given and repaired as
            :func:`read_boundary` repairs it if invalid.
        source (str, optional):
            Where a place name is looked up: ``'census'`` among the
            TIGER/Line places and counties with
            :class:`~brails.types.census_boundary.CensusBoundary`, or
            ``'osm'`` in OpenStreetMap with :class:`RegionBoundary`. Not
            consulted for a region that is not a place name. Defaults to
            ``'census'``.

    Returns:
        AssetInventory:
            A new inventory holding copies of the assets inside the region.

    Raises:
        FileNotFoundError:
            If a path is given and is not a file, before any lookup.
        TypeError:
            If ``region`` is none of the accepted types, or a source is
            neither an inventory nor a path.
        ValueError:
            If the list of sources is empty, the sources hold different
            numbers of worlds or different vector features, or two kept
            asset ids, of one source or of two, have the same string form.
            If the region's bounds are not longitude and latitude, or a
            source inventory holds a housing-unit inventory. For a file, if
            it is not a GeoJSON FeatureCollection, names a ``crs`` other than
            WGS84 or NAD83 longitude and latitude, gives an ``n_worlds`` that
            is not an integer of at least 1, holds a feature without an
            ``id``, with an id that is not a string or a number, with
            properties that are not an object, or with a geometry or
            coordinates BRAILS cannot read, or holds kept features with the
            same ``id``, with an empty list, or with lists of different
            lengths or of a length other than its ``n_worlds``. Also if an
            asset of an inventory has no coordinates or its coordinates form
            no geometry (:func:`reference_point`). Also as :func:`resolve_region`
            raises, for ``source`` and the place name.

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
    sources: list[AssetInventory | Path] = []
    labels: list[str] = []
    items = (
        list(inventory_or_geojson_path)
        if isinstance(inventory_or_geojson_path, (list, tuple))
        else [inventory_or_geojson_path]
    )
    if not items:
        msg = (
            'The list of sources is empty. Pass an inventory, the path of a '
            'GeoJSON inventory, or a list of them.'
        )
        raise ValueError(msg)
    listed = isinstance(inventory_or_geojson_path, (list, tuple))
    for index, item in enumerate(items):
        if isinstance(item, AssetInventory):
            sources.append(item)
            labels.append(
                f'inventory {index + 1} of the list' if listed else 'the inventory'
            )
        elif isinstance(item, (str, os.PathLike)):
            path = Path(item)
            if not path.is_file():
                msg = f'{path} is not a file. Pass the path of a GeoJSON inventory.'
                raise FileNotFoundError(msg)
            sources.append(path)
            labels.append(str(path))
        else:
            kind = type(item).__name__
            article = 'an' if kind[0].lower() in 'aeiou' else 'a'
            if listed:
                msg = (
                    f'Item {index + 1} of the list is {article} {kind}. Pass an '
                    'AssetInventory or the path of a GeoJSON inventory.'
                )
            else:
                msg = (
                    f'The first argument is {article} {kind}. Pass an '
                    'AssetInventory, the path of a GeoJSON inventory, or a list '
                    'or tuple of them.'
                )
            raise TypeError(msg)

    boundary, name, _ = read_boundary(resolve_region(region, source))
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
    shapely.prepare(boundary)

    result = AssetInventory()
    # The number of worlds each source holds, for the sources that state one.
    worlds: dict[str, int] = {}
    vector_features: dict[str, list[str]] = {}
    # The source and the id of each kept asset, by the id's string form.
    kept_from: dict[str, tuple[str, Hashable]] = {}
    messages = []
    total = 0
    for inventory_or_path, label in zip(sources, labels):
        kept: dict[Hashable, Asset] = {}
        if isinstance(inventory_or_path, AssetInventory):
            if inventory_or_path.housing_unit_inventory is not None:
                msg = (
                    f'{label[0].upper() + label[1:]} holds a housing-unit '
                    'inventory, which filter_to_region does not divide. Filter the '
                    'inventory before assigning housing units, or remove them '
                    'first with remove_housing_unit_inventory().'
                )
                raise ValueError(msg)
            for asset_id in find_inside(inventory_or_path.inventory, boundary):
                kept[asset_id] = deepcopy(inventory_or_path.inventory[asset_id])
            count = len(inventory_or_path.inventory)
            worlds[label] = inventory_or_path.n_pw
            vector_features[label] = list(inventory_or_path.vector_features)
        else:
            path = inventory_or_path
            lengths: set[int] = set()
            count = 0
            # How many kept features were keyed by their unique_id.
            numbered = 0
            stated: dict[str, int] = {}
            for feature in _stream_features(path, stated):
                count += 1
                if not isinstance(feature, dict) or 'id' not in feature:
                    msg = (
                        f'Feature {count - 1} of {path} has no id. Give every '
                        'feature a unique id, as write_to_geojson does.'
                    )
                    raise ValueError(msg)
                asset_id = feature['id']
                if not isinstance(asset_id, (str, int, float)) or isinstance(
                    asset_id, bool
                ):
                    msg = (
                        f'Feature {count - 1} of {path} has the id '
                        f'{asset_id!r}. Give every feature a string or a '
                        'number as its id.'
                    )
                    raise ValueError(msg)  # noqa: TRY004, a malformed file
                properties = feature.get('properties') or {}
                if not isinstance(properties, dict):
                    msg = (
                        f'Feature {asset_id!r} of {path} has properties that '
                        'are not an object. Write its properties as a JSON '
                        'object.'
                    )
                    raise ValueError(msg)  # noqa: TRY004, a malformed file
                # A feature of the integer-id form, an integer id beside a
                # string unique_id, is keyed by its unique_id.
                has_unique_id = (
                    isinstance(asset_id, int)
                    and not isinstance(asset_id, bool)
                    and isinstance(properties.get('unique_id'), str)
                )
                if has_unique_id:
                    unique_id = properties.pop('unique_id')
                    if not unique_id:
                        msg = (
                            f'Feature {asset_id!r} of {path} has an empty '
                            'unique_id. Write the file again with '
                            'write_to_geojson, which gives every asset its id '
                            'there.'
                        )
                        raise ValueError(msg)
                    asset_id = unique_id
                geometry = feature.get('geometry')
                remedy = (
                    'Remove the feature from the file, or give it a Point, '
                    'LineString, Polygon or MultiPolygon in longitude and '
                    'latitude (EPSG:4326).'
                )
                try:
                    coordinates = GeoTools.parse_geojson_geometry(geometry)
                except (
                    TypeError,
                    ValueError,
                    NotImplementedError,
                    IndexError,
                ) as error:
                    kind = (
                        geometry.get('type') if isinstance(geometry, dict) else None
                    )
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
                if not find_inside({asset_id: asset}, boundary):
                    continue
                if asset_id in kept:
                    field = 'unique_id' if has_unique_id else 'id'
                    msg = (
                        f'Two features of {path} inside the region have the '
                        f'{field} {asset_id!r}. Give every feature a unique '
                        f'{field}, as write_to_geojson does.'
                    )
                    raise ValueError(msg)
                for key, value in properties.items():
                    if not is_probabilistic_feature(
                        key, value, result.vector_features
                    ):
                        continue
                    if not value:
                        msg = (
                            f'Feature {asset_id!r} of {path} holds an empty list '
                            f'under {key!r}, which gives no possible world. '
                            'Write a single value or one value per world in '
                            'its place.'
                        )
                        raise ValueError(msg)
                    lengths.add(len(value))
                kept[asset_id] = asset
                numbered += has_unique_id
            if 0 < numbered < len(kept):
                msg = (
                    f'Of the features of {path} inside the region, {numbered} '
                    'have the integer-id form (an integer id beside a string '
                    f'unique_id) and {len(kept) - numbered} do not. Write the '
                    'file again with write_to_geojson, in one form.'
                )
                raise ValueError(msg)
            # The file's n_worlds and the lengths of its kept lists, which
            # must agree.
            counts = lengths | set(stated.values())
            if len(counts) > 1:
                msg = (
                    f'The features of {path} inside the region hold lists of '
                    + ' and '.join(str(length) for length in sorted(lengths))
                    + (
                        ' values, but the file gives n_worlds as '
                        f'{stated["n_worlds"]}. Write the file again with '
                        'write_to_geojson from an inventory whose features pass '
                        'validate_possible_worlds.'
                        if len(lengths) == 1
                        else ' values, so they share no number of possible worlds. '
                        'A file written from one inventory holds lists of one '
                        'length. Write the file again from a single inventory '
                        'whose features pass validate_possible_worlds.'
                    )
                )
                raise ValueError(msg)
            if counts:
                worlds[label] = counts.pop()
            vector_features[label] = list(result.vector_features)

        if len(set(worlds.values())) > 1:
            msg = (
                'The sources hold different numbers of possible worlds: '
                + ', '.join(f'{n} in {where}' for where, n in worlds.items())
                + '. Pass sources of one run, such as the county files of one '
                'testbed, which share their number of worlds.'
                + (
                    ' A source of 1 world, such as one world of a run written '
                    'alone, is not merged with sources of more worlds. For an '
                    'inventory read with read_from_geojson from a file without '
                    'n_worlds, which leaves n_pw at 1, pass the file by its path.'
                    if 1 in worlds.values()
                    else ''
                )
            )
            raise ValueError(msg)
        first = labels[0]
        if set(vector_features[label]) != set(vector_features[first]):
            msg = (
                f'{label} has the vector features {vector_features[label]} and '
                f'{first} has {vector_features[first]}. A file has the default '
                f'{list(result.vector_features)}. Give each inventory those '
                'vector_features before filtering it with a file, give the '
                'inventories the same vector_features, or pass every source '
                'as a file.'
            )
            raise ValueError(msg)
        for asset_id in kept:
            key = str(asset_id)
            if key in kept_from:
                other_label, other_id = kept_from[key]
                if other_label == label:
                    msg = (
                        f'{label} holds the ids {other_id!r} and {asset_id!r} '
                        'inside the region, which write_to_geojson writes as '
                        f'the one id {key!r}. Give the assets ids whose string '
                        'forms differ.'
                    )
                else:
                    msg = (
                        f'The asset {asset_id!r} lies inside the region in both '
                        f'{other_label} and {label}, so the sources overlap. '
                        'Pass sources that hold no asset twice, such as the '
                        'county files of one testbed, which the county lines '
                        'divide, or filter each source alone.'
                    )
                raise ValueError(msg)
            kept_from[key] = (label, asset_id)
        result.inventory.update(kept)
        total += count
        if len(sources) > 1:
            noun = 'asset' if count == 1 else 'assets'
            messages.append(
                f'Kept {len(kept)} of {count} {noun} of {label} inside {name}.'
            )

    result.vector_features = list(vector_features[labels[0]])
    result.n_pw = next(iter(worlds.values()), 1)
    noun = 'asset' if total == 1 else 'assets'
    messages.append(f'Kept {len(result.inventory)} of {total} {noun} inside {name}.')
    for message in messages:
        print_ascii_fallback(message)
    return result


def find_inside(
    assets: Mapping[Hashable, Asset], boundary: BaseGeometry
) -> list[Hashable]:
    """
    Return the ids of the assets whose reference point lies inside a boundary.

    The point is :func:`reference_point`'s, and :func:`shapely.contains_xy` decides,
    so a point on the boundary is outside. A boundary that encloses no area, such as
    the repair of a ring that doubles back, holds no asset.

    Args:
        assets (Mapping[Hashable, Asset]):
            The assets, by id.
        boundary (BaseGeometry):
            The boundary, in longitude and latitude.

    Returns:
        list[Hashable]:
            The ids of the assets inside, in the order of ``assets``.

    Raises:
        ValueError:
            As :func:`reference_point` raises.
    """
    if boundary.area == 0:
        return []
    ids = list(assets)
    points = np.array(
        [reference_point(assets[a], a) for a in ids], dtype=float
    ).reshape(-1, 2)
    inside = shapely.contains_xy(boundary, points[:, 0], points[:, 1])
    return [a for a, is_inside in zip(ids, inside) if is_inside]


def read_boundary(
    region: RegionBoundary | Polygon | MultiPolygon,
) -> tuple[BaseGeometry, str, object]:
    """
    Return the boundary of a region, the name it is printed by, and its id.

    A :class:`RegionBoundary` is asked for its boundary once. An invalid boundary is
    repaired with :func:`shapely.make_valid`, of which the polygonal parts are kept;
    a repair without them, which encloses no area, is kept whole.

    Args:
        region (RegionBoundary | Polygon | MultiPolygon):
            The region, as :func:`resolve_region` returns it.

    Returns:
        tuple[BaseGeometry, str, object]:
            The boundary; the name printed for the region, ``'the region'`` for a
            polygon; and the id the region's ``get_boundary`` gives, such as a
            GEOID or an OpenStreetMap id, ``None`` for a polygon.

    Raises:
        TypeError:
            If ``region`` is neither a region nor a Shapely polygon or multipolygon.
    """
    if isinstance(region, RegionBoundary):
        boundary, name, identifier = region.get_boundary()
    elif isinstance(region, (Polygon, MultiPolygon)):
        boundary, name, identifier = region, 'the region', None
    else:
        msg = (
            'region must be a place name, a RegionBoundary, or a Shapely '
            f'Polygon or MultiPolygon, not {type(region).__name__}.'
        )
        raise TypeError(msg)
    if not boundary.is_valid:
        repaired = shapely.make_valid(boundary)
        parts = [
            part
            for part in shapely.get_parts(repaired)
            if isinstance(part, (Polygon, MultiPolygon))
        ]
        boundary = shapely.union_all(parts) if parts else repaired
    return boundary, name, identifier


def resolve_region(
    region: str | RegionBoundary | Polygon | MultiPolygon,
    source: str = 'census',
    *,
    vintage: int = DEFAULT_TIGER_VINTAGE,
) -> RegionBoundary | Polygon | MultiPolygon:
    """
    Return a census or OpenStreetMap region for a place name, and others as given.

    Args:
        region (str | RegionBoundary | Polygon | MultiPolygon):
            A place name with its state, or a region, which is returned as it is
            for the caller to check, a value of another type included.
        source (str, optional):
            Where a place name is looked up: ``'census'`` for
            :class:`~brails.types.census_boundary.CensusBoundary`, ``'osm'`` for
            :class:`RegionBoundary` of the name. Not consulted for a region that
            is not a place name. Defaults to ``'census'``.
        vintage (int, optional):
            The year of the TIGER/Line release a census lookup reads. Defaults to
            :data:`~brails.utils.tiger_files.DEFAULT_TIGER_VINTAGE`.

    Returns:
        RegionBoundary | Polygon | MultiPolygon:
            The :class:`~brails.types.census_boundary.CensusBoundary` or the
            :class:`RegionBoundary` of a place name, and ``region`` itself
            otherwise.

    Raises:
        ValueError:
            If ``source`` is another value, whatever the region, or if
            ``CensusBoundary`` refuses the name.
    """
    if source not in ('census', 'osm'):
        msg = f"source must be 'census' or 'osm', not {source!r}."
        raise ValueError(msg)
    if not isinstance(region, str):
        return region
    if source == 'osm':
        return RegionBoundary({'type': 'locationName', 'data': region})
    return CensusBoundary(region, vintage=vintage)


def _stream_features(  # noqa: C901
    path: Path, stated: dict[str, int]
) -> Iterator[dict[str, Any]]:
    """
    Yield the features of a GeoJSON FeatureCollection one at a time.

    The file is read as UTF-8, with or without a byte order mark,
    :data:`READ_CHUNK_CHARS` characters at a time, and each value is parsed
    with :meth:`json.JSONDecoder.raw_decode`, so the top-level members may
    come in any order. A value that ends at the end of what has been read, or
    before a character that could continue a number, is parsed again with
    more of the file, so a number cut at a chunk's edge is not taken short.
    The members ``type``, ``crs`` and ``n_worlds`` are checked when they are
    read, so a file that writes them before ``features``, as
    ``write_to_geojson`` does, is refused before its first feature is
    yielded.

    Args:
        path (Path):
            The file.
        stated (dict[str, int]):
            Receives the file's ``n_worlds`` under that key, if the file gives
            one.

    Yields:
        dict[str, Any]:
            Each element of the ``features`` array, as parsed.

    Raises:
        ValueError:
            If the file is not a JSON object, holds invalid JSON (a trailing
            comma or a member name that is not a string among it), has no
            ``features`` member or two, has a ``type`` other than
            ``FeatureCollection``, has a ``crs`` other than one of
            :data:`GEOGRAPHIC_CRS_NAMES` and :data:`NAD83_CRS_NAMES`, or has
            an ``n_worlds`` that is not an integer of at least 1.
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
                    if key == 'n_worlds':
                        stated['n_worlds'] = check_n_worlds(member, path)
                    if key == 'type' and member != 'FeatureCollection':
                        msg = (
                            f'{path} has the type {member!r}, not '
                            'FeatureCollection. Pass a file that holds a '
                            'FeatureCollection of the inventory.'
                        )
                        raise ValueError(msg)
                    # A null crs, which the 2008 GeoJSON specification
                    # allowed, is taken as no crs.
                    if key == 'crs' and member is not None:
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
                        if name in NAD83_CRS_NAMES:
                            print_ascii_fallback(
                                f'{path} gives its coordinates in NAD83 '
                                f'({name}). They are used as they are, as '
                                'WGS84 longitudes and latitudes, from which '
                                'they differ slightly.'
                            )
                        elif name not in GEOGRAPHIC_CRS_NAMES:
                            msg = (
                                f'{path} gives the crs '
                                f'{json.dumps(member)[:200]}. filter_to_region '
                                'reads longitude and latitude in WGS84 '
                                '(EPSG:4326) or NAD83 (EPSG:4269) only, named '
                                'as one of '
                                + ', '.join(GEOGRAPHIC_CRS_NAMES + NAD83_CRS_NAMES)
                                + '. Write the file in EPSG:4326, as '
                                'write_to_geojson does, give its crs one of '
                                'those names if it is one of those systems, '
                                'or remove the crs member if its coordinates '
                                'are already longitude and latitude.'
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
