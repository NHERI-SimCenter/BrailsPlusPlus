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
#
# Contributors:
# Adam Zsarnoczay

"""
This module defines classes associated with scraping US Census data.

.. autosummary::

    CensusTractScraper
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import requests
import shapely
from pyproj import CRS
from requests import exceptions
from shapely.geometry import shape

from brails.constants import (
    BLOCK_GROUP_KEY,
    GEOGRAPHIC_CRS,
    GEOID_LENGTH,
    TRACT_GEOID_LENGTH,
    TRACT_KEY,
)
from brails.types.asset_inventory import AssetInventory
from brails.utils.id_preview import preview_ids
from brails.utils.region_filter import reference_point

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import geopandas as gpd

CENSUS_TRACT_QUERY_URL = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Census2020/MapServer/6/query'
CENSUS_BLOCK_GROUP_QUERY_URL = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Census2020/MapServer/8/query'

# The census geographies the scraper assigns.
LEVEL_TRACT = 'tract'
LEVEL_BLOCK_GROUP = 'block_group'
LEVELS = (LEVEL_TRACT, LEVEL_BLOCK_GROUP)

# Columns of the table get_census_geographies returns: an asset on an edge or
# a vertex shared by several polygons, the GEOID it received, and the number
# of polygons its point touches.
TIE_COLUMNS = ('asset_id', 'geoid', 'polygons')


class CensusTractScraper:
    """
    A class to get census tract and block-group data from the US Census.

    An asset is located by the centroid of its exterior rings for a
    footprint (:meth:`AssetInventory.centroid_and_plan_area`), by the point
    itself for a point asset, and by the Shapely centroid for any other
    geometry that encloses no area. The location takes the GEOID of the
    polygon that contains or touches it, and a location on an edge or a
    vertex shared by several polygons takes the lowest GEOID, so the result
    does not depend on the order of the assets. A location in no polygon is
    an error.
    """

    def __init__(self) -> None:
        """Initialize an instance of the class."""

    def _fetch_features(
        self,
        lon: float,
        lat: float,
        url: str = CENSUS_TRACT_QUERY_URL,
        retries: int = 3,
        timeout: int = 10,
        delay: float = 1,
    ) -> list[dict]:
        """
        Fetch the census polygons that intersect a point from TIGERweb.

        Retries only on transient errors:
        - exceptions.ConnectionError (network failure)
        - Timeout (request took too long)
        - 5xx Server Errors (API server is temporarily down)

        Fails immediately on permanent errors:
        - 4xx Client Errors (our request is bad)

        Args:
            lon (float):
                Longitude of the point.
            lat (float):
                Latitude of the point.
            url (str, optional):
                The query endpoint of a TIGERweb layer. Defaults to
                :data:`CENSUS_TRACT_QUERY_URL`.
            retries (int, optional):
                Number of retry attempts for failed requests. Defaults to 3.
            timeout (int, optional):
                Request timeout in seconds. Defaults to 10.
            delay (float, optional):
                Delay between retries in seconds. Defaults to 1.

        Returns:
            list[dict]:
                The GeoJSON features of the polygons containing or touching
                the point, each with its GEOID, empty for a point outside
                every polygon of the layer.

        Raises:
            requests.exceptions.HTTPError:
                On a 4xx response.
            requests.exceptions.ConnectionError:
                If every attempt fails with a transient error.
        """
        query_params = {
            'geometry': f'{lon},{lat}',
            'geometryType': 'esriGeometryPoint',
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': 'GEOID',
            'returnGeometry': 'true',
            'f': 'geojson',
        }

        for attempt in range(retries):
            try:
                response = requests.get(url, params=query_params, timeout=timeout)

                # This will raise an HTTPError if the status is 4xx or 5xx
                response.raise_for_status()

                # A point outside every polygon gets no features.
                return response.json().get('features') or []

            except (exceptions.ConnectionError, exceptions.Timeout) as e:
                # This is a transient network error. We should retry.
                print(
                    f'  WARNING: Network error ({type(e).__name__}). '
                    f'Retrying ({attempt + 1}/{retries})...'
                )

            except exceptions.HTTPError as e:
                # Check if this is a 5xx Server Error (which we can retry)
                # or a 4xx Client Error (which we cannot)
                if 500 <= e.response.status_code < 600:  # noqa: PLR2004
                    print(
                        f'  WARNING: Server error ({e.response.status_code}). '
                        f'Retrying ({attempt + 1}/{retries})...'
                    )
                else:
                    # This is a 4xx error. Our request is bad. Do not retry.
                    raise

            except exceptions.JSONDecodeError:
                # The server returned something that wasn't JSON. This is a server problem.
                print(
                    f'  WARNING: Failed to decode JSON response. '
                    f'Retrying ({attempt + 1}/{retries})...'
                )

            # Wait for a moment before the next retry
            time.sleep(delay)

        # If we exit the loop, all retries have failed. Raise a final error.
        msg = (
            f'Failed to fetch data for point ({lat}, {lon}) after {retries} '
            f'attempts.'
        )
        raise exceptions.ConnectionError(msg)

    def get_census_tracts(self, asset_inventory: AssetInventory) -> dict[str, Any]:
        """
        Identify the census tract corresponding to each asset.

        The tracts are queried from TIGERweb's Census 2020 tract layer, and
        each asset's GEOID is written to its :data:`~brails.constants.TRACT_KEY`
        feature in place.

        Args:
            asset_inventory (AssetInventory):
                The inventory of assets to process.

        Returns:
            downloaded_tracts_cache (dict):
                The GEOID and geometry of each census tract that received an
                asset.

        Raises:
            TypeError:
                If ``asset_inventory`` is not an ``AssetInventory``.
            ValueError:
                If an asset has no coordinates, its coordinates form no
                geometry, or it lies outside every tract. Nothing is written
                then.
            requests.exceptions.HTTPError:
                On a 4xx response from TIGERweb.
            requests.exceptions.ConnectionError:
                If a query fails after its retries.
        """
        # Validate input type early
        if not isinstance(asset_inventory, AssetInventory):
            raise TypeError('asset_inventory must be an instance of AssetInventory')

        # Guard: empty inventory -> nothing to process
        if len(asset_inventory.get_asset_ids()) == 0:
            print('Input asset inventory is empty, skipping census tract lookup.')
            return {}

        asset_ids, geoids, polygons, _ = self._assign(
            asset_inventory, LEVEL_TRACT, None
        )
        for asset_id, geoid in zip(asset_ids, geoids):
            asset_inventory.add_asset_features(asset_id, {TRACT_KEY: geoid})
        # the tracts that received an asset, not every tract an edge fetched
        return {geoid: polygons[geoid] for geoid in sorted(set(geoids))}

    def get_census_geographies(
        self,
        inventory: AssetInventory,
        *,
        levels: Sequence[str] = LEVELS,
        polygons: gpd.GeoDataFrame | None = None,
        tract_key: str = TRACT_KEY,
        block_group_key: str = BLOCK_GROUP_KEY,
        geoid_column: str = 'GEOID',
    ) -> tuple[AssetInventory, pd.DataFrame]:
        """
        Give every asset of an inventory its census tract, block group, or both.

        Without ``polygons`` the block groups are queried from TIGERweb's
        Census 2020 block-group layer when ``levels`` holds
        :data:`LEVEL_BLOCK_GROUP`, and a tract's GEOID is the first
        :data:`~brails.constants.TRACT_GEOID_LENGTH` digits of its block
        group's; otherwise the tracts are queried from the tract layer.
        Given polygons are read at the level their GEOID length names.

        Args:
            inventory (AssetInventory):
                The assets, with coordinates in
                :data:`~brails.constants.GEOGRAPHIC_CRS`. It is not modified.
            levels (Sequence[str], optional):
                The geographies written, from :data:`LEVELS`. Defaults to
                both.
            polygons (geopandas.GeoDataFrame, optional):
                One polygon per tract or per block group, with a CRS, such
                as a TIGER/Line file. A CRS other than
                :data:`~brails.constants.GEOGRAPHIC_CRS` is reprojected. It
                is not modified. Defaults to ``None``, which queries
                TIGERweb.
            tract_key (str, optional):
                The feature a tract's GEOID is written to. Defaults to
                :data:`~brails.constants.TRACT_KEY`.
            block_group_key (str, optional):
                The feature a block group's GEOID is written to. Defaults to
                :data:`~brails.constants.BLOCK_GROUP_KEY`.
            geoid_column (str, optional):
                The column of ``polygons`` holding the GEOIDs, as strings of
                :data:`~brails.constants.TRACT_GEOID_LENGTH` digits for
                tracts or :data:`~brails.constants.GEOID_LENGTH` digits for
                block groups. Defaults to ``'GEOID'``.

        Returns:
            tuple[AssetInventory, pandas.DataFrame]:
                A copy of the inventory in which every asset holds the keys
                of ``levels``, and a table of the assets on an edge or a
                vertex shared by several polygons, with the columns of
                :data:`TIE_COLUMNS`, in ascending order of the asset id as
                a string.

        Raises:
            TypeError:
                If ``inventory`` is not an ``AssetInventory``.
            ValueError:
                If ``levels`` is empty or holds a value outside
                :data:`LEVELS` or a value twice; if ``polygons`` is empty or
                has no CRS, its GEOID column is missing or numeric, or holds
                a value that is not a string of the tract or the block-group
                length or values of both lengths; if ``polygons`` holds
                tracts and ``levels`` asks for block groups; or if an asset
                has no coordinates, coordinates that form no geometry, or a
                location outside every polygon. The message of the last
                names the assets.
            requests.exceptions.HTTPError:
                On a 4xx response from TIGERweb.
            requests.exceptions.ConnectionError:
                If a query fails after its retries.
        """
        if not isinstance(inventory, AssetInventory):
            msg = 'inventory must be an instance of AssetInventory.'
            raise TypeError(msg)
        levels = tuple(levels)
        if (
            not levels
            or len(set(levels)) != len(levels)
            or set(levels) - set(LEVELS)
        ):
            msg = (
                f'levels must name one or both of {LEVELS} once each, not '
                f'{levels!r}.'
            )
            raise ValueError(msg)

        if polygons is None:
            level = LEVEL_BLOCK_GROUP if LEVEL_BLOCK_GROUP in levels else LEVEL_TRACT
            frame = None
        else:
            level, frame = _read_polygons(polygons, geoid_column)
            if level == LEVEL_TRACT and LEVEL_BLOCK_GROUP in levels:
                msg = (
                    'The polygons are tracts, whose GEOIDs hold no block '
                    'group; pass block groups or drop LEVEL_BLOCK_GROUP from '
                    'levels.'
                )
                raise ValueError(msg)

        result = deepcopy(inventory)
        if not result.inventory:
            return result, pd.DataFrame(columns=list(TIE_COLUMNS))
        asset_ids, geoids, _, ties = self._assign(result, level, frame)
        for asset_id, geoid in zip(asset_ids, geoids):
            features = {}
            if LEVEL_TRACT in levels:
                features[tract_key] = geoid[:TRACT_GEOID_LENGTH]
            if LEVEL_BLOCK_GROUP in levels:
                features[block_group_key] = geoid
            result.add_asset_features(asset_id, features)
        return result, ties

    def _assign(
        self,
        inventory: AssetInventory,
        level: str,
        frame: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[list[Hashable], list[str], dict[str, Any], pd.DataFrame]:
        """
        Locate every asset in the polygons of one level.

        Args:
            inventory (AssetInventory):
                A non-empty inventory.
            level (str):
                :data:`LEVEL_TRACT` or :data:`LEVEL_BLOCK_GROUP`.
            frame (tuple[numpy.ndarray, numpy.ndarray] | None):
                The GEOIDs and polygons :func:`_read_polygons` returns, or
                ``None`` to query TIGERweb's layer of ``level``.

        Returns:
            tuple[list, list[str], dict[str, Any], pandas.DataFrame]:
                The asset ids in ascending order as strings, the GEOID of
                each, the polygons read or fetched by GEOID, and the table
                of ties with the columns of :data:`TIE_COLUMNS`.

        Raises:
            ValueError:
                If an asset has no coordinates, coordinates that form no
                geometry, or a location in no polygon.
        """
        asset_ids = sorted(inventory.inventory, key=str)
        points = shapely.points(
            np.array([reference_point(inventory.inventory[a], a) for a in asset_ids])
        )
        if frame is None:
            url = (
                CENSUS_BLOCK_GROUP_QUERY_URL
                if level == LEVEL_BLOCK_GROUP
                else CENSUS_TRACT_QUERY_URL
            )
            geoids, polygons = self._fetch_polygons(points, url)
        else:
            geoids, polygons = frame
        order = np.argsort(geoids, kind='stable')
        geoids = geoids[order]
        polygons = polygons[order]

        point_index, polygon_index = shapely.STRtree(polygons).query(
            points, predicate='intersects'
        )
        # Polygons are sorted by GEOID, so the lowest index is the lowest GEOID.
        chosen = np.full(len(asset_ids), -1, dtype=np.int64)
        hits = np.zeros(len(asset_ids), dtype=np.int64)
        for p, g in zip(point_index.tolist(), polygon_index.tolist()):
            hits[p] += 1
            if chosen[p] < 0 or g < chosen[p]:
                chosen[p] = g

        outside = [str(asset_ids[i]) for i in np.flatnonzero(chosen < 0)]
        if outside:
            name = 'block group' if level == LEVEL_BLOCK_GROUP else 'tract'
            preview = preview_ids(outside)
            msg = (
                f'{len(outside)} asset(s) lie outside every {name} polygon: '
                f'{preview}. Census geographies cover the United States and '
                'its territories; check the coordinates.'
            )
            raise ValueError(msg)

        assigned = [str(geoids[g]) for g in chosen]
        ties = pd.DataFrame(
            [
                (asset_ids[i], assigned[i], int(hits[i]))
                for i in np.flatnonzero(hits > 1)
            ],
            columns=list(TIE_COLUMNS),
        )
        return asset_ids, assigned, dict(zip(geoids.tolist(), polygons)), ties

    def _fetch_polygons(
        self, points: np.ndarray, url: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Query TIGERweb for the polygons that hold a set of points.

        A point is queried unless it lies in the interior of a polygon
        already fetched, so a point on a shared edge or vertex fetches every
        polygon it touches.

        Args:
            points (numpy.ndarray):
                Shapely points in :data:`~brails.constants.GEOGRAPHIC_CRS`.
            url (str):
                The query endpoint of the layer.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]:
                The GEOIDs fetched, as an object array of strings, and their
                polygons as an array of Shapely geometries.

        Raises:
            requests.exceptions.HTTPError:
                On a 4xx response.
            requests.exceptions.ConnectionError:
                If a query fails after its retries.
        """
        fetched: dict[str, Any] = {}
        settled = np.zeros(len(points), dtype=bool)
        for index, point in enumerate(points):
            if settled[index]:
                continue
            for feature in self._fetch_features(lon=point.x, lat=point.y, url=url):
                polygon = shape(feature['geometry'])
                fetched[str(feature['properties']['GEOID'])] = polygon
                settled |= shapely.contains_properly(polygon, points)
        return (
            np.array(list(fetched), dtype=object),
            np.array(list(fetched.values()), dtype=object),
        )


def _read_polygons(
    polygons: gpd.GeoDataFrame, geoid_column: str
) -> tuple[str, tuple[np.ndarray, np.ndarray]]:
    """
    Check census polygons and return their level, GEOIDs and geometries.

    Args:
        polygons (geopandas.GeoDataFrame):
            The polygons as given to
            :meth:`CensusTractScraper.get_census_geographies`.
        geoid_column (str):
            The column holding the GEOIDs.

    Returns:
        tuple[str, tuple[numpy.ndarray, numpy.ndarray]]:
            :data:`LEVEL_TRACT` or :data:`LEVEL_BLOCK_GROUP`, and the GEOIDs
            as an object array of strings with the polygons, in
            :data:`~brails.constants.GEOGRAPHIC_CRS`, as an array of Shapely
            geometries, in the order of the rows.

    Raises:
        ValueError:
            In the cases
            :meth:`CensusTractScraper.get_census_geographies` lists for
            ``polygons``.
    """
    if polygons.crs is None:
        msg = 'polygons has no CRS; set one before assigning census geographies.'
        raise ValueError(msg)
    if len(polygons) == 0:
        msg = 'polygons holds no polygon.'
        raise ValueError(msg)
    if geoid_column not in polygons.columns:
        msg = f'polygons has no column {geoid_column!r}.'
        raise ValueError(msg)
    column = polygons[geoid_column]
    if pd.api.types.is_numeric_dtype(column):
        msg = (
            f'The GEOID column {geoid_column!r} is numeric; read it as strings '
            'so that leading zeros are kept.'
        )
        raise ValueError(msg)
    geoids = column.tolist()
    lengths = {TRACT_GEOID_LENGTH: LEVEL_TRACT, GEOID_LENGTH: LEVEL_BLOCK_GROUP}
    bad = [
        value
        for value in geoids
        if not (
            isinstance(value, str)
            and len(value) in lengths
            and value.isascii()
            and value.isdigit()
        )
    ]
    if bad:
        msg = (
            f'The GEOID column {geoid_column!r} holds values that are not '
            f'strings of {TRACT_GEOID_LENGTH} digits (tracts) or '
            f'{GEOID_LENGTH} digits (block groups), such as {bad[0]!r}.'
        )
        raise ValueError(msg)
    found = {len(value) for value in geoids}
    if len(found) > 1:
        msg = (
            f'The GEOID column {geoid_column!r} mixes tract and block-group '
            'GEOIDs; pass one level.'
        )
        raise ValueError(msg)
    frame = polygons
    if not CRS.from_user_input(frame.crs).equals(
        CRS.from_user_input(GEOGRAPHIC_CRS)
    ):
        frame = frame.to_crs(GEOGRAPHIC_CRS)
    return lengths[found.pop()], (
        np.array(geoids, dtype=object),
        np.asarray(frame.geometry.array),
    )
