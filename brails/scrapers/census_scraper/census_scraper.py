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
#

"""
This module defines classes associated with scraping US Census data.

.. autosummary::

    CensusScraper
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import geopandas as gpd
import requests
from requests import exceptions
from shapely.geometry import shape

from brails.types.asset_inventory import AssetInventory

CENSUS_TRACT_QUERY_URL = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Census2020/MapServer/6/query'


class CensusScraper:
    """A class to get data from the US Census."""

    def __init__(self) -> None:
        """Initialize an instance of the class."""

    def _fetch_tract_geometry(
        self,
        lon: float,
        lat: float,
        retries: int = 3,
        timeout: int = 10,
        delay: float = 1,
    ) -> Dict:
        """
        Fetches a Census Tract GeoJSON from the TIGERweb API.

        Retries only on transient errors:
        - exceptions.ConnectionError (network failure)
        - Timeout (request took too long)
        - 5xx Server Errors (API server is temporarily down)

        Fails immediately on permanent errors:
        - 4xx Client Errors (our request is bad)
        - Successful response with no 'features' (violates our logic)

        Attributes:
            lon (float):
                Longitude coordinate of the asset centroid
            lat (float):
                Latitude coordinate of the asset centroid
            retries (int, optional):
                Number of retry attempts for failed requests. Defaults to 3.
            timeout (int, optional):
                Request timeout in seconds. Defaults to 10.
            delay (float, optional):
                Delay between retries in seconds. Defaults to 1.

        Returns:
            dict:
                GeoJSON feature containing the census tract geometry and properties
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
                response = requests.get(
                    CENSUS_TRACT_QUERY_URL, params=query_params, timeout=timeout
                )

                # This will raise an HTTPError if the status is 4xx or 5xx
                response.raise_for_status()

                # If the request was successful, decode the JSON
                response_data = response.json()

                # Validate the response data
                features = response_data.get('features')
                if not features:  # This checks for None OR an empty list []
                    # This is a permanent failure. The API worked, but returned no data.
                    msg = (
                        f"No 'features' found for point ({lat}, {lon}). "
                        f'The point is likely outside census boundaries.'
                    )
                    raise ValueError(msg)

                # If all checks pass, return the first feature and exit the function
                return features[0]

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
                    # Raise the error to stop the script immediately.
                    print(
                        f'  ERROR: Client error ({e.response.status_code}). Halting.'
                    )
                    raise  # Re-raise the HTTPError to stop execution

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

    def get_census_tracts(self, asset_inventory: AssetInventory) -> Dict[str, Any]:
        """
        Identify the census tract corresponding to each asset.

        Args:
            asset_inventory (AssetInventory):
                The inventory of assets to process.

        Returns:
            downloaded_tracts_cache (dict):
                The downloaded census tract labels and geometries.

        """
        # Validate input type early
        if not isinstance(asset_inventory, AssetInventory):
            raise TypeError('asset_inventory must be an instance of AssetInventory')

        # Guard: empty inventory -> nothing to process
        if len(asset_inventory.get_asset_ids()) == 0:
            print('Input asset inventory is empty, skipping census tract lookup.')
            return {}

        # Prepare a GeoDataFrame that is easier to work with
        geojson_data = asset_inventory.get_geojson()
        asset_gdf = gpd.GeoDataFrame.from_features(
            geojson_data['features'], crs='EPSG:4326'
        )
        asset_gdf.set_index('id', inplace=True)

        # Create a copy that will serve as our "to-do list." We will shrink this list.
        asset_gdf_to_do = asset_gdf.copy()

        # Create an empty gdf that will store the results
        asset_gdf_with_census_tracts = gpd.GeoDataFrame(columns=asset_gdf.columns)

        # And a dictionary to cache polygons we've already downloaded
        downloaded_tracts_cache = {}

        print(f'Starting job. Total points to process: {len(asset_gdf_to_do)}')

        # Initialize error state
        error_occurred = False
        error_details: Optional[BaseException] = None

        try:
            while not asset_gdf_to_do.empty:
                first_point = asset_gdf_to_do.iloc[0]
                point_geom = first_point.geometry

                print(
                    f'\nProcessing new tract. Points remaining: {len(asset_gdf_to_do)}'
                )
                print(f'  Picking point: ({point_geom.y}, {point_geom.x})')

                # This function will either return a valid 'feature' or raise an exception.
                feature = self._fetch_tract_geometry(
                    lon=point_geom.x, lat=point_geom.y
                )

                # If the code continues, 'feature' is guaranteed to be valid.
                tract_geoid = feature['properties']['GEOID']

                # Cache the polygon geometry for this tract GEOID
                print(f'  API Call success. Caching new GEOID: {tract_geoid}')
                tract_polygon = shape(feature['geometry'])
                downloaded_tracts_cache[tract_geoid] = tract_polygon

                # Find all points within this polygon
                points_within_this_tract = asset_gdf_to_do[
                    asset_gdf_to_do.intersects(tract_polygon)
                ]

                print(
                    f'  Local search found {len(points_within_this_tract)} '
                    f'points inside this tract.'
                )

                points_within_this_tract = points_within_this_tract.assign(
                    TRACT_GEOID=tract_geoid
                )
                asset_gdf_with_census_tracts = gpd.pd.concat(
                    [asset_gdf_with_census_tracts, points_within_this_tract]
                )

                asset_gdf_to_do = asset_gdf_to_do.drop(
                    points_within_this_tract.index
                )

        except (
            ValueError,
            exceptions.HTTPError,
            exceptions.ConnectionError,
            KeyError,
        ) as e:
            # Mark error state and capture details; do not print here
            error_occurred = True
            error_details = e

        # Post-processing based on success/failure
        if not error_occurred:
            # Success path: update original inventory only for processed assets
            for asset_id in asset_gdf_with_census_tracts.index:
                asset_inventory.add_asset_features(
                    asset_id,
                    {
                        'TRACT_GEOID': asset_gdf_with_census_tracts.loc[
                            asset_id, 'TRACT_GEOID'
                        ]
                    },
                )

            # Success summary
            print('\n--- Job Finished ---')
            print(
                f'Total points successfully processed: {len(asset_gdf_with_census_tracts)}'
            )
            print(f'Total points remaining (unprocessed): {len(asset_gdf_to_do)}')
            print(f'Total unique API calls made: {len(downloaded_tracts_cache)}')
        else:
            # Failure summary and critical error report
            print('\n--- CRITICAL ERROR ---')
            print('The script has halted due to a permanent or unrecoverable error.')
            print(
                f'Error Type: {type(error_details).__name__ if error_details else "Unknown"}'
            )
            print(f'Details: {error_details}')
            print('No further processing will be done.')
            print('\n--- Job Halted ---')
            print(
                f'Total points successfully processed: {len(asset_gdf_with_census_tracts)}'
            )
            print(f'Total points remaining (unprocessed): {len(asset_gdf_to_do)}')
            print(f'Total unique API calls made: {len(downloaded_tracts_cache)}')

        return downloaded_tracts_cache
