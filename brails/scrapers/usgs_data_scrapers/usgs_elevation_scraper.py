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
# Barbaros Cetiner
#
# Last updated:
# 08-06-2025

"""
This module defines classes associated with scraping USGS 3DEP elevation data.

.. autosummary::

    USGSElevationScraper
"""

import concurrent.futures
import random
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import requests
from shapely.geometry import Point
from tqdm.auto import tqdm

from brails.constants import DEFAULT_UNITS
from brails.types.asset_inventory import Asset, AssetInventory
from brails.types.region_boundary import RegionBoundary
from brails.utils import ArcgisAPIServiceHelper, UnitConverter


API_ENDPOINT = "https://epqs.nationalmap.gov/v1/json"

ELEVATION_FEATURE_NAMES = {
    "centroid": "centroid_elevation",
    "all": "all_elevations",
    "average": "average_elevation",
    "min": "min_elevation",
    "max": "max_elevation",
    "median": "median_elevation",
    "stddev": "stddev_elevation"
}

DEFAULT_ELEVATION_UNIT = 'ft'


class USGSElevationScraper:
    """
    A class to get elevation data using the USGS Elevation Point Query Service.

    This class provides functionality to fetch elevation data for individual
    coordinates, assets (which may consist of multiple coordinates), and
    geographic regions. It supports multiple elevation data aggregation modes
    such as centroid elevation, average, min, max, median, and standard
    deviation of elevations across asset coordinates. Elevation data
    is retrieved from the USGS service and can be converted to desired length
    units.

    The class is designed to work efficiently by fetching data in parallel and
    includes support for sampling random points within polygonal region
    boundaries for broader elevation analysis.

    Discussion of accuracy:
    https://www.usgs.gov/faqs/how-accurate-are-elevations-generated-elevation-point-query-service-national-map

    Attributes:
        units (dict):
            Dictionary specifying units of measurement for length
            (e.g., 'ft', 'm'), parsed from the input configuration or set to
            defaults.
        inventory (AssetInventory):
            The asset inventory instance containing assets with coordinates and
            features to be augmented with elevation data.

    Methods:
        supported_asset_elevation_modes(print_modes: bool = True) -> Set[str]:
            Returns the set of supported elevation modes, optionally printing
            them.
        get_asset_elevation_data(
            asset_inventory: AssetInventory,
            modes: Union[str, List[str]] = "centroid"
        ) -> AssetInventory:
            Computes and adds elevation features to assets based on specified
            modes.
        get_region_elevation_data(
            region,
            num_points: int,
            seed: Optional[int] = None
        ) -> AssetInventory:
            Samples elevation data within a specified geographic region.
        get_elevation_usgs(x: float, y: float) -> dict:
            Fetches elevation data for a single coordinate from the USGS
            Elevation Point Query Service.
        fetch_all_elevations(
            coords: Iterable[Tuple[float, float]]
        ) -> List[Dict[str, Any]]:
            Fetches elevations concurrently for a list of coordinates.

    Notes:
        - Methods that modify the asset inventory typically do so in place,
          updating the provided AssetInventory instance directly.
        - Elevation values retrieved from USGS are in feet by default, with
          unit conversions applied according to the configured units.
    """

    def __init__(self, input_dict: Dict[str, Any] = None):
        """
        Initialize an instance of the class with specified units.

        This constructor allows you to specify the length unit through the
        optional `input_dict`. If `input_dict` is not provided, the length
        unit defaults to 'ft' (feet). The inventory is also initialized when
        an instance is created.

        Args:
            input_dict (dict, optional):
                A dictionary that may contain a key 'length' specifying the
                length unit to be used. If 'length' is provided in the
                dictionary, it will be used to set the length unit (the value
                will be converted to lowercase). If the dictionary is not
                provided or if 'length' is not specified, 'ft' (feet) will be
                used as the default length unit.
        """
        # Parse units from input_dict or fall back to default units:
        self.units = UnitConverter.parse_units(input_dict or {}, DEFAULT_UNITS)
        self.inventory = AssetInventory()

    @staticmethod
    def supported_asset_elevation_modes(print_modes: bool = True) -> Set[str]:
        """
        Return the set of supported elevation modes.

        Args:
            print_modes (bool, optional):
                If True, prints the supported modes. Defaults to True.

        Returns:
            Set[str]:
                A set containing the supported elevation mode keys.
        """
        allowed_modes = set(ELEVATION_FEATURE_NAMES.keys())

        if print_modes:
            print(f"Supported modes: {', '.join(sorted(allowed_modes))}")

        return allowed_modes

    def get_asset_elevation_data(
        self,
        asset_inventory: AssetInventory,
        modes: Union[str, List[str]] = "centroid"
    ) -> AssetInventory:
        """
        Compute elevation features for assets in the provided AssetInventory.

        Based on the specified `modes`, this method fetches elevation data
        either at the centroid of each asset or for all coordinates, then
        computes summary statistics (e.g., average, min, max, median, stddev).

        All elevation values are converted from feet to the target length unit
        before being stored under the `features` dictionary of each asset.

        Args:
            asset_inventory (AssetInventory):
                The inventory of assets to process for elevation data.
            modes (Union[str, List[str]]):
                One or more of the following modes:
                  - "centroid": elevation at asset centroid
                  - "all": elevation for all asset coordinates
                  - "average", "min", "max", "median", "stddev": summary
                     statistics

        Returns:
            AssetInventory:
                The modified inventory with added elevation features.
        """
        self.inventory = asset_inventory
        normalized_modes = self._normalize_and_filter_modes(modes)

        if not normalized_modes:
            print(
                'No valid elevation modes were provided. Skipping elevation '
                'processing.'
            )
            return self.inventory

        # If 'centroid' is requested, handle separately
        if 'centroid' in normalized_modes:
            self._process_centroid_elevations()
            normalized_modes.remove('centroid')

        # Handle any other coordinate-based elevation modes
        if normalized_modes:
            self._compute_asset_elevation_features(list(normalized_modes))

        return self.inventory

    def get_region_elevation_data(
        self,
        region: RegionBoundary,
        num_points: int,
        seed: Optional[int] = None
    ) -> AssetInventory:
        """
        Sample elevation data within a region boundary.

        Args:
            region:
                RegionBoundary object with a `get_boundary()` method.
            num_points (int):
                Total number of points to sample within the region boundary.
            seed (int, optional):
                Random seed for reproducibility. Default is None.

        Returns:
            AssetInventory:
                An inventory of sampled points with elevation values stored in
                each asset's features.
        """
        # Set random seed if provided for reproducibility
        if seed is not None:
            random.seed(seed)

        # Extract polygon boundary:
        polygon, _, _ = region.get_boundary()

        # Use rejection sampling to find points within polygon:
        minx, miny, maxx, maxy = polygon.bounds
        sampled_coords = []

        max_attempts = num_points * 20  # Prevent infinite loop
        attempts = 0
        while len(sampled_coords) < num_points and attempts < max_attempts:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            point = Point(x, y)

            if polygon.contains(point):
                sampled_coords.append((x, y))
            attempts += 1

        # Create an AssetInventory to hold the sampled elevation points:
        inventory = AssetInventory()

        if not sampled_coords:
            print('No points could be sampled inside the polygon.')
            return inventory

        # Fetch elevation data for the sampled coordinates:
        elevation_results = self.fetch_all_elevations(sampled_coords)
        target_length_unit = self.units['length']
        for asset_id, ((x, y), result) in enumerate(
            zip(sampled_coords,
                elevation_results)
        ):
            elevation = result['elevation']
            if elevation is not None:
                elevation = UnitConverter.convert_length(
                    elevation, DEFAULT_ELEVATION_UNIT, target_length_unit)

            asset = Asset(
                asset_id=asset_id,
                coordinates=[[x, y]],
                features={"elevation": elevation}
            )
            inventory.add_asset(asset)

        return inventory

    @staticmethod
    def get_elevation_usgs(x: float, y: float) -> dict:
        """
        Get elevation data for a coordinate using USGS Elevation Point Service.

        This method sends a request to the USGS service using the provided
        geographic coordinates and returns a dictionary containing the
        longitude, latitude, and elevation (in feet). If the request fails or
        the elevation value is unavailable, the elevation is returned as
        `None`.

        Args:
            x (float):
                Longitude in decimal degrees (WGS84).
            y (float):
                Latitude in decimal degrees (WGS84).

        Returns:
            dict: A dictionary in the format:
                {
                    "x": <longitude>,
                    "y": <latitude>,
                    "elevation": <elevation in feet or None>
                }
        """
        params = {
            "x": x,
            "y": y,
            "units": DEFAULT_ELEVATION_UNIT,
            "wkid": 4326,
            "includeDate": "false"
        }

        result = {"x": x, "y": y, "elevation": None}

        try:
            response = ArcgisAPIServiceHelper._make_request_with_retry(
                API_ENDPOINT,
                params=params
            )
            if response.ok:
                data = response.json()
                result["elevation"] = data.get("value")
            else:
                print(
                    f'Elevation data download failed for ({x}, {y}). '
                    f'Status code: {response.status_code}')
        except requests.RequestException as e:
            print(f"Request error for ({x}, {y}): {e}")

        return result

    @staticmethod
    def fetch_all_elevations(
        coords: Iterable[Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """
        Fetch elevations in parallel for a list of (x, y) coordinate pairs.

        Fetch elevation data in parallel for a list of (x, y) coordinate pairs
        using the USGS Elevation service.

        Args:
            coords (Iterable[Tuple[float, float]]):
                An iterable of (x, y) coordinate pairs.

        Returns:
            List[Dict[str, Any]]:
                A list of result dictionaries, each containing elevation data.
        """
        # Convert the iterable to a list to support indexing and length
        # calculation:
        coords_list = list(coords)

        # Pre-allocate a results list with the same length as coords_list to
        # preserve order:
        results = [None] * len(coords_list)

        # Print a blank line for clean tqdm output formatting:
        print('\n')

        # Create a progress bar with the total number of coordinates:
        pbar = tqdm(total=len(coords_list),
                    desc='Downloading USGS elevation data')

        # Use a ThreadPoolExecutor to fetch elevation data concurrently:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    USGSElevationScraper.get_elevation_usgs, x, y
                ): i
                for i, (x, y) in enumerate(coords_list)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                pbar.update(1)
                try:
                    result = future.result()
                except Exception as e:
                    coord = coords_list[index]
                    result = {"x": coord[0], "y": coord[1], "elevation": None}
                    print(f"Exception for {coord}: {e}")
                results[index] = result

        pbar.close()
        return results

    def _normalize_and_filter_modes(
        self,
        modes: Union[str, List[str]]
    ) -> Set[str]:
        """
        Normalize mode entries and filter out unsupported modes.

        This method prints a warning if an unsupported modes is identified and
        is omitted.

        Args:
            modes (str or list of str):
                Mode(s) to normalize and filter.

        Returns:
            Set[str]:
                Normalized set of valid mode strings.
        """
        # If a single string is provided, convert it into a list for uniform
        # processing:
        if isinstance(modes, str):
            modes = [modes]

        # Validate that all entries in the list are strings:
        if not all(isinstance(mode, str) for mode in modes):
            raise TypeError("All mode entries must be strings.")

        # Retrieve the set of allowed mode names (lowercase):
        allowed_modes = USGSElevationScraper.supported_asset_elevation_modes(
            print_modes=False
        )

        # Normalize modes by stripping whitespace and converting to lowercase:
        normalized_modes = {mode.strip().lower() for mode in modes}

        # Identify any invalid modes that are not supported:
        invalid_modes = normalized_modes - allowed_modes

        # Warn the user about any unsupported modes that will be ignored:
        if invalid_modes:
            print(
                'Warning: The following modes are unsupported and will be '
                f"ignored: {', '.join(invalid_modes)}"
            )

        # Return only the valid normalized modes that are supported:
        return normalized_modes & allowed_modes

    def _process_centroid_elevations(self) -> None:
        """
        Compute and assign centroid-based elevation values for all assets.

        For each asset in the current inventory, this method calculates the
        centroid coordinates, fetches the elevation from the USGS Elevation
        Point Service, and stores the converted value under the key defined by
        `ELEVATION_FEATURE_NAMES['centroid']` in the asset's `features`
        dictionary.

        Invalid centroids (with None coordinates) are skipped.
        Elevation values are converted from feet to the configured target unit.

        Returns:
            None. Modifies the AssetInventory in place.
        """
        # Get feature name:
        feature_name = ELEVATION_FEATURE_NAMES['centroid']

        # Get list of assets:
        assets = list(self.inventory.inventory.values())

        # Extract centroid coordinates and track valid indices:
        centroids = [asset.get_centroid()[0] for asset in assets]
        valid_indices = [i for i, (x, y) in enumerate(centroids)
                         if x is not None and y is not None]
        valid_coords = [centroids[i] for i in valid_indices]

        # Fetch elevation data for valid centroids:
        elevation_results = self.fetch_all_elevations(valid_coords)
        target_length_unit = self.units['length']

        # Assign elevations back to the corresponding assets (in-place):
        for i, idx in enumerate(valid_indices):
            elevation = elevation_results[i].get('elevation')
            if elevation is not None:
                converted = UnitConverter.convert_length(
                    elevation, DEFAULT_ELEVATION_UNIT, target_length_unit)
                assets[idx].features[feature_name] = converted

    def _compute_asset_elevation_features(
        self,
        modes: Union[str, List[str]]
    ) -> None:
        """
        Compute requested elevation features for every asset.

        For every asset, elevation data is fetched for all of its coordinates,
        then elevation metrics are computed based on the requested modes.
        Supported modes are:
        - 'all': list of elevation values per coordinate
        - 'average': arithmetic mean of valid elevations
        - 'min': minimum elevation
        - 'max': maximum elevation
        - 'median': median elevation
        - 'stddev': standard deviation of elevations

        Elevation values are converted to the specified unit and stored in each
        asset's `features` dictionary under the appropriate key.

        Args:
            modes (str or List[str]):
                One or more of ['all', 'average', 'min', 'max', 'median',
                                'stddev'].

        Returns:
            None. Modifies AssetInventory in place by updating asset features.
        """
        # Normalize and validate input modes:
        mode_set = self._normalize_and_filter_modes(modes)
        if not mode_set:
            print('No valid modes specified; skipping elevation processing.')
            return

        assets = list(self.inventory.inventory.values())
        coord_list = []
        asset_point_map = defaultdict(list)

        # Collect all coordinates from assets and track their indices per
        # asset:
        for asset in assets:
            for coord in asset.coordinates:
                x, y = coord
                asset_point_map[asset].append(len(coord_list))
                coord_list.append((x, y))

        # If no coordinates found, exit early:
        if not coord_list:
            print('No coordinates found in assets; skipping elevation '
                  'processing.')
            return

        # Fetch elevation data once for all collected coordinates:
        point_results = self.fetch_all_elevations(coord_list)

        # Process results per asset and assign features based on requested
        # modes:
        for asset in assets:
            indices = asset_point_map[asset]
            elevations = []
            target_length_unit = self.units['length']

            # Collect and convert elevations for each coordinate of the asset:
            for i in indices:
                result = point_results[i]
                elev = result['elevation']
                if elev is not None:
                    elev = UnitConverter.convert_length(
                        elev,
                        DEFAULT_ELEVATION_UNIT,
                        target_length_unit
                    )
                elevations.append(elev)

            valid_elevations = [e for e in elevations if e is not None]

            # Assign features based on requested modes:
            if 'all' in mode_set:
                asset.features[ELEVATION_FEATURE_NAMES['all']] = elevations

            if 'average' in mode_set:
                avg = sum(valid_elevations) / len(valid_elevations) \
                    if valid_elevations else None
                asset.features[ELEVATION_FEATURE_NAMES['average']] = avg

            if 'min' in mode_set:
                min_val = min(valid_elevations) if valid_elevations else None
                asset.features[ELEVATION_FEATURE_NAMES['min']] = min_val

            if 'max' in mode_set:
                max_val = max(valid_elevations) if valid_elevations else None
                asset.features[ELEVATION_FEATURE_NAMES['max']] = max_val

            if 'median' in mode_set:
                median_val = statistics.median(
                    valid_elevations) if valid_elevations else None
                asset.features[ELEVATION_FEATURE_NAMES['median']] = median_val

            if 'stddev' in mode_set:
                if len(valid_elevations) >= 2:
                    stddev_val = statistics.stdev(valid_elevations)
                elif len(valid_elevations) == 1:
                    stddev_val = 0.0
                else:
                    stddev_val = None
                asset.features[ELEVATION_FEATURE_NAMES['stddev']] = stddev_val
