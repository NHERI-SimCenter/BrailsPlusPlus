# Copyright (c) 2024 The Regents of the University of California
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
# Frank McKenna
#
# Last updated:
# 06-06-2025

"""
This module defines MS_FootprintScraper class downloading Microsoft footprints.

.. autosummary::

    MS_FootprintScraper
"""

import gzip
import json
import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import Any, Dict, List, Tuple
import pandas as pd
import requests
from shapely.geometry import Polygon
from tqdm import tqdm
from brails.scrapers.footprint_scraper import FootprintScraper
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import AssetInventory

# Constants:
ZOOM_LEVEL = 9
DATASET_URL = ('https://minedbuildings.z5.web.core.windows.net/'
               'global-buildings/dataset-links.csv.')


class MS_FootprintScraper(FootprintScraper):
    """
    A scraper class for downloading building footprints from Microsoft data.

    This class is designed to retrieve building footprint data for a specified
    region using quadkeys, download the data from Microsoft's dataset, and
    provide an inventory of building footprints and their attributes, such as
    building height. It supports multiple units for building height (i.e., feet
    or meters).

    Attributes:
        length_unit (str):
            The unit of measurement for building height, either 'ft' (feet) or
            'm' (meters). The default is 'ft' (feet) if not specified.

    Methods:
        get_footprints(region: RegionBoundary) -> AssetInventory:
            Retrieves building footprints and their attributes for the
            specified region.
    """

    def __init__(self, input_data: Dict[str, Any]):
        """
        Initialize the scraper object with a length unit.

        Args:
            input (dict):
                A dictionary that may contain the 'length' unit. If 'length'
                unit is not provided, feet ('ft') is used by default.
        """
        self.length_unit = input_data.get('length', 'ft')

    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        Retrieve building footprints and associated attributes for a region.

        This method takes a `RegionBoundary` object, extracts the region's
        boundary, determines the relevant quadkeys for that region, downloads
        the building footprint data, and returns an `AssetInventory`
        containing the building footprints and other attributes (e.g.,
        building height).

        Args:
            region (RegionBoundary):
                The region of interest containing the geographical boundary.

        Returns:
            AssetInventory:
                An inventory of buildings within the specified region,
                including their footprints and attributes (such as building
                height).
        """
        # Get the boundary polygon and related info from the region:
        bpoly, queryarea_printname, _ = region.get_boundary()

        # Get quadkeys corresponding to the bounding polygon of the region:
        quadkeys = self._bbox2quadkeys(bpoly)

        # Initialize a dictionary to hold building attributes (e.g., height):
        attributes = {'buildingheight': []}

        # Download building footprints and the corresponding building heights:
        (footprints, attributes['buildingheight']) = self._download_ms_tiles(
            quadkeys, bpoly
        )

        # Print the number of footprints found for the region:
        print(f'\nFound a total of {len(footprints)} building footprints in '
              f'{queryarea_printname}')

        # Create and return an AssetInventory containing the building
        # footprints and attributes:
        return self._create_asset_inventory(footprints,
                                            attributes,
                                            self.length_unit)

    def _download_ms_tiles(
        self,
        quadkeys: List[int],
        bpoly: Polygon
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Download building footprint data for given quadkeys and polygon.

        This method downloads building footprint data corresponding to the
        provided list of quadkeys, filters the data to include only those
        footprints that intersect with the provided bounding polygon, and
        returns the footprints and building heights in the specified length
        unit (feet or meters).

        Args:
            quadkeys (list[int]):
                A list of quadkeys representing the tiles for which to download
                building data.
            bpoly (Polygon):
                A Shapely Polygon object representing the bounding polygon to
                filter the building footprints.

        Returns:
            tuple[list[list[float]], list[float]]:
                A tuple containing two lists:
                - The first list contains building footprint coordinates
                    (list of [x, y] coordinates).
                - The second list contains building heights (in feet or meters,
                    depending on the `length_unit`). Heights are `None` if
                    unavailable.
        """
        # Load the tile data:
        dftiles = pd.read_csv(DATASET_URL)

        # Define the conversion factor based on the length unit:
        conv_factor = 3.28084 if self.length_unit == "ft" else 1

        # Build a map of quadkeys:
        url_map = {}
        for quadkey in quadkeys:
            rows = dftiles[dftiles["QuadKey"] == quadkey]
            if rows.empty:
                continue
            if len(rows) > 1:
                rows.loc[:, "Size"] = rows["Size"].apply(self._parse_file_size)
                url = rows.loc[rows["Size"].idxmax()]["Url"]
            else:
                url = rows.iloc[0]["Url"]
            url_map[quadkey] = url

        # Pre-bind the bpoly and conv_factor arguments
        fetch_fn = partial(self._fetch_and_filter, bpoly=bpoly,
                           conv_factor=conv_factor)

        # Run downloads in parallel:
        footprints = []
        bldg_heights = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Map over URLs
            results = executor.map(fetch_fn, url_map.values())

            for fp_chunk, ht_chunk in tqdm(results,
                                           total=len(url_map),
                                           desc="Downloading tiles"):
                footprints.extend(fp_chunk)
                bldg_heights.extend(ht_chunk)

        return footprints, bldg_heights

    def _fetch_and_filter(
            self,
            url: str,
            bpoly: Polygon,
            conv_factor: float
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Download and filter building footprint data from an Microsoft tile URL.

        This method performs an HTTP GET request to download a GeoJSON file
        containing building footprints. It parses each feature, checks whether
        it intersects with the specified bounding polygon, and extracts the
        footprint geometry and height if applicable. Heights are converted to
        the desired unit using the provided conversion factor.

        Args:
        url (str):
            URL to the Microsoft building footprint tile (in GeoJSON format)
        bpoly (shapely.geometry.Polygon):
            Bounding polygon used to spatially filter downloaded footprints
        conv_factor (float)
            Conversion factor for building height (e.g., convert from m to ft)

        Returns:
        Tuple[List[List[float]], List[float]]:
            A tuple containing:
            - A list of building footprint coordinates (each as a list of
              [lon, lat] pairs).
            - A list of corresponding building heights (converted, or `None`
              if unavailable).

        Notes:
        - Footprints with height = -1 are treated as missing and returned with
          `None`.
        - If an error occurs during the request or parsing, the function logs
          the error and returns empty lists.
        """
        bpoly_bounds = bpoly.bounds

        try:
            resp = requests.get(url, timeout=30)

            # Check if content is gzip-encoded
            if resp.headers.get(
                    'Content-Encoding') == 'gzip' or url.endswith('.gz'):
                with gzip.GzipFile(fileobj=BytesIO(resp.content)) as gz:
                    lines = gz.read().decode('utf-8').splitlines()
            else:
                lines = resp.text.splitlines()

            data = [json.loads(line) for line in lines]
            fp, ht = [], []
            for row in data:
                coords = row["geometry"]["coordinates"][0]
                fp_poly = Polygon(coords)
                if not self._bbox_overlap(fp_poly.bounds, bpoly_bounds):
                    continue
                if fp_poly.intersects(bpoly):
                    fp.append(coords)
                    h = row["properties"]["height"]
                    ht.append(round(h * conv_factor, 1) if h != -1 else None)
            return fp, ht

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return [], []

# TODO: Refactor this method. Maybe in GeoTools?
    def _bbox_overlap(self, bounds1, bounds2) -> bool:
        """
        Check if two bounding boxes overlap.

        Args:
        bounds1 (Tuple[float, float, float, float]):
            Bounds of the first geometry (minx, miny, maxx, maxy)
        bounds2 (Tuple[float, float, float, float]):
            Bounds of the second geometry (minx, miny, maxx, maxy)

        Returns:
        bool:
            True if bounding boxes overlap, False otherwise
        """
        return not (
            bounds1[2] < bounds2[0] or  # maxx1 < minx2
            bounds1[0] > bounds2[2] or  # minx1 > maxx2
            bounds1[3] < bounds2[1] or  # maxy1 < miny2
            bounds1[1] > bounds2[3]     # miny1 > maxy2
        )

    def _bbox2quadkeys(self, bpoly: Polygon) -> List[int]:
        """
        Convert the bounding box of a polygon to a list of unique quadkeys.

        This method calculates the bounding box of the given polygon,
        determines the tile coordinates within that bounding box, and
        converts those coordinates into unique quadkeys. The resulting
        quadkeys represent the tiles that cover the area of the bounding box.

        Args:
            bpoly (Polygon):
                A Shapely polygon representing the bounding polygon for an area

        Returns:
            List[int]:
                A list of unique quadkeys representing the tiles that cover
                the bounding box.
        """
        # Get the bounds of the bounding box (min_lon, min_lat, max_lon,
        # max_lat):
        bbox = bpoly.bounds

        # Define the four corners of the bounding box as (longitude, latitude)
        # pairs:
        bbox_coords = [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ]

        # Get the tile coordinates for the bounding box at the specified zoom
        # level:
        (xtiles, ytiles) = self._determine_tile_coords(bbox_coords)

        # Convert (xtile, ytile) pairs to quadkeys:
        quadkeys = []
        for xtile in xtiles:
            for ytile in ytiles:
                quadkeys.append(self._xy2quadkey(xtile, ytile))

        # Remove duplicates by converting the list to a set and back to a list:
        quadkeys = list(set(quadkeys))

        return quadkeys

    def _determine_tile_coords(
            self,
            bbox: List[Tuple[float, float]]
    ) -> Tuple[List[int], List[int]]:
        """
        Determine the tile coordinates for a bounding box.

        This method takes a list of bounding box coordinates
        (latitude, longitude pairs), converts them to tile coordinates
        and returns the x and y tile coordinate ranges for the bounding box.

        Args:
            bbox (list[tuple[float, float]]):
                A list of tuples, where each tuple contains a pair of
                coordinates (longitude, latitude) representing the corners of
                the bounding box.

        Returns:
            tuple[List[int], List[int]]:
                A tuple containing two lists:
                - The first list contains the x tile coordinates.
                - The second list contains the y tile coordinates.
        """
        xlist = []
        ylist = []
        for vert in bbox:
            # Unpack latitude and longitude:
            (lat, lon) = (vert[1], vert[0])

            # Convert unpacked coordinates to tile coordinates at the desired
            # zoom level:
            x, y = self._deg2num(lat, lon, ZOOM_LEVEL)
            xlist.append(x)
            ylist.append(y)

            # Ensure that x and y ranges include all tiles in the bounding box:
            xlist = list(range(min(xlist), max(xlist) + 1))
            ylist = list(range(min(ylist), max(ylist) + 1))

        return xlist, ylist

    @staticmethod
    def _deg2num(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """
        Convert geographic coordinates to tile coordinates at a zoom level.

        Args:
            lat (float):
                Latitude in decimal degrees.
            lon (float):
                Longitude in decimal degrees.
            zoom (int):
                Zoom level, which determines the tile grid resolution.

        Returns:
            tuple[int, int]:
                A tuple containing the x and y tile coordinates at the given
                zoom level.
        """
        lat_rad = math.radians(lat)
        n = 2**zoom
        x_tile = int((lon + 180) / 360 * n)
        y_tile = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
        return (x_tile, y_tile)

    @staticmethod
    def _xy2quadkey(x_tile: int, y_tile: int) -> int:
        """
        Convert tile coordinates (x, y) to a quadkey.

        Quadkey is a string that uniquely identifies a tile in a tile pyramid.
        The quadkey is generated by interleaving the binary representations of
        the x and y tile coordinates.

        Args:
            x_tile (int):
                The x tile coordinate.
            y_tile (int):
                The y tile coordinate.

        Returns:
            int:
                The quadkey as an integer.
        """
        # Convert x and y tiles to binary, remove '0b' prefix:
        x_tile_binary = str(bin(x_tile))[2:]
        y_tile_binary = str(bin(y_tile))[2:]
        zpad = len(x_tile_binary) - len(y_tile_binary)

        # Pad tile binary strings:
        if zpad < 0:
            x_tile_binary = x_tile_binary.zfill(len(x_tile_binary) - zpad)
        elif zpad > 0:
            y_tile_binary = y_tile_binary.zfill(len(y_tile_binary) + zpad)

        # Interleave x and y binary strings:
        quadkeybin = "".join(i + j for i, j in zip(y_tile_binary,
                                                   x_tile_binary))
        quadkey = ""

        # Convert the interleaved binary string to a quadkey:
        for i in range(0, int(len(quadkeybin) / 2)):
            quadkey += str(int(quadkeybin[2 * i: 2 * (i + 1)], 2))

        return int(quadkey)  # Return the quadkey as an integer

# TODO: Refactor this method. Maybe in a new class called FileHandling?
    @staticmethod
    def _parse_file_size(size_string: str) -> float:
        """
        Parse a file size string into a float representing the size in bytes.

        This method interprets a size string that may contain units such as
        'GB', 'MB', 'KB', or 'B'. The method converts the string to its
        equivalent size in bytes.

        Args:
            size_string (str):
                A string representing the file size, which may include a unit
                such as 'GB', 'MB', 'KB', or 'B' (e.g., '10GB', '500MB').

        Returns:
            float:
                The size in bytes as a float. The returned value represents
                the file size converted to bytes, where 1 GB = 1e9 bytes,
                1 MB = 1e6 bytes, 1 KB = 1e3 bytes, and 1 B = 1 byte.
        """
        size_string = size_string.lower()
        if "gb" in size_string:
            multiplier = 1e9
            size_unit = "gb"
        elif "mb" in size_string:
            multiplier = 1e6
            size_unit = "mb"
        elif "kb" in size_string:
            multiplier = 1e3
            size_unit = "kb"
        else:
            multiplier = 1
            size_unit = "b"

        # Remove the unit from the string and convert to float, then multiply
        # by the appropriate factor:
        return float(size_string.replace(size_unit, '')) * multiplier
