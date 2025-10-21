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
#
# Last updated:
# 10-14-2025

"""
This module defines GoogleSatellite class downloading Google satellite imagery.

.. autosummary::

    GoogleSatellite
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import requests
from PIL import Image
from requests.adapters import HTTPAdapter, Retry
from shapely.geometry import Polygon
from tqdm import tqdm

from brails.types.asset_inventory import AssetInventory
import brails.types.image_set as brails_image_set
from brails.scrapers.image_scraper import ImageScraper


# Constants:
GOOGLE_TILE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
ZOOM_LEVEL = 20
TILE_SIZE = 256
RESIZED_IMAGE_SIZE = (640, 640)

FOOTPRINT_BUFFER_RATIO = 0.1

REQUESTS_RETRY_STRATEGY = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504],
)


class GoogleSatellite(ImageScraper):
    """
    A class for downloading satellite imagery from Google tilemaps.

    This class is a subclass to the `ImageScraper` and provides functionality
    to obtain satellite images for assets defined in an AssetInventory. The
    images are retrieved based on the coordinates of the assets and saved to a
    specified directory.

    Methods:
        get_images(inventory: AssetInventory, save_directory: str) -> ImageSet:
            Retrieves satellite images for the assets in the given inventory
            and saves them to the specified save_directory.
    """

    def get_images(
        self,
        inventory: AssetInventory,
        save_directory: str
    ) -> brails_image_set.ImageSet:
        """
        Get satellite images of buildings given footprints in AssetInventory.

        Args:
              inventory (AssetInventory):
                  AssetInventory for which the images will be retrieved
            save_directory (str):
                Path to the folder where the retrieved images will be saved

        Returns:
              ImageSet:
                  An ImageSet for the assets in the inventory

        Raises:
            ValueError:
                If the provided inventory is not an instance of AssetInventory
        """
        # Validate inputs:
        if not isinstance(inventory, AssetInventory):
            raise ValueError('Invalid AssetInventory provided.')

        # Ensure consistency in dir_path, i.e remove ending / if given:
        dir_path = Path(save_directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f'Images will be saved to: {dir_path.resolve()}')

        # Create the footprints from the items in AssetInventory
        # Keep the asset keys in a list for later use:
        image_set = brails_image_set.ImageSet()
        image_set.dir_path = str(dir_path)

        asset_footprints = []
        asset_keys = []
        for key, asset in inventory.inventory.items():
            asset_footprints.append(asset.coordinates)
            asset_keys.append(key)

        # Get the images:
        satellite_images = self._download_images(asset_footprints,
                                                 dir_path)

        for index, image_path in enumerate(satellite_images):
            if image_path.exists():
                img = brails_image_set.Image(image_path.name)
                image_set.add_image(asset_keys[index], img)
            else:
                print(f'Image for asset {asset_keys[index]} could not be'
                      'downloaded.')

        return image_set

    def _download_images(
        self,
        footprints: List[List[Tuple[float, float]]],
        save_dir: Path
    ) -> List[Path]:
        """
        Download satellite images for a list of footprints.

        Args:
            footprints (List[List[Tuple[float, float]]]):
                List of asset footprints.
            save_dir (Path):
                Directory to save images.

        Returns:
            List[Path]:
                List of paths to the downloaded images.
        """
        # Compute building footprints, parse satellite image names, and
        # save them in the class object. Also, create the list of inputs
        # required for downloading satellite images in parallel:
        satellite_image_paths = []
        inps = []

        for footprint in footprints:
            centroid = Polygon(footprint).centroid
            image_name = str(round(centroid.y, 8)) + str(round(centroid.x, 8))
            image_name.replace('.', '')
            image_path = save_dir / f'imsat_{image_name}.jpg'
            satellite_image_paths.append(image_path)
            inps.append((footprint, image_path))

        # Download satellite images corresponding to each building
        # footprint:
        pbar = tqdm(total=len(footprints),
                    desc='Obtaining satellite imagery')
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._download_satellite_image,
                                footprint, path): footprint
                for footprint, path in inps
            }

            for future in as_completed(futures):
                footprint = futures[future]
                pbar.update(n=1)
                try:
                    future.result()
                except Exception as exc:
                    print(f'Error downloading image for footprint {footprint}:'
                          f' {exc}')

        return satellite_image_paths

    def _download_satellite_image(
        self,
        footprint: List[Tuple[float, float]],
        impath: Path
    ):
        """
        Download and process the satellite image for a single footprint.

        Args:
            footprint (List[Tuple[float, float]]):
                Asset footprint coordinates.
            impath (Path):
                Path to save the processed image.
        """
        bbox_buffered = self._buffer_footprint(footprint)
        x_list, y_list = self._determine_tile_coords(bbox_buffered)

        # Get the tiles for the satellite image:
        tiles, offsets, imbnds = self._fetch_tiles(x_list, y_list)

        # Combine tiles into a single image using the calculated
        # offsets:
        combined_image = self._combine_tiles(tiles,
                                             (len(x_list), len(y_list)),
                                             offsets)

        # Crop combined image around the footprint of the building and
        # pad the resulting image image in horizontal or vertical
        # directions to make it square:
        cropped_padded_image = self._crop_and_pad_image(combined_image,
                                                        bbox_buffered,
                                                        imbnds)

        resized_image = cropped_padded_image.resize(RESIZED_IMAGE_SIZE)
        resized_image.save(impath)
        print(f'Saved image to {impath}')

    def _buffer_footprint(
        self,
        footprint: List[Tuple[float, float]]
    ) -> Tuple[List[float], List[float]]:
        """
        Buffer the footprint to account for inaccuracies.

        Args:
            footprint (List[Tuple[float, float]]):
                Original footprint

        Returns:
            Tuple[List[float], List[float]]:
                Buffered bounding box coordinates
        """
        lon, lat = zip(*footprint)
        minlon, maxlon = min(lon), max(lon)
        minlat, maxlat = min(lat), max(lat)

        londiff = maxlon - minlon
        latdiff = maxlat - minlat

        minlon_buff = minlon - londiff * FOOTPRINT_BUFFER_RATIO
        maxlon_buff = maxlon + londiff * FOOTPRINT_BUFFER_RATIO
        minlat_buff = minlat - latdiff * FOOTPRINT_BUFFER_RATIO
        maxlat_buff = maxlat + latdiff * FOOTPRINT_BUFFER_RATIO

        return ([minlon_buff, minlon_buff, maxlon_buff, maxlon_buff],
                [minlat_buff, maxlat_buff, maxlat_buff, minlat_buff])

    def _determine_tile_coords(
        self,
        bbox_buffered: Tuple[List[float], List[float]]
    ) -> Tuple[List[int], List[int]]:
        """
        Determine tile x,y coordinates containing the buffered bounding box.

        Args:
            bbox_buffered (Tuple[List[float], List[float]]):
                Buffered bounding box.

        Returns:
            Tuple[List[int], List[int]]:
                Lists of x and y tile coordinates
        """
        x_coords, y_coords = [], []
        for lon, lat in zip(bbox_buffered[0], bbox_buffered[1]):
            x, y = self._deg2num(lat, lon, ZOOM_LEVEL)
            x_coords.append(x)
            y_coords.append(y)

        x_list = list(range(min(x_coords), max(x_coords) + 1))
        y_list = list(range(min(y_coords), max(y_coords) + 1))
        return x_list, y_list

    @staticmethod
    def _deg2num(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """
        Convert latitude and longitude to tile numbers.

        Args:
            lat (float):
                Latitude in degrees.
            lon (float):
                Longitude in degrees.
            zoom (int):
                Zoom level.

        Returns:
            Tuple[int, int]:
                Tile x and y numbers.
        """
        lat_rad = math.radians(lat)
        n = 2 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile

    def _fetch_tiles(
        self,
        x_list: List[int], y_list: List[int]
    ) -> Tuple[List[Image.Image], List[Tuple[int, int]], List[float]]:
        """
        Fetch all tiles for the given x and y coordinates.

        Args:
            x_list (List[int]):
                List of x tile coordinates.
            y_list (List[int]):
                List of y tile coordinates.

        Returns:
            Tuple[List[Image.Image], List[Tuple[int, int]], List[float]]:
                List of tile images, their offsets, and image bounds.
        """
        # Get the tiles for the satellite image:
        tiles, offsets, imbnds = [], [], []
        ntiles = (len(x_list), len(y_list))

        session = requests.Session()
        session.mount("https://",
                      HTTPAdapter(max_retries=REQUESTS_RETRY_STRATEGY))
        for y_idx, ycoord in enumerate(y_list):
            for x_idx, xcoord in enumerate(x_list):
                url = GOOGLE_TILE_URL.format(x=xcoord, y=ycoord, z=ZOOM_LEVEL)

                # Download tile using the defined retry strategy:
                response = session.get(url)
                response.raise_for_status()

                # Save downloaded tile as a PIL image in tiles and calculate
                # tile offsets and bounds:
                tile_image = Image.open(BytesIO(response.content))
                tiles.append(tile_image)
                offsets.append((x_idx * TILE_SIZE, y_idx * TILE_SIZE))
                tile_bounds = self._tile_bbox(ZOOM_LEVEL, xcoord, ycoord)
                imbnds = self._update_image_bounds(imbnds,
                                                   tile_bounds,
                                                   ntiles,
                                                   x_idx,
                                                   y_idx)

        return tiles, offsets, imbnds

    @staticmethod
    def _tile_bbox(zoom: int, x_coord: int, y_coord: int) -> List[float]:
        """
        Get the bounding box of a tile.

        Args:
            zoom (int):
                Zoom level
            x_coord (int):
                Tile x number
            y_coord (int):
                Tile y number

        Returns:
            List[float]:
                Bounding box coordinates stored in [south, north, west, east]
                order
        """
        return [
            GoogleSatellite._tile_lat(y_coord, zoom),
            GoogleSatellite._tile_lat(y_coord + 1, zoom),
            GoogleSatellite._tile_lon(x_coord, zoom),
            GoogleSatellite._tile_lon(x_coord + 1, zoom),
        ]

    @staticmethod
    def _tile_lat(y_coord: int, z_coord: int) -> float:
        """
        Calculate latitude from tile y number.

        Args:
            y_coord (int):
                Tile y number
            z_coord (int):
                Zoom level

        Returns:
            float:
                Latitude in degrees
        """
        n = math.pi - (2.0 * math.pi * y_coord) / (2 ** z_coord)
        return math.degrees(math.atan(math.sinh(n)))

    @staticmethod
    def _tile_lon(xcoord: int, zcoord: int) -> float:
        """
        Calculate longitude from tile x number.

        Args:
            xcoord (int):
                Tile x number
            zcoord (int):
                Zoom level

        Returns:
            float:
                Longitude in degrees
        """
        return xcoord / (2 ** zcoord) * 360.0 - 180.0

    @staticmethod
    def _update_image_bounds(imbnds: List[float],
                             tilebnds: List[float],
                             ntiles: Tuple[int, int],
                             xind: int,
                             yind: int) -> List[float]:
        """
        Update image bounds based on tile bounds.

        Args:
            imbnds (List[float]):
                Current image bounds
            tilebnds (List[float]):
                Bounds of the current tile
            ntiles (Tuple[int, int]):
                Number of tiles in x and y directions
            xind(int):
                Current tile x index
            yind (int):
                Current tile y index

        Returns:
            List[float]:
                Updated image bounds
        """
        # If the number of tiles both in x and y directions are
        # greater than 1:
        if ntiles[0] > 1 and ntiles[1] > 1:
            if xind == 0 and yind == 0:
                imbnds.append(tilebnds[2])
                imbnds.append(tilebnds[0])
            elif xind == ntiles[0] - 1 and yind == 0:
                imbnds.append(tilebnds[3])
            elif xind == 0 and yind == ntiles[1] - 1:
                imbnds.append(tilebnds[1])
        # If the total number of tiles is 1:
        elif ntiles[0] == 1 and ntiles[1] == 1:
            imbnds = [tilebnds[2], tilebnds[0],
                      tilebnds[3], tilebnds[1]]
        # If the total number of tiles is 1 in x-direction and
        # greater than 1 in y-direction:
        elif ntiles[0] == 1:
            if yind == 0:
                imbnds.append(tilebnds[2])
                imbnds.append(tilebnds[0])
                imbnds.append(tilebnds[3])
            elif yind == ntiles[1] - 1:
                imbnds.append(tilebnds[1])
        # If the total number of tiles is greater than 1 in
        # x-direction and 1 in y-direction:
        elif ntiles[1] == 1:
            if xind == 0:
                imbnds.append(tilebnds[2])
                imbnds.append(tilebnds[0])
            elif xind == ntiles[0] - 1:
                imbnds.append(tilebnds[3])
                imbnds.append(tilebnds[1])
        return imbnds

    @staticmethod
    def _combine_tiles(
        tiles: List[Image.Image],
        ntiles: Tuple[int, int],
        offsets: List[Tuple[int, int]],
    ) -> Image.Image:
        """
        Combine individual tiles into a single image.

        Args:
            tiles (List[Image.Image]):
                List of tile images
            ntiles (Tuple[int, int]):
                Number of tiles in x and y directions
            offsets (List[Tuple[int, int]]):
                Offsets for pasting tiles

        Returns:
            Image.Image:
                Combined image
        """
        combined_image = Image.new('RGB',
                                   (TILE_SIZE * ntiles[0],
                                    TILE_SIZE * ntiles[1]))
        for ind, image in enumerate(tiles):
            combined_image.paste(image, offsets[ind])
        return combined_image

    def _crop_and_pad_image(
        self,
        combined_im: Image.Image,
        bbox_buffered: Tuple[List[float], List[float]],
        imbnds: List[float]
    ) -> Image.Image:
        """
        Crop and pad the combined image around the footprint.

        Args:
            combined_im (Image.Image):
                The combined satellite image
            bbox_buffered (Tuple[List[float], List[float]]):
                Buffered bounding box
            imbnds (List[float]):
                Image bounds

        Returns:
            Image.Image:
                Cropped and padded image
        """
        # Crop combined image around the footprint of the building:
        im_width, im_height = combined_im.size

        lonrange = imbnds[2] - imbnds[0]
        latrange = imbnds[3] - imbnds[1]

        left = math.floor(
            (bbox_buffered[0][0] - imbnds[0]) / lonrange * im_width
        )
        right = math.ceil(
            (bbox_buffered[0][-1] - imbnds[0]) / lonrange * im_width
        )
        bottom = math.ceil(
            (bbox_buffered[1][0] - imbnds[1]) / latrange * im_height
        )
        top = math.floor(
            (bbox_buffered[1][1] - imbnds[1]) / latrange * im_height
        )

        cropped_im = combined_im.crop((left, top, right, bottom))

        # Pad the image in horizontal or vertical directions to make it
        # square:
        (newdim, indmax, mindim, _) = self._maxmin_and_ind(cropped_im.size)
        padded_im = Image.new('RGB', (newdim, newdim))
        buffer = round((newdim - mindim) / 2)

        if indmax == 1:
            padded_im.paste(cropped_im, (buffer, 0))
        else:
            padded_im.paste(cropped_im, (0, buffer))

        return padded_im

    @staticmethod
    def _maxmin_and_ind(size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Determine the maximum and minimum dimensions and their indices.

        Args:
            size (Tuple[int, int]):
                Width and height of the image

        Returns:
            Tuple[int, int, int, int]:
                (max_dim, index_of_max, min_dim, index_of_min)
        """
        max_dim = max(size)
        min_dim = min(size)
        return max_dim, size.index(max_dim), min_dim, size.index(min_dim)
