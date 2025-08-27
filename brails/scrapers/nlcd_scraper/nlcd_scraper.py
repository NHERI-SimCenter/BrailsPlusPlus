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
# 08-26-2025

"""
Module for scraping and processing USGS NLCD land cover data.

.. autosummary::

    NLCDScraper
"""

from typing import List, Tuple, Optional, Dict, Any
import requests
import rasterio
from rasterio.warp import transform
from rasterio.merge import merge
from rasterio.io import MemoryFile
from tqdm import tqdm
import numpy as np
from brails.types.asset_inventory import AssetInventory
from brails.scrapers.asset_data_augmenter import AssetDataAugmenter

# -------------------------------
# NLCD class dictionaries
# -------------------------------
# Full list of classes as listed by USGS:
# Open Water (11)
# Perennial Ice/Snow (12)
# Developed, Open Space (21)
# Developed, Low Intensity (22)
# Developed, Medium Intensity (23)
# Developed, High Intensity (24)
# Barren Land (Rock/Sand/Clay) (31)
# Unconsolidated Shore (32)
# Deciduous Forest (41)
# Evergreen Forest (42)
# Mixed Forest (43)
# Dwarf Scrub (51)
# Shrub/Scrub (52)
# Grasslands/Herbaceous (71)
# Sedge/Herbaceous (72)
# Lichens (73)
# Moss (74)
# Pasture/Hay (81)
# Cultivated Crops (82)
# Woody Wetlands (90)
# Emergent Herbaceous Wetlands (95)


NLCD_CLASSES_PIXEL = {
    'conus': {
        1: "Open Water",
        2: "Perennial Ice/Snow",
        3: "Developed, Open Space",
        4: "Developed, Low Intensity",
        5: "Developed, Medium Intensity",
        6: "Developed, High Intensity",
        7: "Barren Land (Rock/Sand/Clay)",
        8: "Deciduous Forest",
        9: "Evergreen Forest",
        10: "Mixed Forest",
        11: "Shrub/Scrub",
        12: "Grasslands/Herbaceous",
        13: "Pasture/Hay",
        14: "Cultivated Crops",
        15: "Woody Wetlands",
        16: "Emergent Herbaceous Wetlands"
    },
    'ak': {
        1: "Open Water",
        2: "Perennial Ice/Snow",
        3: "Developed, Open Space",
        4: "Developed, Low Intensity",
        5: "Developed, Medium Intensity",
        6: "Developed, High Intensity",
        7: "Barren Land (Rock/Sand/Clay)",
        8: "Unconsolidated Shore",
        9: "Deciduous Forest",
        10: "Evergreen Forest",
        11: "Mixed Forest",
        12: "Dwarf Scrub",
        13: "Shrub/Scrub",
        14: "Grasslands/Herbaceous",
        15: "Sedge/Herbaceous",
        16: "Lichens",
        17: "Moss",
        18: "Pasture/Hay",
        19: "Cultivated Crops",
        20: "Woody Wetlands",
        21: "Emergent Herbaceous Wetlands"
    }
}


# -------------------------------
# Tile helpers
# -------------------------------
ZOOM_VALUE = 12
TILE_SIZE = 256
ORIGIN_SHIFT = 20037508.342789244
INITIAL_RESOLUTION = 156543.03392804097  # meters/pixel at zoom 0


class NLCDScraper(AssetDataAugmenter):
    """
    Scraper for USGS NLCD land cover data.

    This class downloads tiled NLCD rasters from USGS WMS services, mosaics
    them in memory, and assigns NLCD land cover classes to each asset in an
    asset inventory based on its geographic location.
    """

    def __init__(self, input_dict: Dict[str, Any] = None):
        """
        Initialize an instance of the scraper.

        Args:
            input_dict (Dict[str, Any], optional):
                Optional parameters for the scraper. Supported key:
                    - 'raster_output' (str):
                        Full file path to save the mosaic raster of the
                        scraped land cover data. If not provided or empty, the
                        raster is not saved. Defaults to ''.
        """
        self.inventory = AssetInventory()
        input_dict = input_dict or {}
        self.raster_output = input_dict.get('raster_output', '')

    def populate_feature(
            self,
            input_inventory: AssetInventory
    ) -> AssetInventory:
        """
        Scrape NLCD land cover values for each asset in the inventory.

        Each asset’s centroid is sampled against an in-memory NLCD raster
        mosaic. The resulting fine-grained NLCD class is stored in the asset’s
        ``'land_cover'`` field.

        Args:
            input_inventory (AssetInventory):
                The inventory of assets with location information.

        Returns:
            AssetInventory:
                The same inventory with each asset updated to include a
                ``'land_cover'`` feature.
        """
        self.inventory = input_inventory

        # Get bounding box and retrieve tiles:
        bbox4326 = self.inventory.get_extent().bounds
        tiles, class_dict = self._gwc_wms_tiles(bbox4326, ZOOM_VALUE)
        mosaic_array, mosaic_meta = self._download_and_mosaic(
            tiles,
            self.raster_output
        )

        # Collect all centroids:
        coords = np.array([asset.get_centroid()[0]
                           for asset in self.inventory.inventory.values()])

        # Transform lon/lat to mosaic CRS (EPSG:3857):
        lons, lats = coords[:, 0], coords[:, 1]
        xs, ys = rasterio.warp.transform(
            'EPSG:4326', mosaic_meta['crs'], lons, lats)

        # Stack coordinates as list of (x, y) for rasterio.sample:
        sample_coords = list(zip(xs, ys))

        # Sample raster values at asset centroids:
        with MemoryFile() as memfile:
            with memfile.open(**mosaic_meta) as dataset:
                dataset.write(mosaic_array)
                # Extract the first (and only) band value from each sample:
                sampled_values = [v[0] for v in dataset.sample(sample_coords)]

        # Assign land cover classes to assets:
        assets_list = list(self.inventory.inventory.values())
        for i in tqdm(
            range(len(assets_list)),
            total=len(sampled_values),
            desc='Assigning NLCD classes to each asset'
        ):
            val = sampled_values[i]
            if val in class_dict:
                assets_list[i].features['land_cover'] = class_dict[val]

        return self.inventory

    def _resolution(self, zoom: int) -> float:
        """
        Compute map resolution (meters per pixel) for a given zoom level.

        Args:
            zoom (int):
                Zoom level (0 = whole world, higher = finer resolution).

        Returns:
            float: Resolution in meters per pixel.
        """
        return INITIAL_RESOLUTION / (2 ** zoom)

    def _latlon_to_meters(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        Convert latitude/longitude in EPSG:4326 to Web Mercator meters.

        Args:
            lon (float): Longitude in decimal degrees.
            lat (float): Latitude in decimal degrees.

        Returns:
            tuple[float, float]: (x, y) coordinates in EPSG:3857 meters.
        """
        x, y = transform('EPSG:4326', 'EPSG:3857', [lon], [lat])
        return x[0], y[0]

    def _meters_to_tile(
        self,
        x: float,
        y: float, zoom: int
    ) -> Tuple[int, int]:
        """
        Convert coordinates in EPSG:3857 to a tile index at a given zoom.

        Args:
            x (float): X coordinate in meters (EPSG:3857).
            y (float): Y coordinate in meters (EPSG:3857).
            zoom (int): Zoom level.

        Returns:
            tuple[int, int]: ``(tx, ty)`` tile indices.
        """
        res = self._resolution(zoom)
        tx = int((x + ORIGIN_SHIFT) / (res * TILE_SIZE))
        ty = int((y + ORIGIN_SHIFT) / (res * TILE_SIZE))
        return tx, ty

    def _tile_bounds(
        self,
        tx: int,
        ty: int,
        zoom: int
    ) -> Tuple[float, float, float, float]:
        """
        Compute bounding box of a tile in Web Mercator meters.

        Args:
            tx (int): Tile index in x direction.
            ty (int): Tile index in y direction.
            zoom (int): Zoom level.

        Returns:
            tuple[float, float, float, float]:
                (minx, miny, maxx, maxy) in EPSG:3857 meters.
        """
        res = self._resolution(zoom)
        minx = tx * TILE_SIZE * res - ORIGIN_SHIFT
        maxx = (tx + 1) * TILE_SIZE * res - ORIGIN_SHIFT
        miny = ty * TILE_SIZE * res - ORIGIN_SHIFT
        maxy = (ty + 1) * TILE_SIZE * res - ORIGIN_SHIFT
        return (minx, miny, maxx, maxy)

    def _get_base_url_for_bbox(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Return the USGS NLCD WMS base URL & class dictionary for a given bbox.

        Args:
            bbox (tuple[float, float, float, float]): Bounding box in EPSG:4326
                format as (minLon, minLat, maxLon, maxLat).

        Returns:
            tuple[str or None, dict or None]:
                - Base WMS URL (str) for CONUS or Alaska, or ``None`` if
                  ``bbox`` is outside both regions.
                - Corresponding NLCD class dictionary, or ``None`` if not
                  found.
        """
        minLon, minLat, maxLon, maxLat = bbox

        CONUS = (-125, 24, -66.5, 49.5)
        ALASKA = (-170, 51, -129, 72)

        if (minLon >= CONUS[0] and maxLon <= CONUS[2] and
                minLat >= CONUS[1] and maxLat <= CONUS[3]):
            return (
                'https://dmsdata.cr.usgs.gov/geoserver/gwc/service/wms?'
                'REQUEST=GetMap&SERVICE=WMS&VERSION=1.3.0&FORMAT=image/png'
                '&STYLES=&TRANSPARENT=true'
                '&LAYERS='
                'mrlc_Land-Cover_conus_year_data:Land-Cover_conus_year_data'
                '&TILED=true&TIME=2024-01-01'
            ), NLCD_CLASSES_PIXEL['conus']

        elif (minLon >= ALASKA[0] and maxLon <= ALASKA[2] and
              minLat >= ALASKA[1] and maxLat <= ALASKA[3]):
            return (
                'https://www.mrlc.gov/geoserver/wms?'
                'REQUEST=GetMap&SERVICE=WMS&VERSION=1.3.0&FORMAT=image%2Fpng'
                '&STYLES=&TRANSPARENT=true'
                '&LAYERS='
                'mrlc_display%3ANLCD_2016_Land_Cover_AK'
                '&TILED=true'
                '&SRS=EPSG%3A3857'
                '&jsonLayerId=allalaskaNlcd2016LandCover'
                '&WIDTH=256&HEIGHT=256&CRS=EPSG%3A3857'
            ), NLCD_CLASSES_PIXEL['ak']

        else:
            print('Warning: Bounding box is outside CONUS or Alaska.')
            return None, None

    def _gwc_wms_tiles(
        self,
        bbox4326: Tuple[float, float, float, float],
        zoom: int
    ) -> Tuple[
        List[Tuple[str, Tuple[float, float, float, float]]],
        Optional[Dict[int, str]]
    ]:
        """
        Generate WMS tile URLs and bounding boxes for a bounding box.

        Args:
            bbox4326 (Tuple[float, float, float, float]):
                The bounding box in EPSG:4326 coordinates
                (minLon, minLat, maxLon, maxLat).
            zoom (int):
                Zoom level to determine tile resolution and number of tiles.

        Returns:
            Tuple containing:
                - List of tuples: each with (tile URL, tile bounding box in
                  EPSG:3857)
                - Optional dictionary mapping pixel values to NLCD class names
                  (returns None if bounding box is outside supported areas)
        """
        # Get the base WMS URL and corresponding NLCD class dictionary
        # depending on whether the bounding box falls in CONUS or Alaska:
        base_url, class_dict = self._get_base_url_for_bbox(bbox4326)
        if not base_url:
            # Return empty list and None if bounding box is outside supported
            # areas:
            return [], None

        # Unpack bounding box coordinates:
        minLon, minLat, maxLon, maxLat = bbox4326

        # Convert bounding box coordinates from lon/lat to EPSG:3857 (meters):
        minx, miny = self._latlon_to_meters(minLon, minLat)
        maxx, maxy = self._latlon_to_meters(maxLon, maxLat)

        # Determine tile indices (tx, ty) for min and max coordinates:
        tx_min, ty_min = self._meters_to_tile(minx, miny, zoom)
        tx_max, ty_max = self._meters_to_tile(maxx, maxy, zoom)

        tiles = []
        # Loop through all tiles covering the bounding box:
        for tx in range(tx_min, tx_max + 1):
            for ty in range(ty_min, ty_max + 1):
                # Get the bounding box for this specific tile in EPSG:3857:
                bbox = self._tile_bounds(tx, ty, zoom)

                # Construct the full WMS tile URL with bounding box, size, and
                # CRS:
                url = (f'{base_url}'
                       f'&WIDTH=256&HEIGHT=256'
                       f'&SRS=EPSG:3857&CRS=EPSG:3857'
                       f'&BBOX={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}')

                # Append the URL and corresponding tile bounding box to the
                # list:
                tiles.append((url, bbox))

        # Return the list of tiles and the class dictionary:
        return tiles, class_dict

    def _download_and_mosaic(
        self,
        tiles: List[Tuple[str, Tuple[float, float, float, float]]],
        output_tif: str = ''
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Download WMS tiles and build an in-memory mosaic raster.

        Args:
            tiles (List[Tuple[str, Tuple[float, float, float, float]]]):
                List of tile URLs and their bounding boxes.
            output_tif (str, optional):
                Path to save the mosaic raster on disk. If empty, no file is
                saved.

        Returns:
            Tuple containing:
                - Mosaic as a numpy array (or ``None`` if no tiles)
                - Raster metadata dictionary (or ``None`` if no tiles)
        """
        # Check if there are any tiles to download:
        if not tiles:
            print(
                'No WMS tiles generated for the given bounding box. '
                'Skipping download.'
            )
            return None, None

        # Download each tile and convert to in-memory GeoTIFF:
        datasets = []
        for url, bbox in tqdm(tiles, desc='Downloading tiles'):
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            # Open the PNG response as a rasterio dataset in memory:
            with MemoryFile(resp.content) as memfile:
                with memfile.open(driver='PNG') as src:
                    # Read the raster data (single band):
                    data = src.read(1, masked=True)

                    # Copy the profile & update for GeoTIFF format and bounds:
                    profile = src.profile.copy()
                    profile.update({
                        'driver': 'GTiff',
                        'dtype': src.dtypes[0],
                        'count': src.count,
                        'height': TILE_SIZE,
                        'width': TILE_SIZE,
                        'crs': 'EPSG:3857',
                        'transform': rasterio.transform.from_bounds(
                            *bbox,
                            TILE_SIZE, TILE_SIZE
                        )
                    })

                    # Write to an in-memory GeoTIFF:
                    mem_gtiff = MemoryFile()
                    with mem_gtiff.open(**profile) as tmp_dst:
                        tmp_dst.write(data, 1)

                    # Append in-memory dataset for later merging:
                    datasets.append(mem_gtiff.open())

        # Merge all individual tiles into a single mosaic in memory:
        mosaic, out_transform = merge(datasets)

        # Copy metadata from one tile and update with mosaic info:
        out_meta = datasets[0].meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_transform,
            'crs': 'EPSG:3857'
        })

        # Optionally save to disk:
        if output_tif:
            with rasterio.open(output_tif, 'w', **out_meta) as dest:
                dest.write(mosaic)
            print(f'Mosaic saved to {output_tif}')

        # Close all in-memory tile datasets to free resources:
        for ds in datasets:
            ds.close()

        # Return the mosaic array and updated metadata:
        return mosaic, out_meta
