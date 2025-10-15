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
# 10-14-2025

"""
This is a utility class for datasets created by the RAPID facility at UW.

.. autosummary::

      RAPIDUtils
"""

from pathlib import Path
from typing import Tuple, Union, List, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from tqdm import tqdm

from rasterio import open as rasterio_open
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds, transform_geom

import brails.types.image_set as brails_image_set
if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory


class RAPIDUtils:
    """
    Utility class for datasets created by the RAPID facility at UW.

    This class provides methods to extract asset-specific imagery from
    datasets collected by the RAPID facility at UW.

    Please use the following statement to import the :class:`RAPIDUtils` class.

    .. code-block:: python

        from brails.utils import RAPIDUtils
    """

    def __init__(self, dataset_path: Union[str, Path]):
        """
        Initialize the object with a dataset path and detect its type.

        Args:
            dataset_path (Union[str, Path]):
                Path to the dataset file or folder.
        Raises:
            ValueError: If the provided path is not a valid
        """
        dataset_path = Path(dataset_path)

        # Check if the path points to a TIFF file
        if (dataset_path.suffix.lower() in ['.tif', '.tiff'] and
                dataset_path.is_file()):
            self.dataset_path = dataset_path
            print(
                'Detected orthomosaic data.\n'
                'Applicable methods for this data type are: '
                "'extract_aerial_imagery', 'get_mosaic_bbox_wgs84'."
            )
        else:
            raise ValueError(
                f"Provided dataset path '{dataset_path.name}' is not valid."
            )

        self.dataset_extent = self.get_mosaic_bbox_wgs84()

    def extract_aerial_imagery(
        self,
        asset_inventory: "AssetInventory",
        save_directory: str,
        max_missing_data_ratio: float = 0.2,
        overlay_asset_outline: bool = False
    ) -> brails_image_set.ImageSet:
        """
        Extract aerial imagery patches for each asset from a raster dataset.

        Args:
            asset_inventory (AssetInventory):
                Inventory object containing asset geometries and metadata.
            save_directory (str):
                Directory where extracted images will be saved.
            max_missing_data_ratio (float, optional):
                Maximum allowed proportion of missing data (0-1). Default is
                ``0.2``.
            overlay_asset_outline (bool, optional):
                If ``True``, overlay the asset polygon on the saved image.
                Default is ``False``.

        Returns:
            ImageSet:
                Object containing metadata and paths of saved images.

        Example:
            The following example assumes that a valid orthomosaic raster file
            exists at ``raster_test_data.tif``. Actual outputs may vary
            depending on the specific contents of the raster file.

            Step 1: Import required packages and load the raster dataset.
            Note that :class:`RAPIDUtils` automatically identifies the methods
            applicable for the loaded dataset:

            >>> from brails.utils import RAPIDUtils, Importer
            >>> rapid_utils = RAPIDUtils('raster_test_data.tif')
            Detected orthomosaic data.
            Applicable methods for this data type are:
            'extract_aerial_imagery', 'get_mosaic_bbox_wgs84'.

            Step 2: Retrieve building footprints covering the extent of the
            dataset:

            >>> importer = Importer()
            >>> region_data = {
            ...     "type": "locationPolygon",
            ...     "data": rapid_utils.dataset_extent
            ... }
            >>> region_boundary_object = importer.get_class(
            ...     'RegionBoundary'
            ... )(region_data)
            >>> osm_footprint_scraper = importer.get_class(
            ...     'OSM_FootprintScraper'
            ... )({'length': 'ft'})
            >>> scraper_inventory = osm_footprint_scraper.get_footprints(
            ...     region_boundary_object
            ... )
            Found a total of 503 building footprints in the bounding box:
            (-122.1421632630, 47.69423131358, -122.12908634292, 47.70804716657)

            Step 3: Extract building-level imagery from the orthomosaic
            dataset. Optionally overlay the building footprints to aid computer
            vision applications:

            >>> image_set = rapid_utils.extract_aerial_imagery(
            ...     scraper_inventory,
            ...     'images_raster_test/overlaid_imagery',
            ...     overlay_asset_outline=True
            ... )
            Images will be saved to: /home/bacetiner/Documents/SoftwareTesting/
            images_raster_test/overlaid_imagery
            Extracting aerial imagery...: 100%|██████████| 503/503
            [01:04<00:00,  7.79it/s]
            Extracted aerial imagery for a total of 397 assets.
        """
        base_dir_path = Path(save_directory).resolve()
        base_dir_path.mkdir(parents=True, exist_ok=True)
        print(f'\nImages will be saved to: {base_dir_path}\n')

        image_set = brails_image_set.ImageSet()
        image_set.dir_path = str(base_dir_path)

        with rasterio_open(
            self.dataset_path,
            driver='GTiff',
            num_threads='all_cpus'
        ) as dataset:
            for asset_key, asset in tqdm(
                asset_inventory.inventory.items(),
                desc='Extracting aerial imagery...'
            ):
                asset_geometry = asset.coordinates
                centroid = Polygon(asset_geometry).centroid
                image_name = f'{centroid.y:.8f}_{centroid.x:.8f}'.replace(
                    '.', '')
                image_path = base_dir_path / f'gstrt_{image_name}.jpg'

                projected_coords = [self._project_point_from_wgs84(
                    pt, dataset.crs) for pt in asset_geometry]
                projected_polygon = Polygon(projected_coords)
                buffered_bounds = self._get_buffered_bounds(projected_polygon)

                image_array = self._read_image_window(dataset, buffered_bounds)

                if not self._is_image_valid(
                    image_array,
                    max_missing_data_ratio
                ):
                    continue

                image = np.moveaxis(image_array[:3], 0, -1)  # CHW to HWC
                pil_image = Image.fromarray(image)

                if overlay_asset_outline:
                    projected_coords = list(projected_polygon.exterior.coords)
                    pixel_coords = self._get_polygon_pixel_coords_(
                        dataset, projected_coords, buffered_bounds)
                    draw = ImageDraw.Draw(pil_image)
                    draw.line(pixel_coords +
                              [pixel_coords[0]], fill='red', width=6)

                pil_image.save(image_path)
                img = brails_image_set.Image(image_path.name)
                image_set.add_image(asset_key, img)

        print(
            '\nExtracted aerial imagery for a total of '
            f'{len(image_set.images)} assets.'
        )

        return image_set

    def get_mosaic_bbox_wgs84(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of the raster dataset in WGS84 coordinates.

        This method opens the raster dataset, reads its bounding box, and
        transforms it from the dataset's native CRS to EPSG:4326 (WGS84
        latitude-longitude).

        Returns:
            Tuple[float, float, float, float]:
                The bounding box coordinates in WGS84, returned as
                ``(min_longitude, min_latitude, max_longitude, max_latitude)``.

        Example:
            The following example assumes that a valid orthomosaic raster file
            exists at ``raster_test_data.tif``. Actual outputs may vary
            depending on the specific contents of the raster file.

            >>> from brails.utils import RAPIDUtils
            >>> rapid_utils = RAPIDUtils('raster_test_data.tif')
            Detected orthomosaic data.
            Applicable methods for this data type are:
            'extract_aerial_imagery', 'get_mosaic_bbox_wgs84'.
            >>> bbox_wgs84 = rapid_utils.get_mosaic_bbox_wgs84()
            >>> print(bbox_wgs84)
            (-122.1421632630, 47.69423131358, -122.12908634292, 47.70804716657)
        """
        with rasterio_open(
            self.dataset_path,
            driver='GTiff',
            num_threads='all_cpus'
        ) as dataset:
            return transform_bounds(
                dataset.crs,
                CRS.from_epsg(4326),
                *dataset.bounds
            )

    @staticmethod
    def _project_point_from_wgs84(
        lon_lat_pt: Union[Tuple[float, float], List[float]],
        destination_crs: CRS
    ) -> Tuple[float, float]:
        """
        Convert coordinates of geographic point from WGS84 to a target CRS.

        Args:
            lon_lat_pt (Tuple[float, float] or List[float, float]):
                A point in WGS84 (EPSG:4326) coordinates (lon, lat).
            destination_crs (CRS):
                The target coordinate reference system.

        Returns:
            Tuple[float, float]:
                The transformed point coordinates in the target CRS.
        """
        if not isinstance(lon_lat_pt, (tuple, list)) or len(lon_lat_pt) != 2:
            raise ValueError(
                'lon_lat_pt must be a tuple or list of two numbers (lon, lat).'
            )
        if not all(isinstance(coord, (int, float)) for coord in lon_lat_pt):
            raise ValueError(
                'Both elements of lon_lat_pt must be numbers (int or float).'
            )
        if not isinstance(destination_crs, CRS):
            raise TypeError(
                'destination_crs must be an instance of rasterio.crs.CRS.'
            )

        feature = {"type": "Point", "coordinates": lon_lat_pt}
        projected = transform_geom(
            src_crs=CRS.from_epsg(4326),
            dst_crs=destination_crs,
            geom=feature
        )
        return tuple(projected['coordinates'])

    @staticmethod
    def _get_buffered_bounds(
        polygon: Polygon,
        buffer_ratio: float = 0.2
    ) -> Tuple[float, float, float, float]:
        """
        Compute buffered bounds for a polygon using a buffer ratio.

        Args:
            polygon (Polygon):
                The input polygon geometry.
            buffer_ratio (float, optional):
                Proportional buffer size based on max polygon extent.
                Defaults to 0.2.

        Returns:
            Tuple[float, float, float, float]:
                Buffered bounding box as (minx, miny, maxx, maxy).
        """
        bounds = polygon.bounds
        buffer_dist = max(
            abs(bounds[0] - bounds[2]),
            abs(bounds[1] - bounds[3])
        ) * buffer_ratio
        return polygon.buffer(buffer_dist).bounds

    @staticmethod
    def _read_image_window(
        dataset: DatasetReader,
        bounds: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Read a windowed portion of a raster image defined by geographic bounds.

        Args:
            dataset (DatasetReader):
                Opened raster dataset.
            bounds (Tuple[float, float, float, float]):
                Bounding box (minx, miny, maxx, maxy).

        Returns:
            np.ndarray:
                Image array read from the specified window.
        """
        row1, col1 = dataset.index(bounds[0], bounds[1])
        row2, col2 = dataset.index(bounds[2], bounds[3])
        window = Window(
            min(col1, col2),
            min(row1, row2),
            abs(col2 - col1),
            abs(row2 - row1)
        )
        return dataset.read(window=window)

    @staticmethod
    def _is_image_valid(
        image_array: np.ndarray,
        max_missing_data_ratio: float
    ) -> bool:
        """
        Determine if the image contains acceptable levels of missing data.

        Args:
            image_array (np.ndarray):
                Input image array.
            max_missing_data_ratio (float):
                Maximum allowed ratio of zero-valued pixels.

        Returns:
            bool:
                True if valid, False if missing data exceeds threshold.
        """
        total_pixels = image_array.size
        zero_pixels = np.count_nonzero(image_array == 0)
        zero_ratio = (zero_pixels / total_pixels) if total_pixels else 1
        return zero_ratio < max_missing_data_ratio

    @staticmethod
    def _get_polygon_pixel_coords_(
        dataset: DatasetReader,
        polygon_coords: List[Tuple[float, float]],
        buffered_bounds: Tuple[float, float, float, float]
    ) -> List[Tuple[int, int]]:
        """
        Convert geographic polygon coordinates to image pixel coordinates.

        Args:
            dataset (DatasetReader): Raster dataset.
            polygon_coords (List[Tuple[float, float]]):
                List of polygon (x, y) coordinates.
            buffered_bounds (Tuple[float, float, float, float]):
                Bounds of the buffered window.

        Returns:
            List[Tuple[int, int]]:
                List of (col, row) pixel coordinates relative to the window.
        """
        pixel_coords_global = [dataset.index(x, y) for x, y in polygon_coords]
        window = from_bounds(*buffered_bounds, transform=dataset.transform)
        row_off, col_off = int(window.row_off), int(window.col_off)

        return [(col - col_off, row - row_off) for row, col in
                pixel_coords_global]
