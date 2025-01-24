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
# 12-19-2024

"""
This module defines the class object for downloading FEMA USA Structures data.

.. autosummary::

    USA_FootprintScraper
"""

import concurrent.futures
import logging
from tqdm import tqdm
from shapely.geometry import Polygon
from brails.scrapers.footprint_scraper import FootprintScraper
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import AssetInventory
from brails.utils import GeoTools
from brails.utils import ArcgisAPIServiceHelper

# Configure logging:
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_ENDPOINT = ('https://services2.arcgis.com/FiaPA4ga0iQKduv3/ArcGIS/'
                'rest/services/USA_Structures_View/FeatureServer/0/query')


class USA_FootprintScraper(FootprintScraper):
    """
    A class to generate footprint data using FEMA USA Structures building data.

    This class interacts with the FEMA USA Structures API to download
    building footprints, attributes (such as height), and additional metadata
    for a given geographic region. The class is built on top of the
    `FootprintScraper` class.

    Attributes:
        length_unit (str):
            Unit of length for building heights (default is 'ft').

    Methods:
        get_footprints(region: RegionBoundary):
            Obtains building footprints and creates an inventory for the
            specified region.
    """

    def __init__(self, input_dict: dict):
        """
        Initialize the class object.

        Args
            input_dict:
                A dictionary specifying length units; if "length" is not
                provided, "ft" is used as the default.
        """
        self.length_unit = input_dict.get('length', 'ft')

    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        Retrieve building footprints and attributes for a specified region.

        This method divides the provided region into smaller cells, if
        necessary,  and then downloads building footprint data for each cell
        using the FEMA USA Structures API. The footprints and attributes are
        returned as an AssetInventory for buildings within the region.

        Args:
            region (RegionBoundary):
                The geographic region for which building footprints and
                attributes are to be obtained.

        Returns:
            AssetInventory:
                An inventory of buildings in the region, including their
                footprints and associated attributes (e.g., height).

        Raises:
            TypeError:
                If the 'region' argument is not an instance of the BRAILS++
                'RegionBoundary' class.

        Notes:
            - The region is split into smaller cells if the bounding area
              contains more than the maximum allowed number of elements per
              cell.
            - If the `plot_cells` flag is set to `True`, the cell boundaries
              are plotted and saved as an image.
            - The method creates a polygon mesh for the region, splits it if
              needed, and downloads building data for each cell in the region.
        """
        if not isinstance(region, RegionBoundary):
            raise TypeError("The 'region' argument must be an instance of the "
                            "'RegionBoundary' class.")

        # Obtain the boundary polygon, print name, and OSM ID of the region:
        bpoly, queryarea_printname, _ = region.get_boundary()

        # Get the number of elements allowed per cell by the API:
        api_tools = ArcgisAPIServiceHelper(API_ENDPOINT)

        plot_cells = False  # Set to True for debugging purposes to plot cells

        # Split the region polygon into smaller cells (initial stage):
        print("\nMeshing the defined area...")
        preliminary_cells = api_tools.split_polygon_into_cells(bpoly)

        # If there are multiple cells, categorize and split them based on the
        # element count in each cell:
        if len(preliminary_cells) > 1:
            final_cells = []
            split_cells = preliminary_cells.copy()

            # Continue splitting cells until all cells are within the element
            # limit:
            while len(split_cells) != 0:
                cells_to_keep, split_cells = \
                    api_tools.categorize_and_split_cells(split_cells)
                final_cells += cells_to_keep
            print(f'\nMeshing complete. Split {queryarea_printname} into '
                  '{len(final_cells)} cells')

        # If only one cell, no splitting is needed:
        else:
            final_cells = preliminary_cells.copy()
            print(f'\nMeshing complete. Covered {queryarea_printname} '
                  'with a single rectangular cell')

        # If plot_cells is True, generate a visualization of the final mesh:
        if plot_cells:
            mesh_final_fout = queryarea_printname.replace(
                " ", "_") + "_Mesh_Final.png"
            GeoTools.plot_polygon_cells(bpoly, final_cells, mesh_final_fout)

        # Download building footprints and attributes for element inside the
        # boundaing polygon, bpoly:
        footprints, attributes = self._get_usastruct_data_for_region(
            final_cells, bpoly
        )
        print(f'\nFound a total of {len(footprints)} building footprints in'
              f' {queryarea_printname}')

        # Create and return the building AssetInventory:
        return self._create_asset_inventory(footprints,
                                            attributes,
                                            self.length_unit)

    def _get_usastruct_data_for_region(self,
                                       final_cells: list[Polygon],
                                       bpoly: Polygon
                                       ) -> tuple[list[list[float]],
                                                  dict[str, list[float]]]:
        """
        Download building attribute data for a specified region.

        Args:
            final_cells (list[Polygon]):
                List of polygons representing the cells for which building
                attribute data needs to be downloaded.
            bpoly (Polygon):
                A boundary polygon used to filter out building centroids that
                fall outside of this region.

        Returns:
            tuple:
                A tuple containing:
                - footprints (list[list[float]]): A list of building
                  footprints, where each footprint is represented as a list of
                  coordinates.
                - attributes (dict[str, list[float]]): A dictionary containing
                  building attributes, with the key `"buildingheight"` mapped
                  to a list of building heights.

        Raises:
            Exception:
                Prints error messages for cells or centroids that result in an
                exception during data download or processing.

        Notes:
            - This method uses multithreading to parallelize the downloading
              process for each cell.
            - Duplicates in building data are removed, and centroids are
              computed for all footprints.
            - Building heights are converted to the specified length unit (`ft`
              or meters) if provided in `self.length_unit`.
            - Buildings with centroids falling outside the boundary polygon
              (`bpoly`) are excluded from the final results.
        """
        # Download building attribute data for each cell:
        pbar = tqdm(
            total=len(final_cells),
            desc="Obtaining the building attributes for each cell",
        )
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(self._download_ustruct_bldgattr, cell): cell
                for cell in final_cells
            }
            for future in concurrent.futures.as_completed(future_to_url):
                cell = future_to_url[future]
                pbar.update(1)
                try:
                    results[cell] = future.result()
                except Exception as exc:
                    results[cell] = None
                    logger.warning("%r generated an exception: %s", cell, exc)
        pbar.close()

        # Parse the API results into building id, footprints and height
        # information:
        ids, footprints, bldgheight = [], [], []
        for cell in tqdm(final_cells):
            res = results[cell]
            ids += res[0]
            footprints += res[1]
            bldgheight += res[2]

        # Remove the duplicate footprint data by recording the API
        # outputs to a dictionary:
        data = {}
        for ind, bldgid in enumerate(ids):
            data[bldgid] = [footprints[ind], bldgheight[ind]]

        # Define length unit conversion factor:
        if self.length_unit == "ft":
            conv_factor = 3.28084
        else:
            conv_factor = 1

        # Calculate building centroids and save the API outputs into
        # their corresponding variables:
        footprints = []
        attributes = {"buildingheight": []}
        centroids = []
        for value in data.values():
            fp = value[0]
            centroids.append(Polygon(fp).centroid)
            footprints.append(fp)
            heightout = value[1]
            if heightout is not None:
                attributes["buildingheight"].append(
                    round(heightout * conv_factor, 1))
            else:
                attributes["buildingheight"].append(None)

        # Identify building centroids and that fall outside of bpoly:
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(bpoly.contains, cent): cent
                for cent in centroids
            }
            for future in concurrent.futures.as_completed(future_to_url):
                cent = future_to_url[future]
                try:
                    results[cent] = future.result()
                except Exception as exc:
                    results[cell] = None
                    logger.warning("%r generated an exception: %s", cell, exc)

        # Save the indices for building that fall outside bpoly:
        ind_remove = []
        for ind, cent in enumerate(centroids):
            if not results[cent]:
                ind_remove.append(ind)

        # Remove data corresponding to centroids that fall outside bpoly:
        for i in sorted(ind_remove, reverse=True):
            del footprints[i]
            del attributes["buildingheight"][i]

        return footprints, attributes

    def _download_ustruct_bldgattr(self,
                                   cell: Polygon) -> tuple[list, list, list]:
        """
        Download building attributes and footprints within a given polygon.

        The method interacts with the ArcGIS REST API to fetch building
        information within the bounding box of the provided polygon. It handles
        retries for failed requests and processes the data to extract relevant
        attributes.

        Args:
            cell (Polygon):
                A polygon representing the geographic area for which building
                attributes are requested.

        Returns:
            tuple: A tuple containing three lists:
                - List of FEMA USA Structures building IDs (`ids`).
                - List of building footprints (`footprints`), represented as
                  polygons.
                - List of building heights (`bldgheight`), with `None` for
                  invalid or missing heights.
        """
        # Parse the response JSON and extract features:
        api_tools = ArcgisAPIServiceHelper(API_ENDPOINT)
        datalist = api_tools.download_attr_from_api(cell,
                                                    ['BUILD_ID', 'HEIGHT'])

        # Process each feature in the response:
        ids = []
        footprints = []
        bldgheight = []
        for data in datalist:
            footprint = data['geometry']['rings'][0]
            bldgid = data['attributes']['BUILD_ID']

            # Only process if the building ID has not been encountered before:
            if bldgid not in ids:
                ids.append(bldgid)
                footprints.append(footprint)

                # Try converting height to float, fallback to None if it fails:
                height = data['attributes'].get('HEIGHT')
                try:
                    height = float(height) if height is not None else None
                except (ValueError, TypeError):
                    height = None

                bldgheight.append(height)

        return ids, footprints, bldgheight
