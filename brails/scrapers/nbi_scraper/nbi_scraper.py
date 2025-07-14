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
# 06-07-2025

"""
This module defines a class for retrieving bridge data from NBI.

.. autosummary::

    NBI_Scraper
"""

import concurrent.futures
from tqdm import tqdm
from typing import Any, Dict, List
from shapely.geometry import Point, Polygon
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils import GeoTools, ArcgisAPIServiceHelper

# Define global variables:
API_ENDPOINT = ('https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/'
                'services/NTAD_National_Bridge_Inventory/FeatureServer/0/query'
                )
ASSET_TYPE = 'bridge'
METRIC_ATTR = ['MIN_VERT_CLR_010',
               'APPR_WIDTH_MT_032',
               'NAV_VERT_CLR_MT_039',
               'NAV_HORR_CLR_MT_040',
               'HORR_CLR_MT_047',
               'MAX_SPAN_LEN_MT_048',
               'STRUCTURE_LEN_MT_049',
               'LEFT_CURB_MT_050A',
               'RIGHT_CURB_MT_050B',
               'ROADWAY_WIDTH_MT_051',
               'DECK_WIDTH_MT_052',
               'VERT_CLR_OVER_MT_053',
               'VERT_CLR_UND_054B',
               'LAT_UND_MT_055B',
               'LEFT_LAT_UND_MT_056',
               'IMP_LEN_MT_076',
               'MIN_NAV_CLR_MT_116']


class NBI_Scraper:
    """
    A class for scraping and processing National Bridge Inventory (NBI) data.

    The class handles tasks such as fetching and processing bridge data,
    meshing the region into smaller cells for efficient data retrieval, and
    organizing the results into an AssetInventory.

    Attributes:
        length_unit (str):
            The unit of length (default is 'ft').
        inventory (AssetInventory):
            An inventory object that holds the assets for this instance.

    Methods:
        get_assets(region: RegionBoundary) -> AssetInventory:
            Retrieves bridge inventory data within a given region boundary,
            processes the data, and returns the inventory of assets.
    """

    def __init__(self, input_dict: Dict[str, Any] = None):
        """
        Initialize an instance of the class with a specified length unit.

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
        # If input_dict is None, default to an empty dictionary
        if input_dict is None:
            input_dict = {}

        # Use the provided dictionary or default to 'ft'
        self.length_unit = input_dict.get('length', 'ft').lower()
        # TODO: Validate length input so that the user is notified of
        # unsupported inputs.
        self.inventory = AssetInventory()

    def get_assets(self, region: RegionBoundary) -> AssetInventory:
        """
        Retrieve bridge inventory within a given region.

        This method processes the region boundary, splits it into smaller
        cells, fetches the necessary bridge data for each cell, and returns the
        processed bridge asset inventory.

        Args:
            region (RegionBoundary):
                A `RegionBoundary` object representing the geographic region
                for which to retrieve bridge inventory data.

        Returns:
            AssetInventory:
                An `AssetInventory` object containing the bridge assets within
                the specified region.

        Raises:
            TypeError:
                If the `region` argument is not an instance of the
                `RegionBoundary` class. The error message will indicate that
                the 'region' argument must be an instance of the
                `RegionBoundary` class.
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
                  f'{len(final_cells)} cells')

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

        # Download bridge data for each cell:
        results = self._fetch_data_for_cells(final_cells)

        # Process the downloaded data and save it in an AssetInventory:
        self._process_data(results, final_cells, bpoly)
        return self.inventory

    def _fetch_data_for_cells(
            self,
            final_cells: List[Polygon]
    ) -> Dict[Polygon, Any]:
        """
        Fetch bridge data for each polygonal cell using multithreading.

        This method distributes the task of data retrieval across multiple
        threads to efficiently query bridge attributes for a list of
        geographic cells.

        Args:
            final_cells (List[Polygon]):
                A list of Shapely Polygon objects representing the cells into
                which the region was divided for API querying.

        Returns:
            Dict[Polygon, Any]:
                A dictionary where each key is a Polygon cell and each value is
                the data retrieved for that cell (or None if an error
                occurred).
        """
        pbar = tqdm(
            total=len(final_cells),
            desc="Obtaining the bridge attributes for each cell",
        )
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(self._download_data, cell): cell
                for cell in final_cells
            }
            for future in concurrent.futures.as_completed(future_to_url):
                cell = future_to_url[future]
                pbar.update(1)
                try:
                    results[cell] = future.result()
                except Exception as exc:
                    results[cell] = None
                    print(f'{cell} generated an exception: {exc}')
        pbar.close()

        return results

    @staticmethod
    def _download_data(cell: Polygon) -> List[Dict[str, Any]]:
        """
        Download attribute data for a given cell using the ArcGIS API.

        This method queries the ArcGIS API for all available attributes within
        the specified polygon cell.

        Args:
            cell (Polygon):
                A Shapely Polygon object representing the geographic cell for
                which to retrieve bridge data.

        Returns:
            List[Dict[str, Any]]:
                A list of dictionaries, each representing the attribute data
                for a bridge within the cell. Each dictionary includes
                'geometry' and 'attributes' keys as returned by the ArcGIS API.
        """
        api_tools = ArcgisAPIServiceHelper(API_ENDPOINT)
        return api_tools.download_attr_from_api(cell, 'all')

    def _process_data(self,
                      results: Dict[Polygon, List[Dict[str, Any]]],
                      final_cells: List[Polygon],
                      boundary_polygon: Polygon) -> None:
        """
        Process the downloaded NBI data and store it in an AssetInventory.

        This method filters bridge data retrieved from each cell based on
        whether the geometry lies within the region boundary, converts units if
        needed, and constructs asset objects for the inventory.

        Args:
            results (Dict[Polygon, List[Dict[str, Any]]]):
                A mapping of each polygonal cell to a list of attribute
                dictionaries, each representing a bridge.
            final_cells (List[Polygon]):
                The list of polygonal cells that define the meshed region.
            boundary_polygon (Polygon):
                The main geographic boundary of the region. Used to filter
                bridges that fall outside the defined region.
        """
        # Get the data for cells that are fully contained within the bounding
        # polygon:
        # Identify the cells that are inside the bounding polygon and record
        # their data:
        data, data_to_filter = [], []
        for cell in final_cells:
            if boundary_polygon.contains(cell):
                data.extend(results.get(cell, []))
            else:
                data_to_filter.extend(results.get(cell, []))

        # Filter the data within the cells that are not contained in the
        # bounding polygon such that only the points withing the bounding
        # polygon are retained:
        for item in data_to_filter:
            if boundary_polygon.contains(Point(item['geometry']['x'],
                                               item['geometry']['y'])):
                data.append(item)

        # Display the number of elements detected:
        print(f'\nFound a total of {len(data)} bridges.')

        # Save the results in the inventory:
        for index, item in enumerate(data):
            geometry = [[item['geometry']['x'], item['geometry']['y']]]
            asset_features = {**item['attributes'], 'type': ASSET_TYPE}
            if self.length_unit == 'ft':
                for feature in METRIC_ATTR:
                    feature_Value = asset_features.get(feature)
                    if feature_Value is not None:
                        asset_features[feature] = feature_Value*3.28084

            # Create the Asset and add it to the inventory:
            asset = Asset(index, geometry, asset_features)
            self.inventory.add_asset(index, asset)

        # TODO: Remove duplicate bridges resulting from the way NBI stores data
        # across different states. Solution likely requires using
        # structure_number_008, othr_state_struc_no_099, and
        # other_state_pcnt_098b fields.
