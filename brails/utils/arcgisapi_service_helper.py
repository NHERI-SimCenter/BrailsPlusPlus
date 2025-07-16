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
# 06-11-2025

"""
This module defines a class for retrieving data from ArcGIS services APIs.

.. autosummary::

    ArcgisAPIServiceHelper
"""

import math
import concurrent.futures
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm
from shapely.geometry import Polygon
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from brails.utils import GeoTools


REQUESTS_RETRY_STRATEGY = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504],
)


class ArcgisAPIServiceHelper:
    """
    A helper class for interacting with an ArcGIS API service.

    This class provides methods to query the service, fetch element counts,
    split polygons into cells, and categorize cells based on their element
    count.

    This class allows users to interact with an ArcGIS API endpoint to retrieve
    information about elements within specific polygons. It supports
    functionality for:
    - Retrieving the maximum number of elements that can be returned per query.
    - Categorizing and splitting polygon cells based on the element count.
    - Dividing polygons into smaller rectangular cells to ensure a balance of
      elements per cell.
    - Fetching specified attribute data for given cells in parallel.

    The class includes methods for making API requests with retry logic,
    handling transient network errors, and parsing the responses to determine
    element counts within specific regions.

    Attributes:
        api_endpoint_url (str):
            The base URL of the ArcGIS API endpoint used for requests.
        max_elements_per_cell (int):
            The maximum number of elements the API allows per query, fetched
            from the ArcGIS service.

    Methods:
        fetch_max_records_per_query() -> int:
            Retrieves the maximum number of records returned by the API in a
            single query.
        categorize_and_split_cells(preliminary_cells: list[Polygon]
                                   ) -> tuple[list[Polygon], list[Polygon]]:
            Categorizes and splits polygons based on their element count.
            Returns two lists: cells to keep and cells to split.
        split_polygon_into_cells(bpoly: Polygon, element_count: int = -1,
                                 plot_mesh: str = '') -> list[Polygon]:
            Divides a polygon into smaller cells if the element count exceeds
            the threshold.
        download_attr_from_api(cell: Polygon,
                               requested_fields: list[str] | str) -> list:
            Fetches specific attribute fields from the API for a given polygon.
        get_element_counts(bpoly: Polygon) -> int:
            Retrieves the count of elements within a given polygon's bounding
            box.
        fetch_data_for_cells(final_cells: list, download_func: Callable,
                             desc: str) -> dict:
            Concurrently applies a download function to a list of cells with
            progress tracking.

    Example Usage:
        # Instantiate the helper class with an API endpoint:
        api_helper = ArcgisAPIServiceHelper("https://api.example.com/query")

        # Fetch the maximum number of records per query:
        max_records = api_helper.fetch_max_records_per_query()

        # Split polygons into cells based on element count:
        cells_to_keep, cells_to_split = api_helper.categorize_and_split_cells(
            preliminary_cells)

    Notes:
        - The class assumes a uniform distribution of elements across polygons
          when splitting.
        - The retry strategy uses exponential backoff to handle transient
          network issues.
        - The method `get_element_counts` provides a count of elements within
          the bounding box of a polygon, which is essential for categorizing
          and splitting polygons.

    Raises:
        ValueError: If an invalid Polygon or parameter is passed to a method.
        NotImplementedError: If the requested functionality is not supported
        by the API.
        HTTPError: If the HTTP request fails after retries.
    """

    def __init__(self, api_endpoint_url: str) -> None:
        """
        Initialize the ArcgisAPIServiceHelper class object.

        This constructor initializes an instance of the
        `ArcgisAPIServiceHelper` class. It accepts the base URL of the ArcGIS
        API endpoint and automatically fetches the maximum number of elements
        allowed per query by calling the `fetch_max_records_per_query` method.

        Args:
            api_endpoint_url (str):
                The base URL of the ArcGIS API endpoint to be used for making
                API requests.

        Initializes the following attributes:
            api_endpoint_url (str):
                The provided API endpoint URL.
            max_elements_per_cell (int):
                The maximum number of elements allowed in a single cell,
                fetched from the API.
        """
        self.api_endpoint_url = api_endpoint_url
        try:
            self.max_elements_per_cell = self.fetch_max_records_per_query()
        except Exception as e:
            raise ValueError(
                f"An error occurred while accessing the API endpoint: {e}. "
                "Please verify that the provided URL and parameters are "
                "correct."
            ) from e

    def fetch_max_records_per_query(self) -> int:
        """
        Retrieve the maximum number of records returned by the API per query.

        This function sends a request to the specified API endpoint and parses
        the response to determine the maximum number of records that can be
        returned in a single query. If the API does not provide this
        information or returns a value of 0, it raises an error.


        Returns:
            int:
                The maximum number of elements the API allows per query.

        Raises:
            ValueError:
                If the API returns a value of 0 for the maximum number of
                elements, indicating an issue with the API or its response.
            HTTPError:
                If the HTTP request fails (e.g., due to connectivity issues or
                a server error).

        Example:
            >>> api_tools = ArcgisAPIServiceHelper(
                "https://api.example.com/data/query")
            >>> max_records = api_tools.fetch_max_records_per_query()
            >>> print(max_records)
            1000

        Notes:
            - The function expects the API to return a JSON response containing
              a `maxRecordCount` key.
            - A retry strategy is implemented for the HTTPS request to handle
              transient network issues.
        """
        # Get the number of elements allowed per cell by the API:
        api_reference_url = (
            self.api_endpoint_url[:-6] if self.api_endpoint_url.endswith(
                '/query')
            else self.api_endpoint_url
        ) + '?f=pjson'

        # Set up a session with retry logic:
        response = self._make_request_with_retry(api_reference_url)

        # Return the maximum record count from the API response. If the
        # response does not include the 'maxRecordCount' key, count
        # defaults to 0 and a ValueError is raised:
        max_record_count = response.json().get('maxRecordCount', 0)
        if max_record_count == 0:
            raise ValueError('The API reported a value of 0 for the maximum '
                             'number of elements returned per query, '
                             'indicating a potential problem with the '
                             'response or the API service.')

        return max_record_count

    def download_attr_from_api(
            self,
            cell: Polygon,
            requested_fields: Union[List[str], str]
    ) -> List:
        """
        Download specified fields from the API for a given cell.

        Args:
            cell (Polygon):
                A Shapely Polygon object representing the area of interest.
            requested_fields (list[str] or str):
                A list of attribute names or the string 'all'.

        Returns:
            list[dict]:
                A list of features (attributes) fetched from the API.

        Raises:
            ValueError:
                If the input 'cell' is not a valid Polygon or
                'requested_fields' is not valid.
        """
        # Validate the 'cell' input and get its bounding box coordinates:
        if not isinstance(cell, Polygon):
            raise ValueError("Invalid input: The 'cell' parameter must be a "
                             'valid Shapely Polygon object.')
        bbox = cell.bounds

        # Reformat the requested fields for API consumption:
        error_message = (
            "Invalid value for 'requested_fields'.\n"
            'Allowable values are:\n'
            '- A list of string values representing the requested attribute '
            'names.\n'
            "- The string 'all' to request all attributes available from the "
            'API.'
        )
        if isinstance(requested_fields, str):
            if requested_fields.lower() == 'all':
                output_fields = '*'
            else:
                raise ValueError(error_message)
        elif isinstance(requested_fields, list):
            # Ensure all elements in the list are strings
            if not all(isinstance(field, str) for field in requested_fields):
                raise ValueError(
                    "All elements in the 'requested_fields' list must be "
                    'strings.'
                )
            output_fields = ','.join(requested_fields)
        else:
            raise ValueError(error_message)

        # Parameters for querying the ArcGIS API, specifying the geometry and
        # desired fields:
        params = {
            'geometry': f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}',
            'geometryType': 'esriGeometryEnvelope',
            'inSR': '4326',
            'outSR': '4326',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': output_fields,
            'f': 'json',
        }

        response = self._make_request_with_retry(self.api_endpoint_url, params)

        # Parse the response JSON and extract features:
        datalist = response.json().get('features', [])

        # If no features are returned, adjust the bounding box definition
        # and try again:
        if not datalist:
            params['geometry'] = f'{bbox[0]},{bbox[1]},{bbox[3]},{bbox[2]}'
            response = self._make_request_with_retry(self.api_endpoint_url,
                                                     params)
            datalist = response.json().get('features', [])

        return datalist

    def download_all_attr_for_region(
        self,
        region,
        plot_cells=False,
        task_description='Obtaining attributes for each cell'
    ):
        """
        Download all attribute data for the specified region.

        This method:
            - Retrieves the boundary polygon of the region.
            - Splits region into smaller cells for manageable data querying.
            - Refines the mesh by recursively splitting oversized cells.
            - Optionally plots the final mesh.
            - Downloads data (e.g., bridge attributes) for each cell using a
              provided data-fetching function.

        Args:
            region:
                A Region object that provides a `.get_boundary()` method
                returning a polygon, region name, and optional OSM ID.
            plot_cells (bool, optional):
                If True, generates and saves a visualization of the final
                meshed cells.
            task_description (str, optional):
                A message string that describes the task being performed,
                passed to the data-fetching method for logging or display.

        Returns:
            Tuple:
                - downloaded_data (Dict[Polygon, List[Dict[str, Any]]]):
                    Mapping from each final mesh cell polygon to a list of
                    attribute dictionaries (e.g., representing bridges or other
                    assets).
                - final_cells (List[Polygon]):
                    List of polygons representing the final meshed cells used
                    for data querying.
        """
        # Get the region's boundary polygon and printable name:
        boundary_polygon, region_name, _ = region.get_boundary()

        print("\nMeshing the defined area...")
        initial_cells = self.split_polygon_into_cells(boundary_polygon)

        # Refine mesh by splitting cells that exceed element limits:
        if len(initial_cells) > 1:
            final_cells = []
            cells_to_process = initial_cells.copy()

            while cells_to_process:
                cells_to_keep, cells_to_process = \
                    self.categorize_and_split_cells(cells_to_process)
                final_cells.extend(cells_to_keep)

            print(
                f'\nMeshing complete. Split {region_name} into '
                f'{len(final_cells)} cells.')
        else:
            final_cells = initial_cells.copy()
            print(
                f'\nMeshing complete. Covered {region_name} with a single '
                'rectangular cell.')

        # Optionally plot the final mesh of cells:
        if plot_cells:
            mesh_plot_path = region_name.replace(" ", "_") + "_Mesh_Final.png"
            GeoTools.plot_polygon_cells(
                boundary_polygon, final_cells, mesh_plot_path)

        # Download bridge attributes for each mesh cell:
        downloded_data = self.fetch_data_for_cells(
            final_cells,
            self.download_all_attr_from_api,
            desc=task_description
        )
        return downloded_data, final_cells

    def download_all_attr_from_api(self, cell: Polygon
                                   ) -> List[Dict[str, Any]]:
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
        return self.download_attr_from_api(cell, 'all')

    def categorize_and_split_cells(
            self,
            preliminary_cells: List[Polygon]
    ) -> Tuple[List[Polygon], List[Polygon]]:
        """
        Categorize/split a list of polygon cells based on their element count.

        This method processes a list of polygons (preliminary_cells) by first
        obtaining the number of elements contained within each polygon. If a
        polygon contains more elements than the specified maximum allowed per
        cell (max_elements_per_cell), the polygon is split into smaller cells.
        The method returns two lists:
        - A list of cells that are kept as is (those that do not exceed the
          element threshold).
        - A list of split cells (those that exceeded the element threshold).

        Args:
            preliminary_cells (list[Polygon]):
                A list of Shapely Polygon objects representing the preliminary
                cells to be processed.

        Returns:
            tuple: A tuple containing two lists of Polygon objects:
                - The first list contains the cells to keep (those with
                  elements <= max_elements_per_cell).
                - The second list contains the split cells (those with
                  elements > max_elements_per_cell).
        """
        # Download the element count for each cell:
        results = self.fetch_data_for_cells(
            preliminary_cells,
            self.get_element_counts,
            desc="Obtaining the number of elements in each cell"
        )

        # Iterate through each cell in the preliminary cells list:
        cells_to_split = []
        cells_to_keep = []
        for cell in preliminary_cells:
            # Get the element count for the current cell:
            element_count = results.get(cell)

            # Skip cells that have an element count of 0 or None:
            if element_count == 0 or element_count is None:
                continue

            # If the element count exceeds the max allowed, add the cell to the
            # 'to split' list:
            if element_count > self.max_elements_per_cell:
                cells_to_split.append(cell)
            # Otherwise, add the cell to the 'to keep' list:
            else:
                cells_to_keep.append(cell)

        # Iterate through each cell that needs to be split:
        split_cells = []
        for cell in cells_to_split:
            # Call the method to split the polygon of the cell into smaller
            # cells. The method returns a list of new cells (rectangles) after
            # splitting:
            rectangles = self.split_polygon_into_cells(
                cell, element_count=results[cell])

            # Add the resulting split cells (rectangles) to the split_cells
            # list:
            split_cells += rectangles

        return cells_to_keep, split_cells

    def fetch_data_for_cells(
            self,
            final_cells: List[Any],
            download_func: Callable[[Any], Any],
            desc: str = "Obtaining the attributes for each cell"
    ) -> Dict[Any, Any]:
        """
        Download data for a list of cells in parallel using provided function.

        Args:
            final_cells (List[Any]):
                List of cells to process.
            download_func (Callable[[Any], Any]):
                Function to download data for a single cell.
            desc (str):
                Description for the progress bar.

        Returns:
            Dict[Any, Any]:
                Dictionary mapping each cell to its downloaded data
                (or None if failed).
        """
        results = {}
        pbar = tqdm(total=len(final_cells), desc=desc)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_cell = {
                executor.submit(download_func, cell): cell
                for cell in final_cells
            }
            for future in concurrent.futures.as_completed(future_to_cell):
                cell = future_to_cell[future]
                pbar.update(1)
                try:
                    results[cell] = future.result()
                except Exception as exc:
                    results[cell] = None
                    print(f'{cell} generated an exception: {exc}')

        pbar.close()
        return results

    def split_polygon_into_cells(
            self,
            bpoly: Polygon,
            element_count: int = -1,
            plot_mesh: str = ''
    ) -> List[Polygon]:
        """
        Divide a polygon into smaller cells based on its element count.

        If the number of elements exceeds the specified threshold
        (max_elements_per_cell), the polygon will be divided into multiple
        rectangular cells with a margin of error. The grid is generated under
        the assumption that the elements are uniformly distributed across the
        polygon, and it can be split based on the ratio of element_count to
        max_elements_per_cell to get cells that do not exceed the  max elements
        specified per cellapproximate balance. Please note that, this method
        does not guarantee that each cell will contain fewer than the specified
        maximum number of elements.

        Args:
            bpoly (Polygon):
                The polygon to be split into rectangular cells.
            element_count (int, optional):
                The total number of elements in the polygon. If not provided,
                the method will compute this using `get_element_counts`.
            plot_mesh (str, optional):
                If provided, the generated mesh will be plotted using
                `GeoTools.plot_polygon_cells`.

        Returns:
            list:
                A list of polygons (or bounding boxes) representing the
                rectangular grid cells that cover the input polygon.

        Notes:
            - If the element count is below or equal to
              `max_elements_per_cell`, the entire polygon's envelope is
              returned as a single cell.
            - If the element count exceeds `max_elements_per_cell`, the polygon
              is divided into smaller cells based on the bounding box aspect
              ratio.
        """
        if element_count == -1:
            # Get the number of elements in the input polygon bpoly:
            element_count = self.get_element_counts(bpoly)

        # If the element count exceeds the number of elements allowed per cell:
        if element_count > self.max_elements_per_cell:

            # Calculate the number of cells required to cover the polygon area
            # with 20 percent margin of error:
            ncells_required = round(
                1.2 * element_count / self.max_elements_per_cell)

            # Get the coordinates of the bounding box for input polygon bpoly:
            bbox = bpoly.bounds

            # Calculate the horizontal and vertical dimensions of the bounding
            # box:
            xdist = GeoTools.haversine_dist((bbox[0], bbox[1]),
                                            (bbox[2], bbox[1]))
            ydist = GeoTools.haversine_dist((bbox[0], bbox[1]),
                                            (bbox[0], bbox[3]))

            # Determine the bounding box aspect ratio (defined as a number
            # greater than 1) and the longer side direction of the bounding
            # box:
            if xdist > ydist:
                bbox_aspect_ratio = math.ceil(xdist / ydist)
                long_side = 1  # The box is longer in its horizontal direction
            else:
                bbox_aspect_ratio = math.ceil(ydist / xdist)
                long_side = 2  # The box is longer in its vertical direction

            # # Calculate the number of cells required on the short side of the
            # bounding box, n, using the relationship
            # ncells_required = bbox_aspect_ratio*n^2:
            n = math.ceil(math.sqrt(ncells_required / bbox_aspect_ratio))

            # Based on the calculated n value, determine the number of rows and
            # columns of cells required:
            if long_side == 1:
                rows, cols = bbox_aspect_ratio * n, n
            else:
                rows, cols = n, bbox_aspect_ratio * n

            # Determine the coordinates of each cell covering bpoly:
            rectangles = GeoTools.mesh_polygon(bpoly, rows, cols)

        # If the element count is within the limit, use the polygon's envelope:
        else:
            rectangles = [bpoly.envelope]

        # Plot the generated mesh if requested:
        if plot_mesh:
            GeoTools.plot_polygon_cells(bpoly, rectangles, plot_mesh)

        return rectangles

    def get_element_counts(self, bpoly: Polygon) -> int:
        """
        Get the count of elements within the bounding box of the given polygon.

        Args:
            bpoly (Polygon):
                The polygon marking the boundaries of a region.

        Returns:
            int:
                The count of elements within the bounding box, or 0 if an
                error occurs.
        """
        # Get the coordinates of the bounding box for input polygon bpoly:
        bbox = bpoly.bounds

        # Set API parameters required to get the element counts:
        params = {
            'geometry': f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}',
            'geometryType': 'esriGeometryEnvelope',
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelIntersects',
            'returnCountOnly': 'true',
            'f': 'json',
        }

        # Set up a session with retry logic and query the API:
        response = self._make_request_with_retry(self.api_endpoint_url, params)

        # Return the count from the API response. If the response does not
        # include the 'count' key, count defaults to 0:
        return response.json().get('count', 0)

    @staticmethod
    def _make_request_with_retry(
            url: str,
            params: Optional[Dict] = None
    ) -> requests.Response:
        """
        Make a GET request to the ArcGIS API with retry logic.

        This method uses a session with retry logic to send a GET request to
        the specified URL. If query parameters are provided, they are included
        in the request; otherwise, the request is made without parameters.

        Args:
            url (str):
                The URL for the API endpoint.
            params (dict, optional):
                A dictionary of query parameters to include in the API request.
                If None, the request will be made without any query parameters.

        Args:
            params (dict, optional):
                The query parameters to be included in the API request.
                If None, the request will be made without parameters.

        Returns:
            requests.Response:
                The response object from the API request.

        Raises:
            HTTPError: If the HTTP request returns an unsuccessful status code.
        """
        with requests.Session() as session:
            session.mount(
                'https://', HTTPAdapter(max_retries=REQUESTS_RETRY_STRATEGY))

            if params:
                # If params are provided, include them in the GET request
                response = session.get(url, params=params)
            else:
                # If no params, make the request without query parameters
                response = session.get(url)

            response.raise_for_status()

        return response
