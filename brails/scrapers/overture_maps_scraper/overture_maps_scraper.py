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
# 01-08-2025

"""
This module defines the class scraping data from Overture Maps.

.. autosummary::

    OvertureMapsScraper
"""

import re
import urllib.request

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import fs
from shapely import wkb, Polygon, MultiPolygon
from shapely.coords import CoordinateSequence

from brails.types.asset_inventory import Asset, AssetInventory
from brails.types.region_boundary import RegionBoundary

TYPE_THEME_MAP = {
    'address': 'addresses',
    'bathymetry': 'base',
    'building': 'buildings',
    'building_part': 'buildings',
    'division': 'divisions',
    'division_area': 'divisions',
    'division_boundary': 'divisions',
    'place': 'places',
    'segment': 'transportation',
    'connector': 'transportation',
    'infrastructure': 'base',
    'land': 'base',
    'land_cover': 'base',
    'land_use': 'base',
    'water': 'base',
}

EXCLUDE_COLUMNS = ['geometry', 'id', 'bbox', 'version', 'sources']


class OvertureMapsScraper:
    """
    A class for interacting with Overture Maps data.

    This scraper enables the retrieval of geospatial data from Overture Maps
    based on defined region boundaries and requested element types. The class
    supports data filtering, conversion, and parsing into a structured format
    (`AssetInventory`) for further use.

    Attributes:
        length_unit (str): The unit of measurement for lengths (default is
                           'ft').

    Methods:
        __init__(input_dict: dict):
            Initializes the scraper with a specified length unit.

        get_available_element_types(return_types: bool = False) -> list | None:
            Retrieves and optionally returns a list of element types supported
            by the scraper.

        get_elements(region: RegionBoundary, requested_elements: list
                     ) -> AssetInventory:
            Queries and retrieves geospatial elements within a specified region
            and returns them as an inventory.
    """

    def __init__(self, input_dict: dict):
        """
        Initialize the class object with length units.

        Args:
            input_dict(dict):
                A dictionary specifying length units. If not provided, 'ft' is
                assumed by default..
        """
        self.length_unit = input_dict.get('length', 'ft')

    @staticmethod
    def get_available_element_types(return_types: bool = False) -> list | None:
        """
        Retrieve list of element types that can be returned by the scraper.

        This method returns a list of element types that can be returned by
        this scraper and enabled in Overture Maps. Optionally, it can return
        the list of  available element types when the `return_types` flag is
        set to True.

        Args:
            return_types(bool):
                If True, the method will return the list of available element
                types. If False (default), the method only prints the available
                types.

        Returns:
            list | None:
                A list of available element types if `return_types` is True,
                otherwise None.

        Example:
            get_available_element_types()     # Prints the types, returns None.
            get_available_element_types(True)  # Returns a list of available
                                                element types.
        """
        available_types = ', '.join(f"'{key}'" for key in TYPE_THEME_MAP)

        print('The elements types that can be returned by this scraper are: '
              f'{available_types}')
        # TODO: Fix this so that it either returns values or not
        if return_types:
            return available_types

    def get_elements(self,
                     region: RegionBoundary,
                     requested_elements: list) -> AssetInventory:
        """
        Retrieve geospatial elements within a specified region.

        Args:
            region(RegionBoundary):
                The geographical boundary defining the region of interest.
                Must provide methods to retrieve the boundary polygon, its
                name, and identifier.
            requested_elements(list):
                A list of element types to query, where the first element
                determines the data theme and dataset type for the request.

        Returns:
            AssetInventory:
                An inventory of assets, including their geometries and
                attributes, parsed from the requested data within the
                specified region.

        Raises:
            ValueError:
                If no data is available for the specified region or if the
                requested data cannot be retrieved.

        Workflow:
            1. Determines the dataset path based on the requested element
               type and theme.
            2. Queries the dataset to filter elements within the specified
               region boundary using bounding box coordinates.
            3. Downloads the data and converts it into a pandas DataFrame.
            4. Parses geometries(e.g., Polygon, MultiPolygon) and their
               attributes from the data.
            5. Stores each parsed geometry and its associated attributes in an
               `AssetInventory` object.

        Notes:
            - The function supports filtering based on bounding boxes for
              performance optimization.
            - Geometries are converted from Well-Known Binary(WKB) to Shapely
              objects for processing.
            - Attributes are extracted from the dataset, excluding predefined
              columns in `EXCLUDE_COLUMNS`.
            - Geometries include Polygons and MultiPolygons. Non-supported
              geometries are skipped.

        Example:
            >> > region = RegionBoundary(...)
            >> > requested_elements = ['building', 'road']
            >> > inventory = get_elements(region, requested_elements)
            >> > print(inventory)
        """
        # TODO: Run queries and create an inventory for multiple requested
        # elements
        # TODO: Return only the elements within the RegionBoundary

        overture_type = requested_elements[0]
        theme = TYPE_THEME_MAP[overture_type]

        release = self._get_latest_release()

        path = (f'overturemaps-us-west-2/release/{release}/theme={theme}'
                f'/type={overture_type}/')

        bpoly, _, _ = region.get_boundary()
        bbox = bpoly.bounds
        xmin, ymin, xmax, ymax = bbox
        spatial_filter = (
            (pc.field("bbox", "xmin") < xmax)
            & (pc.field("bbox", "xmax") > xmin)
            & (pc.field("bbox", "ymin") < ymax)
            & (pc.field("bbox", "ymax") > ymin)
        )

        print('\nDownloading requested data from Overture Maps...')
        dataset = ds.dataset(
            path, filesystem=fs.S3FileSystem(
                anonymous=True, region="us-west-2")
        )
        batches = dataset.to_batches(filter=spatial_filter)

        non_empty_batches = (b for b in batches if b.num_rows > 0)

        geoarrow_schema = self._geoarrow_schema_adapter(dataset.schema)
        reader = pa.RecordBatchReader.from_batches(
            geoarrow_schema, non_empty_batches)

        dfs = []
        for record_batch in reader:
            # Convert each RecordBatch to a pandas DataFrame:
            df = record_batch.to_pandas()

            if 'geometry' in df.columns:
                df['geometry'] = df['geometry'].apply(
                    lambda x: wkb.loads(x) if x is not None else None)

            dfs.append(df)

        if len(dfs) == 0:
            raise ValueError('No data available for the specified region')

        # Concatenate all DataFrames if needed:
        final_df = pd.concat(dfs, ignore_index=True)
        print('Requested data downloaded.\n')

        print('Parsing downloaded data...')
        final_df = final_df[final_df.columns.drop(EXCLUDE_COLUMNS)]

        inventory_final = AssetInventory()
        asset_id = 1
        for _, row in final_df.iterrows():
            geometry = row['geometry']

            coordinates = []
            if isinstance(geometry, Polygon):
                coords = self._parse_coordinate_sequence(
                    geometry.exterior.coords)
                coordinates.append(coords)
            elif isinstance(geometry, MultiPolygon):
                polygons = list(geometry.geoms)
                for polygon in polygons:
                    coords = self._parse_coordinate_sequence(
                        polygon.exterior.coords)
                    coordinates.append(coords)
            else:
                continue

            attributes = row.dropna().drop(['geometry']).to_dict()

            for coords in coordinates:
                inventory_final.add_asset(asset_id, Asset(
                    asset_id, coords, attributes))
                asset_id += 1
            # TODO: Include asset type in attributes
            # TODO: Unit conversion
        print('Data successfully parsed.')

        return inventory_final

    @staticmethod
    def _get_latest_release():
        """
        Fetch version number from the title of Overture releases webpage.

        Returns:
            str:
                The version number of the latest release or raises an error if
                not found.

        Raises:
            ValueError:
                If the version number cannot be extracted from the title.
        """
        url = 'https://docs.overturemaps.org/release/latest/'

        try:
            # Fetch the webpage content
            with urllib.request.urlopen(url) as response:
                # Decode the response using the page's encoding
                charset = response.headers.get_content_charset() or 'utf-8'
                html = response.read().decode(charset)

            # Use a regex to extract the title tag content
            title_match = re.search(r'<title data-rh=true>(.*?)\s?\|', html)

            if title_match:
                version = title_match.group(1).strip()
                if version:
                    return version

                raise ValueError('Version number is empty.')

            raise ValueError('The version number for the latest release could '
                             'not be extracted from the title.')

        except urllib.error.URLError as e:
            raise ValueError(f'Error fetching the webpage: {e}') from e
        except Exception as e:
            raise ValueError(f'An error occurred: {e}') from e

    @staticmethod
    def _geoarrow_schema_adapter(schema: pa.Schema) -> pa.Schema:
        """
        Convert a geoarrow-compatible schema to a proper geoarrow schema.

        This assumes there is a single "geometry" column with WKB formatting

        Args:
            schema: pa.Schema

        Returns:
            pa.Schema
            A copy of the input schema with the geometry field replaced with
            a new one with the proper geoarrow ARROW: extension metadata

        """
        geometry_field_index = schema.get_field_index("geometry")
        geometry_field = schema.field(geometry_field_index)
        geoarrow_geometry_field = geometry_field.with_metadata(
            {b"ARROW:extension:name": b"geoarrow.wkb"}
        )

        geoarrow_schema = schema.set(
            geometry_field_index, geoarrow_geometry_field)

        return geoarrow_schema

    @staticmethod
    def _parse_coordinate_sequence(sequence: CoordinateSequence):
        """
        Convert a coordinate sequence into a list of lists.

        Args:
            sequence(shapely.coords.CoordinateSequence):
                A Shapely CoordinateSequence of coordinates.

        Returns:
            list: A list of lists, where each inner list represents a
            coordinate from the input sequence.

        Example:
            Input: [(1, 2), (3, 4), (5, 6)]
            Output: [[1, 2], [3, 4], [5, 6]]
        """
        sequence_list = list(sequence)
        return [list(item) for item in sequence_list]
