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
# 01-06-2025

"""
This module defines the class scraping data from Overture Maps.

.. autosummary::

    OvertureMapsScraper
"""

import urllib.request
import re
import pandas as pd

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as fs

from shapely import wkb


class OvertureMapsScraper
   def __init__(self, input_dict: dict):
        """
        Initialize the class object with length units.

        Args:
            input_dict (dict):
                A dictionary specifying length units. If not provided, 'ft' is
                assumed by default..
        """
        self.length_unit = input_dict.get('length', 'ft')

    def get_latest_release():
        """
        Fetches the version number from the title of a webpage.

        Returns:
            str:
                The version number of the latest release or raises an error if not
                found.

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
                else:
                    raise ValueError('Version number is empty.')
            else:
                raise ValueError('The version number for the latest release could '
                                 'not be extracted from the title.')

        except urllib.error.URLError as e:
            raise ValueError(f'Error fetching the webpage: {e}')
        except Exception as e:
            raise ValueError(f'An error occurred: {e}')

    def geoarrow_schema_adapter(schema: pa.Schema) -> pa.Schema:
        """
        Convert a geoarrow-compatible schema to a proper geoarrow schema

        This assumes there is a single "geometry" column with WKB formatting

        Parameters
        ----------
        schema: pa.Schema

        Returns
        -------
        pa.Schema
        A copy of the input schema with the geometry field replaced with
        a new one with the proper geoarrow ARROW:extension metadata

        """
        geometry_field_index = schema.get_field_index("geometry")
        geometry_field = schema.field(geometry_field_index)
        geoarrow_geometry_field = geometry_field.with_metadata(
            {b"ARROW:extension:name": b"geoarrow.wkb"}
        )

        geoarrow_schema = schema.set(
            geometry_field_index, geoarrow_geometry_field)

        return geoarrow_schema

    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        type_theme_map = {
            "address": "addresses",
            "bathymetry": "base",
            "building": "buildings",
            "building_part": "buildings",
            "division": "divisions",
            "division_area": "divisions",
            "division_boundary": "divisions",
            "place": "places",
            "segment": "transportation",
            "connector": "transportation",
            "infrastructure": "base",
            "land": "base",
            "land_cover": "base",
            "land_use": "base",
            "water": "base",
        }

        #bbox = (-118.50647, 34.01793, -118.47420, 34.03404)

        overture_type = 'building'
        theme = type_theme_map[overture_type]

        release = get_latest_release()

        path = f'overturemaps-us-west-2/release/{release}/theme={theme}/type={overture_type}/'

        bpoly, queryarea_printname, osmid = region.get_boundary()
        bbox_coords = bpoly.bounds
        bbox = (bbox_coords[1],bbox_coords[0],bbox_coords[3],bbox_coords[2])
        xmin, ymin, xmax, ymax = bbox
        filter = (
            (pc.field("bbox", "xmin") < xmax)
            & (pc.field("bbox", "xmax") > xmin)
            & (pc.field("bbox", "ymin") < ymax)
            & (pc.field("bbox", "ymax") > ymin)
        )


        dataset = ds.dataset(
            path, filesystem=fs.S3FileSystem(
                anonymous=True, region="us-west-2")
        )
        batches = dataset.to_batches(filter=filter)

        non_empty_batches = (b for b in batches if b.num_rows > 0)

        geoarrow_schema = geoarrow_schema_adapter(dataset.schema)
        reader = pa.RecordBatchReader.from_batches(
            geoarrow_schema, non_empty_batches)

        dfs = []
        for record_batch in reader:
            # Convert each RecordBatch to a pandas DataFrame
            df = record_batch.to_pandas()

            if 'geometry' in df.columns:
                df['geometry'] = df['geometry'].apply(
                    lambda x: wkb.loads(x) if x is not None else None)

            dfs.append(df)

        # Concatenate all DataFrames if needed
        final_df = pd.concat(dfs, ignore_index=True)

        footprints = []
        exclude_columns = ['geometry', 'id', 'bbox', 'version', 'sources']
        attributes = {col: []
            for col in df.columns if col not in exclude_columns}
        for index, row in final_df.iterrows():
            geometry = row['geometry']
            if geometry is not None:
                coords = list(geometry.exterior.coords)  # Extract coordinates
                footprints.append(coords)
                attributes_dict = row.drop(exclude_columns).to_dict()
                for key, value in attributes_dict.items():
                    attributes[key].append(value)

        return self._create_asset_inventory(footprints,
                                            attributes,
                                        self.length_unit)
