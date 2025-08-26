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
# 07-13-2025

"""
This module define the base class for scraping OvertureMaps data.

.. autosummary::

    OvertureMapsScraper
"""

from abc import ABC
import re
import urllib.request
from typing import Optional, List, Tuple, Union
from tqdm import tqdm

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as fs


from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.geometry import MultiLineString
from shapely.geometry.base import BaseGeometry


import numpy as np

# Global constants for base paths, URLs, and type-theme mapping:
S3_BASE_PATH = 'overturemaps-us-west-2/release'
OVERTURE_RELEASES_URL = ('https://raw.githubusercontent.com/OvertureMaps/data/'
                         'main/overture_releases.yaml')

TYPE_THEME_MAP = {
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


class OvertureMapsScraper(ABC):
    """
    A base class for accessing and processing data from Overture Maps.

    This is a base class and is **not intended to be instantiated directly**.
    To use it, either:

    - Import and instantiate a subclass
      (e.g., ``Importer().get_class('OvertureMapsFootprintScraper')``)
    - Or subclass it yourself and implement additional logic.

    Direct imports of this base class are typically only needed when creating
    new Overture Maps scrapers.

    Provided static methods include:

    - Fetching release version names from the Overture Maps release index.
    - Constructing S3 paths to specific dataset partitions.
    - Reading datasets from S3 into a Pandas DataFrame with optional
      spatial filtering.
    - Normalizing bounding box coordinates to a consistent format.
    - Formatting source metadata from NumPy arrays or lists of dictionaries.
    """

    @staticmethod
    def fetch_release_names(print_releases: bool = False) -> List[str]:
        """
        Fetch the list of release names from the OvertureMaps releases YAML.

        Args:
            print_releases (bool, optional):
                If ``True``, print the list of releases. Defaults to ``False``.

        Returns:
            List[str]:
                A list of release version strings.
        """
        with urllib.request.urlopen(OVERTURE_RELEASES_URL) as resp:
            text = resp.read().decode("utf-8")
            releases = re.findall(r'release:\s*"([^"]+)"', text)

        if print_releases:
            print("Available releases:")
            for release in releases:
                print(f"  - {release}")

        return releases

    @staticmethod
    def normalize_bbox_order(
        bbox: Union[Tuple[float, float, float, float], list]
    ) -> Tuple[float, float, float, float]:
        """
        Reorder bbox coordinates to ensure (xmin, ymin, xmax, ymax) format.

        Args:
            bbox (tuple or list):
                A sequence of four numeric coordinates (x1, y1, x2, y2).

        Returns:
            Tuple[float, float, float, float]:
                Bounding box reordered as (xmin, ymin, xmax, ymax).
        """
        if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
            raise ValueError(
                "bbox must be a tuple or list of exactly four elements")

        x1, y1, x2, y2 = bbox
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        return xmin, ymin, xmax, ymax

    @staticmethod
    def _dataset_path(release: str, overture_type: str) -> str:
        """
        Return the S3 path of the Overture dataset to use.

        Args:
            overture_type (str):
                The overture sub-partition type.
            release (str):
                The release version string.

        Returns:
            str:
                The full S3 path string.
        """
        theme = TYPE_THEME_MAP[overture_type]
        return f"{S3_BASE_PATH}/{release}/theme={theme}/type={overture_type}/"

    @staticmethod
    def _read_to_pandas(
        overture_release: str,
        overture_type: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        connect_timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Read an Overture dataset from S3 into a Pandas DataFrame.

        Args:
            overture_release (str):
                Dataset release version specifying which data to retrieve.
            overture_type (str):
                The dataset type to load.
            bbox (Optional[Tuple[float, float, float, float]]):
                Optional bounding box (xmin, ymin, xmax, ymax) to filter data
                spatially.
            connect_timeout (Optional[int]):
                Timeout in seconds for establishing the S3 connection.
            request_timeout (Optional[int]):
                Timeout in seconds for data requests.

        Returns:
            pd.DataFrame:
                The dataset content as a Pandas DataFrame.
        """
        path = OvertureMapsScraper._dataset_path(
            overture_release, overture_type
        )

        if bbox is not None:
            bbox = OvertureMapsScraper.normalize_bbox_order(bbox)
            xmin, ymin, xmax, ymax = bbox
            filter_expr = (
                (pc.field("bbox", "xmin") < xmax) &
                (pc.field("bbox", "xmax") > xmin) &
                (pc.field("bbox", "ymin") < ymax) &
                (pc.field("bbox", "ymax") > ymin)
            )
        else:
            filter_expr = None

        filesystem = fs.S3FileSystem(
            anonymous=True,
            region="us-west-2",
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
        )

        dataset = ds.dataset(path, filesystem=filesystem)

        batches = []
        batch_iterator = dataset.to_batches(
            batch_size=10000, filter=filter_expr)

        for batch in tqdm(batch_iterator, desc="Reading dataset batches"):
            batches.append(batch)

        table = pa.Table.from_batches(batches)
        return table.to_pandas()

    @staticmethod
    def _format_sources(sources_array):
        """
        Format a NumPy array or list of dictionaries into a single string.

        Each dictionary's non-empty values are formatted as key: value pairs,
        separated by commas. Multiple dictionaries are separated by semicolons.

        Args:
            sources_array (Union[np.ndarray, List[Dict]]):
                Array or list of source dictionaries.

        Returns:
            Optional[str]:
                Formatted string or None if input is empty or invalid.
        """
        if sources_array is None:
            return None

        # Convert to list if it is a NumPy array:
        if isinstance(sources_array, np.ndarray):
            sources_list = sources_array.tolist()
        elif isinstance(sources_array, list):
            sources_list = sources_array
        else:
            return None

        if not sources_list:
            return None

        formatted_parts = []
        for source in sources_list:
            if isinstance(source, dict):
                parts = [f"{k}: {v}" for k,
                         v in source.items() if v not in (None, "", [])]
                if parts:
                    formatted_parts.append(", ".join(parts))

        return "; ".join(formatted_parts) if formatted_parts else None
