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
Base class for reading Overture Maps data.

.. autosummary::

    OvertureMapsScraper
"""

from abc import ABC
import re
import urllib.request
from typing import Optional, List, Sequence, Tuple, Union
from tqdm import tqdm

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as fs


from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.geometry import MultiLineString
from shapely.geometry.base import BaseGeometry


import numpy as np

from brails.constants import PROVIDER_ID_FEATURES

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

# The footprint_provider value of an Overture footprint, and the feature
# holding its Overture id.
OVERTURE_PROVIDER = 'overture'
OVERTURE_ID_KEY = PROVIDER_ID_FEATURES[OVERTURE_PROVIDER][0]

# Overture's building columns that other modules read: the height, the
# number of floors, the class and the sources.
HEIGHT_KEY = 'height'
NUM_FLOORS_KEY = 'num_floors'
CLASS_KEY = 'class'
SOURCES_KEY = 'sources'

# The separator of the entries of the sources string that _format_sources
# writes, and the property an entry names when it is the source of the
# height.
SOURCES_SEPARATOR = '; '
HEIGHT_PROPERTY = '/properties/height'


class OvertureMapsScraper(ABC):
    """
    Base class of the Overture Maps scrapers, with the release lookup.

    The class is not instantiated directly, and
    :class:`~brails.scrapers.overture_maps_scraper.overture_maps_footprint_scraper.OvertureMapsFootprintScraper`
    derives from it. The README beside this module, "Releases", states how
    a release is resolved.
    """

    # Releases the S3 bucket holds, cached after the first listing.
    _available_releases: Optional[List[str]] = None

    @staticmethod
    def fetch_release_names(print_releases: bool = False) -> List[str]:
        """
        Fetch the release names from Overture's published release list.

        The list need not match the releases the S3 bucket holds, which
        :meth:`fetch_available_releases` gives.

        Args:
            print_releases (bool, optional):
                If ``True``, print the list of releases. Defaults to ``False``.

        Returns:
            List[str]:
                A list of release version strings.

        Raises:
            OSError:
                If the list cannot be fetched, as ``urllib.error.URLError``.
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
    def fetch_available_releases(refresh: bool = False) -> List[str]:
        """
        List the releases the Overture S3 bucket currently holds.

        Args:
            refresh (bool, optional):
                List the bucket again instead of reusing the result cached
                by the first listing. Defaults to ``False``.

        Returns:
            List[str]:
                Release names, in reverse order, which is newest first for
                date-named releases.

        Raises:
            OSError:
                If the bucket cannot be listed.
        """
        if OvertureMapsScraper._available_releases is not None and not refresh:
            return list(OvertureMapsScraper._available_releases)

        filesystem = fs.S3FileSystem(anonymous=True, region="us-west-2")
        entries = filesystem.get_file_info(
            fs.FileSelector(S3_BASE_PATH, recursive=False)
        )
        releases = sorted(
            (
                entry.base_name
                for entry in entries
                if entry.type == fs.FileType.Directory and entry.base_name
            ),
            reverse=True,
        )

        OvertureMapsScraper._available_releases = releases
        return list(releases)

    @staticmethod
    def resolve_release(requested: Optional[str] = None) -> str:
        """
        Determine which release to read, checking it against the bucket.

        Where the bucket cannot be listed or lists no release, a warning is
        printed, ``requested`` is returned unchecked, and without it the
        first release of the published list is returned.

        Args:
            requested (str, optional):
                The release to read. When omitted, the first release
                :meth:`fetch_available_releases` gives, with a printed
                notice.

        Returns:
            str:
                The release name to read from.

        Raises:
            ValueError:
                If the bucket holds no release named ``requested``, or if
                the bucket cannot be listed or lists no release and the
                published list is empty.
            OSError:
                If the bucket cannot be listed or lists no release,
                ``requested`` is omitted and the published list cannot be
                fetched.
        """
        listing_error = None
        try:
            available = OvertureMapsScraper.fetch_available_releases()
        except Exception as error:  # noqa: BLE001 - any failure falls back
            available, listing_error = [], error

        if available:
            if requested and requested not in available:
                raise ValueError(
                    f"Overture release '{requested}' is not available. "
                    f"s3://{S3_BASE_PATH} currently holds: "
                    f"{', '.join(available)}. Only the most recent releases "
                    "are retained, so a release that worked before may have "
                    "been removed."
                )
            if requested:
                return requested
            print(
                "'overtureRelease' not defined in input. Falling back to the "
                f"newest available release: {available[0]}"
            )
            return available[0]

        # The bucket could not be listed, so nothing here can be confirmed
        # readable. Fall back to the published list.
        print(
            f"WARNING: could not list s3://{S3_BASE_PATH} ({listing_error}). "
            "Falling back to the published release list, which may name "
            "releases that are no longer readable."
        )
        if requested:
            return requested

        published = OvertureMapsScraper.fetch_release_names()
        if not published:
            raise ValueError(
                "No Overture release could be determined: the S3 bucket "
                "could not be listed and the published release list is empty."
            )
        print(f"Falling back to published release: {published[0]}")
        return published[0]

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

        Raises:
            ValueError:
                If ``bbox`` is not a tuple or list of four elements.
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
            release (str):
                The release version string.
            overture_type (str):
                The Overture type, a key of ``TYPE_THEME_MAP``.

        Returns:
            str:
                The full S3 path string.

        Raises:
            KeyError:
                If ``overture_type`` is not a key of ``TYPE_THEME_MAP``.
        """
        theme = TYPE_THEME_MAP[overture_type]
        return f"{S3_BASE_PATH}/{release}/theme={theme}/type={overture_type}/"

    @staticmethod
    def _read_table(
        overture_release: str,
        overture_type: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        connect_timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
        paths: Optional[Sequence[str]] = None,
    ) -> pa.Table:
        """
        Read an Overture dataset from S3, or from local files, as a table.

        The rows kept are those whose ``bbox`` column overlaps ``bbox``. The
        same filter applies to the S3 dataset and to local files.

        Args:
            overture_release (str):
                Dataset release version specifying which data to retrieve.
                Not used when ``paths`` is given.
            overture_type (str):
                The dataset type to load. Not used when ``paths`` is given.
            bbox (Optional[Tuple[float, float, float, float]]):
                Optional bounding box (xmin, ymin, xmax, ymax) to filter data
                spatially.
            connect_timeout (Optional[int]):
                Timeout in seconds for establishing the S3 connection.
            request_timeout (Optional[int]):
                Timeout in seconds for data requests.
            paths (Optional[Sequence[str]]):
                Local Parquet files to read in place of the S3 dataset. Each
                file must hold Overture's ``bbox`` column when ``bbox`` is
                given.

        Returns:
            pa.Table:
                The rows read, with the columns and the schema metadata of
                the dataset.
        """
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

        if paths is not None:
            dataset = ds.dataset(
                list(paths), filesystem=fs.LocalFileSystem(), format="parquet"
            )
        else:
            path = OvertureMapsScraper._dataset_path(
                overture_release, overture_type
            )
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

        return pa.Table.from_batches(batches)

    @staticmethod
    def _format_sources(sources_array):
        """
        Format a NumPy array or list of dictionaries into a single string.

        Each dictionary's non-empty values are formatted as key: value pairs,
        joined by a comma and a space. Multiple dictionaries are joined by
        SOURCES_SEPARATOR.

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

        return SOURCES_SEPARATOR.join(formatted_parts) if formatted_parts else None
