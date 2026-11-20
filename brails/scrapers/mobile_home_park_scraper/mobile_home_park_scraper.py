# Copyright (c) 2026 The Regents of the University of California
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
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.

"""
Read the mobile-home park points of HIFLD.

.. autosummary::

      MobileHomeParkScraper
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
import shapely

from brails.constants import GEOGRAPHIC_CRS
from brails.types.asset_inventory import Asset, AssetInventory
from brails.utils.http_session import create_retrying_session, get_whole_body

if TYPE_CHECKING:
    import requests

    from brails.types.region_boundary import RegionBoundary

# The bucket of SeerAI's archive of HIFLD layers on Source Cooperative.
PARK_BUCKET_URL = 'https://data.source.coop/seerai'

# The folder of the bucket that holds the parts of the park points' Parquet file.
PARK_PREFIX = 'hifld/mobile-home-parks/mobilehomeparks/mobilehomeparks.parquet/'

# The URL of the listing of the park file's folder.
PARK_LISTING_URL = f'{PARK_BUCKET_URL}?list-type=2&prefix={PARK_PREFIX}'

# The HIFLD column of a park's id, by which the points are keyed.
PARK_ID_COLUMN = 'MHPID'

# The XML namespace of the bucket's listing, which follows Amazon S3's.
_LISTING_NAMESPACE = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def _download(session: requests.Session, url: str, timeout: float) -> bytes:
    """
    Return the body of a file of the archive.

    Args:
        session (requests.Session):
            The session the request is sent through.
        url (str):
            The file's URL.
        timeout (float):
            Seconds to wait for the server to answer.

    Returns:
        bytes:
            The body.

    Raises:
        requests.ConnectionError:
            If the connection keeps failing after the retries, or the body is cut
            short.
        requests.HTTPError:
            If the server answers with an error status, after the retries.
    """
    response = get_whole_body(session, url, timeout)
    response.raise_for_status()
    return response.content


class MobileHomeParkScraper:
    """
    Read the mobile-home park points of HIFLD that lie inside a region.

    The points come from SeerAI's archive of HIFLD on Source Cooperative, read into
    memory on each call. The README beside this module describes the archive and
    which write of it is read.

    Args:
        timeout (float, optional):
            Seconds to wait for the server to answer each request. Defaults to 120.
        retries (int, optional):
            How many times a failed request is tried again. Defaults to 3.

    Attributes:
        marker (str | None):
            The name of the archive's marker whose parts the last call of
            :meth:`get_assets` read, or ``None`` before a call.
        written (str | None):
            That marker's time of modification, or ``None`` before a call.
    """

    def __init__(self, *, timeout: float = 120, retries: int = 3) -> None:
        """Keep the request settings."""
        self.timeout = timeout
        self.retries = retries
        self.marker = None
        self.written = None

    def get_assets(self, region: RegionBoundary) -> AssetInventory:
        """
        Return the park points strictly inside a region.

        Args:
            region (RegionBoundary):
                The region, whose boundary is in EPSG:4326.

        Returns:
            AssetInventory:
                One point asset per park, keyed by its :data:`PARK_ID_COLUMN` as an
                integer, with the HIFLD columns as features.

        Raises:
            RuntimeError:
                If the archive shows no complete write to read.
            ValueError:
                If a part states a CRS other than longitude and latitude, or a park
                inside the region has a :data:`PARK_ID_COLUMN` that is not a whole
                number or that another park inside has too.
            requests.ConnectionError:
                If the connection keeps failing after the retries, or a body is cut
                short.
            requests.HTTPError:
                If the server answers with an error status, after the retries.
        """
        boundary, _, _ = region.get_boundary()
        with create_retrying_session(self.retries) as session:
            listing = _download(session, PARK_LISTING_URL, self.timeout)
            # Ruff flags every ElementTree parse call (S314). Expat 2.4.1 and later,
            # bundled with Python 3.10 to 3.12, refuses runaway entity expansion, and
            # ElementTree reads no external entity; pyexpat.EXPAT_VERSION shows the
            # version a build uses.
            root = ET.fromstring(listing)  # noqa: S314
            modified = {
                entry.findtext('s3:Key', '', _LISTING_NAMESPACE).removeprefix(
                    PARK_PREFIX
                ): entry.findtext('s3:LastModified', '', _LISTING_NAMESPACE)
                for entry in root.iterfind('s3:Contents', _LISTING_NAMESPACE)
            }
            markers = sorted(
                (time, name)
                for name, time in modified.items()
                if name.startswith('_committed_')
            )
            parts = []
            if root.findtext('s3:IsTruncated', '', _LISTING_NAMESPACE) == 'true':
                problem = 'holds more files than one listing returns'
            elif '_SUCCESS' not in modified:
                problem = 'has no _SUCCESS marker'
            elif not markers:
                problem = 'has no _committed_ marker'
            else:
                self.written, self.marker = markers[-1]
                marker = _download(
                    session,
                    f'{PARK_BUCKET_URL}/{PARK_PREFIX}{self.marker}',
                    self.timeout,
                )
                parts = json.loads(marker)['added']
                problem = f'has a newest marker, {self.marker}, that lists no part'
            if not parts:
                msg = (
                    f'The folder of the park file in the HIFLD archive {problem} '
                    f'({PARK_LISTING_URL}), so it shows no complete write to read. '
                    'The archive may be in the middle of a write; try again later.'
                )
                raise RuntimeError(msg)
            frames = []
            for part in parts:
                body = _download(
                    session, f'{PARK_BUCKET_URL}/{PARK_PREFIX}{part}', self.timeout
                )
                frame = gpd.read_parquet(io.BytesIO(body))
                # The archive's parts state no CRS; their points are in longitude and
                # latitude.
                if frame.crs not in (None, GEOGRAPHIC_CRS, 'OGC:CRS84'):
                    msg = (
                        f'The part {part} of the park file states the CRS '
                        f'{frame.crs.to_string()}, not longitude and latitude.'
                    )
                    raise ValueError(msg)
                frames.append(frame)

        parks = pd.concat(frames, ignore_index=True)
        lon, lat = parks.geometry.x, parks.geometry.y
        inside = shapely.contains_xy(boundary, lon, lat)
        parks, lon, lat = parks[inside], lon[inside], lat[inside]
        ids = parks[PARK_ID_COLUMN]
        unfit = (ids % 1 != 0) | ids.duplicated(keep=False)
        if unfit.any():
            msg = (
                f'The archive holds {PARK_ID_COLUMN} values inside the region that '
                'are not whole numbers or are held by more than one park: '
                f'{sorted(set(ids[unfit].tolist()))[:5]}.'
            )
            raise ValueError(msg)

        features = parks.drop(columns=parks.geometry.name).assign(
            **{PARK_ID_COLUMN: ids.astype(int)}
        )
        inventory = AssetInventory()
        for x, y, values in zip(
            lon.tolist(), lat.tolist(), features.to_dict('records')
        ):
            park_id = values[PARK_ID_COLUMN]
            inventory.add_asset(park_id, Asset(park_id, [[x, y]], values))
        return inventory
