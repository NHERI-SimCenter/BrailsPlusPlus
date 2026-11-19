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
Read the Census Bureau's TIGER/Line files into memory.

.. autosummary::

      read_tiger_file
      tiger_file_url
"""

from __future__ import annotations

import io

import geopandas as gpd
import requests

from brails.utils.http_session import create_retrying_session, get_whole_body

DEFAULT_TIGER_VINTAGE = 2023
"""The year of the TIGER/Line release read when no vintage is given."""

TIGER_ROOT = 'https://www2.census.gov/geo/tiger'
"""The folder of the TIGER/Line releases on the Census Bureau's server."""

_downloads: dict[str, bytes] = {}


def tiger_file_url(
    layer: str, area: str, vintage: int = DEFAULT_TIGER_VINTAGE
) -> str:
    """
    Return the URL of a TIGER/Line shapefile.

    The file is ``tl_<vintage>_<area>_<layer>.zip`` in the folder ``<layer>`` of the
    release, as in ``TIGER2023/PLACE/tl_2023_06_place.zip``, with the layer's name in
    lower case.

    Args:
        layer (str):
            The layer's folder in the release, such as ``'PLACE'``, ``'COUNTY'``,
            ``'STATE'`` or ``'BG'``.
        area (str):
            The two-digit FIPS code of a state for a file published by state, such
            as ``'06'``, or ``'us'`` for a national file.
        vintage (int, optional):
            The year of the TIGER/Line release. Defaults to
            :data:`DEFAULT_TIGER_VINTAGE`.

    Returns:
        str:
            The URL under :data:`TIGER_ROOT`.
    """
    return (
        f'{TIGER_ROOT}/TIGER{vintage}/{layer}/'
        f'tl_{vintage}_{area}_{layer.lower()}.zip'
    )


def read_tiger_file(
    layer: str,
    area: str,
    vintage: int = DEFAULT_TIGER_VINTAGE,
    *,
    timeout: float = 120,
    retries: int = 3,
    notice: str = '',
) -> gpd.GeoDataFrame:
    """
    Return the features of a TIGER/Line shapefile.

    The file, at the URL :func:`tiger_file_url` gives, is downloaded the first time
    it is asked for in a process and read from memory, with names decoded as the
    zip's ``.cpg`` file states. Its bytes are kept for later calls once they have
    been read, so a file that does not read is downloaded again on the next call.
    Nothing is written to disk.

    Args:
        layer (str):
            The layer's folder in the release, as for :func:`tiger_file_url`.
        area (str):
            The state's FIPS code or ``'us'``, as for :func:`tiger_file_url`.
        vintage (int, optional):
            The year of the TIGER/Line release. Defaults to
            :data:`DEFAULT_TIGER_VINTAGE`.
        timeout (float, optional):
            Seconds to wait for the server to answer. Defaults to 120.
        retries (int, optional):
            How many times a download is tried again after a failure that
            :func:`~brails.utils.http_session.create_retrying_session` retries.
            Defaults to 3.
        notice (str, optional):
            A line printed before the file is downloaded, and not when it is read
            from memory. Defaults to ``''``, which prints nothing.

    Returns:
        gpd.GeoDataFrame:
            The file's features in its own coordinate reference system, NAD83
            (EPSG:4269) for the files of the Census Bureau.

    Raises:
        ValueError:
            If the server has no such file, naming its URL and the vintage.
        requests.ConnectionError:
            If the connection keeps failing after the retries, or the file's
            body is cut short as
            :func:`~brails.utils.http_session.get_whole_body` finds it; the
            message then says to try again.
        requests.HTTPError:
            If the server answers with another error status, after the retries.
    """
    url = tiger_file_url(layer, area, vintage)
    if url in _downloads:
        return gpd.read_file(io.BytesIO(_downloads[url]))
    if notice:
        print(notice)
    with create_retrying_session(retries) as session:
        response = get_whole_body(session, url, timeout)
    if response.status_code == requests.codes.not_found:
        msg = (
            f'The Census Bureau has no TIGER/Line {vintage} file at {url}. '
            'Check that the vintage is a published TIGER/Line release and '
            'that it holds this layer for this area.'
        )
        raise ValueError(msg)
    response.raise_for_status()
    features = gpd.read_file(io.BytesIO(response.content))
    _downloads[url] = response.content
    return features
