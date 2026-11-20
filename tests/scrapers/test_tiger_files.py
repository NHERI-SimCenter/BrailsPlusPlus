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


"""Tests for reading TIGER/Line files into memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pytest
import requests
from pyogrio.errors import DataSourceError
from shapely.geometry import box

from brails.scrapers.us_census_scrapers import tiger_files
from brails.scrapers.us_census_scrapers.tiger_files import (
    read_tiger_file,
    tiger_file_url,
)
from tests.local_files import zip_shapefile

if TYPE_CHECKING:
    from pathlib import Path

PLACE_URL = 'https://www2.census.gov/geo/tiger/TIGER2023/PLACE/tl_2023_06_place.zip'


pytestmark = pytest.mark.usefixtures('no_network')


@pytest.fixture
def server(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, tuple[int, bytes]], list[tuple[str, float]]]:
    """
    Serve files from a dict in place of the Census Bureau's server.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Replaces ``requests.Session.get`` and empties the downloads kept.

    Returns:
        tuple[dict[str, tuple[int, bytes]], list[tuple[str, float]]]:
            The status and content served for each URL, which a test fills, any other
            URL answering 404, and the URL and timeout of each request.
    """
    files: dict[str, tuple[int, bytes]] = {}
    requests_made: list[tuple[str, float]] = []

    def get(
        _session: requests.Session,
        url: str,
        *,
        timeout: float,
        params: object = None,  # noqa: ARG001
    ) -> requests.Response:
        requests_made.append((url, timeout))
        response = requests.Response()
        response.url = url
        response.status_code, response._content = files.get(url, (404, b''))
        return response

    monkeypatch.setattr(requests.Session, 'get', get)
    monkeypatch.setattr(tiger_files, '_downloads', {})
    return files, requests_made


def zipped_places(folder: Path) -> bytes:
    """
    Return a zipped shapefile of two places, one named with an accent.

    Args:
        folder (Path):
            An empty folder for the shapefile's parts.

    Returns:
        bytes:
            The zip, holding the parts the way TIGER/Line files do, with a ``.cpg``
            file that names UTF-8.
    """
    places = gpd.GeoDataFrame(
        {
            'GEOID': ['0639003', '0606000'],
            'NAME': ['La Cañada Flintridge', 'Berkeley'],
        },
        geometry=[box(-118.2, 34.2, -118.1, 34.3), box(-122.3, 37.8, -122.2, 37.9)],
        crs='EPSG:4269',
    )
    return zip_shapefile(places, folder, 'tl_2023_06_place')


def test_reads_names_and_crs_from_bytes(server: tuple, tmp_path: Path) -> None:
    """A zipped shapefile is read from memory with its UTF-8 names and its CRS."""
    files, _ = server
    files[PLACE_URL] = (200, zipped_places(tmp_path))
    places = read_tiger_file('PLACE', '06', 2023)
    assert list(places['NAME']) == ['La Cañada Flintridge', 'Berkeley']
    assert places.crs.to_epsg() == 4269
    assert places.geometry.iloc[1].equals(box(-122.3, 37.8, -122.2, 37.9))


def test_url_of_a_national_file() -> None:
    """A national file is named with 'us' in its layer's folder."""
    assert tiger_file_url('COUNTY', 'us', 2023) == (
        'https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip'
    )


def test_one_download_per_url(server: tuple, tmp_path: Path) -> None:
    """A file asked for twice is downloaded once, and another URL again."""
    files, requests_made = server
    content = zipped_places(tmp_path)
    files[PLACE_URL] = (200, content)
    other = PLACE_URL.replace('2023', '2024')
    files[other] = (200, content)
    read_tiger_file('PLACE', '06', 2023)
    read_tiger_file('PLACE', '06', 2023)
    read_tiger_file('PLACE', '06', 2024, timeout=5)
    assert requests_made == [(PLACE_URL, 120), (other, 5)]


@pytest.mark.usefixtures('server')
def test_missing_file_names_url_and_vintage() -> None:
    """A file the server does not have is refused with its URL and vintage."""
    url = 'https://www2.census.gov/geo/tiger/TIGER2031/COUNTY/tl_2031_us_county.zip'
    with pytest.raises(ValueError, match=f'no TIGER/Line 2031 file at {url}'):
        read_tiger_file('COUNTY', 'us', 2031)
    assert tiger_files._downloads == {}


def test_server_error_raised(server: tuple) -> None:
    """A 503 answer is raised as HTTPError, and nothing is kept."""
    files, _ = server
    files[PLACE_URL] = (503, b'')
    with pytest.raises(requests.HTTPError, match='503'):
        read_tiger_file('PLACE', '06')
    assert tiger_files._downloads == {}


def test_retrying_session_and_notice(
    server: tuple,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A download goes through a retrying session, after the notice, printed once."""
    files, _ = server
    files[PLACE_URL] = (200, zipped_places(tmp_path))
    sessions = []

    def create_retrying_session(retries: int) -> requests.Session:
        sessions.append(retries)
        return requests.Session()

    monkeypatch.setattr(
        tiger_files, 'create_retrying_session', create_retrying_session
    )
    read_tiger_file('PLACE', '06', 2023, retries=5, notice='Downloading places.')
    read_tiger_file('PLACE', '06', 2023, notice='Downloading places.')
    assert sessions == [5]
    assert capsys.readouterr().out == 'Downloading places.\n'


def test_body_cut_short_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A download whose body is cut short is reported with advice to try again."""

    def get(*_args: object, **_kwargs: object) -> requests.Response:
        raise requests.exceptions.ChunkedEncodingError

    monkeypatch.setattr(requests.Session, 'get', get)
    monkeypatch.setattr(tiger_files, '_downloads', {})
    with pytest.raises(requests.ConnectionError, match='cut short .* try again'):
        read_tiger_file('PLACE', '06')
    assert tiger_files._downloads == {}


def test_bytes_that_do_not_read_not_kept(server: tuple) -> None:
    """Bytes that geopandas cannot read are not kept, so the next call downloads."""
    files, requests_made = server
    files[PLACE_URL] = (200, b'not a zip')
    for _ in range(2):
        with pytest.raises(DataSourceError):
            read_tiger_file('PLACE', '06')
    assert tiger_files._downloads == {}
    assert len(requests_made) == 2
