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

"""Tests for MobileHomeParkScraper in brails.scrapers.mobile_home_park_scraper."""

from __future__ import annotations

import io

import geopandas as gpd
import pytest
import requests
from shapely.geometry import Point, box

from brails.scrapers import MobileHomeParkScraper as ExportedScraper
from brails.scrapers.mobile_home_park_scraper.mobile_home_park_scraper import (
    PARK_BUCKET_URL,
    PARK_ID_COLUMN,
    PARK_LISTING_URL,
    PARK_PREFIX,
    MobileHomeParkScraper,
)
from brails.types.geometry_region import GeometryRegion
from brails.utils import Importer

REGION = GeometryRegion(box(-122.3, 37.8, -122.2, 37.9))
FOLDER = f'{PARK_BUCKET_URL}/{PARK_PREFIX}'


def build_listing(names: dict[str, str], *, truncated: bool = False) -> bytes:
    """
    Return a listing of the park folder in the bucket's XML.

    Args:
        names (dict[str, str]):
            The time each file of the folder was last modified, by its name.
        truncated (bool, optional):
            Whether the listing is marked as truncated, holding fewer files than the
            folder.

    Returns:
        bytes:
            The listing as the bucket's server sends it.
    """
    entries = ''.join(
        f'<Contents><Key>{PARK_PREFIX}{name}</Key>'
        f'<LastModified>{time}</LastModified></Contents>'
        for name, time in names.items()
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><ListBucketResult '
        'xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f'<IsTruncated>{str(truncated).lower()}</IsTruncated>{entries}'
        '</ListBucketResult>'
    ).encode()


def build_part(
    parks: list[tuple[float, str, float, float]], crs: str | None = None
) -> bytes:
    """
    Return a part of the park file as Parquet bytes.

    Args:
        parks (list[tuple[float, str, float, float]]):
            The id, name, longitude and latitude of each park.
        crs (str, optional):
            The CRS the part states. Defaults to ``None``, which states none, as the
            archive's parts do.

    Returns:
        bytes:
            The part, as the archive stores it.
    """
    frame = gpd.GeoDataFrame(
        {
            PARK_ID_COLUMN: [park[0] for park in parks],
            'NAME': [park[1] for park in parks],
            'STATUS': 'OPEN',
        },
        geometry=[Point(x, y) for _, _, x, y in parks],
        crs=crs,
    )
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return buffer.getvalue()


# A folder written twice: the older marker names the stale part, the newer one the
# current part, whose parks lie inside the region, on its western edge and east of
# it. The newer marker's name sorts first, so only its time tells it apart.
NEWEST = '_committed_10'
LISTING = {
    '_SUCCESS': '2025-11-17T21:19:11.000Z',
    '_committed_9': '2025-01-02T00:00:00.000Z',
    NEWEST: '2025-11-17T21:19:11.000Z',
    '_started_10': '2025-11-17T21:19:11.000Z',
    'part-stale.parquet': '2025-01-02T00:00:00.000Z',
    'part-current.parquet': '2025-11-17T21:19:11.000Z',
}
CURRENT_PARKS = [
    (101.0, 'Inside', -122.25, 37.85),
    (102.0, 'On the edge', -122.3, 37.85),
    (103.0, 'East', -122.0, 37.85),
]


pytestmark = pytest.mark.usefixtures('no_network')


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, bytes], list]:
    """
    Serve the folder above in place of requests.Session.get.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Replaces requests.Session.get.

    Returns:
        tuple[dict[str, bytes], list]:
            The body served at each URL, which a test may change, and the URL and
            timeout of each request, in order.
    """
    bodies = {
        PARK_LISTING_URL: build_listing(LISTING),
        f'{FOLDER}_committed_9': b'{"added":["part-stale.parquet"],"removed":[]}',
        f'{FOLDER}{NEWEST}': b'{"added":["part-current.parquet"],"removed":[]}',
        f'{FOLDER}part-stale.parquet': build_part(
            [(201.0, 'Stale', -122.25, 37.85)]
        ),
        f'{FOLDER}part-current.parquet': build_part(CURRENT_PARKS),
    }
    calls = []

    def get(
        _session: requests.Session,
        url: str,
        timeout: float | None = None,
        params: object = None,  # noqa: ARG001
    ) -> requests.Response:
        calls.append((url, timeout))
        response = requests.Response()
        response.status_code = 200
        response.url = url
        response._content = bodies[url]
        return response

    monkeypatch.setattr(requests.Session, 'get', get)
    return bodies, calls


def test_reads_only_the_parts_the_newest_marker_names(served: tuple) -> None:
    """The listing, the newest marker by time and its part are read, and no other."""
    _, calls = served
    scraper = MobileHomeParkScraper(timeout=7)
    parks = scraper.get_assets(REGION)
    assert calls == [
        (PARK_LISTING_URL, 7),
        (f'{FOLDER}{NEWEST}', 7),
        (f'{FOLDER}part-current.parquet', 7),
    ]
    assert 201 not in parks.inventory
    assert (scraper.marker, scraper.written) == (NEWEST, LISTING[NEWEST])


@pytest.mark.usefixtures('served')
def test_parks_strictly_inside_keyed_by_integer_id() -> None:
    """Only a park strictly inside is kept, under its integer id with its columns."""
    parks = MobileHomeParkScraper().get_assets(REGION)
    assert list(parks.inventory) == [101]
    park = parks.inventory[101]
    assert park.coordinates == [[-122.25, 37.85]]
    assert park.features == {PARK_ID_COLUMN: 101, 'NAME': 'Inside', 'STATUS': 'OPEN'}
    assert type(park.features[PARK_ID_COLUMN]) is int


@pytest.mark.parametrize(
    ('listing', 'added', 'problem'),
    [
        ({k: v for k, v in LISTING.items() if k != '_SUCCESS'}, None, '_SUCCESS'),
        (
            {k: v for k, v in LISTING.items() if not k.startswith('_committed_')},
            None,
            'no _committed_ marker',
        ),
        (LISTING, b'{"added":[],"removed":[]}', 'lists no part'),
    ],
)
def test_folder_without_a_complete_write_is_refused(
    served: tuple, listing: dict, added: bytes | None, problem: str
) -> None:
    """A folder without _SUCCESS, a marker or a part in it raises, naming the gap."""
    bodies, _ = served
    bodies[PARK_LISTING_URL] = build_listing(listing)
    if added is not None:
        bodies[f'{FOLDER}{NEWEST}'] = added
    with pytest.raises(RuntimeError, match=f'{problem}.*try again later'):
        MobileHomeParkScraper().get_assets(REGION)


def test_body_cut_short_is_a_connection_error(
    served: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A part cut short raises the ConnectionError of get_whole_body."""
    bodies, _ = served
    serve = requests.Session.get

    def get_short(
        session: requests.Session, url: str, **kwargs: object
    ) -> requests.Response:
        response = serve(session, url, **kwargs)
        response.headers['Content-Length'] = str(len(bodies[url]) + 1)
        return response

    monkeypatch.setattr(requests.Session, 'get', get_short)
    with pytest.raises(requests.ConnectionError, match='cut short.*try again'):
        MobileHomeParkScraper().get_assets(REGION)


def test_truncated_listing_is_refused(served: tuple) -> None:
    """A listing marked as truncated raises RuntimeError."""
    bodies, _ = served
    bodies[PARK_LISTING_URL] = build_listing(LISTING, truncated=True)
    with pytest.raises(RuntimeError, match='more files than one listing returns'):
        MobileHomeParkScraper().get_assets(REGION)


def test_part_in_a_projected_crs_is_refused(served: tuple) -> None:
    """A part stating a CRS other than longitude and latitude raises ValueError."""
    bodies, _ = served
    bodies[f'{FOLDER}part-current.parquet'] = build_part(
        [(101.0, 'Inside', 550000.0, 4180000.0)], crs='EPSG:32610'
    )
    with pytest.raises(ValueError, match='EPSG:32610'):
        MobileHomeParkScraper().get_assets(REGION)


@pytest.mark.parametrize(
    'parks',
    [
        [(101.5, 'Fraction', -122.25, 37.85)],
        [(float('nan'), 'No id', -122.25, 37.85)],
        [(101.0, 'First', -122.25, 37.85), (101.0, 'Second', -122.24, 37.85)],
    ],
)
def test_park_id_that_is_not_one_whole_number_is_refused(
    served: tuple, parks: list
) -> None:
    """A park id inside that is not whole or is held twice raises ValueError."""
    bodies, _ = served
    bodies[f'{FOLDER}part-current.parquet'] = build_part(parks)
    with pytest.raises(ValueError, match=PARK_ID_COLUMN):
        MobileHomeParkScraper().get_assets(REGION)


def test_class_is_exported_and_found_by_the_importer() -> None:
    """brails.scrapers exports the class, and the Importer finds it by its name."""
    assert ExportedScraper is MobileHomeParkScraper
    assert Importer().get_class('MobileHomeParkScraper') is MobileHomeParkScraper
