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

"""Tests for Overture Maps release discovery and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.fs as pafs
import pytest

from brails.scrapers.overture_maps_scraper import overture_maps_scraper
from brails.scrapers.overture_maps_scraper.overture_maps_footprint_scraper import (
    OvertureMapsFootprintScraper,
)
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import (
    OvertureMapsScraper,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

BUCKET_RELEASES = ['2026-06-17.0', '2026-07-22.0']


class FakeFileInfo:
    """Stand-in for a pyarrow FileInfo entry."""

    def __init__(self, base_name: str, file_type: pafs.FileType) -> None:
        """
        Store the entry name and type.

        Args:
            base_name (str):
                The final path component.
            file_type (pafs.FileType):
                Directory or File.
        """
        self.base_name = base_name
        self.type = file_type


class FakeFileSystem:
    """Stand-in for pyarrow's S3FileSystem that serves a fixed listing."""

    def __init__(self, entries: list[FakeFileInfo], counter: list[int]) -> None:
        """
        Store the listing this filesystem returns.

        Args:
            entries (list[FakeFileInfo]):
                Entries returned by every listing call.
            counter (list[int]):
                Single-element list incremented on each listing, so tests can
                assert how often the bucket was hit.
        """
        self.entries = entries
        self.counter = counter

    def get_file_info(self, selector: object) -> list[FakeFileInfo]:  # noqa: ARG002
        """
        Return the fixed listing.

        Args:
            selector (object):
                Ignored; the listing does not depend on it.

        Returns:
            list[FakeFileInfo]:
                The stored entries.
        """
        self.counter[0] += 1
        return self.entries


@pytest.fixture(autouse=True)
def _clear_release_cache() -> Iterator[None]:
    """Keep the class-level release cache from leaking between tests."""
    OvertureMapsScraper._available_releases = None
    yield
    OvertureMapsScraper._available_releases = None


@pytest.fixture
def listing_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """
    Serve a fixed bucket listing and count how often it is read.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Fixture used to replace the S3 filesystem.

    Returns:
        list[int]:
            Single-element call counter.
    """
    counter = [0]
    entries = [
        FakeFileInfo(name, pafs.FileType.Directory) for name in BUCKET_RELEASES
    ]
    entries.append(FakeFileInfo('index.json', pafs.FileType.File))

    monkeypatch.setattr(
        overture_maps_scraper.fs,
        'S3FileSystem',
        lambda **_kwargs: FakeFileSystem(entries, counter),
    )
    return counter


def _fail_listing(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """
    Make the bucket listing raise.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Fixture used to replace the S3 filesystem.
        error (Exception):
            The error the constructor raises.
    """

    def _raise(**_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(overture_maps_scraper.fs, 'S3FileSystem', _raise)


def test_available_releases_are_sorted_newest_first(
    listing_calls: list[int],
) -> None:
    """Directory entries come back newest first, non-directories dropped."""
    assert OvertureMapsScraper.fetch_available_releases() == [
        '2026-07-22.0',
        '2026-06-17.0',
    ]
    assert listing_calls[0] == 1


def test_available_releases_are_cached(listing_calls: list[int]) -> None:
    """A second call reuses the cached listing."""
    OvertureMapsScraper.fetch_available_releases()
    OvertureMapsScraper.fetch_available_releases()

    assert listing_calls[0] == 1


def test_refresh_relists_the_bucket(listing_calls: list[int]) -> None:
    """Passing refresh bypasses the cache."""
    OvertureMapsScraper.fetch_available_releases()
    OvertureMapsScraper.fetch_available_releases(refresh=True)

    assert listing_calls[0] == 2


def test_caller_gets_a_copy_of_the_cached_list(listing_calls: list[int]) -> None:
    """Mutating the returned list does not corrupt the cache."""
    first = OvertureMapsScraper.fetch_available_releases()
    first.append('2020-01-01.0')

    assert OvertureMapsScraper.fetch_available_releases() == BUCKET_RELEASES[::-1]
    assert listing_calls[0] == 1


def test_resolve_release_defaults_to_the_newest(listing_calls: list[int]) -> None:
    """An unspecified release resolves to the newest one in the bucket."""
    assert OvertureMapsScraper.resolve_release() == '2026-07-22.0'
    assert listing_calls[0] == 1


def test_resolve_release_accepts_an_available_release(
    listing_calls: list[int],
) -> None:
    """A release the bucket holds is returned unchanged."""
    assert OvertureMapsScraper.resolve_release('2026-06-17.0') == '2026-06-17.0'
    assert listing_calls[0] == 1


def test_resolve_release_rejects_a_retired_release(
    listing_calls: list[int],
) -> None:
    """A retired release raises, naming what is available."""
    with pytest.raises(ValueError, match='2026-04-15.0') as caught:
        OvertureMapsScraper.resolve_release('2026-04-15.0')

    message = str(caught.value)
    assert '2026-07-22.0' in message
    assert '2026-06-17.0' in message
    assert listing_calls[0] == 1


def test_retired_release_is_not_silently_replaced(
    listing_calls: list[int],
) -> None:
    """Rejection is the behavior, not substitution of the newest release."""
    with pytest.raises(ValueError):  # noqa: PT011
        OvertureMapsScraper.resolve_release('2026-04-15.0')

    assert listing_calls[0] == 1


def test_unlistable_bucket_falls_back_to_the_published_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the bucket cannot be listed, the published list is used."""
    _fail_listing(monkeypatch, OSError('no route to host'))
    monkeypatch.setattr(
        OvertureMapsScraper,
        'fetch_release_names',
        staticmethod(lambda *_args, **_kwargs: ['2026-04-15.0', '2026-03-18.0']),
    )

    assert OvertureMapsScraper.resolve_release() == '2026-04-15.0'


def test_unlistable_bucket_keeps_an_explicit_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit release survives a failed listing; it cannot be checked."""
    _fail_listing(monkeypatch, OSError('no route to host'))

    assert OvertureMapsScraper.resolve_release('2026-07-22.0') == '2026-07-22.0'


def test_no_release_anywhere_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither source usable, resolution fails loudly."""
    _fail_listing(monkeypatch, OSError('no route to host'))
    monkeypatch.setattr(
        OvertureMapsScraper,
        'fetch_release_names',
        staticmethod(lambda *_args, **_kwargs: []),
    )

    with pytest.raises(ValueError, match='No Overture release'):
        OvertureMapsScraper.resolve_release()


def test_footprint_scraper_defaults_to_an_available_release(
    listing_calls: list[int],
) -> None:
    """The constructor picks a release the bucket actually holds."""
    scraper = OvertureMapsFootprintScraper({'length': 'ft'})

    assert scraper.overture_release == '2026-07-22.0'
    assert listing_calls[0] == 1


def test_footprint_scraper_rejects_a_retired_release(
    listing_calls: list[int],
) -> None:
    """A retired release fails at construction, not mid-download."""
    with pytest.raises(ValueError, match='not available'):
        OvertureMapsFootprintScraper(
            {'length': 'ft', 'overtureRelease': '2026-04-15.0'}
        )

    assert listing_calls[0] == 1


@pytest.mark.live
def test_live_default_release_is_readable() -> None:
    """The bucket lists releases and the default resolves into that set.

    This is the drift alarm for the release layout. When it fails, check
    whether Overture changed the bucket path in ``S3_BASE_PATH`` or stopped
    partitioning releases as directories.
    """
    available = OvertureMapsScraper.fetch_available_releases(refresh=True)

    assert available
    assert OvertureMapsScraper.resolve_release() in available


@pytest.mark.live
def test_live_published_list_is_not_a_safe_default() -> None:
    """The published list still disagrees with the bucket.

    Recorded because this disagreement is the defect the resolution logic
    exists for: the published list named retired releases and trailed the
    bucket by months. If this ever passes, Overture fixed the release YAML and
    the fallback path matters less.
    """
    available = set(OvertureMapsScraper.fetch_available_releases(refresh=True))
    published = OvertureMapsScraper.fetch_release_names()

    assert published
    assert published[0] not in available
