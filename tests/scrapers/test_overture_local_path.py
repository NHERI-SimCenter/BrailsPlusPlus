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

"""Offline tests of reading Overture buildings from local GeoParquet files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from shapely.geometry import box

from brails.scrapers.overture_maps_scraper import overture_maps_scraper
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import (
    OvertureMapsScraper,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

BOX = (-122.46, 37.76, -122.40, 37.78)


def test_local_read_applies_the_bucket_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local read keeps the rows the bucket read keeps from the same file."""
    # Two rows overlap BOX and one lies east of it.
    cells = [
        box(-122.45, 37.77, -122.449, 37.771),
        box(-122.41, 37.765, -122.409, 37.766),
        box(-122.39, 37.77, -122.389, 37.771),
    ]
    table = pa.table(
        {
            'id': ['ov-0', 'ov-1', 'ov-2'],
            'geometry': [cell.wkb for cell in cells],
            'bbox': [
                dict(zip(('xmin', 'ymin', 'xmax', 'ymax'), cell.bounds))
                for cell in cells
            ],
        }
    )
    folder = tmp_path / 'type=building'
    folder.mkdir()
    pq.write_table(table, folder / 'part-0.parquet')
    # The bucket read, served from the same folder on the local filesystem.
    monkeypatch.setattr(
        overture_maps_scraper.fs, 'S3FileSystem', lambda **_: pafs.LocalFileSystem()
    )
    monkeypatch.setattr(
        OvertureMapsScraper, '_dataset_path', staticmethod(lambda *_: str(folder))
    )

    bucket = OvertureMapsScraper._read_table('2026-10-21.0', 'building', bbox=BOX)
    local = OvertureMapsScraper._read_table(
        '2026-10-21.0', 'building', bbox=BOX, paths=[str(folder / 'part-0.parquet')]
    )

    assert bucket['id'].to_pylist() == ['ov-0', 'ov-1']
    assert local.equals(bucket)
