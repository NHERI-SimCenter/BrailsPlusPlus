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

import json
import urllib.request
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Polygon, box

from brails.scrapers.overture_maps_scraper import overture_maps_scraper
from brails.scrapers.overture_maps_scraper.overture_maps_footprint_scraper import (
    BRAILS_METADATA_KEY,
    OvertureMapsFootprintScraper,
)
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import (
    OvertureMapsScraper,
)
from tests.scrapers.conftest import Region

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from brails.types.asset_inventory import AssetInventory

BOX = (-122.46, 37.76, -122.40, 37.78)
RELEASE = '2026-10-21.0'

# The south-east half of BOX. Its bounding box is BOX.
TRIANGLE = Polygon([(-122.46, 37.76), (-122.40, 37.76), (-122.40, 37.78)])

# Four houses of 12 by 10 m and a 4 by 4 m shed inside TRIANGLE, and a house
# inside BOX but north-west of TRIANGLE.
HOUSE_LON, HOUSE_LAT = 0.0001366, 0.0000899
SHED_LON, SHED_LAT = 0.0000455, 0.000036
FOOTPRINTS = [
    *(
        box(lon, 37.761, lon + HOUSE_LON, 37.761 + HOUSE_LAT)
        for lon in (-122.45, -122.44, -122.43, -122.42)
    ),
    box(-122.41, 37.761, -122.41 + SHED_LON, 37.761 + SHED_LAT),
    box(-122.455, 37.775, -122.455 + HOUSE_LON, 37.775 + HOUSE_LAT),
]
IDS = [f'ov-{index}' for index in range(len(FOOTPRINTS))]
INSIDE = IDS[:5]

NAMES_TYPE = pa.struct(
    [('primary', pa.string()), ('common', pa.map_(pa.string(), pa.string()))]
)
SOURCES_TYPE = pa.list_(
    pa.struct(
        [
            ('property', pa.string()),
            ('dataset', pa.string()),
            ('record_id', pa.string()),
            ('confidence', pa.float64()),
        ]
    )
)
BBOX_TYPE = pa.struct(
    [(key, pa.float64()) for key in ('xmin', 'xmax', 'ymin', 'ymax')]
)


def building_table(
    indices: list[int], overture_type: str = 'building', release: str | None = None
) -> pa.Table:
    """
    Return fixture rows as a local file of Overture rows holds them.

    The rows carry the ``theme`` and ``type`` columns.

    Args:
        indices (list[int]):
            Positions in ``FOOTPRINTS`` of the rows to return.
        overture_type (str, optional):
            The value of the ``type`` column. Defaults to ``'building'``.
        release (str, optional):
            A release to record under ``BRAILS_METADATA_KEY``, as
            ``download`` does. Defaults to none.

    Returns:
        pa.Table:
            The rows, with GeoParquet metadata.
    """
    count = len(indices)
    columns = {
        'id': pa.array([IDS[index] for index in indices]),
        'names': pa.array(
            [{'primary': 'Hall', 'common': [('en', 'Hall')]}, *[None] * (count - 1)],
            NAMES_TYPE,
        ),
        'sources': pa.array(
            [
                [
                    {
                        'property': '',
                        'dataset': 'OpenStreetMap',
                        'record_id': f'w{index}',
                        'confidence': None,
                    }
                ]
                for index in indices
            ],
            SOURCES_TYPE,
        ),
        'height': pa.array([4.0 + index for index in indices], pa.float64()),
        'class': pa.array(['house'] * count),
        'has_parts': pa.array([False] * count),
        'geometry': pa.array(
            [FOOTPRINTS[index].wkb for index in indices], pa.binary()
        ),
        'bbox': pa.array(
            [
                dict(zip(('xmin', 'ymin', 'xmax', 'ymax'), FOOTPRINTS[index].bounds))
                for index in indices
            ],
            BBOX_TYPE,
        ),
        'theme': pa.array(['buildings'] * count),
        'type': pa.array([overture_type] * count),
    }
    metadata = {
        b'schema_versions': b'{"overture-schema-system": "2.0.0"}',
        b'geo': json.dumps(
            {
                'version': '1.1.0',
                'primary_column': 'geometry',
                'columns': {
                    'geometry': {'encoding': 'WKB', 'geometry_types': ['Polygon']}
                },
            }
        ).encode(),
    }
    if release is not None:
        metadata[BRAILS_METADATA_KEY] = json.dumps(
            {'overture_release': release}
        ).encode()
    return pa.table(columns).replace_schema_metadata(metadata)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Make any attempt to reach the S3 bucket or the release list fail.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Fixture used to replace the S3 filesystem and ``urlopen``.
    """

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError('network access')

    monkeypatch.setattr(overture_maps_scraper.fs, 'S3FileSystem', refuse)
    monkeypatch.setattr(urllib.request, 'urlopen', refuse)


@pytest.fixture
def local_file(tmp_path: Path) -> str:
    """
    Write the six fixture buildings to a local file.

    Args:
        tmp_path (Path):
            The test's temporary folder.

    Returns:
        str:
            The path of the file, which records no release.
    """
    path = tmp_path / 'local_building.parquet'
    pq.write_table(building_table(list(range(len(FOOTPRINTS)))), path)
    return str(path)


@pytest.fixture
def bucket(mocker: MockerFixture, local_file: str) -> type:
    """
    Serve the local file's rows as the S3 bucket read gives them.

    Args:
        mocker (MockerFixture):
            Fixture used to patch the release lookup and the read.
        local_file (str):
            The fixture file whose rows the bucket serves.

    Returns:
        type:
            The scraper class, patched.
    """
    mocker.patch.object(
        OvertureMapsFootprintScraper, 'resolve_release', return_value=RELEASE
    )
    rows = pq.read_table(local_file).drop_columns(['theme', 'type'])
    read_table = OvertureMapsScraper._read_table

    def read(
        release: str, overture_type: str, bbox: Any = None, paths: Any = None
    ) -> pa.Table:
        if paths is None:
            return rows
        return read_table(release, overture_type, bbox=bbox, paths=paths)

    mocker.patch.object(
        OvertureMapsFootprintScraper, '_read_table', side_effect=read
    )
    return OvertureMapsFootprintScraper


def contents(inventory: AssetInventory) -> dict[Any, tuple[Any, dict]]:
    """
    Return each asset's coordinates and features by key.

    Args:
        inventory (AssetInventory):
            The inventory to describe.

    Returns:
        dict:
            The coordinates and features of each asset, by key.
    """
    return {
        key: (asset.coordinates, asset.features)
        for key, asset in inventory.inventory.items()
    }


def overture_ids(inventory: AssetInventory) -> list[str]:
    """
    Return the Overture ids of an inventory's assets, sorted.

    Args:
        inventory (AssetInventory):
            The inventory to describe.

    Returns:
        list[str]:
            The ``overture_id`` of every asset, sorted.
    """
    return sorted(
        asset.features['overture_id'] for asset in inventory.inventory.values()
    )


@pytest.mark.parametrize('id_scheme', ['index', 'geohash'])
def test_local_read_equals_the_bucket_read(
    bucket: type, local_file: str, id_scheme: str
) -> None:
    """The local file and the bucket serving its rows give one inventory."""
    from_bucket = bucket({'id_scheme': id_scheme}).get_footprints(Region(TRIANGLE))
    from_file = bucket(
        {
            'id_scheme': id_scheme,
            'local_path': local_file,
            'overtureRelease': RELEASE,
        }
    ).get_footprints(Region(TRIANGLE))

    assert contents(from_file) == contents(from_bucket)
    assert overture_ids(from_file) == (
        INSIDE if id_scheme == 'index' else INSIDE[:4]
    )


def test_local_read_needs_no_network(
    local_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a local path neither the bucket nor the release list is contacted."""
    calls = []

    def record(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))
        raise AssertionError('network access')

    monkeypatch.setattr(overture_maps_scraper.fs, 'S3FileSystem', record)
    monkeypatch.setattr(urllib.request, 'urlopen', record)
    scraper = OvertureMapsFootprintScraper(
        {'local_path': local_file, 'overtureRelease': RELEASE}
    )

    assert overture_ids(scraper.get_footprints(Region(TRIANGLE))) == INSIDE
    assert scraper.overture_release == RELEASE
    assert calls == []


def test_release_comes_from_the_input_or_the_file(
    local_file: str, tmp_path: Path
) -> None:
    """A local read needs a release for every file, takes the files', and refuses a conflict."""
    with pytest.raises(ValueError, match='No Overture release'):
        OvertureMapsFootprintScraper({'local_path': local_file})

    recorded = tmp_path / 'recorded.parquet'
    pq.write_table(building_table([0, 1], release='2026-09-24.0'), recorded)
    scraper = OvertureMapsFootprintScraper({'local_path': str(recorded)})
    assert scraper.overture_release == '2026-09-24.0'

    with pytest.raises(ValueError, match='more than one Overture release'):
        OvertureMapsFootprintScraper(
            {'local_path': str(recorded), 'overtureRelease': RELEASE}
        )

    # A file that records no release does not take another file's.
    with pytest.raises(ValueError, match='No Overture release is known for .*local'):
        OvertureMapsFootprintScraper({'local_path': [str(recorded), local_file]})


def test_a_part_file_without_a_type_column_is_refused(tmp_path: Path) -> None:
    """A part file as the bucket stores it, with building_id and no type, raises."""
    parts = tmp_path / 'part-0.parquet'
    table = building_table([0, 1]).drop_columns(['theme', 'type'])
    pq.write_table(table.append_column('building_id', pa.array(IDS[2:4])), parts)

    with pytest.raises(ValueError, match='building_part'):
        OvertureMapsFootprintScraper(
            {'local_path': str(parts), 'overtureRelease': RELEASE}
        )


def test_a_file_of_building_parts_is_refused(tmp_path: Path) -> None:
    """Rows whose type is building_part raise instead of becoming assets."""
    parts = tmp_path / 'local_building_part.parquet'
    pq.write_table(building_table([0, 1], overture_type='building_part'), parts)
    scraper = OvertureMapsFootprintScraper(
        {'local_path': str(parts), 'overtureRelease': RELEASE}
    )

    with pytest.raises(ValueError, match='building_part'):
        scraper.get_footprints(Region(TRIANGLE))


def test_a_building_in_two_files_is_kept_once(tmp_path: Path) -> None:
    """A building that two files share becomes one asset."""
    paths = [str(tmp_path / 'west.parquet'), str(tmp_path / 'east.parquet')]
    pq.write_table(building_table([0, 1, 2]), paths[0])
    pq.write_table(building_table([2, 3, 4]), paths[1])
    scraper = OvertureMapsFootprintScraper(
        {'local_path': paths, 'overtureRelease': RELEASE}
    )

    assert overture_ids(scraper.get_footprints(Region(TRIANGLE))) == INSIDE


def test_download_reads_back_as_the_bucket_inventory(
    bucket: type, tmp_path: Path
) -> None:
    """A file written by download gives the bucket's inventory and its release."""
    path = str(tmp_path / 'region.parquet')
    scraper = bucket({'id_scheme': 'geohash'})
    assert scraper.download(Region(TRIANGLE), path) == len(INSIDE)
    assert scraper.inventory.inventory == {}

    from_file = bucket({'id_scheme': 'geohash', 'local_path': path})

    assert from_file.overture_release == RELEASE
    assert contents(from_file.get_footprints(Region(TRIANGLE))) == contents(
        scraper.get_footprints(Region(TRIANGLE))
    )


def test_download_writes_the_raw_rows_as_geoparquet(
    bucket: type, local_file: str, tmp_path: Path
) -> None:
    """The file keeps every column with its Arrow type and declares GeoParquet 1.1.0."""
    path = str(tmp_path / 'region.parquet')
    bucket({}).download(Region(TRIANGLE), path)
    written = pq.read_table(path)
    source = pq.read_table(local_file)

    assert written.schema.remove_metadata().equals(source.schema.remove_metadata())
    assert written['id'].to_pylist() == INSIDE
    assert set(written['theme'].to_pylist()) == {'buildings'}
    assert set(written['type'].to_pylist()) == {'building'}
    metadata = written.schema.metadata
    assert metadata[b'schema_versions'] == source.schema.metadata[b'schema_versions']
    geo = json.loads(metadata[b'geo'])
    assert geo['version'] == '1.1.0'
    assert geo['columns']['geometry']['encoding'] == 'WKB'
    assert geo['columns']['geometry']['covering']['bbox']['xmin'] == ['bbox', 'xmin']
    entry = json.loads(metadata[BRAILS_METADATA_KEY])
    assert entry == {
        'overture_release': RELEASE,
        'source': 's3://overturemaps-us-west-2/release/2026-10-21.0/'
        'theme=buildings/type=building/',
    }
    frame = gpd.read_parquet(path)
    assert len(frame) == len(INSIDE)
    assert frame.crs.to_string() == 'OGC:CRS84'


def test_an_empty_region_gives_an_empty_inventory_and_file(
    bucket: type, local_file: str, tmp_path: Path
) -> None:
    """A region holding no building gives empty inventories and an empty export."""
    empty = Region(box(-122.30, 37.70, -122.29, 37.71))
    assert bucket({}).get_footprints(empty).inventory == {}
    local = bucket({'local_path': local_file, 'overtureRelease': RELEASE})
    assert local.get_footprints(empty).inventory == {}

    path = str(tmp_path / 'empty.parquet')
    assert local.download(empty, path) == 0
    entry = json.loads(pq.read_schema(path).metadata[BRAILS_METADATA_KEY])
    assert entry['source'] == ['local_building.parquet']
    assert bucket({'local_path': path}).get_footprints(empty).inventory == {}


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    """A local path that does not exist raises at construction."""
    with pytest.raises(FileNotFoundError):
        OvertureMapsFootprintScraper(
            {
                'local_path': str(tmp_path / 'absent.parquet'),
                'overtureRelease': RELEASE,
            }
        )


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
