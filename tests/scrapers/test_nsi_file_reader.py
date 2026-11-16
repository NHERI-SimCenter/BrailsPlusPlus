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


"""Tests for the NSI parquet reader and writer, and the parser's raw-pull file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from brails.scrapers import NSIFileReader
from brails.scrapers.nsi_parser.nsi_conventions import NSI_RAW_SCHEMA, to_simcenter
from brails.scrapers.nsi_parser.nsi_file_reader import write_raw_records
from brails.scrapers.nsi_parser.nsi_parser import NSI_Parser
from tests.scrapers.conftest import Region, serve_features
from tests.scrapers.data.nsi_fixture_generator import FIXTURE_PATH as API_SAMPLE
from tests.scrapers.data.nsi_pooled_fixture_generator import (
    FIXTURE_PATH as NSI_SAMPLE,
)

if TYPE_CHECKING:
    from brails.types.asset_inventory import AssetInventory

# The fields the NSI API serves, and the column the sample adds.
NSI_FIELDS = frozenset(NSI_RAW_SCHEMA.names)
SAMPLE_COLUMNS = NSI_FIELDS | {'query_fips'}


def snapshot(inventory: AssetInventory) -> list[tuple]:
    """
    Reduce an inventory to its ids, coordinates and features, in order.

    Args:
        inventory (AssetInventory):
            The inventory.

    Returns:
        list[tuple]:
            One ``(id, coordinates, features)`` per asset.
    """
    return [
        (asset_id, asset.coordinates, asset.features)
        for asset_id, asset in inventory.inventory.items()
    ]


def typed(features: dict[str, Any]) -> dict[str, tuple[type, Any]]:
    """
    Pair each feature value with its type, so that 5 and 5.0 differ.

    Args:
        features (dict[str, Any]):
            The features.

    Returns:
        dict[str, tuple[type, Any]]:
            Each value as ``(type, value)``.
    """
    return {name: (type(value), value) for name, value in features.items()}


def write_table(path: Path, columns: dict[str, list]) -> Path:
    """
    Write a small parquet table.

    Args:
        path (Path):
            The file to write.
        columns (dict[str, list]):
            Column values by name.

    Returns:
        Path:
            The path written.
    """
    pq.write_table(pa.table(columns), path)
    return path


# ---------------------------------------------------------------------
# the Bay Area sample
# ---------------------------------------------------------------------


class TestBayAreaSample:
    """The reader on NSI records sampled from several Bay Area counties."""

    def test_the_sample_covers_the_cases_it_was_cut_for(self) -> None:
        """Several counties, both point sources, and stacked coordinates."""
        frame = pq.read_table(NSI_SAMPLE).to_pandas()

        assert len(frame) == 200
        assert set(frame.columns) == SAMPLE_COLUMNS
        assert frame['query_fips'].nunique() > 1
        assert set(frame['source']) == {'P', 'H'}
        assert frame.duplicated(['x', 'y'], keep=False).sum() > 0

    def test_the_declared_types_are_the_served_types(self) -> None:
        """Each field of the sample has the type the schema declares."""
        schema = pq.read_schema(NSI_SAMPLE)

        for field in NSI_RAW_SCHEMA:
            stored = schema.field(field.name).type
            if pa.types.is_large_string(stored):
                stored = pa.string()
            assert stored == field.type, field.name

    def test_every_field_reaches_the_features(self) -> None:
        """Each asset carries every non-null cell of its row, under its NSI name."""
        records = pq.read_table(NSI_SAMPLE).to_pylist()

        inventory = NSIFileReader().read(NSI_SAMPLE)

        assert inventory.get_all_asset_features() == SAMPLE_COLUMNS | {'type'}
        assert len(inventory.inventory) == len(records)
        for record in records:
            _, features = inventory.get_asset_features(record['fd_id'])
            expected = {k: v for k, v in record.items() if v is not None}
            assert features == {**expected, 'type': 'building'}

    def test_assets_are_points_keyed_by_ascending_fd_id(self) -> None:
        """The key is fd_id and the point is the record's x and y."""
        records = pq.read_table(NSI_SAMPLE).to_pylist()

        inventory = NSIFileReader().read(NSI_SAMPLE)

        ids = inventory.get_asset_ids()
        assert ids == sorted(record['fd_id'] for record in records)
        for record in records:
            _, coordinates = inventory.get_asset_coordinates(record['fd_id'])
            assert coordinates == [[record['x'], record['y']]]

    def test_the_result_does_not_depend_on_row_order(self, tmp_path: Path) -> None:
        """Rows in another order give the identical inventory."""
        table = pq.read_table(NSI_SAMPLE)
        order = list(range(1, table.num_rows, 2))[::-1] + list(
            range(0, table.num_rows, 2)
        )
        shuffled = tmp_path / 'shuffled.parquet'
        pq.write_table(table.take(order), shuffled)

        original = NSIFileReader().read(NSI_SAMPLE)
        reordered = NSIFileReader().read(shuffled)

        assert snapshot(reordered) == snapshot(original)

    def test_simcenter_format_converts_each_record(self) -> None:
        """With the flag on, each record's non-null cells go through the conversion."""
        records = pq.read_table(NSI_SAMPLE).to_pylist()

        inventory = NSIFileReader().read(NSI_SAMPLE, simcenter_format=True)

        for record in records:
            _, features = inventory.get_asset_features(record['fd_id'])
            cells = {k: v for k, v in record.items() if v is not None}
            assert features == {**to_simcenter(cells), 'type': 'building'}


# ---------------------------------------------------------------------
# the reader's rules
# ---------------------------------------------------------------------


class TestReaderRules:
    """Null cells, the region predicate, and the tables the reader refuses."""

    def test_a_null_cell_is_omitted(self, tmp_path: Path) -> None:
        """A null is a gap, not a feature with a placeholder value."""
        path = write_table(
            tmp_path / 'nulls.parquet',
            {
                'fd_id': [2, 1],
                'x': [0.5, 0.25],
                'y': [0.5, 0.25],
                'resunits': [None, 4],
                'usastrucid': ['', None],
            },
        )

        inventory = NSIFileReader().read(path)

        assert inventory.get_asset_features(1)[1] == {
            'fd_id': 1,
            'x': 0.25,
            'y': 0.25,
            'resunits': 4,
            'type': 'building',
        }
        # An empty string is a served value and is kept as it is.
        assert inventory.get_asset_features(2)[1]['usastrucid'] == ''
        assert 'resunits' not in inventory.get_asset_features(2)[1]

    def test_a_null_cell_is_omitted_before_conversion(self, tmp_path: Path) -> None:
        """A null occupancy code yields no occupancy feature at all."""
        path = write_table(
            tmp_path / 'null_occtype.parquet',
            {
                'fd_id': [1],
                'x': [0.5],
                'y': [0.5],
                'occtype': pa.array([None], pa.string()),
            },
        )

        _, features = (
            NSIFileReader().read(path, simcenter_format=True).get_asset_features(1)
        )

        assert 'OccupancyClass' not in features
        assert 'occtype' not in features

    def test_region_keeps_interior_points_only(self, tmp_path: Path) -> None:
        """A point on the boundary is dropped."""
        path = write_table(
            tmp_path / 'region.parquet',
            {'fd_id': [1, 2, 3], 'x': [0.5, 1.0, 2.0], 'y': [0.5, 0.5, 0.5]},
        )
        region = Region(box(0.0, 0.0, 1.0, 1.0))

        inventory = NSIFileReader().read(path, region=region)

        assert inventory.get_asset_ids() == [1]

    def test_a_repeated_fd_id_raises(self, tmp_path: Path) -> None:
        """Two rows under one id would leave one asset."""
        path = write_table(
            tmp_path / 'repeated.parquet',
            {'fd_id': [7, 7, 8], 'x': [0.1, 0.2, 0.3], 'y': [0.1, 0.2, 0.3]},
        )

        with pytest.raises(ValueError, match=r'1 fd_id value\(s\).*: 7\.'):
            NSIFileReader().read(path)

    def test_a_null_fd_id_raises(self, tmp_path: Path) -> None:
        """The error names the file and the row of a null fd_id."""
        path = write_table(
            tmp_path / 'null_id.parquet',
            {'fd_id': [1, None], 'x': [0.1, 0.2], 'y': [0.1, 0.2]},
        )

        with pytest.raises(ValueError, match=r'null_id\.parquet .* at row\(s\) 1\.'):
            NSIFileReader().read(path)

    def test_a_missing_coordinate_column_raises(self, tmp_path: Path) -> None:
        """A table without x or y cannot place its points."""
        path = write_table(tmp_path / 'no_y.parquet', {'fd_id': [1], 'x': [0.1]})

        with pytest.raises(ValueError, match=r'no_y\.parquet lacks the column'):
            NSIFileReader().read(path)

    def test_an_out_of_range_coordinate_raises(self, tmp_path: Path) -> None:
        """A point Asset would silently drop is refused instead."""
        path = write_table(
            tmp_path / 'range.parquet',
            {'fd_id': [1, 2], 'x': [0.1, 200.0], 'y': [0.1, float('nan')]},
        )

        with pytest.raises(ValueError, match=r'1 row\(s\) with x outside .*: 1\.'):
            NSIFileReader().read(path)


# ---------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------


class TestRawRecordWriter:
    """The declared types, and the records the writer refuses."""

    def test_each_value_reads_back_under_its_declared_type(
        self, tmp_path: Path
    ) -> None:
        """An integer stays an integer, and in a float field it becomes a float."""
        record = {'fd_id': 1, 'x': 0.5, 'y': 0.5, 'num_story': 2, 'static_bfe': 5}
        path = tmp_path / 'types.parquet'

        write_raw_records([record], path)
        _, features = NSIFileReader().read(path).get_asset_features(1)

        assert typed(features) == typed(
            {**record, 'static_bfe': 5.0, 'type': 'building'}
        )

    def test_a_field_outside_the_schema_is_kept(self, tmp_path: Path) -> None:
        """A field NSI adds later is written with its type inferred."""
        path = tmp_path / 'extra.parquet'

        write_raw_records([{'fd_id': 1, 'x': 0.5, 'y': 0.5, 'newfield': 3}], path)

        assert pq.read_table(path).column_names == [
            *NSI_RAW_SCHEMA.names,
            'newfield',
        ]
        assert pq.read_table(path)['newfield'].to_pylist() == [3]

    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('num_story', '2'),
            ('num_story', 2.5),
            ('num_story', True),
            ('occtype', 5),
        ],
    )
    def test_a_value_of_the_wrong_type_raises(
        self, field: str, value: object, tmp_path: Path
    ) -> None:
        """A value is refused rather than coerced, truncated or written as text."""
        record = {'fd_id': 1, 'x': 0.5, 'y': 0.5, field: value}

        with pytest.raises(TypeError, match=f'field {field} of record 1'):
            write_raw_records([record], tmp_path / 'bad.parquet')

    def test_a_repeated_fd_id_is_refused(self, tmp_path: Path) -> None:
        """The writer will not store a file the reader would refuse."""
        records = [
            {'fd_id': 7, 'x': 0.1, 'y': 0.1},
            {'fd_id': 7, 'x': 0.2, 'y': 0.2},
        ]

        with pytest.raises(ValueError, match='records would be lost: 7'):
            write_raw_records(records, tmp_path / 'bad.parquet')

    def test_a_record_without_coordinates_is_refused(self, tmp_path: Path) -> None:
        """The writer will not store a record the reader could not place."""
        record = {'fd_id': 1, 'x': 0.1}

        with pytest.raises(ValueError, match='lacks y'):
            write_raw_records([record], tmp_path / 'bad.parquet')


# ---------------------------------------------------------------------
# the parser's raw-pull file read back
# ---------------------------------------------------------------------

# Covers the stored API sample, an Inner Sunset block.
SAMPLE_REGION = Region(box(-122.47, 37.76, -122.46, 37.77))


class TestRawPullFile:
    """A pull written by the parser reads back as the inventory it returned."""

    def test_the_file_reads_back_as_the_pull(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Same ids and features up to the stated differences, and the points to 5e-7."""
        features = json.loads(Path(API_SAMPLE).read_text())['features']
        # One record served without a field, and one with a null.
        del features[3]['properties']['bldheight']
        features[4]['properties']['firmzone'] = None
        serve_features(monkeypatch, features)
        path = tmp_path / 'pull.parquet'

        pulled = NSI_Parser().get_raw_data(SAMPLE_REGION, raw_parquet_path=path)
        read = NSIFileReader().read(path)

        assert read.get_asset_ids() == sorted(pulled.get_asset_ids())
        assert len(read.inventory) == len(features)
        floats = {
            field.name
            for field in NSI_RAW_SCHEMA
            if pa.types.is_floating(field.type)
        }
        for asset_id in read.get_asset_ids():
            _, read_features = read.get_asset_features(asset_id)
            _, pulled_features = pulled.get_asset_features(asset_id)
            # a write and read-back drops nulls and stores float fields as floats
            stored = {
                name: float(value) if name in floats else value
                for name, value in pulled_features.items()
                if value is not None
            }
            assert typed(read_features) == typed(stored)
            ((x_read, y_read),) = read.get_asset_coordinates(asset_id)[1]
            ((x_pulled, y_pulled),) = pulled.get_asset_coordinates(asset_id)[1]
            assert abs(x_read - x_pulled) <= 5e-7
            assert abs(y_read - y_pulled) <= 5e-7
        assert (
            'bldheight'
            not in read.get_asset_features(features[3]['properties']['fd_id'])[1]
        )
        assert (
            'firmzone'
            not in read.get_asset_features(features[4]['properties']['fd_id'])[1]
        )

    def test_an_empty_pull_writes_a_readable_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An empty pull writes a file that reads as an empty inventory."""
        serve_features(monkeypatch, [])
        path = tmp_path / 'pull.parquet'

        NSI_Parser().get_raw_data(SAMPLE_REGION, raw_parquet_path=path)

        assert NSIFileReader().read(path).inventory == {}
