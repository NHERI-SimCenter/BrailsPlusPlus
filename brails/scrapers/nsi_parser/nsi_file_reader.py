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
Build assets from raw NSI records, store the records as parquet tables and read
them back as asset inventories.

.. autosummary::

      NSIFileReader
      write_raw_records
      record_to_asset
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely

from brails.scrapers.nsi_parser.nsi_conventions import (
    ASSET_TYPE,
    NSI_RAW_SCHEMA,
    to_simcenter,
)
from brails.types.asset_inventory import Asset, AssetInventory

if TYPE_CHECKING:
    from collections.abc import Iterable

    from brails.types.region_boundary import RegionBoundary

logger = logging.getLogger(__name__)

# The columns a table must carry.
REQUIRED_COLUMNS = ('fd_id', 'x', 'y')

# The largest absolute longitude and latitude, the ranges Asset accepts.
_MAX_X = 180
_MAX_Y = 90

# The Python values each declared type accepts, and the name an error gives
# them. A bool is an int in Python but is refused for every type.
_ACCEPTED = {
    pa.int64(): ((int,), 'an integer'),
    pa.float64(): ((int, float), 'a number'),
    pa.string(): ((str,), 'a string'),
}

# The number of rows converted to Python objects at once, which bounds
# transient memory.
_BATCH_ROWS = 100_000

# The number of repeated ids or offending rows an error names before
# counting the rest.
_PREVIEW = 10


def record_to_asset(
    record: dict[str, Any],
    coordinates: list[list[float]],
    *,
    simcenter_format: bool,
    omit_nulls: bool,
) -> Asset:
    """
    Build the point asset of one NSI record, keyed by its ``fd_id``.

    Args:
        record (dict[str, Any]):
            The NSI properties of the record. It is not modified.
        coordinates (list[list[float]]):
            The point, as ``[[x, y]]``.
        simcenter_format (bool):
            When True, the features pass through
            :func:`~brails.scrapers.nsi_parser.nsi_conventions.to_simcenter`.
        omit_nulls (bool):
            When True, a field holding None is left out of the features.

    Returns:
        Asset:
            The asset, with the record's fields and ``type`` as features.
    """
    if omit_nulls:
        features = {
            name: value for name, value in record.items() if value is not None
        }
    else:
        features = dict(record)
    if simcenter_format:
        features = to_simcenter(features)
    features['type'] = ASSET_TYPE
    return Asset(record['fd_id'], coordinates, features)


def write_raw_records(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """
    Write raw NSI records to a parquet table under their NSI names.

    The table has one row per record, in the order given. It has one column
    per field of
    :data:`~brails.scrapers.nsi_parser.nsi_conventions.NSI_RAW_SCHEMA`,
    stored under its declared type, and then one per field outside the
    schema, in the order first seen and with its type inferred. A field a
    record lacks or holds as None is written as null.

    Args:
        records (Iterable[dict[str, Any]]):
            The NSI properties of each record, as the API serves them.
            Each must carry ``fd_id``, ``x`` and ``y``.
        path (str or Path):
            The parquet file to write. Its folder must exist.

    Raises:
        ValueError:
            If a record lacks ``fd_id``, ``x`` or ``y``, or if two records
            share an ``fd_id``.
        TypeError:
            If a value does not fit the declared type of its field, or if a
            field outside the schema mixes types.
    """
    rows = list(records)
    for row in rows:
        missing = [name for name in REQUIRED_COLUMNS if row.get(name) is None]
        if missing:
            message = f'An NSI record lacks {", ".join(missing)}: {row}'
            raise ValueError(message)

    counts = Counter(row['fd_id'] for row in rows)
    repeated = sorted(fd_id for fd_id, count in counts.items() if count > 1)
    if repeated:
        raise ValueError(_repeated_ids_message(repeated, 'the records to write'))

    declared = {field.name: field.type for field in NSI_RAW_SCHEMA}
    undeclared: dict[str, None] = {}
    for row in rows:
        for name, value in row.items():
            field_type = declared.get(name)
            if field_type is None:
                undeclared[name] = None
            elif value is not None:
                _check_value(name, value, field_type, row['fd_id'])

    fields = list(NSI_RAW_SCHEMA)
    for name in undeclared:
        try:
            inferred = pa.array([row.get(name) for row in rows]).type
        except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
            message = (
                f'The NSI field {name}, which NSI_RAW_SCHEMA does not declare, '
                f'mixes types: {error}'
            )
            raise TypeError(message) from error
        fields.append(pa.field(name, inferred))

    table = pa.Table.from_pylist(rows, schema=pa.schema(fields))
    pq.write_table(table, Path(path))


def _check_value(
    name: str, value: object, field_type: pa.DataType, fd_id: object
) -> None:
    """
    Refuse a value that its field's declared type would not store exactly.

    Args:
        name (str):
            The field name.
        value (object):
            The served value, not None.
        field_type (pa.DataType):
            The type the schema declares for the field.
        fd_id (object):
            The id of the record, for the message.

    Raises:
        TypeError:
            If the value is a bool or is not of a type the field accepts.
    """
    accepted, description = _ACCEPTED[field_type]
    if isinstance(value, bool) or not isinstance(value, accepted):
        message = (
            f'The NSI field {name} of record {fd_id} holds {value!r}, which is '
            f'not {description}.'
        )
        raise TypeError(message)


def _repeated_ids_message(repeated: list[Any], where: str) -> str:
    """
    Describe the repeated ids found in a set of records.

    Args:
        repeated (list[Any]):
            The repeated ids, sorted.
        where (str):
            What holds the records, such as a file path.

    Returns:
        str:
            The error message, naming the first ids.
    """
    shown = ', '.join(str(fd_id) for fd_id in repeated[:_PREVIEW])
    more = len(repeated) - _PREVIEW
    suffix = f' and {more} more' if more > 0 else ''
    return (
        f'{len(repeated)} fd_id value(s) occur more than once in {where}, so '
        f'records would be lost: {shown}{suffix}.'
    )


def _rows_message(rows: np.ndarray) -> str:
    """
    Name the first of a set of row positions.

    Args:
        rows (np.ndarray):
            The zero-based row positions.

    Returns:
        str:
            The first positions, then a count of the rest.
    """
    shown = ', '.join(str(row) for row in rows[:_PREVIEW])
    more = len(rows) - _PREVIEW
    return f'{shown} and {more} more' if more > 0 else shown


class NSIFileReader:
    """
    Read a parquet table of NSI records as a point inventory keyed by fd_id.

    The table may be one written by :meth:`NSI_Parser.get_raw_data
    <brails.scrapers.nsi_parser.nsi_parser.NSI_Parser.get_raw_data>` with
    ``raw_parquet_path``. BRAILS is compatible with the records of the NSI
    2026 Base only.

    To import the :class:`NSIFileReader` class, use:

    .. code-block:: python

        from brails.scrapers import NSIFileReader
    """

    def read(
        self,
        path: str | Path,
        *,
        region: RegionBoundary | None = None,
        simcenter_format: bool = False,
    ) -> AssetInventory:
        """
        Read the table at ``path``, optionally limited to ``region``.

        Args:
            path (str or Path):
                The parquet file.
            region (RegionBoundary, optional):
                When given, only the points strictly inside the region's
                boundary polygon are kept. The test uses each row's ``x``
                and ``y``, which can differ by up to 5e-7 degree from the
                served point geometry, so a point that close to the boundary
                can fall on the other side. Defaults to None, which keeps
                every row.
            simcenter_format (bool):
                When True, each record passes through
                :func:`brails.scrapers.nsi_parser.nsi_conventions.to_simcenter`.
                Defaults to False, which keeps the NSI names and values.

        Returns:
            AssetInventory:
                One point asset per kept row, inserted in ascending
                ``fd_id``, with coordinates ``[[x, y]]`` and the row's
                non-null cells and ``type`` as features.

        Raises:
            ValueError:
                If the table lacks ``fd_id``, ``x`` or ``y``, if any of
                them is null in a row, if a coordinate is out of range, or
                if an ``fd_id`` occurs more than once.
        """
        table = pq.read_table(Path(path))
        self._check_table(table, path)

        if region is not None:
            bpoly, _, _ = region.get_boundary()
            inside = shapely.contains_xy(
                bpoly,
                table['x'].to_numpy(),
                table['y'].to_numpy(),
            )
            table = table.filter(pa.array(np.asarray(inside, dtype=bool)))

        table = table.sort_by('fd_id')

        inventory = AssetInventory()
        for batch in table.to_batches(max_chunksize=_BATCH_ROWS):
            for row in batch.to_pylist():
                asset = record_to_asset(
                    row,
                    [[row['x'], row['y']]],
                    simcenter_format=simcenter_format,
                    omit_nulls=True,
                )
                inventory.add_asset(row['fd_id'], asset)

        logger.info('Read %d NSI records from %s.', len(inventory.inventory), path)
        return inventory

    @staticmethod
    def _check_table(table: pa.Table, path: str | Path) -> None:
        """
        Refuse a table that cannot be keyed and placed without loss.

        Args:
            table (pa.Table):
                The table as read from the file.
            path (str or Path):
                The file, for the messages.

        Raises:
            ValueError:
                As documented for
                :meth:`~brails.scrapers.nsi_parser.nsi_file_reader.NSIFileReader.read`.
        """
        missing = [
            name for name in REQUIRED_COLUMNS if name not in table.schema.names
        ]
        if missing:
            message = (
                f'The NSI table {path} lacks the column(s) {", ".join(missing)}.'
            )
            raise ValueError(message)

        for name in REQUIRED_COLUMNS:
            column = table[name]
            if column.null_count:
                rows = pc.indices_nonzero(pc.is_null(column)).to_numpy()
                message = (
                    f'The NSI table {path} has {column.null_count} null value(s) '
                    f'in {name}, at row(s) {_rows_message(rows)}.'
                )
                raise ValueError(message)

        x = table['x'].to_numpy()
        y = table['y'].to_numpy()
        # Written so that a NaN fails the test, as it fails Asset's own check.
        valid = (np.abs(x) <= _MAX_X) & (np.abs(y) <= _MAX_Y)
        if not valid.all():
            rows = np.flatnonzero(~valid)
            message = (
                f'The NSI table {path} has {len(rows)} row(s) with x outside '
                f'-180 to 180 or y outside -90 to 90: {_rows_message(rows)}.'
            )
            raise ValueError(message)

        if pc.count_distinct(table['fd_id']).as_py() != table.num_rows:
            counts = pc.value_counts(table['fd_id'])
            repeated = sorted(
                entry['values']
                for entry in counts.to_pylist()
                if entry['counts'] > 1
            )
            raise ValueError(_repeated_ids_message(repeated, str(path)))
