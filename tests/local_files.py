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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.


"""Local files in the forms of TIGER/Line and Overture, for the tests to read."""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry

# The column types of Overture's building rows that the tests fill.
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


def zip_shapefile(
    frame: gpd.GeoDataFrame, folder: Path, name: str = 'layer'
) -> bytes:
    """
    Return a frame as a zipped shapefile, the form of a TIGER/Line file.

    Args:
        frame (geopandas.GeoDataFrame):
            The features.
        folder (Path):
            A folder for the shapefile's parts, created if it does not exist.
        name (str, optional):
            The shapefile's name inside the zip. Defaults to ``'layer'``.

    Returns:
        bytes:
            The zip, holding the parts with a ``.cpg`` file that names UTF-8.
    """
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_file(folder / f'{name}.shp', encoding='UTF-8')
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as parts:
        for part in sorted(folder.glob(f'{name}.*')):
            parts.write(part, part.name)
    return archive.getvalue()


def overture_table(
    ids: list[str],
    geometries: list[BaseGeometry],
    *,
    heights: list[float] | None = None,
    names: list[dict | None] | None = None,
    sources: list[list[dict] | None] | None = None,
    overture_type: str = 'building',
    metadata: dict[bytes, bytes] | None = None,
) -> pa.Table:
    """
    Return rows as a local file of Overture building rows holds them.

    Args:
        ids (list[str]):
            The Overture ids.
        geometries (list[BaseGeometry]):
            The footprints, in longitude and latitude.
        heights (list[float], optional):
            The heights in meters. Defaults to 4 m each.
        names (list[dict | None], optional):
            The ``names`` of each row. Defaults to none.
        sources (list[list[dict] | None], optional):
            The ``sources`` of each row. Defaults to none.
        overture_type (str, optional):
            The value of the ``type`` column. Defaults to ``'building'``.
        metadata (dict[bytes, bytes], optional):
            Further schema metadata, such as the release a file records.

    Returns:
        pa.Table:
            The rows, with the ``theme`` and ``type`` columns and GeoParquet
            metadata.
    """
    count = len(ids)
    table = pa.table(
        {
            'id': pa.array(ids),
            'names': pa.array(names or [None] * count, NAMES_TYPE),
            'sources': pa.array(sources or [None] * count, SOURCES_TYPE),
            'height': pa.array(heights or [4.0] * count, pa.float64()),
            'class': pa.array(['house'] * count),
            'has_parts': pa.array([False] * count),
            'geometry': pa.array([g.wkb for g in geometries], pa.binary()),
            'bbox': pa.array(
                [
                    dict(zip(('xmin', 'ymin', 'xmax', 'ymax'), g.bounds))
                    for g in geometries
                ],
                BBOX_TYPE,
            ),
            'theme': pa.array(['buildings'] * count),
            'type': pa.array([overture_type] * count),
        }
    )
    geo = {
        'version': '1.1.0',
        'primary_column': 'geometry',
        'columns': {'geometry': {'encoding': 'WKB', 'geometry_types': ['Polygon']}},
    }
    return table.replace_schema_metadata(
        {
            b'schema_versions': b'{"overture-schema-system": "2.0.0"}',
            b'geo': json.dumps(geo).encode(),
            **(metadata or {}),
        }
    )
