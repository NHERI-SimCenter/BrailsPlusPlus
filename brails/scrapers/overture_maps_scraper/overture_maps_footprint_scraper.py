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
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 08-14-2025

"""
Scraper of Overture Maps building footprints.

.. autosummary::

    OvertureMapsFootprintScraper
"""
import json
import os
from typing import Dict, Any, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely
from shapely import wkb

from brails.constants import DEFAULT_UNITS
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import AssetInventory
from brails.utils import GeoTools, UnitConverter
from brails.scrapers.footprint_scraper import FootprintScraper
from brails.scrapers.overture_maps_scraper.overture_maps_scraper import \
    OvertureMapsScraper, TYPE_THEME_MAP

# Define global variables:
OVERTURE_TYPE = 'building'
ASSET_TYPE = 'building'

# Schema metadata key under which ``download`` records the release and the
# source read:
BRAILS_METADATA_KEY = b'brails'

# Attributes with associated units and the units they are defined in within
# the dataset:
DIMENSIONAL_ATTR = {'height': 'm',
                    'min_height': 'm',
                    'roof_height': 'm'}


class OvertureMapsFootprintScraper(OvertureMapsScraper, FootprintScraper):
    """
    Scraper of the Overture Maps buildings in a region.

    The rows come from the Overture S3 bucket, or from local GeoParquet
    files given as ``'local_path'``, which :meth:`download` writes. Only
    rows of Overture's ``building`` type are read. The README beside this
    module states what is read and how the release is resolved.

    Attributes:
        units (dict):
            The units parsed from ``input_dict``. ``units['length']`` is the
            length unit of plan areas and heights.
        inventory (AssetInventory):
            The inventory :meth:`get_footprints` adds its assets to.
        overture_release (str):
            The Overture release read.
        local_paths (list of str or None):
            The local files read in place of the S3 bucket, or ``None``.

    Example:
        With ``region``, ``scraper_class`` and ``scraper`` as in the example
        of :meth:`get_footprints`, record the release an inventory was built
        from and name it in a later run, to read the same release for as
        long as the bucket holds it:

        >>> release = scraper.overture_release
        >>> later = scraper_class({'overtureRelease': release})

        Read Parquet files that record no release, naming the release they
        hold, here ``files_release``:

        >>> local = scraper_class(
        ...     {
        ...         'local_path': ['buildings_west.parquet',
        ...                        'buildings_east.parquet'],
        ...         'overtureRelease': files_release,
        ...     }
        ... )
        >>> inventory = local.get_footprints(region)
    """

    def __init__(self, input_dict: Dict[str, Any] = None):
        """
        Set the units and the source, and resolve the release.

        Args:
            input_dict (dict, optional):
                Scraper options, all optional:

                - ``'length'`` (str): the length unit, such as ``'ft'`` or
                  ``'m'``. Defaults to ``DEFAULT_UNITS['length']``.
                - ``'overtureRelease'`` (str): the release to read. Without
                  ``'local_path'`` it is checked by
                  :meth:`~brails.scrapers.overture_maps_scraper.overture_maps_scraper.OvertureMapsScraper.resolve_release`,
                  which also gives the default.
                - ``'id_scheme'`` (str) and ``'min_area'`` (float): described
                  in :class:`~brails.scrapers.footprint_scraper.FootprintScraper`.
                - ``'local_path'`` (str or sequence of str): GeoParquet files
                  of Overture ``building`` rows read in place of the S3
                  bucket, which is then not contacted. Each file must hold
                  Overture's ``bbox`` column, and a file without it fails
                  when it is read. The release is ``'overtureRelease'`` or
                  the one each file records. The README beside this module,
                  "Local files and their release", states the rules.

        Raises:
            FileNotFoundError:
                If a file of ``'local_path'`` does not exist.
            ValueError:
                If ``'length'`` or ``'weight'`` is not a supported unit, or
                ``'id_scheme'`` or ``'min_area'`` is invalid; if
                ``'local_path'`` names no file, a path that is not a Parquet
                file, or a file holding a ``building_id`` column; if no
                release is known for a file; if the files and
                ``'overtureRelease'`` name more than one release; or,
                without ``'local_path'``, as ``resolve_release`` raises it.
            OSError:
                Without ``'local_path'``, as ``resolve_release`` raises it.
        """
        input_dict = input_dict or {}
        
        # Parse units from input_dict or fall back to default units:
        self.units = UnitConverter.parse_units(input_dict, DEFAULT_UNITS)
        super().__init__(
            'OvertureMapsFootprintScraper',
            input_dict,
            self.units['length'],
        )
        self.inventory = AssetInventory()

        local_path = input_dict.get('local_path')
        if local_path is None:
            self.local_paths = None
            # Parse Overture Maps release from input dict or fall back to the
            # newest release the S3 bucket holds:
            self.overture_release = self.resolve_release(
                input_dict.get('overtureRelease')
            )
        else:
            if isinstance(local_path, (str, os.PathLike)):
                local_path = [local_path]
            self.local_paths = [os.fspath(path) for path in local_path]
            if not self.local_paths:
                raise ValueError("'local_path' names no file.")
            missing = [
                path for path in self.local_paths if not os.path.exists(path)
            ]
            if missing:
                raise FileNotFoundError(
                    f"Overture file not found: {', '.join(missing)}"
                )

            # The release declared in input_dict and the releases the files
            # record must name one release:
            declared = input_dict.get('overtureRelease')
            releases = {declared} if declared else set()
            unrecorded = []
            for path in self.local_paths:
                if not os.path.isfile(path):
                    raise ValueError(f"{path} is not a file.")
                try:
                    schema = pq.read_schema(path)
                    entry = (schema.metadata or {}).get(BRAILS_METADATA_KEY)
                    release = (
                        json.loads(entry).get('overture_release')
                        if entry else None
                    )
                except (pa.ArrowInvalid, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"{path} is not a GeoParquet file of Overture rows: "
                        f"{error}"
                    ) from error
                # Only Overture's building_part rows carry building_id,
                # including part files without a type column:
                if 'building_id' in schema.names:
                    raise ValueError(
                        f"{path} holds Overture building_part rows. Only "
                        f"'{OVERTURE_TYPE}' rows can be read."
                    )
                if release:
                    releases.add(release)
                else:
                    unrecorded.append(path)
            if not declared and unrecorded:
                raise ValueError(
                    "No Overture release is known for "
                    f"{', '.join(unrecorded)}: pass 'overtureRelease'. Files "
                    "written by download record their release."
                )
            if len(releases) > 1:
                raise ValueError(
                    "The local files and 'overtureRelease' name more than "
                    f"one Overture release: {', '.join(sorted(releases))}."
                )
            self.overture_release = releases.pop()

    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        Retrieve the Overture buildings whose geometry intersects a region.

        The assets are added to ``self.inventory``. Each carries its row's
        columns as features, with ``id`` as ``overture_id``, heights in
        ``units['length']`` and ``names`` dropped. A building found in
        several local files becomes one asset. The README beside this
        module, "The features of an asset", lists every change.

        Args:
            region (RegionBoundary):
                The region whose buildings are retrieved.

        Returns:
            AssetInventory:
                ``self.inventory``, with the buildings of the region added.

        Raises:
            ValueError:
                If a local file holds rows of a type other than 'building',
                or, under ``'geohash'`` or with ``'min_area'`` above 0, if a
                vertex lies outside EPSG:4326.

        Example:
            Read the buildings of Berkeley, California, from the release
            ``resolve_release`` picks:

            >>> from brails import Importer
            >>> importer = Importer()
            >>> region = importer.get_class('RegionBoundary')(
            ...     {'type': 'locationName', 'data': 'Berkeley, CA'}
            ... )
            >>> scraper_class = importer.get_class(
            ...     'OvertureMapsFootprintScraper'
            ... )
            >>> scraper = scraper_class({'length': 'ft'})
            >>> inventory = scraper.get_footprints(region)
        """
        bpoly, _,  _ = region.get_boundary()
        bbox = bpoly.bounds

        dataset = self._read_buildings(bbox).to_pandas()

        # Convert WKB to Shapely geometries:
        dataset["geometry"] = dataset["geometry"].apply(
            lambda wkb_bytes: wkb.loads(wkb_bytes))

        print('\nFinding the assets within the specified area...\n')

        # Filter rows that intersect with the bounding polygon:
        intersects_mask = [geom.intersects(bpoly)
                           for geom in dataset["geometry"]]
        intersecting_rows = dataset[np.asarray(intersects_mask, dtype=bool)].copy()

        # Format sources
        intersecting_rows['sources'] = intersecting_rows['sources'].apply(
            self._format_sources)

        # Drop problematic columns for GeoJSON export:
        intersecting_rows.drop(columns=["names"], inplace=True)

        # Convert geometry to list of coordinates:
        intersecting_rows["geometry_coords"] = \
            intersecting_rows["geometry"].apply(
                GeoTools.geometry_to_list_of_lists
        )
        intersecting_rows["overture_id"] = intersecting_rows["id"]
        intersecting_rows.reset_index(drop=True, inplace=True)

        # Assemble the features of each footprint:
        features = []
        for index, row in intersecting_rows.iterrows():
            # Filter attributes:
            attributes = {
                k: v for k, v in row.items()
                if k not in ("geometry", "geometry_coords", 'bbox', 'id')
                and v is not None
                and not (isinstance(v, float) and np.isnan(v))
            }

            # Convert dimensional attributes:
            for attr, attr_unit in DIMENSIONAL_ATTR.items():
                unit_type = UnitConverter.get_unit_type(attr_unit)
                target_unit = self.units[unit_type]
                if attr in attributes:
                    try:
                        attributes[attr] = UnitConverter.convert_length(
                            attributes[attr],
                            attr_unit,
                            target_unit)
                    except Exception as e:
                        print(
                            f"Warning: could not convert {attr} for index "
                            f"{index}: {e}"
                        )

            # Merge with asset type:
            features.append({**attributes, "type": ASSET_TYPE})

        # Key the footprints kept and build the inventory:
        _, kept = self._build_keyed_inventory(
            list(intersecting_rows["geometry_coords"]),
            features,
            self.units['length'],
            provider='overture',
            centroids=[
                geometry.centroid for geometry in intersecting_rows["geometry"]
            ],
            inventory=self.inventory,
        )
        print(
            f'Found a total of {len(kept)} assets within the '
            'specified area.'
        )
        return self.inventory

    def download(self, region: RegionBoundary, path: str) -> int:
        """
        Write the Overture building rows of a region to a GeoParquet file.

        The rows are those whose geometry intersects the region, read from
        the S3 bucket or from the files of ``'local_path'``, with every
        column they hold and the ``theme`` and ``type`` columns. The file
        records the release under ``BRAILS_METADATA_KEY``. With the same
        options, a scraper reading it as ``'local_path'`` builds the same
        inventory over the same region. ``self.inventory`` is left
        unchanged. The README beside this module, "The file download
        writes", states the format and the metadata.

        Args:
            region (RegionBoundary):
                The region whose buildings are written.
            path (str):
                The GeoParquet file to write, replaced if it exists.

        Returns:
            int:
                The number of rows written.

        Raises:
            ValueError:
                If a local file holds rows of a type other than 'building'.

        Example:
            With ``region``, ``scraper_class`` and ``scraper`` as in the
            example of :meth:`get_footprints`, download the region once and
            build inventories from the file, without reading the bucket
            again, keyed by geohash:

            >>> count = scraper.download(region, 'berkeley_buildings.parquet')
            >>> local = scraper_class(
            ...     {'local_path': 'berkeley_buildings.parquet',
            ...      'id_scheme': 'geohash'}
            ... )
            >>> inventory = local.get_footprints(region)
        """
        bpoly, _, _ = region.get_boundary()
        table = self._read_buildings(bpoly.bounds)

        geometries = shapely.from_wkb(
            table['geometry'].to_numpy(zero_copy_only=False)
        )
        inside = shapely.intersects(geometries, bpoly)
        table = table.filter(pa.array(inside))
        geometries = geometries[inside]
        count = table.num_rows
        table = table.append_column(
            'theme', pa.array([TYPE_THEME_MAP[OVERTURE_TYPE]] * count,
                              pa.string())
        )
        table = table.append_column(
            'type', pa.array([OVERTURE_TYPE] * count, pa.string())
        )

        # GeoParquet metadata of the geometry column; the bbox of an empty
        # file is left out:
        column = {
            'encoding': 'WKB',
            'geometry_types': sorted(
                {geometry.geom_type for geometry in geometries}
            ),
            'covering': {'bbox': {
                key: ['bbox', key] for key in ('xmin', 'ymin', 'xmax', 'ymax')
            }},
        }
        if count:
            column['bbox'] = [
                float(value) for value in shapely.total_bounds(geometries)
            ]
        if self.local_paths is None:
            source = 's3://' + self._dataset_path(
                self.overture_release, OVERTURE_TYPE)
        else:
            source = [os.path.basename(path) for path in self.local_paths]
        metadata = {
            **(table.schema.metadata or {}),
            b'geo': json.dumps({
                'version': '1.1.0',
                'primary_column': 'geometry',
                'columns': {'geometry': column},
            }).encode(),
            BRAILS_METADATA_KEY: json.dumps({
                'overture_release': self.overture_release,
                'source': source,
            }).encode(),
        }
        # Row groups of 100,000 rows let a later read by bbox skip groups:
        pq.write_table(
            table.replace_schema_metadata(metadata),
            path,
            compression='zstd',
            row_group_size=100_000,
        )
        print(f'Wrote {count} Overture buildings to {path}')
        return count

    def _read_buildings(
        self, bbox: Tuple[float, float, float, float]
    ) -> pa.Table:
        """
        Read the Overture building rows whose bbox overlaps a bounding box.

        The rows come from the S3 bucket, or from the files of
        ``'local_path'``. Rows read from local files lose the ``theme`` and
        ``type`` columns, which the bucket holds as partition values, and a
        building found in several files is kept once, at its first
        occurrence by ``id``.

        Args:
            bbox (Tuple[float, float, float, float]):
                The bounding box (xmin, ymin, xmax, ymax) to read.

        Returns:
            pa.Table:
                The rows read, in the columns of a read from the bucket.

        Raises:
            ValueError:
                If a local file holds rows of a type other than 'building'.
        """
        if self.local_paths is None:
            return self._read_table(
                self.overture_release, OVERTURE_TYPE, bbox=bbox
            )

        table = self._read_table(
            self.overture_release,
            OVERTURE_TYPE,
            bbox=bbox,
            paths=self.local_paths,
        )
        if 'type' in table.column_names:
            other_types = set(
                pc.unique(table['type']).drop_null().to_pylist()
            ) - {OVERTURE_TYPE}
            if other_types:
                raise ValueError(
                    "The local files hold Overture rows of type "
                    f"{', '.join(sorted(other_types))}. Only "
                    f"'{OVERTURE_TYPE}' rows can be read."
                )
        table = table.drop_columns(
            [name for name in ('theme', 'type') if name in table.column_names]
        )
        first = pc.index_in(pc.unique(table['id']), value_set=table['id'])
        return table.take(first)
