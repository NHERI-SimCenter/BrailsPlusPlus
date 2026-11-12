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
This module define the concrete class for scraping OvertureMaps footprint data.

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
    Scraper for extracting Overture Maps buildings.

    The scraper reads Overture's ``building`` type only. Overture's
    ``building_part`` type describes portions of a building that is already
    present as a ``building`` row, such as a tower above its podium, and is
    not read.

    This class extends
    :class:`~brails.scrapers.overture_maps_scraper.overture_maps_scraper.OvertureMapsScraper`
    and :class:`~brails.scrapers.footprint_scraper.FootprintScraper`, and
    provides methods to load and filter Overture asset
    datasets based on a region boundary. Assets are stored in an
    :class:`~brails.types.asset_inventory.AssetInventory` and can be exported
    as GeoJSON. The rows are read from the Overture S3 bucket, or from local
    GeoParquet files given as ``'local_path'``. :meth:`download` writes
    the rows of a region to such a file.

    Attributes:
        units (str):
            Length unit used for spatial calculations. Parsed from
            ``input_dict`` or defaults to ft.
        inventory (AssetInventory):
            Stores the assets retrieved by the scraper.
        overture_release (str):
            The specific Overture Maps release being used. Falls back to the
            newest release the S3 bucket holds if not specified in
            ``input_dict``. With ``'local_path'``, the release declared in
            ``input_dict`` or recorded in the files.
        local_paths (list of str or None):
            The local files read in place of the S3 bucket, or ``None``.
    """

    def __init__(self, input_dict: Dict[str, Any] = None):
        """
        Initialize the class with user-defined or default settings.

        This constructor sets the unit system, initializes an empty asset
        inventory, and determines which Overture Maps release to use.
        Configuration options can be provided via the optional ``input_dict``.

        Args:
            input_dict (dict, optional):
                A configuration dictionary that may include:

                - 'length' (str): Length unit to use (e.g., 'ft', 'm').
                  Defaults to the unit defined in ``DEFAULT_UNITS`` if not
                  provided.
                - 'overtureRelease' (str): Specific Overture Maps release name
                  to use, checked against the releases the S3 bucket holds. If
                  omitted, the newest available release is selected, with a
                  notice. See
                  :meth:`~brails.scrapers.overture_maps_scraper.overture_maps_scraper.OvertureMapsScraper.resolve_release`.
                - 'id_scheme' (str) and 'min_area' (float): described in
                  :class:`~brails.scrapers.footprint_scraper.FootprintScraper`.
                - 'local_path' (str or sequence of str): GeoParquet files of
                  Overture ``building`` rows, such as files written by
                  :meth:`download`, read in place of the S3 bucket. Each
                  file must hold Overture's ``bbox`` column. The bucket is
                  not contacted, and the release is not checked against it.
                  The release is ``'overtureRelease'``, or without it the
                  release :meth:`download` recorded in every file. The
                  releases given must agree.

        Raises:
            FileNotFoundError:
                If a file of ``'local_path'`` does not exist.
            ValueError:
                If ``'local_path'`` names no file, or a path that is not a
                Parquet file, or a file holding a ``building_id`` column,
                which only Overture's ``building_part`` rows have; if
                ``'overtureRelease'`` is absent and a file records no
                release; or if the files and ``'overtureRelease'`` name more
                than one release.

        Notes:
            - Units are parsed and validated through
              :meth:`~brails.utils.unit_converter.UnitConverter.parse_units`.
            - An empty :class:AssetInventory` is created upon initialization.
            - If ``input_dict`` is not provided, all defaults will be used.
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
        Retrieve Overture building assets within a specified region.

        This method loads the Overture rows of the 'building' type that are
        within the boundary of the given region, from the S3 bucket or from
        the files of ``'local_path'``. Resulting assets are saved as an
        :class:`~brails.types.asset_inventory.AssetInventory`.

        Args:
            region (RegionBoundary)
                A BRAILS++ region boundary object defining the coordinate
                information for the region of interest.

        Returns:
            AssetInventory: An inventory of assets within the specified region.

        Raises:
            ValueError:
                If a local file holds rows of a type other than 'building'.

        Notes:
            - Only assets that intersect the provided region are included.
            - A building found in several local files becomes one asset.
            - Columns named "names" are dropped to avoid GeoJSON export issues.
            - Asset attributes exclude ``None`` and ``NaN`` values.
            - The inventory is stored in ``self.inventory`` and also returned.

        Example:
            First import the BRAILS++ :class:`~brails.utils.importer.Importer`
            class.

            >>> from brails import Importer
            >>> importer = Importer()

            Create a region boundary object for Berkeley, CA.

            >>> region_data = {"type": "locationName", "data": 'Berkeley, CA'}
            >>> region_boundary_class = importer.get_class('RegionBoundary')
            >>> region_boundary_object = region_boundary_class(region_data)

            Get building inventory data using the '2024-07-22.0' release of
            Overture Maps for Berkeley, CA.

            >>> scraper_class = importer.get_class(
            ... 'OvertureMapsFootprintScraper'
            ... )
            >>> scraper_object = scraper_class(
            ...     input_dict={'overtureRelease': '2024-07-22.0'}
            ... )
            No length unit specified. Using default: 'ft'.
            No weight unit specified. Using default: 'lb'.
            >>> inventory = scraper_object.get_footprints(
            ...     region_boundary_object
            ... )
            Searching for Berkeley, CA...
            Found Berkeley, Alameda County, California, United States
            Reading dataset batches: 339it [00:39,  8.49it/s]
            Finding the assets within the specified area...
            Found a total of ... assets within the specified area.

            Write the obtained inventory in ``berkeley_buildings.geojson``.

            >>> _ = inventory.write_to_geojson('berkeley_buildings.geojson')
            Wrote ... assets to /home/bacetiner/Documents/SoftwareTesting/
            berkeley_buildings.geojson
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

        # TODO: Figure out how to parse names
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

        The rows written are those whose geometry intersects the boundary
        of the region, read from the S3 bucket or from the files of
        ``'local_path'``. They keep every column of the release with its
        Arrow type, and gain ``theme`` and ``type``, the constant string
        columns that the bucket holds as partition values. The file is
        GeoParquet 1.1.0: WKB geometry in OGC:CRS84, Overture's ``bbox``
        column declared as the covering, and zstd compression. Its schema
        metadata keeps the metadata of the rows read and records, under
        ``BRAILS_METADATA_KEY``, the release and the source read: the S3
        path, or the names of the local files.

        A scraper given the file as ``'local_path'`` reads its release from
        it, and its ``get_footprints`` over the same region gives the
        inventory this scraper's ``get_footprints`` gives. The method builds
        no inventory and leaves ``self.inventory`` unchanged.

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
            Write the buildings of Berkeley, CA from the '2026-10-21.0'
            release, then build the inventory from the file.

            >>> from brails import Importer
            >>> importer = Importer()
            >>> region_boundary_object = importer.get_class('RegionBoundary')(
            ...     {'type': 'locationName', 'data': 'Berkeley, CA'}
            ... )
            >>> scraper_class = importer.get_class(
            ...     'OvertureMapsFootprintScraper'
            ... )
            >>> scraper_object = scraper_class(
            ...     input_dict={'overtureRelease': '2026-10-21.0'}
            ... )
            No length unit specified. Using default: 'ft'.
            No weight unit specified. Using default: 'lb'.
            >>> count = scraper_object.download(
            ...     region_boundary_object, 'berkeley_buildings.parquet'
            ... )
            Searching for Berkeley, CA...
            Found Berkeley, Alameda County, California, United States
            Reading dataset batches: ...
            Wrote ... Overture buildings to berkeley_buildings.parquet
            >>> local_scraper = scraper_class(
            ...     input_dict={'local_path': 'berkeley_buildings.parquet'}
            ... )
            No length unit specified. Using default: 'ft'.
            No weight unit specified. Using default: 'lb'.
            >>> inventory = local_scraper.get_footprints(region_boundary_object)
            Searching for Berkeley, CA...
            Found Berkeley, Alameda County, California, United States
            Reading dataset batches: ...
            Finding the assets within the specified area...
            Found a total of ... assets within the specified area.
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
