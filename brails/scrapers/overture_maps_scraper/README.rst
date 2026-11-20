Overture Maps scraper
=====================

``OvertureMapsFootprintScraper`` reads the buildings of a region from Overture
Maps and returns them as an asset inventory. It derives from
``OvertureMapsScraper``, which finds the available releases and reads an
Overture type from the S3 bucket or from local files. The two classes live in
``overture_maps_footprint_scraper.py`` and ``overture_maps_scraper.py`` beside
this page. This page states what the scraper reads, how it chooses a release
and the file ``download`` writes. The docstrings state the arguments, the
return values and the errors, and the class docstring and those of
``get_footprints`` and ``download`` give usage examples.

What the scraper reads
----------------------

Overture publishes its data as releases in a public S3 bucket, which the
scraper reads anonymously. The folder ``S3_BASE_PATH``,
``overturemaps-us-west-2/release``, holds one subfolder per release. Within a
release, ``TYPE_THEME_MAP`` gives the theme of each Overture type. Buildings
are the ``building`` type of the ``buildings`` theme, so the scraper reads the
folder ``<release>/theme=buildings/type=building/`` under ``S3_BASE_PATH``.

The region's boundary decides which rows are read, in two steps. The read
keeps the rows whose ``bbox`` column, the bounding box Overture stores with
each row, overlaps the bounding box of the boundary. Of those, the scraper
keeps the rows whose geometry intersects the boundary itself. A building that
crosses the boundary is therefore kept whole.

Releases
--------

A scraper reads one release, ``overture_release``. The release list Overture
publishes, at ``OVERTURE_RELEASES_URL``, and the bucket need not agree, so a
release is checked against the bucket.

Without ``'local_path'``, the release is resolved by ``resolve_release``:

- A release given as ``'overtureRelease'`` is used if the bucket holds it. If
  it does not, the scraper raises ``ValueError`` and names the releases the
  bucket holds.
- Without ``'overtureRelease'``, the first release that
  ``fetch_available_releases`` gives is used, with a printed notice. That
  function sorts the release names in reverse, which puts the newest first
  for date-named releases.
- Where the bucket cannot be listed or lists no release, the scraper prints a
  warning. A given release is then used unchecked, and without one the
  scraper takes the first release of the published list, raising
  ``ValueError`` if that list is empty.

A release the bucket does not hold raises instead of being replaced by a
newer one because a named release records which data an inventory was built
from. Reading another release would make that record false. A release the
bucket no longer holds can still be read from a local file, such as one
``download`` wrote while the bucket held it.

Buildings and building parts
----------------------------

The scraper reads Overture's ``building`` type and refuses ``building_part``
rows. For a caller this means:

- Each asset is one ``building`` row.
- The height of an asset is that of its ``building`` row. The heights of
  ``building_part`` rows are not read.
- The scraper takes a ``building_id`` column as the mark of ``building_part``
  rows. No asset carries it, and a local file holding it raises when the
  scraper is constructed.
- Rows of a local file whose ``type`` is not ``'building'`` raise when they
  are read.

Where several ``building`` rows outline one building, the scraper returns
each row as an asset. ``StackedFootprintMerge`` merges a footprint nested in a
larger one where its stacking condition holds, and its docstring states the
condition.

The features of an asset
------------------------

Each building kept becomes an asset of ``type`` ``'building'``, whose
coordinates are its geometry as BRAILS coordinate lists. Its features are the
columns of its row, with these changes:

- ``id`` is kept as ``overture_id``, and ``footprint_provider`` is
  ``'overture'``.
- ``geometry`` and ``bbox`` are not features, and ``names`` is dropped.
- ``sources`` becomes one string. The non-empty fields of each source are
  written as ``key: value`` pairs joined by a comma and a space, and the
  sources are joined by ``SOURCES_SEPARATOR``.
- ``height``, ``min_height`` and ``roof_height``, which ``DIMENSIONAL_ATTR``
  lists as meters, are converted to the scraper's length unit, ``'length'``
  in ``input_dict``, feet by default. A value that cannot be converted keeps
  its meters, with a printed warning.
- A value that is ``None`` or NaN is left out.

The file download writes
------------------------

``download`` writes to a GeoParquet file the rows ``get_footprints`` reads for
a region, before any feature is changed. The rows keep every column of the
release with its Arrow type, ``names`` and ``bbox`` included. Two string
columns are added, ``theme`` with the value ``'buildings'`` and ``type`` with
the value ``'building'``, which the bucket holds as partition values. The file
is compressed with zstd, in row groups of 100,000 rows.

The schema metadata keeps the metadata of the rows read and adds two entries:

- ``geo``, the GeoParquet 1.1.0 metadata. ``geometry`` is the primary column,
  encoded as WKB, with the geometry types present and, for a file that holds
  rows, their bounding box. The ``bbox`` column is declared as the covering of
  ``geometry``. No coordinate reference system is stated, which GeoParquet
  reads as OGC:CRS84, longitude and latitude.
- ``brails``, the key ``BRAILS_METADATA_KEY`` names, a JSON object with
  ``overture_release``, the release read, and ``source``, the S3 path read or
  the list of the local file names read.

A scraper given the file as ``'local_path'`` takes the release from the file
and does not contact the bucket. With the same options and region, its
``get_footprints`` gives the same inventory as the scraper that wrote the
file. A region inside the one downloaded can be read from the file as well. A
region that extends beyond it gives only the buildings the file holds.

Local files and their release
-----------------------------

``'local_path'`` names GeoParquet files of Overture ``building`` rows, read in
place of the bucket. The files can be written by ``download`` or come from
elsewhere if they meet the conditions below. Each must hold Overture's
``bbox`` column, which the read filters on as it does in the bucket.

The release comes from ``'overtureRelease'`` or from the files:

- Without ``'overtureRelease'``, every file must record a release, as the
  files ``download`` writes do. A file that records none raises
  ``ValueError``, even when another file records one.
- With ``'overtureRelease'``, a file need not record a release.
- The releases given, by ``'overtureRelease'`` and by the files that record
  one, must be the same release, or the scraper raises ``ValueError``.

The release is taken as given or recorded and is not compared with the rows.
