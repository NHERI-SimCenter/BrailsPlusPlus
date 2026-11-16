National Structure Inventory
============================

``NSI_Parser`` and ``NSIFileReader`` read the building records of the
National Structure Inventory (NSI), the point inventory of structures that
the U.S. Army Corps of Engineers maintains. ``NSI_Parser`` fetches the
records of a region from the NSI API, and ``NSIFileReader`` reads them from
a parquet table. The fields BRAILS expects and the feature names they take
are constants in ``nsi_conventions.py`` beside this page. This page states
how the records reach an inventory, what each field becomes, what NSI's
``fullrep`` and ``sqft`` hold, and which records BRAILS reads. The
docstrings state the arguments, the return values, and the errors.

How the records reach an inventory
----------------------------------

``NSI_Parser.get_raw_data`` fetches the records inside a region and returns
one point asset per record, keyed by ``fd_id``, NSI's identifier of the
structure. Given ``raw_parquet_path``, it also writes the records to a
parquet table under their NSI names, through ``write_raw_records``, and
``NSIFileReader.read`` reads such a table back. The table has one column for
each field of ``NSI_RAW_SCHEMA``, the fields the NSI 2026 API serves, stored
under the type the schema declares, and one column for each other field a
record carries.

Both keep NSI's field names and values by default. With
``simcenter_format=True``, each record passes through ``to_simcenter``,
which renames the fields listed in ``NSI_TO_SIMCENTER`` to SimCenter feature
names, maps the building type codes, and decodes the occupancy code. A field
the mapping does not list keeps its NSI name.

The NSI footprint join, ``NSIFootprintJoin`` in
``brails/aggregators/points_to_polygons/``, takes the points under their NSI
names, as ``NSIFileReader`` reads them, and links each point to at most one
building footprint. A footprint that receives points takes the fields of one
of its records, the governing record, through ``to_simcenter``, with that
record's ``OccupancyClass`` renamed ``NSIOccupancyClass``. The footprint's own
``OccupancyClass`` is the governing record's class or, for single-family and
multi-family records, the class of the footprint's unit count, summed from
the records' ``resunits``. Its ``GrossArea`` and ``ReplacementCost`` are the
value where one record states it and the sum where several do. The join
returns the stated values in one inventory, ``observed``, and the derived
ones, sums included, in another, ``derived``. Its docstrings state how it
links the points and which record governs.

The fields and the features they become
---------------------------------------

The table lists the fields ``NSI_TO_SIMCENTER`` renames, with NSI's
description of each from its `technical documentation for the 2026 Base`_,
shortened.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - NSI field
     - Feature
     - NSI's description
   * - ``x``
     - ``Longitude``
     - The x coordinate of the structure, in WGS84
   * - ``y``
     - ``Latitude``
     - The y coordinate of the structure, in WGS84
   * - ``sqft``
     - ``GrossArea``
     - The estimated square footage of the structure
   * - ``num_story``
     - ``NumberOfStories``
     - The number of stories of the structure
   * - ``med_yr_blt``
     - ``YearBuiltTractMedian``
     - The median year built of the structures in the Census tract
   * - ``fullrep``
     - ``ReplacementCost``
     - The full, not depreciated, estimated replacement value of the
       structure
   * - ``bldgtype``
     - ``BuildingType``
     - The building type, commonly associated with the exterior wall
   * - ``occtype``
     - ``OccupancyClass``
     - The occupancy type of the structure
   * - ``found_type``
     - ``FoundationType``
     - The foundation type, as a one-letter code
   * - ``found_ht``
     - ``FoundationHeight``
     - The foundation height above the ground elevation, in feet
   * - ``ground_elv``
     - ``GroundElevation``
     - The ground elevation at the structure, in feet (NAVD88)
   * - ``firmzone``
     - ``FloodZone``
     - The estimated 2025 flood zone of the structure

``BUILDING_TYPE_MAP`` maps the ``bldgtype`` code ``H``, manufactured
housing, to ``MH``. The other codes (``W``, ``S``, ``C``, ``M``), wood,
steel, concrete, and masonry, match the Hazus general building types and pass
unchanged.

``OccupancyClass`` is the part of ``occtype`` before the hyphen, so
``RES1-1SNB`` becomes ``RES1``, and the RES3 subtypes, such as ``RES3A``,
which carry no hyphen, stay as they are. For single-family homes NSI's
occupancy code adds a subtype after the hyphen, which NSI's occupancy table
lists: the number of stories or ``SL`` for split level, then ``WB`` with
basement or ``NB`` with none, as in ``RES1-1SNB`` and ``RES1-SLWB``.
``to_simcenter`` reads two features from it: ``SplitLevel``, true for ``SL``,
and ``Basement``, true for ``WB`` and false for ``NB``. A feature the
occupancy code does not state is left out, not defaulted. The story count in
the subtype is not read, and ``NumberOfStories`` comes from ``num_story``.

``med_yr_blt`` becomes ``YearBuiltTractMedian`` and not ``YearBuilt``.
Every building of a tract carries the same value, which is not an
observation of the building's own year. ``YearBuiltSampler`` draws a year
for each building around it.

.. _technical documentation for the 2026 Base:
   https://www.hec.usace.army.mil/confluence/nsi/technicalreferences/latest/technical-documentation

The value fields
----------------

``fullrep`` is NSI's full replacement value, its replacement cost new: the
cost to construct a comparable new structure, in August 2025 price levels. NSI
estimates it as a cost per square foot for the occupancy type, which generally
falls as the structure's square footage grows, times that square footage. The
cost is adjusted for the county's construction wages and, for single-family
homes, for the income of the Census tract relative to that of its Census
division. NSI's documentation states that no adjustments are made "for
differences in material costs or differences in regional construction
practices". ``fullrep`` excludes contents, which NSI states separately as
``val_cont``. BRAILS applies no price-level adjustment.

``val_struct`` is NSI's depreciated replacement value of the structure.
NSI's documentation describes the depreciation as based on the structure's
age and the likely level of maintenance and reinvestment in its area.
``val_struct`` keeps its NSI name, and no SimCenter feature is taken from
it.

``sqft`` and the footprint
--------------------------

``sqft`` becomes ``GrossArea``, and no NSI field becomes ``PlanArea``.
Beside ``sqft``, NSI states ``ftprntsqft``, the square footage of the
footprint polygon used for the record, and ``ftprntid``, the footprint's
identifier, which stacked structures share, so ``sqft`` can be compared with
the gross area computed from NSI's own footprint. Over the 165,152 records
for San Francisco in the NSI 2026 Base, ``sqft`` divided by the area of
NSI's own footprint (``ftprntsqft``) times its stories (``num_story``) has a
median of 0.9252 and quartiles of 0.5546 and 1.0. The 137,869 records alone
on their NSI footprint have a median of 1.0 and a lower quartile of 0.706,
and the 27,283 records (16.5 percent) that share a footprint with another
record have a median of 0.118. On the 7,849 NSI footprints of San Francisco
that hold several records (the 27,283 records above), the sum of ``sqft``
over a footprint's records divided by its area times its stories lies
between 0.99998 and 1.00002. The ratio of ``sqft`` to the gross area computed
from the footprints the join assigns NSI records to has not been measured.
What these ratios mean for a value computed from a rate is in
``brails/imputers/replacement_cost/README.rst``.

Of the 2,243,922 records of the NSI 2026 Base for the nine San Francisco Bay
Area counties, every record holds a ``fullrep`` and a ``sqft``, and both are
positive.

Scope
-----

BRAILS is compatible with the records of the NSI 2026 Base only. Some of the
fields of ``NSI_RAW_SCHEMA`` first appear in the public records of that
release: NSI's documentation lists ``fullrep`` as a new field, and
``resunits`` and ``ftprntsqft`` as fields moved from the private to the public
set. A table from an earlier release has no ``fullrep``, so no
``ReplacementCost`` is taken from it, and ``NSIFileReader`` reads such a table
without a warning.

``NSI_Parser.GetNSIData`` and ``NSI_Parser.get_filtered_data_given_inventory``
do not use these conventions. They match the records to footprints
themselves and rename the fields through ``nsi2brailsmap``, which
``NSI_Parser`` sets when it is constructed. That mapping takes ``sqft`` as
``fparea``, the building's plan area in those methods' output, and
``val_struct``, the depreciated value, as ``repaircost``. This page
describes ``to_simcenter`` and the join, not that mapping.
