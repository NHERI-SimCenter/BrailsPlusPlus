Block-group unit budget
=======================

``BlockGroupUnitBudget`` places the housing units a census table counts in a
block group, less those of the footprints that already hold observed records,
on the remaining footprints. The tables it packages, ``B25024_BINS``,
``MOBILE_HOME_BIN`` and ``BIN_CLASSES``, are in ``acs_unit_budget.py``
beside this page. The classes, unit ranges and midpoints it reads,
``UNIT_CLASSES``, ``COUNT_BANDS`` and ``UNIT_MIDPOINTS``, are in
``brails/aggregators/points_to_polygons/nsi_footprint_join.py``. This page
states where those tables come from, why the bins take Hazus occupancy
classes, and the scope of the defaults. The class docstring states the
procedure.

The census table and the classes
--------------------------------

The counts come from the American Community Survey table B25024, Units in
Structure, published by the U.S. Census Bureau as 5-year estimates down to the
block group. Its universe is housing units, occupied and vacant. Line 001 is
the total, and lines 002 to 011 split it by the number of units in the
structure: one unit detached (002) and attached (003), 2 units (004), 3 or 4
(005), 5 to 9 (006), 10 to 19 (007), 20 to 49 (008), 50 or more (009), mobile
homes (010), and boats, RVs and vans (011). ``B25024_BINS`` sums lines 002 and
003 into the one-unit bin and takes lines 004 to 009 one bin each.
``MOBILE_HOME_BIN`` is line 010. Lines 001 and 011 are not read.

``BIN_CLASSES`` gives each bin a Hazus occupancy class: RES1 for one unit and
RES3A to RES3F for 2, 3 or 4, 5 to 9, 10 to 19, 20 to 49 and 50 or more units.
A mobile home becomes ``MOBILE_HOME_CLASS``, RES2. The match is exact because
B25024's size bins coincide with the unit ranges of the Hazus multi-family
classes, which ``COUNT_BANDS`` holds from the Hazus Inventory Technical Manual,
and the budget uses Hazus because the footprints' observed records and the
inferers that read the budget's output use it. ``UNIT_MIDPOINTS`` gives the
number of units a structure of each class is taken to hold when the budget
counts the structures a bin needs: the middle of the class's range, and for
the open 50-or-more class a fixed 100 units, which the manual does not give
and which was not tuned.

The classes are a choice. The same procedure assigns another taxonomy, such as
one used outside the United States, when ``BIN_CLASSES``, ``UNIT_CLASSES``,
``COUNT_BANDS`` and ``UNIT_MIDPOINTS`` are replaced, with
``MOBILE_HOME_CLASS`` and ``RELABEL_CLASS``, and when the reached footprints
and the imputer's status classes use the same taxonomy. The last three of the
four tables belong to the NSI footprint join, whose counting of a record's
units the budget shares, so replacing them changes the join as well. A census
table of another country replaces ``B25024_BINS`` and ``MOBILE_HOME_BIN`` the
same way. These are module constants, so either change is a change to the
code, not an argument.

The margins of error change no allocation and bound only the diagnostic. The
margin of the one-unit bin, which sums two lines, and of the total of the size
bins is the square root of the summed squared margins, the approximation the
Census Bureau gives in Understanding and Using American Community Survey Data:
What All Data Users Need to Know (2020), chapter 8. A negative or missing
margin, which the Census API uses to code special cases, reads as zero.

Scope
-----

None of the four defaults below was calibrated against data. Each is a setting
to tune for the study area.

``threshold_sqft``, 1,000 sq ft of plan area, guards against keeping small
structures without housing units, such as garages and sheds, which the budget
drops unless they lie within the mobile-home park radius.

``mh_park_radius_m``, 300 m, is how far from a mobile-home park point a
footprint may become a mobile home, and within it no small footprint is
dropped.

``moe_factor``, 2, sets the width of the diagnostic's bounds. The ACS states
its margins at 90 percent confidence, so a margin is about 1.645 standard
errors and ``2 M`` about 3.3. The flag then marks only gross disagreement with
the estimate, and a factor of 1 flags at the 90 percent level. Under the
default sharing of the deficit, the total of the size bins falls below its
bounds only where the budget could not place units, apart from the fractions
of a unit that whole units leave, and above them only where the reached units
already exceed the estimate. A single bin can fall below its bounds where the
deficit is shared, since sharing gives a bin fewer units than its residual.
Under ``per_bin`` the total can also rise above its bounds while the reached
units do not exceed the estimate, since each bin's residual stands alone.

``min_unit_area_sqft``, 300 sq ft of gross floor area per unit, sets each
footprint's capacity and guards against placing more units in a building than
its floor area can hold. Gross floor area includes walls, corridors and shared
space, so the value is meant to rule out only implausible densities. The
budget reads a footprint's gross floor area from its ``GrossArea`` feature, in
square feet. The NSI footprint join writes that feature from NSI records, and
any other source of floor areas can supply it. A footprint with neither
``GrossArea`` nor a story count counts as one story, which errs toward too few
units in a tall building rather than too many in a low one. The log's
``gross_area_source`` column shows which footprints that applied to.
