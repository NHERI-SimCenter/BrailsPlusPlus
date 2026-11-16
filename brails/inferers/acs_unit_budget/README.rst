ACS unit budget
===============

``ACSUnitBudget`` places the housing units a census table counts in a
block group, less those of the footprints that already hold observed records,
on the remaining footprints. The tables it packages, ``B25024_BINS``,
``MOBILE_HOME_BIN`` and ``BIN_CLASSES``, are in ``acs_unit_budget.py``
beside this page. The classes, unit ranges and midpoints it reads,
``UNIT_COUNT_CLASSES``, ``COUNT_BANDS`` and ``UNIT_MIDPOINTS``, are in
``brails/utils/unit_count_rules.py``. This page
states where those tables come from, why the bins take Hazus occupancy
classes, how the budget allocates the units, and the scope of the defaults.
The docstrings state what a caller passes and receives.

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
one used outside the United States, when ``BIN_CLASSES``,
``UNIT_COUNT_CLASSES``, ``COUNT_BANDS`` and ``UNIT_MIDPOINTS`` are replaced,
with ``MOBILE_HOME_CLASS`` and ``RELABEL_CLASS``, and when the reached
footprints and the imputer's status classes use the same taxonomy. The last
three of the four tables are the unit-count rules, which the NSI footprint
join applies as well, so replacing them changes the join too. A census
table of another country replaces ``B25024_BINS`` and ``MOBILE_HOME_BIN`` the
same way. These are module constants, so either change is a change to the
code, not an argument.

The margins of error change no allocation and bound only the diagnostic. The
margin of the one-unit bin, which sums two lines, and of the total of the size
bins is the square root of the summed squared margins, the approximation the
Census Bureau gives in Understanding and Using American Community Survey Data:
What All Data Users Need to Know (2020), chapter 8, formula (1). A negative or
missing margin, which the Census API uses to code special cases, reads as zero.

The method
----------

``allocate`` first measures every footprint, the targets and the reached alike:
its plan area, its gross floor area and capacity as the class docstring defines
them, and the great-circle distance in meters from its centroid to the nearest
mobile-home park point, infinite for a footprint without area or where no point
is given. The imputer then gives a residential probability, with the reached
footprints as donors, to each target with area that lies in a block group with
a budget or, under ``drop_without_acs``, is small. A target is small when its
plan area is under ``threshold_sqft`` and it lies beyond ``mh_park_radius_m``
of every mobile-home park point. A reached footprint's class is its
``OccupancyClass``, and its units are its ``NumberOfUnits`` as
``unit_contribution`` in ``brails/utils/unit_count_rules.py`` counts a
record's.

The footprints are grouped by the block group they hold under
``BLOCKGROUP_GEOID``, and those without one form a group without a budget.
The groups are every block group that holds a target and every block group
with a budget that holds a reached footprint, taken in ascending order of
the GEOID as a string. A block group without a budget leaves its targets as
the class docstring states, and the result's ``gaps`` holds a
``GAP_NO_ACS`` row counting those not dropped. Each block group with a
budget runs the steps below, in this order.

The mobile homes come first. The table's mobile-home count, less the units
of the reached ``MOBILE_HOME_CLASS`` footprints, floored at zero and rounded
down, is the number of footprints made ``MOBILE_HOME_CLASS``, each with
``MOBILE_HOME_UNITS`` units. They are taken nearest first, equal distances in
ascending order of the id as a string, from the footprints within
``mh_park_radius_m`` of a mobile-home park point: the targets with a
capacity of at least one unit, and the reached ``RELABEL_CLASS`` footprints
that ``allocate`` names in ``relabel_sources``. A reached footprint made a
mobile home keeps its source's unit count, so its capacity is not checked,
and its class is withheld from its source. Where too few footprints qualify,
the shortfall is recorded under ``GAP_MOBILE_HOME_SHORTFALL``. The mobile
homes take no part in the steps that follow. The diagnostic's mobile-home
rows are recorded here.

Each structure-size bin's residual is then its estimate less the units of
the reached footprints of its class in ``BIN_CLASSES``, floored at zero, and
its whole residual is the residual rounded down. Flooring at zero hides a
bin whose reached units exceed its estimate, so the residuals can sum to
more than the block group lacks as a whole, the estimates' sum less the
reached units' sum, floored at zero. Where they do and ``per_bin`` is not
set, the residuals are replaced by the whole units of that deficit, at most
the sum of the whole residuals, shared among the bins in proportion to their
whole residuals by largest remainders, equal remainders going to the bin of
fewer units per structure. No bin gains.

The candidates are the targets other than mobile homes with area and a
capacity of at least one unit. They are ranked by residential probability,
highest first and unknown last, then by plan area, largest first, then by id
as a string.

Each bin needs as many structures as its residual fills at the midpoint of its
class in ``UNIT_MIDPOINTS``, a fraction. The sum of these needs, rounded half
up and capped at the number of candidates, is shared among the bins in
proportion to their needs by largest remainders the same way. Structures beyond
the candidates are recorded under ``GAP_CANDIDATE_SHORTFALL``.

The bins then reserve their dwellings, from the largest bin down. A bin's least
and greatest unit counts are those of its class in ``COUNT_BANDS``, the least
being at least one and the open 50-or-more class having no greatest. A bin
keeps no more structures than its whole residual fills at its least unit count,
and no more than the candidates left whose capacity holds that count. The
structures it gives up are recorded under ``GAP_STRUCTURES_WITHOUT_UNITS`` and
``GAP_CANDIDATE_SHORTFALL`` respectively, and a bin left with a positive whole
residual and no structure under ``GAP_BIN_WITHOUT_STRUCTURE``. Its even share
is its whole residual over its structures, rounded up and capped at its
greatest unit count. The bin reserves one candidate per structure: the
highest-ranked of those left whose capacity holds the even share, then, where
too few do, those of greatest capacity among the ones whose capacity holds its
least unit count, equal capacities by rank. The reserved candidates are the
dwellings.

Each bin's whole residual is split evenly over its structures, the extra units
to the first, and each part is capped at the bin's greatest unit count. Units
the cap removes are recorded under ``GAP_UNITS_CLIPPED``. In each world the
dwellings are then put in a random order, descending ``ln(u) / A``, with ``u``
the dwelling's uniform keyed by its id, the seed, ``PURPOSE_UNIT_BUDGET`` and
the world, and ``A`` its gross floor area in square meters, equal values in
ascending order of the id as a string. Each next dwelling of the order is drawn
from those not yet in it with probability proportional to its gross floor area.
From the largest bin down, each bin's structures go to the first dwellings left
in that order whose capacity holds the bin's even share, then, where too few
do, to those of greatest capacity among the ones whose capacity holds its least
unit count, equal capacities in that order. The reservation leaves enough
dwellings for every bin. The parts go to these dwellings in the order they were
chosen, each part cut to the dwelling's capacity, and the units cut move, in
the same order, to the bin's dwellings with room left below their capacity and
the bin's greatest unit count. Units that find no room are recorded under
``GAP_UNITS_OVER_CAPACITY``; they are the same in every world.

Last, a target that became neither a mobile home nor a dwelling is dropped
when it is small and passed to the non-residential stage otherwise. The
diagnostic then compares, in each world, the reached and placed units of each
size bin and of their total with the estimate, bounded by ``moe_factor``
times the margin on either side.

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
