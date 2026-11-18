ACS year-built sampler
======================

``ACSYearBuiltSampler`` draws a construction year for each residential asset
from two tables of the American Community Survey (ACS): the housing units by
year built in the asset's block group, and the housing units by year built and
structure size in its tract. The variable lists and the packaged defaults are
constants in ``acs_year_built_sampler.py`` beside this page. This page states
which table cells the sampler reads, how it combines them, and the scope of
the defaults. The class docstring states what a caller passes and receives.

The tables
----------

The sampler reads the ACS 2024 five-year estimates of the U.S. Census Bureau,
which pool the surveys of 2020 to 2024. The caller passes three tables as data
frames: B25034 per block group, B25034 per tract and B25127 per tract, each
indexed by GEOID string with the ACS variable ids as columns, the form the
Census API returns. The package holds the variable ids, not the data.

Table B25034, Year Structure Built, counts housing units, occupied and vacant,
in ten bins: 1939 or earlier, the eight decades from 1940 to 1949 through
2010 to 2019, and 2020 or later. ``B25034_BINS`` lists the bins with their
variables, ``B25034_011E`` for the oldest to ``B25034_002E`` for the newest.
The total, ``B25034_001E``, is not read.

Table B25127, Tenure by Year Structure Built by Units in Structure, counts
occupied housing units only. It splits them by tenure, by six coarser year
bins (``COARSE_YEARS``: 1939 or earlier, 1940 to 1959, 1960 to 1979, 1980 to
1999, 2000 to 2019, 2020 or later) and by six classes of units in structure,
which the table labels "1, detached or attached", "2 to 4", "5 to 19", "20
to 49", "50 or more" and "Mobile home, boat, RV, van, etc." and
``SIZE_CLASSES`` names ``'1 unit'``, ``'2 to 4'``, ``'5 to 19'``,
``'20 to 49'``, ``'50 or more'`` and ``'Manufactured home'``. The Census
Bureau publishes it down to the tract, not the block group.
``B25127_CELLS`` names its 36 cells of year bin by class, each the sum of
an owner-occupied variable (lines 004 to 044) and a renter-occupied one
(lines 047 to 087). The totals and subtotals are not read. Because B25127
leaves out vacant units and B25034 includes them, the sampler takes from
B25127 only how years are distributed within each size class, and B25034
sets how many units fall in each bin.

An asset's size class comes from ``size_class``: RES2 is a mobile home
whatever its unit count, and any other class is placed by its unit count
(``NumberOfUnits`` by default) against ``SIZE_CLASS_BOUNDS``. A negative
estimate, the code the Census API gives an estimate it does not publish, reads
as zero, and ``fit_report`` counts such values per block group.

The method
----------

The sampler fits, for each block group, one distribution of construction years
per size class, and draws each asset's year from the distribution of its block
group and size class. B25034 sets how the block group's units fall over the ten
B25034 bins of year built, and B25127 sets how the years of each size class
spread over the tract's six coarse bins. The fit splits each coarse bin by the
B25034 shares and then adjusts the size-class distributions, weighted by the
block group's units of each size class, toward those shares. The section
"Concepts" of ``brails/compilers/README.rst`` describes the world a fit reads
and the keyed random numbers of a draw under "Possible worlds".

``fit`` reads one world's inventory and sums, for each block group among its
residential assets, the units of the assets in each size class. It then fits
each block group separately.

The block group's B25034 counts, divided by their total, give its shares of
the ten bins. A block group missing from the table, or holding no units in it,
takes its tract's shares. Where the tract holds none either, the block group
is not fitted, and ``check`` refuses its assets.

The tract's B25127 counts give, for each size class, a distribution over the
six coarse bins. Each class row is smoothed toward the same row pooled over
every tract of the county in the table: ``smoothing`` pseudo-units of the
pooled distribution are added to the tract's counts before the row is
normalized. A tract missing from B25127 takes the county's pooled rows alone,
and a county with no tract in the table leaves the block group unfitted.

Each coarse bin's probability is then split among the B25034 bins it holds,
in proportion to the block group's shares. A B25034 bin with a zero share
gets no probability, and a class row left with none takes the shares.

Last, the class rows are raked by iterative proportional fitting. Each row is
weighted by the units of its class in the block group, and the rows are
adjusted until the expected units per B25034 bin, summed over the classes,
equal the bin's share times the block group's units. All six rows are fitted,
including those of classes with no asset in the block group, which carry no
weight in the raking. ``fit_report`` records the passes, the final residual
and whether the raking converged. A block group that does not converge keeps
its last rows.

``draw`` gives each asset a year from the fitted row of its block group and
class. The row's probability for a bin is spread evenly over the bin's years,
and one uniform number per asset picks the year from the result. The uniform
carries the purpose label ``ACS_YEAR_PURPOSE``. A call on an inventory of one
asset reads that asset's row from the fit, so it draws what the asset would
draw in the fitted inventory.

Defaults and scope
------------------

``OLDEST_YEAR``, 1890, is the first year of the open oldest bin, 1939 or
earlier, which the table leaves without one. It is not a published bound.
It sets the years drawn in that bin, and the Hazus earthquake inferers of
BRAILS return the same result for any of them: all 50 years from 1890 to
1939 fall in the earliest era of every design-level table, whose earliest
break is 1940, and in the earliest era of the structure type tables, which
ends before 1950. The bound will be revisited when BRAILS supports a method
that tells those years apart. ``NEWEST_YEAR``, 2024, the last survey year
of the estimates, closes the newest bin, 2020 or later, at five years.

``DEFAULT_SMOOTHING``, 20 pseudo-units, keeps a tract's empty B25127 cells
from ruling a combination out. Without it, a tract that holds no structure of
50 or more units built from 2000 to 2019 would forbid one in every block group
of the tract, and the raking could have no solution. The value was tested
on San Francisco, on a building table built on the city's own footprint
layer, with 142,542 buildings in the test: the absolute error of the
building counts per B25034 bin, summed over the ten bins, was 31,149
buildings at 5 pseudo-units, 31,669 at 20 and 32,480 at 50. The error
moves by about 4 percent over that range, 5 scoring lowest, and the
default stays at 20. A caller passes another value as ``smoothing``. At 0
nothing is added, and a class row with no counts in the tract becomes
uniform over the six bins.

``RAKE_TOLERANCE``, 1e-12, is the bin residual, as a share of the block
group's units, below which the raking stops, and ``RAKE_ITERATIONS``,
10,000, caps the passes.
Both are settings of the numerical method, not calibrated against data.

The sampler covers ``UNIT_COUNT_CLASSES`` of
``brails/utils/unit_count_rules.py``: RES1, RES2 and RES3A to RES3F, the
Hazus single-family, mobile home and multi-family classes. RES4 to RES6 are
not covered, and ``check`` refuses them. A sampler draws for the one
world it is constructed with, and its fit reads that world's inventory, so
each world of a study takes its own sampler, with the seed of the run.

A draw depends on more than its asset. Through the raking it depends on the
classes and units of the other residential assets of its block group in the
fitted inventory, and through the pooled rows on every tract of its county in
the B25127 table. Fit the sampler on the inventory of the whole study area,
and pass B25127 rows for every tract of each county the study area
touches.
