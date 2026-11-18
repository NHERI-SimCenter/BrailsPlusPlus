Replacement cost imputer
========================

``ReplacementCostImputer`` values buildings, the feature ``ReplacementCost``,
from the buildings whose replacement cost is stated. It fills in a missing
value and recomputes a stated one, except a value from a local source, which
it keeps. Its constants are in ``replacement_cost_imputer.py`` beside this
page, and the five defaults of its same-class neighbor draw are those of
``ObservedDonorImputer``, which
``brails/imputers/observed_donor_imputer/README.rst`` states. This page states
what the value and the rate are, the procedure, and the scope of the rates and
of the defaults.

The imputer values a building as a rate per square foot times the building's
gross area, which it computes from the footprint as the plan area times the
number of stories. The rates come from the buildings whose replacement cost is
stated: a rate is a building's ``ReplacementCost`` divided by its
``GrossArea``, for a value from NSI, from a local source such as the user's
own inventory, or derived, a code that also marks the values this imputer
computes from a building's own rate. Every building is valued this way,
including those whose replacement cost is stated, except that a replacement
cost from a local source is kept as it is. An inventory can hold several
possible worlds, each one realization of the features whose values are
uncertain, such as the story count, and the imputer values each world on its
own.

A building with a rate of its own takes that rate. A building without one
takes the rate of its donor, the building from which ``ObservedDonorImputer``
copied its missing features, and failing that the rate of a nearby building
of the same occupancy class, drawn at random among the nearest by location,
plan area, and story count (the same-class neighbor draw). A building that
gets no rate, or whose footprint encloses no area, gets no value, and the
imputer's log records why.

The value and the rate
----------------------

A building's value in a world is its rate times its gross area computed from
the footprint: its plan area in square feet times its ``NumberOfStories`` in
that world where that is a finite number above zero, logged as
``AREA_FROM_STORIES``, and otherwise its plan area as one story, logged as
``AREA_FROM_PLAN_AREA``. ``ACSUnitBudget``, which deals housing units to
footprints, computes the same gross area for a footprint without a
``GrossArea``, and the two labels are the budget's too, in
``brails/constants.py``. The budget uses a stated ``GrossArea`` where there is
one. A computed value is a float, not rounded, in the units of the
``ReplacementCost`` its rate came from. Given ``target_area_key``, the imputer
multiplies the rate by the gross area that feature holds, read from the frozen
stage ``area_stage`` where ``impute_stage`` is given one, and logs it as
``AREA_FROM_FEATURE``. A building without a finite number above zero in that
feature gets no value, unless it keeps a local value. The same-class neighbor
draw still locates a building by its footprint, so a building that would take
a neighbor's rate and whose footprint encloses no area gets no value either.

``AssetInventory.merge`` records the source of each building's
``ReplacementCost`` in its source string, the feature ``SourceString``, as one
of the codes of ``SOURCE_CODES``, or ``SOURCE_CODE_NONE`` where no source
holds a value ("The source string" under "Concepts" in
``brails/compilers/README.rst``). This imputer reads the codes of the value
and of the occupancy class, and writes the value's code by
``OUTCOME_VALUE_CODES``, as the Procedure states. A building has a rate of its
own where the code of its ``ReplacementCost`` is among ``RATE_VALUE_CODES``
(``L``, ``N``, and ``R``) and its ``ReplacementCost`` and ``GrossArea`` are
finite numbers above zero. The rate is the value divided by ``GrossArea``. A
value with any other code, or without a ``GrossArea``, gives no rate, and a
local value without one is still kept. A value this imputer computed from a
building's own rate is also coded ``R``, so an inventory the imputer has
valued gives rates again from those values, and valuing it a second time
changes them. No area computed from the footprint enters a rate, so a rate
does not change with a world's story count. A rate never multiplies
``GrossArea``, since the own rate of a value times the building's
``GrossArea`` is that value again. The imputer applies no adjustment to a
rate.

``ReplacementCost`` often comes from the National Structure Inventory (NSI)
through ``NSIFootprintJoin``, which also gives ``GrossArea``. The join
returns the value one NSI record of a footprint states and the sum over
several records in two inventories, and ``AssetInventory.merge`` codes them
``N`` and ``R`` when they are merged under the labels ``nsi`` and
``derived``. ``brails/scrapers/nsi_parser/README.rst`` defines the NSI fields
the two features are read from, in "The fields and the features they
become", and states, in its section on NSI's footprint, how the
``GrossArea`` NSI states compares with the area of NSI's own footprint times
NSI's story count. A local inventory can supply either feature from any
other source.

A value NSI states is divided by ``GrossArea``, and the rate multiplies the
gross area computed from the footprint, so the value computed for the same
building is NSI's value divided by the ratio of ``GrossArea`` to that gross
area. For a building with one NSI record whose footprint and story count in
the inventory agree with NSI's, the value is NSI's for the median of the San
Francisco records alone on their NSI footprint, whose ratio is 1.0, and about
1.42 times NSI's value or more for the quarter of those records with a ratio
of 0.706 or less. On a footprint that holds several NSI records, where its
footprint and story count in the inventory agree with NSI's, the value is the
join's sum (the NSI README's section on NSI's footprint gives the ratio). No
scale factor is applied.

Procedure
---------

``impute_stage`` reads two inventories that earlier steps froze with
``StageFreeze``: each was frozen once under its name, and the imputer changes
neither. The first holds the buildings. The second, the donor stage, is the
inventory that ``imputation_donors_from_log`` builds from the log of
``ObservedDonorImputer.impute_stage``. It names, for each building and world,
the donor whose copied values the building kept. In each world, the imputer
reads that world's realization of both inventories. The section "Concepts" of
``brails/compilers/README.rst`` describes these terms under "Possible worlds",
"Stages and freezing", and "Donors".

The imputer first reads the own rate of every building of the stage, so
every rate comes from the values as frozen and none from a value the imputer
writes. It then values each building in ascending order of its id as a
string, an order the result does not depend on. A building whose
``ReplacementCost`` is coded ``L`` keeps that value unchanged, logged as
``OUTCOME_LOCAL_VALUE``. A building whose coordinates enclose no area gets
no value (``OUTCOME_NO_AREA``). Every other building takes the first rate
that exists of three, and its value is that rate times its gross area
computed from the footprint.

1. The building's own rate, logged as ``OUTCOME_OWN_RATE``.
2. The own rate of the donor that the donor stage names for the building
   in the world, the donor whose copied values the building kept, logged
   as ``OUTCOME_IMPUTATION_DONOR``. The donor's occupancy class is not
   compared with the building's.
3. A rate from the same-class neighbor draw, described below. A building
   without an occupancy class does not take the draw and gets no value
   (``OUTCOME_NO_CLASS``).

A building without a rate gets no value. Every stated value other than a local
one is replaced. The log holds one row per building and world, with the gross
area computed from the footprint as ``footprint_gross_area_sqft`` and the rate
the value came from as ``rate``, which on the row of a kept local value is the
building's own rate, where it has one. ``OUTCOME_VALUE_CODES`` gives the
source code of the value by the row's outcome: ``L`` for a local value, ``R``
for the building's own rate, ``I`` for another building's, and none for an
outcome it does not list. ``impute``, which reads no donor stage, values an
inventory the same way with no donor named, removes the value of a building
left without a rate, and writes the value's code into each source string,
``SOURCE_CODE_NONE`` where there is no value.

The same-class neighbor draw
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The draw builds, per occupancy class, one pool over the world's buildings of
that occupancy class that have an own rate, a local value's rate included, and
whose occupancy class is coded ``L``, ``F``, ``N``, or ``R``
(``NEIGHBOR_CLASS_CODES``). It runs the draw of ``ObservedDonorImputer`` with
that imputer's distance, pick and block-group fallback, which
``brails/imputers/observed_donor_imputer/README.rst`` describes, reading the
``NumberOfStories`` of the world. The outcome is that of
``ObservedDonorImputer``: ``OUTCOME_DONOR`` or ``OUTCOME_FEWER_THAN_K`` within
the cap, ``OUTCOME_BLOCK_GROUP`` from the block group, and ``OUTCOME_NO_DONOR``
without either. Each pick's random number is keyed as the section "Possible
worlds" under "Concepts" of ``brails/compilers/README.rst`` describes, with a
purpose label that keeps it apart from every other draw,
``same_class_neighbor:observed_donor`` or
``same_class_neighbor:observed_donor_fallback``, which
``NEIGHBOR_PURPOSE_PREFIX`` forms.

Scope
-----

A building that ``ObservedDonorImputer`` filled takes the rate of the donor
it copied from, so the value comes from the building that supplied its other
missing features. That donor lies within the cap of ``ObservedDonorImputer``
or in the building's block group. Its occupancy class can differ from the
building's where that imputer relaxed its occupancy class filter because
fewer than ``k`` donors matched, or where the building's occupancy class
came from another source, and its rate is used all the same.

The same-class neighbor draw serves every building that has no own rate and no
donor with one. The measurements below are of records of the NSI 2026 Base. A
record's rate is the value that ``brails/scrapers/nsi_parser/README.rst``
calls NSI's full replacement value, over its ``GrossArea``. Within one county,
NSI's rate varies with location only for single-family homes, through the
adjustment NSI makes for the income of the Census tract. Across the 210 San
Francisco tracts that hold a ``RES1-1SNB`` record, the tract medians of the
rate for that subtype run from 220 dollars per square foot at the tenth
percentile to 428 at the ninetieth. Outside ``RES1``, records of one county
with the same NSI occupancy type (NSI's occupancy code, which adds a subtype
for single-family homes) and the same ``GrossArea`` carry the same rate in
each of the nine Bay Area counties, and 357,024 of the 687,615 records outside
``RES1`` share both with another record of their county. Over the 46,183 pairs
of occupancy type and ``GrossArea`` found outside ``RES1`` in more than one
county, the rate differs between counties by at most 9.8 percent. Outside
``RES1`` the rate falls slightly and consistently as ``GrossArea`` grows. Over
all San Francisco records of ``COM1``, ``COM4``, ``RES3A``, ``RES3C``,
``RES3F``, and ``IND2``, doubling ``GrossArea`` lowers the rate by 2.4 to 3.9
percent within each occupancy class, and the correlation of the logarithms of
rate and ``GrossArea`` lies between -0.98 and -0.81. For ``RES1-1SNB``, over
the whole county, the correlation is -0.06, since the tract income adjustment
spreads the rate. Within one tract the rate falls by about 7 percent per
doubling of ``GrossArea``, and the correlation is -0.73.

The draw therefore weighs plan area and story count beside the location, as
``ObservedDonorImputer`` does. To measure how far the two terms narrow the
spread of rates among neighbors, San Francisco's records alone on their NSI
footprint were ranked by those two terms alone, computed from NSI's own
footprint area and story count, with location left out, within the 12
occupancy classes outside ``RES1`` that have 300 or more such records
(43,339 records), at an area weight of 2, a story weight of 1, and the 10
nearest. For up to 1,000 records per occupancy class, with ties in the
distance broken at random, each record's error is the mean relative
difference between its rate and the rates of its 10 nearest other records of
the occupancy class, the mean of ``|r_c / r_t - 1|`` with ``r_t`` its rate
and ``r_c`` a neighbor's. The median of that error over the records of an
occupancy class is at most 5 percent with the plan-area term alone (4.2 to
4.5 percent in ``RES3B`` over two samples of records, and at most 2.2
percent outside the ``RES3`` occupancy classes), and at most 0.3 percent
with both terms. With both terms the result is the same at the imputer's
default weights of 1 and 1.

The occupancy class filter also pools NSI's single-family subtypes, since the
occupancy class ``NSIFootprintJoin`` gives a footprint carries no subtype
(``brails/scrapers/nsi_parser/README.rst``, "The fields and the features
they become"). In San Francisco the median rate for ``RES1-1SNB`` is 316
dollars per square foot and for ``RES1-1SWB`` 381. A candidate whose occupancy
class was itself imputed or drawn is left out of the draw
(``NEIGHBOR_CLASS_CODES``).

The draw's five defaults are those of ``ObservedDonorImputer``, which
``brails/imputers/observed_donor_imputer/README.rst`` states under "Defaults",
and none is calibrated yet for the value. A caller can pass all five to the
constructor, and a run that also imputes occupancy class and stories should
pass the values it gives ``ObservedDonorImputer``.
