Replacement cost imputer
========================

``ReplacementCostImputer`` values every building, in each world, as a rate
per square foot times the building's floor area, and keeps a value from a
local source. Its constants are in ``replacement_cost_imputer.py`` beside
this page, and the defaults of its same-class neighbor draw,
``DEFAULT_DONOR_K`` and ``DEFAULT_DONOR_CAP_M``, are in
``brails/imputers/observed_donor_imputer/observed_donor_imputer.py``. This
page states what the value and the rate are measured in and what the NSI
records they come from hold, the procedure, and the scope of the rates and of
the defaults.

The value and the rate
----------------------

The rates come from ``fullrep`` of the National Structure Inventory (NSI),
which NSI's technical documentation for the 2026 Base defines as the full, not
depreciated, estimated replacement value of the structure. NSI estimates this
replacement cost new, in August 2025 price levels, as a cost per square foot
for the occupancy type, which generally falls as the floor area grows, times
the floor area. The cost is adjusted for the county's construction wages and,
for single-family homes, for the income of the Census tract relative to that
of its Census division. The documentation states that no adjustments are made
"for differences in material costs or differences in regional construction
practices". ``fullrep`` excludes contents, which NSI states separately as
``val_cont``, and it is not NSI's depreciated replacement value,
``val_struct``. BRAILS applies no price-level adjustment. The NSI footprint
join writes ``fullrep`` as ``ReplacementCost``, coded ``N`` where one record
of the footprint states it and ``R`` where the join sums several.
``NSI_VALUE_CODES`` holds those two codes, so only a value NSI states gives a
rate, and no rate carries a depreciation.

A building's own rate is that value divided by its ``GrossArea``, which the
join writes from NSI's ``sqft``, a floor area, as the sum over the same
records. The rate multiplies the building's plan area times its number of
stories, never its ``GrossArea``, since the rate times ``GrossArea`` is NSI's
value itself. The two floor areas differ. Over the 165,152 records for San
Francisco in the NSI 2026 Base, ``sqft`` divided by the area of NSI's own
footprint (``ftprntsqft``) times its stories (``num_story``) has a median of
0.9252 and quartiles of 0.5546 and 1.0. The 137,869 records alone on their
NSI footprint have a median of 1.0 and a lower quartile of 0.706, and the
27,283 records (16.5 percent) that share a footprint with another record have
a median of 0.118. For a building with one NSI record whose Overture
footprint and story count agree with NSI's, the value is therefore NSI's value
at the median ratio, and 1.42 times NSI's value or more for the quarter of
such records with a ratio of 0.706 or less. On the 7,849 NSI footprints of San
Francisco that hold several records (the 27,283 records above), the sum of
``sqft`` over a footprint's records divided by its area times its stories lies
between 0.99998 and 1.00002, so where Overture's footprint and story count
agree with NSI's, the rule returns the join's sum of ``fullrep``. The ratio
against the footprints the join assigns NSI records to has not been measured.
No scale factor is applied.

Of the 2,243,922 records of the NSI 2026 Base for the nine San Francisco Bay
Area counties, every record holds a ``fullrep`` and a ``sqft``, and both are
positive.

Procedure
---------

In each world the imputer reads that world's realization of the frozen stage
and of the donor stage. A building whose ``ReplacementCost`` is coded ``L``
keeps that value, logged as ``OUTCOME_LOCAL_VALUE``, and never gives a rate.
The imputer first reads the own rate of every building of the stage, so
every rate comes from the values as frozen and none from a value the imputer
writes. A building has an own rate where the code of its
``ReplacementCost`` is among ``NSI_VALUE_CODES`` and its ``ReplacementCost``
and ``GrossArea`` are finite numbers above zero.

Each building is then valued in ascending order of its id as a string, an
order the result does not depend on. Its floor area in square feet is its
plan area times its ``NumberOfStories`` in the world where that is a finite
number above zero, logged as ``AREA_FROM_STORIES``, and otherwise its plan
area as one story, logged as ``AREA_FROM_PLAN_AREA``. This is the gross floor
area ``ACSUnitBudget`` reads for a footprint without a ``GrossArea``.
The budget reads a ``GrossArea`` first, and this imputer never does. A
building whose coordinates enclose no area gets no value
(``OUTCOME_NO_AREA``).

The rate is the first that exists of three. The building's own rate is
logged as ``OUTCOME_OWN_RATE``. Without one, the building takes the own rate
of the donor that the donor stage names for it in the world, logged as
``OUTCOME_IMPUTATION_DONOR``. ``imputation_donors_from_log`` names that donor
from the log of ``ObservedDonorImputer.impute_stage``: the building whose
features the observed-donor imputer copied to this one, the last re-draw's
where its validation re-drew. The donor's class is not compared with the
building's. Without a donor's rate, a building without a class gets no value
(``OUTCOME_NO_CLASS``), and one with a class takes the same-class neighbor
draw.

The same-class neighbor draw builds, per class, one pool over the world's
buildings of that class that have an own rate and whose class is coded
``L``, ``F``, ``N`` or ``R`` (``NEIGHBOR_CLASS_CODES``). It runs the
observed-donor imputer's draw with that imputer's distance::

    sqrt((g / location_scale_m)**2 + area_weight * (ln A_t - ln A_c)**2
         + stories_weight * (n_t - n_c)**2),

where g is the great-circle distance in meters between the building and the
candidate, A_t and A_c are their plan areas, and n_t and n_c their
``NumberOfStories`` in the world. The story term is zero where either side
has no story count. One of the ``k`` nearest candidates within ``cap_m`` is
picked with inverse-distance weights, and without one there, a candidate of
the building's block group with equal weights. The outcome is the
observed-donor imputer's: ``OUTCOME_DONOR`` or ``OUTCOME_FEWER_THAN_K`` within
the cap, ``OUTCOME_BLOCK_GROUP`` from the block group, ``OUTCOME_NO_DONOR``
without either. The uniform of each pick depends on the building's id, the
seed and the world, never on the order of the buildings, under the labels
that ``NEIGHBOR_PURPOSE_PREFIX`` forms, ``same_class_neighbor:observed_donor``
and ``same_class_neighbor:observed_donor_fallback``.

The value is the rate times the floor area, a float, not rounded. A building
without a rate gets no value. Every stated value other than a local one is
replaced. The log holds one row per building and world, and
``OUTCOME_VALUE_CODES`` gives the source code of the value by the row's
outcome: ``L`` for a local value, ``R`` for the building's own rate, ``I`` for
another building's, and none for an outcome it does not list. ``impute``,
which has no donor stage, values an inventory the same way with no donor
named, removes the value of a building left without a rate, and writes the
value's code into each source string, ``-`` where there is no value.

Scope
-----

A building that the observed-donor imputer filled takes the rate of the donor
it copied from, so the value comes from the building that supplied its other
missing features. That donor lies within the imputer's cap or in the
building's block group. Its class can differ from the building's where the
imputer relaxed its class filter for want of donors, or where the building's
class came from another source, and its rate is used all the same.

The same-class neighbor draw serves every other building without an own rate.
Within one county, NSI's rate varies with location only for single-family
homes, through the tract income adjustment. Across the 210 San Francisco
tracts that hold a ``RES1-1SNB`` record, the tract medians of NSI's value per
square foot for that subtype run from 220 dollars at the tenth percentile to
428 at the ninetieth. Outside ``RES1``, records of one county with the same
``occtype`` and ``sqft`` carry the same rate in each of the nine Bay Area
counties, and 357,024 of the 687,615 records outside ``RES1`` share both with
another record of their county. Over the 46,183 pairs of ``occtype`` and
``sqft`` found outside ``RES1`` in more than one county, the rate differs
between counties by at most 9.8 percent. Outside ``RES1`` the rate falls
slightly and consistently with floor area. Over all San Francisco records of
``COM1``, ``COM4``, ``RES3A``, ``RES3C``, ``RES3F`` and ``IND2``, doubling
``sqft`` lowers the rate by 2.4 to 3.9 percent within each class, and the
correlation of the logarithms lies between -0.98 and -0.81. For
``RES1-1SNB`` the correlation is -0.06.

The draw therefore weighs plan area and story count beside the location, as
the observed-donor imputer does. To measure what the two terms give, San
Francisco's records alone on their NSI footprint were ranked by those terms
alone, location left out, within the 12 classes outside ``RES1`` that have
300 or more such records (43,339 records), at the area weight of 2, the
story weight of 1 and the 10 nearest that the compiler uses. For up to 1,000
records per class, with ties in the distance broken at random, the median
over those records of the mean relative difference between a record's rate
and the rates of its 10 nearest other records of the class is at most 5
percent with the plan-area term alone (4.3 to 4.5 percent in ``RES3B`` over
two samples of records, and at most 2.2 percent outside the ``RES3``
classes), and at most 0.3 percent with both terms. With both terms the
result is the same at the imputer's default weights of 1 and 1.

The class filter also pools NSI's subtypes, since the join keeps only the
class before the hyphen. In San Francisco the median for ``RES1-1SNB`` is 316
dollars per square foot and for ``RES1-1SWB`` 381. A candidate's class code
excludes a class that was itself imputed or drawn.

The draw's defaults are not yet calibrated for the value: ``k`` and ``cap_m``
default to ``DEFAULT_DONOR_K`` and ``DEFAULT_DONOR_CAP_M``, and
``location_scale_m``, ``area_weight`` and ``stories_weight`` to
``DEFAULT_LOCATION_SCALE_M``, ``DEFAULT_AREA_WEIGHT`` and
``DEFAULT_STORIES_WEIGHT``, 100 meters, 1 and 1, the observed-donor imputer's
own defaults. A caller passes all five to the constructor, and a run that also
imputes class and stories should pass the values it gives the observed-donor
imputer.
