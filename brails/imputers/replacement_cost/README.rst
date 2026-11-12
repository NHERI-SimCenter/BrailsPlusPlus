Replacement cost imputer
========================

``ReplacementCostImputer`` gives a building without a ``ReplacementCost`` the
value per square foot of gross floor area of one donor, times its own gross
floor area. Its constants are in ``replacement_cost_imputer.py`` beside this
page, and the defaults of its separate draw, ``DEFAULT_DONOR_K`` and
``DEFAULT_DONOR_CAP_M``, are in
``brails/imputers/observed_donor_imputer/observed_donor_imputer.py``. This
page states what the value and the rate are measured in, what the NSI records
they come from hold, and the scope of the donors and the defaults. The class
docstring states the procedure.

The value and the rate
----------------------

The value is ``val_struct`` of the National Structure Inventory (NSI), which
NSI's technical documentation for the 2026 Base defines as the depreciated
replacement value of the structure, in August 2025 price levels. It excludes
contents, which NSI states separately as ``val_cont``, and it is not NSI's
full replacement value, ``fullrep``, which the join does not use. BRAILS
applies no price-level adjustment. The NSI footprint join writes
``val_struct`` as ``ReplacementCost``, coded ``N`` where one record of the
footprint states it and ``R`` where the join sums several.
``NSI_VALUE_CODES`` holds those two codes, so only a value NSI states gives a
rate. A donor's rate carries the depreciation NSI gave the donor for its age
and tract.

The rate is that value divided by the building's ``GrossArea``, which the join
writes from NSI's ``sqft``, a floor area. The target's floor area follows the
same definition where it can: its own ``GrossArea``, else its plan area times
its number of stories, else its plan area as one story, the gross floor area
``BlockGroupUnitBudget`` reads. Over the 165,152 records for San Francisco in
the NSI 2026 Base, ``sqft`` divided by the area of NSI's own footprint times
its stories has a median of 0.9252 and quartiles of 0.5546 and 1.0. The 16.5
percent of records that share a footprint with another record have a median of
0.118, and the others a median of 1.0 and a lower quartile of 0.706. Where a
donor's ``sqft`` is the fraction r of its footprint area times its stories, a
target measured by plan area and stories receives 1/r times the value it would
receive if both floor areas were footprint area times stories, about 8 percent
more at the median r. The ratio has not been measured against the footprints
the join assigns NSI records to.

Of the 2,243,922 records of the NSI 2026 Base for the nine San Francisco Bay
Area counties, every record holds a ``val_struct`` and a ``sqft``, and both are
positive.

Scope
-----

A target that the observed-donor imputer filled takes the rate of the donor
it copied from, so the value comes from the building that supplied its other
missing features. That donor lies within the imputer's cap or in the target's
block group. Its class can differ from the target's where the imputer
relaxed its class filter for want of donors, or where the target's class came
from another source, and its rate is used all the same.

Every other target takes the separate draw, which ranks the buildings of the
target's class by location alone. Location matters within one class: across the
210 San Francisco tracts of the same records, the tract medians of NSI's value
per square foot for ``RES1-1SNB`` run from 145 dollars at the tenth percentile
to 348 at the ninetieth. NSI's rate also falls with floor area within a class
and varies with the structure's age through its depreciation, and the separate
draw does not rank by either. The class filter also pools NSI's subtypes, since
the join keeps only the class before the hyphen. In San Francisco the median
for ``RES1-1SNB`` is 254 dollars per square foot and for ``RES1-1SWB`` 304. A
candidate's class must be coded ``L``, ``F``, ``N`` or ``R``
(``DONOR_CLASS_CODES``), so no value rests on a class that was itself imputed
or drawn.

The separate draw's defaults, ``DEFAULT_DONOR_K`` and ``DEFAULT_DONOR_CAP_M``,
are the observed-donor imputer's and are not yet calibrated for the value. A
calibration against masked NSI values is planned. A caller passes ``k`` and
``cap_m`` to the constructor, and a run that also imputes class and stories
should pass the values it gives the observed-donor imputer.
