Residential neighbor year sampler
=================================

``ResidentialNeighborYearSampler`` copies to a building the construction year
of a nearby residential building that holds one in the same possible world. A
building is residential here when its occupancy class is one of the residential
occupancy classes, those of ``UNIT_COUNT_CLASSES``: RES1, RES2 and RES3A to
RES3F, without RES4 to RES6. It runs the donor search of
``ObservedDonorImputer``, in
``brails/imputers/observed_donor_imputer/observed_donor_imputer.py``, on the
residential buildings of a frozen stage, and takes its defaults from that
module. ``residential_neighbor_year_sampler.py`` beside this page holds the
class. This page states where the defaults come from, how the sampler picks a
donor and why the search is configured as it is, and which buildings the
sampler reads. The docstrings state what a caller passes and receives.

The defaults
------------

``k`` and ``cap_m`` default to the observed-donor imputer's
``DEFAULT_DONOR_K``, 20 donors, and ``DEFAULT_DONOR_CAP_M``, 2,000 m, so a
caller that passes the imputer's values to both draws from neighborhoods
of one size. A caller passes other values as ``k`` and ``cap_m``. The
number of donors is not calibrated yet, for the imputer or for copying
years. The cap is the imputer's own bound and is not calibrated yet either.

The method
----------

The sampler gives each building outside the residential occupancy classes
the construction year of a residential building near it. The donors are the
residential buildings of one world of a frozen stage that hold a year. Each
building draws one of its nearest donors, with weights that fall with
distance, or a donor of its block group where none lies within the cap, and
takes that donor's year. The section "Concepts" of
``brails/compilers/README.rst`` describes these terms under "Possible
worlds", "Stages and freezing", and "Donors".

``select_donors`` reads one world of a frozen stage: the world the sampler
is constructed with where the stage holds several, the stage as it is where
it holds one. It takes as donors the buildings whose ``occupancy_key`` is in
``UNIT_COUNT_CLASSES``, in ``brails/utils/unit_count_rules.py``, and whose
``year_built_key`` holds a whole number, an integer or an integral float,
whatever its source. An asset whose
``occupancy_key`` or ``year_built_key`` still holds one value per world in
the world read, as in a stage of one world that holds a list, is refused.

Each call of ``check``, ``draw`` or ``draw_details`` indexes the donors and
the sites of the inventory's buildings. The index leaves out every building
whose coordinates enclose no area: such a donor is never drawn and
``donors`` does not list it, and ``check`` refuses such a building.

Each building then draws one donor among the ``k`` nearest within
``cap_m``, or among all of them where fewer lie within the cap, donors at
equal distance ranked in ascending order of their id as a string. The
weights are one over the distance, taken as at least
``MIN_WEIGHT_DISTANCE`` of the donor search. The distance is location
alone: the great-circle distance in meters between centroids. The
imputer's own distance also compares log plan area and story count, which
ranks a building against others of its kind. Every donor here is of
another kind, a residential building for a building outside the residential
occupancy classes, so those terms would rank the donors by how much they
resemble the building in size, not by where they stand.

A building with no donor within the cap draws one donor of its block
group, read from ``BLOCK_GROUP_KEY``, with equal weights. The fallback
keeps the copied year of such a building inside its block group; in a
block group that covers a large area the donor may lie beyond the cap.
``check`` refuses a building with a donor in neither place, with the
diagnosis ``no_donor``, and says why.

The donor's year is copied. Both the pick among the nearest donors and the
block-group pick are the draw of ``ObservedDonorImputer``, run with the
building's status given as residential. The draw then matches the residential
donors; on its own occupancy class, which lies outside the residential
occupancy classes, it would match none. The pick's uniform is keyed as the
section "Possible worlds" under "Concepts" of ``brails/compilers/README.rst``
describes, by the building's id among other keys, so a building's draw depends
on no other building's. Its purpose label is ``PURPOSE_DONOR`` or, in the block
group, ``PURPOSE_FALLBACK`` of the observed-donor imputer, with
``NEIGHBOR_YEAR_PURPOSE_PREFIX`` and a colon put before it, which keeps it
apart from the imputer's own draws.

Scope
-----

Where the donors' years were drawn, by ``ACSYearBuiltSampler`` or another
sampler, a copied year rests on a draw. The sampler draws for buildings outside
``UNIT_COUNT_CLASSES``, the occupancy classes ``ACSYearBuiltSampler`` covers,
and so for RES4 to RES6, and refuses a building inside them.

NSI's tract median year, ``YearBuiltTractMedian``, is not the year of a
building and is never read as one.

Each world of a study takes its own sampler, with the seed of the run.
