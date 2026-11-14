Neighbor year sampler
=====================

``NeighborYearSampler`` copies to a building the construction year of a
nearby residential building that holds one in the same possible world. It
runs the donor search of ``ObservedDonorImputer``, in
``brails/imputers/observed_donor_imputer/observed_donor_imputer.py``, on the
residential buildings of a frozen stage, and takes its defaults from that
module. ``residential_neighbor_year_sampler.py`` beside this page holds the
class. This page states where the defaults come from, why the search is
configured as it is, and which buildings the sampler reads. The class
docstring states the procedure.

The defaults
------------

``k`` and ``cap_m`` default to the observed-donor imputer's
``DEFAULT_DONOR_K``, 20 donors, and ``DEFAULT_DONOR_CAP_M``, 2,000 m, so a
caller that passes the imputer's values to both draws from neighborhoods
of one size. A caller passes other values as ``k`` and ``cap_m``. The
number of donors is not calibrated yet, for the imputer or for copying
years. It can be calibrated against buildings outside the residential
classes whose recorded year is held out, and it will be revisited when
that comparison is made. The cap is the imputer's own bound.

The distance is location alone: the great-circle distance in meters
between centroids. The imputer's own distance also compares log plan area
and story count, which ranks a building against others of its kind. Every
donor here is of another kind, a residential building for a building
outside the residential classes, so those terms would rank the donors by
how much they resemble the building in size, not by where they stand.

The fallback to the block group keeps the copied year of a building with
no donor within the cap inside that building's block group. ``check``
refuses a building with a donor in neither place and says why.

Scope
-----

The donors are the buildings of the stage in ``RESIDENTIAL_CLASSES``, the
classes ``ACSYearBuiltSampler`` covers, that hold a whole-number year,
whatever its source. Where those years were drawn, by
``ACSYearBuiltSampler`` or another sampler, a copied year rests on a draw.
The sampler draws for buildings outside those classes, RES4 to RES6
included, and refuses a building inside them.

NSI's tract median year, ``YearBuiltTractMedian``, is not the year of a
building and is never read as one. A building whose coordinates enclose no
area is neither drawn for nor a donor.

A sampler reads the donors of the one world it is constructed with, so
each world of a study takes its own sampler, with the seed of the run. A
copied year depends on the residential buildings within the cap of its
building, or in its block group where none lies within the cap, and on no
other copied year.
