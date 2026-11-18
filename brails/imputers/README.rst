Imputers
========

The modules of ``brails.imputers`` fill values that an inventory lacks from the
values other buildings hold. This page describes the interface of the imputers
listed below, the words their pages use, and where each is described in full.
These are the imputers that the classes of ``brails.compilers`` use. The other
imputer in this folder, ``KnnImputer``, has an interface of its own, which this
page does not describe. Possible worlds, frozen stages and donors are
introduced in the section "Concepts" of ``brails/compilers/README.rst``.

The interface
-------------

``ObservedDonorImputer`` and ``ReplacementCostImputer`` derive from
``Imputation``, in ``imputation.py``, and have ``impute``, which takes an
inventory and returns a new one with the imputer's values written: the
missing values filled, and, for ``ReplacementCostImputer``, every stated
value but a local one recomputed. The inventory passed in is not modified.
``ResidentialNeighborYearSampler`` has the interface of an inferer instead (see
the list below).

Both imputers also have ``impute_stage``, which runs them as one stage of a
longer computation. It can read its inputs from a ``StageFreeze`` by stage
name, so it reads them as the earlier stages froze them, and it changes no
frozen stage. ``ObservedDonorImputer`` reads its observed donors there, and
``ReplacementCostImputer`` the buildings, the donor stage that
``imputation_donors_from_log`` builds and, where the caller names one, the
stage that holds the buildings' gross areas. It returns a result that holds the
filled values as an inventory, a log and the seed used. ``impute`` runs the
same procedure on one inventory, and each imputer's page says what it takes
as donors there.

A stage run serves several possible worlds and draws each with that
world's index ("Possible worlds" under "Concepts" in
``brails/compilers/README.rst``). An imputer constructed without a seed
draws one and records it as ``seed``.

The log has a row for each building and world the imputer handled, and
``ObservedDonorImputer`` one for each round it ran. Its column ``outcome``
holds one of the imputer's ``OUTCOME_`` constants and says where the
building's value came from, or why it has none. A building the imputer
cannot fill is logged, not skipped in silence, and arguments the imputer
cannot use raise an error that names the problem.

Shared words
------------

Target, donor and observed donor are defined under "Donors" in the section
"Concepts" of ``brails/compilers/README.rst``. A *purpose label* names the kind
of random choice a uniform random number serves, so that two choices about one
building in one world use different numbers. The *block group* is the census
block group a building lies in, read from ``BLOCK_GROUP_KEY``, where a target
without a donor nearby looks for one.

The modules
-----------

- ``ObservedDonorImputer``, in ``observed_donor_imputer/``, copies a
  target's missing features, by default its occupancy class, units and
  story count, from one nearby observed donor per world. Described in
  ``observed_donor_imputer/README.rst``.
- ``ReplacementCostImputer``, in ``replacement_cost/``, values every building
  as a rate per square foot times its gross area, computed from its footprint
  unless the caller names a feature that holds it, taking the rate from the
  building itself, its donor or a nearby building of the same occupancy class,
  drawn by the same-class neighbor draw. Described in
  ``replacement_cost/README.rst``.
- ``ResidentialNeighborYearSampler``, in
  ``residential_neighbor_year_sampler/``, copies to a building the construction
  year of a nearby residential building. The residential occupancy classes here
  are those of ``UNIT_COUNT_CLASSES``: RES1, RES2 and RES3A to RES3F, without
  RES4 to RES6. It runs the donor search of ``ObservedDonorImputer`` but has
  the interface of an inferer, ``check``, ``draw`` and ``infer``, which
  ``brails/inferers/README.rst`` describes. Described in
  ``residential_neighbor_year_sampler/README.rst``.
