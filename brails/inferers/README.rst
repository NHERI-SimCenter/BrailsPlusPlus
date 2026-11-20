Inferers
========

The modules of ``brails.inferers`` add features to the assets of an inventory,
derived from the features they already hold or drawn from published tables.
This page describes the interface of the inferers listed below, the words their
pages use, and where each is described in full. These are the inferers that the
classes of ``brails.compilers`` use. Other inferers in this folder, such as
``HazusInfererWind`` and ``HazusInfererFlood``, have interfaces of their own,
which this page does not describe. Possible worlds and the keyed random numbers
are introduced in the section "Concepts" of ``brails/compilers/README.rst``.

The interface
-------------

Each of these inferers but ``ACSUnitBudget`` has the three methods below.
Their base class, ``InferenceEngine`` in ``inference_engine.py``, requires
``infer`` alone. A caller can run the first two one after the other or let
the third run both:

- ``check`` diagnoses every asset of an inventory and changes nothing. It
  returns a diagnosis frame, one row per asset id with the columns of
  ``DIAGNOSIS_COLUMNS``.
- ``draw`` computes or draws the feature for every asset, assuming that all
  of them passed ``check``, and returns the values indexed by asset id
  without writing them. An asset that does not pass raises ``ValueError``,
  since it means ``check`` was skipped. A configuration the inferer cannot
  use also raises.
- ``infer`` runs both and returns a copy of the inventory with the values
  written. If any asset fails ``check``, it raises ``ValueError`` naming the
  assets and their diagnoses.

``infer`` writes the feature whether or not the asset already holds one.
Keeping an observed value is the caller's decision.

An inferer never skips an asset in silence. A caller who wants to write
only the assets that can take a value calls ``check``, keeps those
diagnosed ``OK``, and calls ``draw`` or ``infer`` on them. An inferer that
computes its value without drawing, such as ``HazusEQHeightClassInferer``,
keeps the same three methods, so a caller screens and applies every
inferer the same way.

The diagnosis frame
~~~~~~~~~~~~~~~~~~~

``inference_diagnosis.py`` holds the verdicts shared by the inferers of
every hazard, the columns and the helpers ``diagnosis_frame``, which builds
a frame from per-asset verdicts, and ``preview_failures``, which formats the
failing assets for an error message. The column ``diagnosis`` holds one of five
verdicts:

- ``DIAGNOSIS_OK``, ``'OK'``: the inferer can process the asset.
- ``DIAGNOSIS_INVALID``, ``'invalid'``: a feature the inferer reads, a
  driver, is absent or holds a value the inferer cannot use. The caller
  obtains the data, corrects it or imputes a replacement.
- ``DIAGNOSIS_UNSUPPORTED``, ``'unsupported'``: a driver holds a real value
  in a shape the inferer does not handle. The caller excludes the
  asset, since imputing over it would replace data the inventory holds.
- ``DIAGNOSIS_INFEASIBLE``, ``'infeasible'``: every driver is usable and
  the inferer still has nothing to produce. The caller relaxes one of the
  features in ``resolvable_by``.
- ``DIAGNOSIS_NO_DONOR``, ``'no_donor'``: every driver is usable and no asset
  could serve as the donor of the value. The caller widens the search for
  donors or relaxes one of the features in ``resolvable_by``. No inferer
  returns it. ``ResidentialNeighborYearSampler``, an imputer that follows the
  same check contract, does.

``detail`` says what happened in words. ``resolvable_by`` names the
features that would each, alone, let the inferer produce a value. An
unsupported asset names none, and an invalid one names the driver a caller
could supply or re-draw where the inferer reads one driver and can say
which.

Draws
~~~~~

An inferer that draws keys its random numbers as the section "Possible worlds"
under "Concepts" of ``brails/compilers/README.rst`` describes. ``draw`` takes
the caller's attempt number as ``iteration``, and distinct iterations draw
independently for the same asset. An inferer that draws for one world takes
that world when it is constructed.

Shared words
------------

A *driver* is a feature an inferer reads to produce its value. A *verdict*
is the content of the ``diagnosis`` column. To *relax* a feature is to
give up its value so that the inferer can produce one, as
``resolvable_by`` lists.

The modules
-----------

- ``ACSUnitBudget``, in ``acs_unit_budget/``, places the housing units a
  census table counts in a block group, less those of the footprints that
  already hold an occupancy class and a unit count, on the other footprints.
  It runs through ``allocate``, which takes the footprints of whole block
  groups, and has no ``check``, ``draw`` or ``infer``.
  Described in ``acs_unit_budget/README.rst``.
- ``ACSYearBuiltSampler``, in ``acs_year_built_sampler/``, draws a construction
  year for each residential asset from two American Community Survey tables.
  The residential occupancy classes here are those of ``UNIT_COUNT_CLASSES``:
  RES1, RES2 and RES3A to RES3F, without RES4 to RES6. It is fitted on the
  inventory with ``fit`` before ``check``, ``draw`` and ``infer``. Described in
  ``acs_year_built_sampler/README.rst``.
- ``StoriesFromHeightInferer``, in ``stories_from_height_inferer/``,
  converts a footprint height to a story count with the meters per story of
  the height's source. Described in
  ``stories_from_height_inferer/README.rst``.
- ``HazusEQStructureTypeInferer``, ``HazusEQDesignLevelInferer`` and
  ``HazusEQHeightClassInferer``, in ``hazus_inferer_earthquake/``, draw a
  Hazus structure type, assign a design level and assign a height class
  from tables of the Hazus Inventory Technical Manual. The tables are
  described in ``hazus_inferer_earthquake/README.rst``.
- ``ResidentialNeighborYearSampler``, in
  ``brails/imputers/residential_neighbor_year_sampler/``, copies to a
  building the construction year of a nearby residential one, with this
  interface. It takes its donors with ``select_donors`` before ``check``,
  ``draw`` and ``infer``. Described in
  ``brails/imputers/residential_neighbor_year_sampler/README.rst``.
