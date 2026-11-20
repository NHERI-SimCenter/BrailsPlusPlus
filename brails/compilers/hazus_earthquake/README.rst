Hazus earthquake inventory compiler
===================================

``HazusEQInventoryCompiler`` builds the building inventory that a Hazus
earthquake damage and loss assessment reads, for one study area and over
several possible worlds. It is in ``hazus_eq_inventory_compiler.py`` beside
this page, with ``CompilerInputs``, which holds the data of the study area, and
``CompilationResult``, which holds what a run produces. Three classes in the
same folder carry out pieces of the run: ``PairRedraw`` in ``pair_redraw.py``,
and ``ComposedScreen`` and ``ResolutionLoop`` in ``resolution_loop.py``. This
page states what the compiler produces and from which inputs, the settings it
gives the observed-donor imputer, the run step by step in the order the code
runs it, the frozen stages, the possible worlds, the local inputs, the block
groups, a boundary, a compilation from a place, the written file and the scope, then
the three classes. The terms this page shares with the general compilers page, such
as possible world, stage, part, record, view, computed stage, source and donor, are
defined in the section "Concepts" of ``brails/compilers/README.rst``. Its section
"Composition and assembly" states how a source is put together from frozen stages.
The docstrings state what a caller passes and receives.

The inventory and its inputs
----------------------------

A run produces one building for each footprint it keeps. A feature whose
value is uncertain, such as a construction year drawn from census tables, can
take a different value in each possible world. The run computes the
compiler's ``n_worlds`` worlds. A building that a step cannot complete in a
world, or finds inconsistent there, is excluded from that world: it stays in
the inventory, its feature ``Excluded`` is true in that world, and the run's
report names the step that excluded it and the reason. In each world it is
not excluded from, a building holds the features of ``DEFAULT_PROPERTIES``
that apply to it: its occupancy class, unit count, story count, construction
year, design level, structure type, height class, replacement cost, census
block group, plan area, gross area, general building type and source string.
It also holds the other features the run gave it, such as the values of the
NSI records joined to it.

On this page, the residential occupancy classes are those of
``UNIT_COUNT_CLASSES`` in ``brails/utils/unit_count_rules.py``: RES1, RES2
and RES3A to RES3F. RES4 to RES6, such as hotels and nursing homes, are not
among them, so the run gives them the years of residential neighbors and not
census years (see ``s06_residential_year`` and ``s09_nonresidential_year``).

The caller gives the data of the study area in a ``CompilerInputs``:

- the building footprints, as polygons, each carrying ``type`` where the file
  is to be written with ``integer_ids``;
- the points of the National Structure Inventory (NSI), as the NSI reader
  returns them;
- optionally, three tables of the American Community Survey: B25024, the housing
  units by the number of units in their structure, per block group; B25034, the
  housing units by the year their structure was built, per block group and per tract;
  and B25127, the occupied housing units by year built and structure size, per tract.
  A table left as ``None`` is read from the Census Data API (see ``s01_footprints``);
- optionally, the points of the mobile-home parks, which are read from HIFLD where
  they are left as ``None`` (see ``s01_footprints``);
- optionally, a local inventory of values known locally, which takes
  precedence over every other source (see "Local inputs" below);
- optionally, the polygons of the census block groups (see "Block groups"
  below);
- optionally, a boundary, the place to compile inside the area the footprints
  cover (see "A boundary" below);
- optionally, the state of the study area and whether to merge its stacked
  footprints, which a compiler built without these settings applies;
- the Overture release of the footprints, the length unit of their heights
  and a label of the study area for the report.

The constructor of the compiler takes the settings of the run: the seed of
every random draw, the number of worlds, the state, the seismic zone and the
design level method that the Hazus inferers read, the distance cap of the NSI
join and the projected coordinate system of its distances, the options of
three components, whether to merge stacked footprints first, ``acs_vintage``, the ACS
release of the census tables the run reads, and ``block_group_vintage``, the
TIGER/Line release of the block groups it reads. Given no
seed, the compiler draws one, which ``CompilationResult.seed`` and the
report's summary hold, so that the run can be repeated with the same inputs (see
``s01_footprints`` for those the compiler reads). The constructor
checks the settings. It also builds four of the components, so that each can
refuse its own settings and a bad setting raises an error in the
constructor, before any data is read.

The state, the seismic zone and ``merge_stacked`` default to ``None``. A compilation
then applies the state the inputs name, or ``DEFAULT_STATE`` where they name none,
the zone ``DEFAULT_SEISMIC_ZONE``, and the inputs' ``merge_stacked``, which is off
unless set. Before it reads any data, ``compile`` refuses, in this order, inputs that
name a state other than the compiler's, a state the compiler cannot take, and a state
other than ``DEFAULT_STATE`` when the compiler was built without a seismic zone.
Where the inputs name ``DEFAULT_STATE`` and no zone was given, a notice names the
zone applied.

``compile`` returns a ``CompilationResult``. Its ``inventory`` is the compiled
inventory over every world, with each building's ``Excluded`` true in the
worlds it is excluded from. Its ``report`` is a ``CompilationReport``: one row
for each change a step made to a value and one row for each building a step
excluded from a world, as the section "The compilation report" of
``brails/compilers/README.rst`` states. Its ``freeze`` holds every frozen
stage of the run and its ``manifest`` their digests. Its ``logs`` hold, by
name, the logs of the components and those the compiler writes itself, such
as ``gross_area`` and ``schema_check``. Its ``seed`` is the seed of the run's
random draws. ``write`` saves the inventory as GeoJSON with the report's
summary and exclusions and the manifest beside it, and with ``detailed=True``
the report's actions and the logs too (see "The written file" below).
Given a boundary, ``inventory`` holds the buildings inside it and
``area_inventory`` every building (see "A boundary" below).

The imputer's settings
----------------------

The compiler gives ``ObservedDonorImputer`` its own values of three settings
in place of the imputer's defaults: the number of nearest donors from which
one is picked (``k``), the meters of separation counted as one unit of donor
distance (``location_scale_m``) and the weight of the difference of log plan
areas (``area_weight``). ``DEFAULT_IMPUTER_OPTIONS`` holds ``k`` of 10, a
``location_scale_m`` of 500 and an ``area_weight`` of 2. They come from a
SimCenter masking test on the buildings with an NSI record in San Francisco
and Richmond, California, in which the occupancy class, unit count and story
count of about a fifth of them were hidden and imputed again from the others.
Compared with the test's starting values (``k`` of 5, 100 m and a weight of
1), these values gave better overall occupancy class, story and unit scores
in both cities and a lower exact-label agreement for non-residential
buildings. The test measured donor distance on footprint heights, whereas
the imputer compares story counts. The test is therefore a comparison of
settings, and the three values are not calibrated yet. ``imputer_options``
overrides them one by one. The compiler gives the imputer its donors
directly, so the imputer runs without batch feedback, in which targets
filled in one batch become donors for later batches, and ``n_batches`` is
refused. ``ResidentialNeighborYearSampler`` takes the imputer's ``k`` and
``cap_m``, and ``ReplacementCostImputer`` takes those and the imputer's
``location_scale_m``, ``area_weight`` and ``stories_weight``.

The run in outline
------------------

The run has fifteen steps, labeled ``s01`` to ``s15``.

==================================  ========================================
Step                                What it gives the buildings
==================================  ========================================
``s01_footprints``                  block group, plan area and footprint
                                    provenance
``s02_nsi_join``                    the values of the NSI records joined to
                                    them
``s03_story_counts``                story counts from the footprint height
``s04_unit_budget``                 occupancy classes and unit counts from
                                    the census counts of housing units
``s05_imputation``                  what is still missing, from observed
                                    donors
``s06_residential_year``            construction years of the residential
                                    buildings
``s07_pair_redraw``                 new values for implausible pairs of
                                    occupancy class and story count
``s08_structure``                   design level, structure type and height
                                    class
``s09_nonresidential_year``         construction years of the other
                                    buildings
``s10_nonresidential_structure``    what ``s08`` gives, for those buildings
``s11_schema_check``                the check against the Hazus input schema
``s12_gross_area``                  gross area
``s13_replacement_cost``            replacement cost
``s14_building_type``               general building type
``s15_output``                      the compiled inventory
==================================  ========================================

The first five steps place the footprints and complete the occupancy class,
the unit count and the story count of each one. Steps ``s01`` to ``s03`` give
the same values in every world. From ``s04`` on, a step can give a building a
different value in each world. ``s06`` gives a construction year from the
census tables to each residential building without one, and ``s07`` replaces
implausible pairs of occupancy class and story count. ``s08`` gives a design
level and a structure type to every building that holds a year by then, and a
height class to every building with a structure type. ``s09`` and ``s10`` do
the same for the other buildings, whose years are copied from residential
neighbors. The last five steps check each building against the input schema
of the assessment, compute its gross area, give it its replacement cost and its
general building type, and assemble the output.

Each step adds its values to the run as frozen stages, and the steps after it
read them from there. Many steps read each world's merged inventory: the
values of every source in that world, merged in the order of precedence of the
sources, so that each feature takes the value of the source of highest
precedence that holds one. Between steps, the run also freezes two views,
``s05_view`` after ``s05`` and ``s08_view`` after ``s08``. They hold the
merged inventory of every world as it stands, and later steps read them. The
section "Frozen stages" below lists the stages.

The names of most of the stages a step freezes, of the report stages it files
rows under, and of its timings in the report's summary begin with the step's
label. The exception is the six records frozen right after ``s10``: three are
named after ``s12`` and the other three after ``s13`` and ``s14``, because the
values of those steps replace what the records withhold.

From ``s06`` on, a step that cannot complete a building in a world, or finds it
inconsistent, excludes it from that world. Steps ``s08``, ``s09``, ``s10`` and
``s11`` then skip the building in that world. The other steps compute their
values for it as for any building, and ``s15`` leaves the computed gross area
and replacement cost out of its output. Earlier, ``s04`` drops footprints from
every world: a small footprint that becomes no dwelling, as the subsection
``s04_unit_budget`` states. A dropped footprint is not in the compiled
inventory. The report files each exclusion under one of the categories that the
section "Exclusion categories" of ``brails/compilers/README.rst`` defines, with
a diagnosis, as that section states. The subsections below name the diagnosis
of each exclusion.

The run
-------

Each subsection below states what the step reads, what it does, what it
freezes, and which exclusions it files under which report stage.

``s01_footprints``
~~~~~~~~~~~~~~~~~~

The step gives each footprint its census block group and its plan area.
``CensusTractScraper`` gives each footprint the GEOID of the census block group
that holds the centroid of its outline, as ``BLOCKGROUP_GEOID``. The footprints
whose centroid lies on an edge or a corner shared by several block groups are
listed in the log ``footprint_ties``. A footprint outside every block group, or
one whose outline encloses no area, stops the run with ``ValueError``. Each
footprint also receives its plan area in square feet as ``PlanArea``. An
Overture footprint also receives the release and the datasets of its sources.
The release, the datasets and the caller's ``footprint_provider`` make up
``FOOTPRINT_PROVENANCE_FEATURES``.

The step freezes the footprints as ``s01_footprints``, the one part of the
source labeled ``footprint``, which holds every building of the run and its
outline. Where a local inventory is given, the step freezes it as
``s01_local``, the one part of the source labeled ``local``. A local asset
in it takes its footprint's centroid as its coordinates, and a ``GrossArea``
it holds is copied to ``LocalGrossArea``, which the output keeps (see
``s12_gross_area``). The step files no row in the report.

With ``merge_stacked``, which is off by default, ``StackedFootprintMerge``
first merges the stacked Overture footprints of one building, such as a podium
and the towers drawn on it, into one footprint, as the section "The method" of
``brails/aggregators/stacked_footprints/README.rst`` states. Every later step
reads the merged footprints under their new geohash ids. A merged footprint
takes the height of the tallest of the footprints it merges, over the union
of their outlines. Where its story count comes from that height, the podium
therefore counts as many stories as the tower. The merge's table of the
footprints it merged and its log of ids are written as the logs
``stacked_parts`` and ``stacked_asset_ids``, and a notice in the report
counts the merges. Without the option the footprints are taken as given,
keyed by the caller's asset ids.

Where a census table of ``CompilerInputs`` is ``None``, ``ACSTableScraper`` reads it
from the Census Data API, from the ACS five-year release that the compiler's
``acs_vintage`` names. A table that is given is used as it is. The requests cover
every block group or tract of each county that holds one of the footprints' block
groups, so a single footprint across a county line brings in the whole of that
county. Reading a table needs a Census API key, which ``find_census_api_key`` in
``brails/utils/census_api_key.py`` looks up. Before ``s01_footprints`` starts,
``compile`` asks the API whether it accepts the key. Without a key, or with one the
API refuses, the run stops there. The year sampler receives the rows of B25034 for
the footprints' block groups and their tracts, and every tract of B25127, whose
county totals it sums. The run stops if one of those rows holds an estimate the
Census Bureau does not publish. The report records the release among its settings
and, in a notice, the tables and the counties read, and it times the reading as
``s01_acs_tables``.

Where ``CompilerInputs.mh_park_points`` is ``None``, ``MobileHomeParkScraper`` reads
the mobile-home park points of HIFLD that lie within the unit budget's
``mh_park_radius_m`` of the footprints' bounding box, before ``s01_footprints``. The
read needs network access. The run passes the points' longitude and latitude to the
budget, as it does for an ``AssetInventory`` of points given in the field. A notice
counts the points, and the report times the reading as ``s01_mh_park_points``. The
archive names no release, and a later write replaces its points. The settings
therefore record the write that was read, and a run repeats exactly only with the
points given in ``CompilerInputs``.

``s02_nsi_join``
~~~~~~~~~~~~~~~~

The step joins the NSI points to the footprints with ``NSIFootprintJoin``,
within the compiler's ``join_cap_m``, measuring distances in its
``projected_crs``. Each point goes to at most one footprint: one that holds
the point, or the nearest that accepts it within the cap. The class
docstring of ``NSIFootprintJoin`` states the rules.

The join returns the values of each footprint in two inventories. One holds the
values as the NSI records state them, and it is frozen as ``s02_nsi_join``, the
one part of the source labeled ``nsi``. These values include the occupancy
class, story count and building type of the footprint's governing record, the
one record whose values the footprint takes: a housing record first, then the
record that states the largest ``sqft``, then the lowest ``fd_id``. They also
include, where one record states them, the ``sqft`` as ``GrossArea`` and the
replacement cost. The other inventory holds the values the join computes, such
as the sum of the replacement costs of several records or an occupancy class
set by the number of units. It is frozen as ``s02_derived``, a part of the
source labeled ``derived``. A ``GrossArea`` in either inventory is copied to
``NSIGrossArea``, which ``s13_replacement_cost`` divides by. The join's links,
its unlinked points and its counts of points by branch, that is by how a point
reached its footprint (inside it, or by the walk to a footprint within the cap)
or why it was left unlinked, are the logs ``join_links``, ``join_unlinked`` and
``join_branch_counts``. The step files no row in the report.

``s03_story_counts``
~~~~~~~~~~~~~~~~~~~~

The step reads the merged inventory of the sources so far. These sources
hold one world, so the step gives the same counts in every world.
``StoriesFromHeightInferer`` checks each footprint's height, in the length
unit the caller gives, and its verdicts are the log ``stories``. For each
footprint it accepts, it converts the height to a story count, kept as
``StoriesFromHeight``. That count becomes the footprint's
``NumberOfStories`` where no source holds one. The count also replaces a
story count from NSI, unless the footprint's occupancy class is
``NSI_STORIES_CLASS``, the single-family house, whose NSI count is kept. A
local story count is never replaced. The counts are frozen as
``s03_story_counts``, a part of the ``derived`` source, and the NSI counts
they replace as the record ``s03_withheld``. The report holds a ``withheld``
row under ``s03_story_counts`` for each replaced count, in every world. The
step excludes no building.

``s04_unit_budget``
~~~~~~~~~~~~~~~~~~~

The step reads the merged inventory after ``s03`` and the source of each
footprint's occupancy class. A footprint has an observed occupancy class when
the local inventory, NSI or the NSI join's derived values give it one. The
other footprints are the budget's targets. ``ACSUnitBudget``, configured by the
compiler's ``budget_options``, distributes among them the housing units that
table B25024 counts in their block group, less the units of the footprints that
already have an occupancy class, as
``brails/inferers/acs_unit_budget/README.rst`` states. Near a mobile-home park
point, it makes some footprints mobile homes. It converts only targets and
footprints whose single-family occupancy class came from NSI or the join. It
gives other targets a residential occupancy class and a unit count, and it
drops the small targets that it does not make dwellings. To judge how many
units a footprint can hold, the budget takes its gross area to be its
``GrossArea`` where that is a positive number, a local one included, otherwise
its plan area times its story count where that is a positive number, and
otherwise its plan area alone. A local gross area can therefore change the
occupancy classes and unit counts this step gives and the draws of the steps
after it.

In a block group with B25024 counts, the same targets become dwellings, and
the same targets are dropped, in every world. Which dwelling receives which
occupancy class can still differ between worlds. In a block group without
these counts, the budget decides no target's residential status. ``s05``
decides it in each world, so it can differ between worlds. In such a block
group the budget drops a target only under its option ``drop_without_acs``,
and then only a small target whose residential probability is below
``DROP_PROBABILITY``.

The step freezes three stages. ``s04_dropped`` is the record of the dropped
footprints, which leave every source and every world. ``s04_withheld`` is the
record of the occupancy classes the mobile-home rule replaces, each withheld
from its source. ``s04_unit_budget`` is a part of the ``imputed`` source and
holds the budget's values in each world. The report holds a ``withheld`` row
under ``s04_unit_budget`` for each replaced occupancy class, and a row in the
category ``dropped``, with the diagnosis ``not_selected``, for each dropped
footprint, both in every world. The budget's logs are ``budget``,
``budget_diagnostic``, ``budget_gaps`` and ``budget_nonresidential``.

``s05_imputation``
~~~~~~~~~~~~~~~~~~

The step fills the occupancy class, the unit count and the story count that are
still missing. It reads the same merged inventory as ``s04``, with the budget's
values. The observed donors are the footprints that have an observed occupancy
class and hold a story count. Every other footprint the budget kept is a
target. ``ObservedDonorImputer``, with the settings of "The imputer's settings"
above, copies each of those three features that a target lacks from one
observed donor in each world, as the section "Donors" of
``brails/compilers/README.rst`` describes. For a target whose residential
status the budget decided, the imputer takes that status from the budget. For
the other targets, the imputer's own status round decides it, in each world: a
first draw from the donors within the cap that gives the target a residential
status. The imputer checks each copy for an implausible pair of occupancy class
and story count (see "Pair re-draw" below), and it gives a target whose copy
fails this check one more draw, from another donor. A target that fails again
keeps its first values, and its pair stays implausible, since ``s07`` only logs
a pair with an imputed member. A target with an observed occupancy class that
is not a residential occupancy class keeps no imputed unit count. The values
are frozen as ``s05_imputation``, a part of the ``imputed`` source, and the
imputer's draws are the log ``imputer``. The step excludes no building.

The run then freezes the view ``s05_view``. From the imputer's log it freezes
the computed stage ``s05_donors``: the donor whose values each target kept in
each world, which ``s13_replacement_cost`` reads.

``s06_residential_year``
~~~~~~~~~~~~~~~~~~~~~~~~

The step runs world by world on the world's realization of ``s05_view``.
``ACSYearBuiltSampler`` is fitted on that realization from tables B25034 and
B25127 and configured by the compiler's ``sampler_options``. It draws a
construction year for each building of a residential occupancy class that
has none, from the census tables of the building's block group and tract, as
``brails/inferers/acs_year_built_sampler/README.rst`` states. Its fits are
the log ``year_fit``. The years are frozen, over the worlds, as
``s06_residential_year``, a part of the ``drawn`` source.

The sampler refuses a building whose unit count is missing or unusable, or
whose block group the census tables do not cover. Such a building is
excluded from the world under ``s06_residential_year``, as
``missing_input``.

``s07_pair_redraw``
~~~~~~~~~~~~~~~~~~~

The step runs world by world on the world's merged inventory after ``s06``,
excluded buildings included. In each implausible pair of occupancy class and
story count, ``PairRedraw`` replaces the member that came from NSI, as the
section "Pair re-draw" below states. It draws new values from the observed
donors of the world's realization of ``s05_view``, with the imputer of
``s05``. The step freezes three stages over the worlds: the record
``s07_withheld`` of the values replaced, the part ``s07_pair_redraw`` of the
re-drawn values in the ``imputed`` source, and the part ``s07_derived`` of the
story counts from the footprint height that replace NSI's, in the
``derived`` source. Under ``s07_pair_redraw``, the report holds a
``withheld`` row for each value replaced and a ``redrawn`` or ``derived``
row for each new value. The re-draw's plan and outcomes are the log
``pair_redraw``. The step excludes no building. After it, no step changes a
story count.

``s08_structure``
~~~~~~~~~~~~~~~~~

The step runs world by world on the world's merged inventory after ``s07``,
and it skips the buildings already excluded from the world. The design
level and the structure type go to the buildings that hold a construction
year by then: those that ``s06`` gave a year and those whose year comes from
the local inventory. The height class goes to every building that holds a
structure type. The step runs five operations in this order, each filing its
rows under its own report stage.

The design level. ``HazusEQDesignLevelInferer``, with the compiler's
``design_level_method``, ``state`` and ``seismic_zone``, derives a design
level from the construction year for each building that has a year and no
design level. A building it refuses is excluded under ``s08_design_level``.
The design level inferer in BRAILS refuses a building only for a missing or
unreadable year, such as a local ``YearBuilt`` that is not a number, and the
step files that refusal as ``missing_input``. The step files an infeasible
refusal, which that inferer does not give, as ``missing_input`` too. The
composed screen files such a refusal as ``undetermined`` instead (see "The
exclusion category" below).

The composed screen and the resolution loop. Each building that has a year
and no structure type passes through ``ComposedScreen`` and
``ResolutionLoop``, as the sections "Composed screen" and "Resolution loop"
below state. The screen asks the design level and structure type inferers
whether they can accept the building. Where the structure type inferer
refuses it because of a building type that NSI gave, the loop withholds that
type and screens the building again. The report holds the loop's
``withheld`` and ``restored`` rows under ``s08_resolution``. It files an
exclusion under the report stage of the check that refused the building,
``s08_design_level`` or ``s08_structure_type``, or under ``s08_resolution``
where the loop acted and the building was refused again.

The structure type. ``HazusEQStructureTypeInferer`` draws a structure type
for each building the loop admits, from the completed West Coast tables of
the Hazus manual. A general building type the building holds, such as NSI's
letter ``W``, restricts the draw to the structure types of that letter.

The height class. ``HazusEQHeightClassInferer`` gives a height class to each
building that holds a structure type and no height class, whether this step
drew the structure type or the building already held it. A building whose
structure type has no height class in Hazus gets none. The inferer refuses a
building whose structure type is not a Hazus structure type, such as a local
``W1X``, or whose story count is missing or not a whole number of one or
more where its structure type needs one. Such a building is excluded under
``s08_height_class``, as ``missing_input``.

The W1 carve-out. The carve-out applies in the state ``CARVE_OUT_STATE``
under the design level method ``CARVE_OUT_METHOD``, which are the state and
the edition of Table C-10 of the Hazus manual. There, a building whose
structure type this step drew as W1 and whose design level this step derived
as Pre-Code is raised to Moderate-Code, as that table assumes. The report
holds a ``derived`` row under ``s08_carve_out`` for each such building.

A building without a construction year therefore receives only a height
class in this step, and only where it holds its own structure type. A
building excluded within the step keeps what the step gave it before the
refusal, such as a design level derived before its height class was refused.
The design levels, height classes and carve-out levels are frozen, over the
worlds, as ``s08_derived``, a part of the ``derived`` source, and the
structure types as ``s08_structure``, a part of the ``drawn`` source. The
loop's counts per world are the log ``resolution``. The run then freezes the
view ``s08_view``.

``s09_nonresidential_year``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Steps ``s09`` and ``s10`` run together, world by world, on the world's
realization of ``s08_view``. ``s09`` gives a construction year to each
building that is not excluded from the world and has no year, where the
building's occupancy class is missing or is not one of the residential
occupancy classes. ``ResidentialNeighborYearSampler``, with the ``k`` and
``cap_m`` of the imputer of ``s05``, copies to each such building the year
of a nearby residential building that holds one in that world, as
``brails/imputers/residential_neighbor_year_sampler/README.rst`` states. Its
picks are the log ``neighbor_year``. The years are frozen, over the worlds,
as ``s09_nonresidential_year``, a part of the ``imputed`` source.

The sampler refuses a building that has no occupancy class, or that has no
residential building with a year within ``cap_m`` or in its block group.
Such a building is excluded from the world under
``s09_nonresidential_year``: as ``missing_input`` where it has no occupancy
class, and as ``no_plausible_value``, with the sampler's diagnosis
``no_donor``, where it has no such neighbor.

``s10_nonresidential_structure``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The step runs the five operations of ``s08`` for the buildings that ``s09``
gave a year, on the same realization with those years written in. It files
its rows under report stages of its own, ``s10_design_level``,
``s10_resolution``, ``s10_structure_type``, ``s10_height_class`` and
``s10_carve_out``, so the report keeps the buildings of ``s10`` apart from
those of ``s08``. A building that one of the operations refuses is excluded
from the world under the report stage of that operation, in the same
category as in ``s08``. The structure types are frozen as
``s10_nonresidential_structure``, a part of the ``drawn`` source, and the
design levels, height classes and carve-out levels as ``s10_derived``, a
part of the ``derived`` source. The loop's counts per world are the log
``nonresidential_resolution``. The step runs inside the loop of ``s09`` over
the worlds, and the report times it in each world under its own label.

``s11_schema_check``
~~~~~~~~~~~~~~~~~~~~

Steps ``s11`` to ``s15`` run world by world on one merged inventory of the
world's sources, which ``s15`` completes. Before the first of these merged
inventories, right after ``s10``, the run freezes six records of withheld
values. Each removes from every later merged inventory a value that a later
step replaces, and each is named by the step whose value takes its place:

- ``s12_withheld_local``: a local ``GrossArea``;
- ``s12_withheld_join``: the ``GrossArea`` of the NSI join, one record's
  ``sqft`` or the sum of several;
- ``s12_withheld_footprint``: a ``GrossArea`` the footprints carry;
- ``s13_withheld``: the join's ``ReplacementCost`` of each building without a
  local one;
- ``s14_withheld``: NSI's ``BuildingType``;
- ``s14_withheld_footprint``: a ``BuildingType`` the footprints carry.

Every step before ``s11`` merged its sources before these records existed.
The unit budget and the structure type draw therefore read these values, and
so does ``s13_replacement_cost`` when it computes its rates, since it reads
``s08_view``. The local and NSI gross areas stay in every merged inventory
as ``LocalGrossArea`` and ``NSIGrossArea``.

``s11`` checks every building not yet excluded from the world against the Hazus
input schema of the damage and loss model library, ``DLML_DATASET``. The schema
states the features the assessment reads and the values it accepts, such as the
design levels it accepts for each structure type. A building the schema refuses
is excluded from the world under ``s11_schema_check``, as ``fails_schema``,
with the diagnosis ``invalid`` and the schema's messages as the detail. The
messages are the log ``schema_check``, and the report's summary holds the
dataset and the number of buildings refused in each world. The schema reads
none of ``GrossArea``, ``ReplacementCost`` and ``BuildingType``, which the next
steps give, so the check does not depend on them.

``s12_gross_area``
~~~~~~~~~~~~~~~~~~

The step computes, in each world after the check, the gross area of every
building of ``s08_view``, excluded buildings included, from the world's
realization of ``s08_view``. ``footprint_gross_area`` of the replacement cost
imputer's module gives the plan area of the footprint's outline times the
story count in that world, or the plan area alone where the building has no
positive story count. The area therefore differs between worlds where the
story count does. ``s13`` and ``s15`` read each world's areas as the step
gives them, and the areas of every world are frozen as the computed stage
``s12_gross_area`` after the last world (see ``s15_output``). The log
``gross_area`` names for each building and world whether its area came from
the story count (``stories``) or from the plan area alone (``plan_area``). A
building that ``s11`` excludes from a world keeps its row there. The step
excludes no building.

``s13_replacement_cost``
~~~~~~~~~~~~~~~~~~~~~~~~

The step gives every building of ``s08_view`` its replacement cost in the
world, excluded buildings included. ``ReplacementCostImputer`` takes the
``k``, ``cap_m`` and distance weights of the imputer of ``s05``. It reads the
world's realization of ``s08_view`` and the donors of ``s05_donors``. It
computes each building's replacement cost as a rate per square foot times
its gross area of ``s12_gross_area`` in the world, as
``brails/imputers/replacement_cost/README.rst`` states. A building takes its
own rate where it has one. Otherwise it takes the rate of the donor whose
values ``s05`` copied to it, or of a nearby building of the same occupancy
class. A rate is a replacement cost divided by ``NSIGrossArea``, the
``sqft`` of the building's NSI records. The replacement cost is NSI's, the
join's sum of NSI's or a local one. A local ``ReplacementCost`` therefore
gives a rate only to a building whose NSI records state a ``sqft``, and a
local ``GrossArea`` enters neither a rate nor the area a rate multiplies.
The step keeps a local ``ReplacementCost`` as it is. Where a local
``ReplacementCost`` gives a rate, that rate also serves each building that
takes its rate from the local building.

A building not yet excluded for which the imputer finds no donor is excluded
from the world under ``s13_replacement_cost``, as ``no_plausible_value``, with
the diagnosis ``no_donor``. A building not yet excluded has an occupancy class,
which the schema requires, and a footprint that encloses an area, so the step
files no other outcome. A building that the step left without a replacement
cost for want of an occupancy class or of an area would be excluded as
``missing_input``, with the diagnosis ``invalid``. The imputer gives no other
outcome without a value; one would be filed as ``no_plausible_value``, with
``infeasible``. A building excluded earlier is not filed again. The imputer's
log is the log ``value``, and the replacement costs that are not local ones are
frozen, over the worlds, as the computed stage ``s13_replacement_cost`` (see
``s15_output``).

``s14_building_type``
~~~~~~~~~~~~~~~~~~~~~

The step gives every building without a local ``BuildingType``, excluded
buildings included, the letter of its structure type in the world: ``W``,
``S``, ``C``, ``M`` or ``MH``. The letter follows the pairing of
``get_hazus_general_type_membership`` in
``brails/inferers/hazus_inferer_earthquake/hazus_rulesets.py``, the same
pairing that the structure type draw restricts itself with. A building without
a structure type gets no letter. NSI's letter, which the draw read, is not
written. A local letter is kept. A building not yet excluded whose local letter
is not the letter of its structure type in a world, such as a local ``W``
beside a ``C1``, is excluded from that world under ``s14_building_type``, as
``conflict``, with the diagnosis ``infeasible``. The structure type draw keeps
to a local letter. A local letter can therefore contradict only a structure
type that the building held before the draw: a local structure type, or one its
footprint carries. Every earlier step has run in that world by then, so such a
building has served as a donor like any other. The letters are frozen, over the
worlds, as the computed stage ``s14_building_type`` (see ``s15_output``).

``s15_output``
~~~~~~~~~~~~~~

The step writes the values of the last steps into the world's merged
inventory. Every building receives ``Excluded``, true where the report lists
it as excluded from the world, and the letter of ``s14_building_type``. Each
building not excluded that has no local replacement cost also receives its
replacement cost of ``s13_replacement_cost``, with its source code in the
source string. By ``OUTCOME_VALUE_CODES``, the code is ``R`` for a
replacement cost from the building's own rate and ``I`` for one from another
building's rate. Each building not excluded, one with a local
``ReplacementCost`` included, receives its area of ``s12_gross_area`` as
``GrossArea``.

A building excluded from the world receives no computed replacement cost and
no gross area there. It keeps a local ``ReplacementCost``, its
``LocalGrossArea`` and ``NSIGrossArea``, the letter of its structure type or
its local letter, and every other value its sources hold in that world.
Steps ``s08``, ``s09`` and ``s10`` gave it nothing in a world it was excluded
from before them, so a building excluded at ``s06`` has, for example, no
structure type drawn in that world.

Last, every building not excluded must hold each property of
``DEFAULT_PROPERTIES`` that applies to it: ``NumberOfUnits`` applies to the
residential occupancy classes only, and ``HeightClass`` to the structure
types for which Hazus defines one. A building that lacks one is a defect of
the run. The log ``MISSING_DEFAULTS_LOG`` names each such building, world and
property, a notice in the report counts them, and ``write`` refuses to write
the GeoJSON. The worlds are added one by one, with
``AssetInventory.add_world``, to the compiled inventory, which is frozen as
the view ``s15_output``. The computed stages ``s12_gross_area``,
``s13_replacement_cost`` and ``s14_building_type`` are frozen at the same
time, from the values ``s12`` to ``s14`` gave in every world. No step reads
them.

Frozen stages
-------------

The run freezes each stage once, under its name, in the order of the
subsections above, except the six records of withheld values, which it
freezes before ``s11``, and the computed stages of ``s12`` to ``s14``, which
it freezes with ``s15_output``. The names are the module's ``STAGE_``
constants. The stages fall into the four kinds that the section "Stages and
freezing" of ``brails/compilers/README.rst`` defines:

- Parts, each holding the values one step adds to one source:
  ``s01_footprints`` (``footprint``), ``s01_local`` (``local``),
  ``s02_nsi_join`` (``nsi``), ``s02_derived``, ``s03_story_counts``,
  ``s07_derived``, ``s08_derived`` and ``s10_derived`` (``derived``),
  ``s04_unit_budget``, ``s05_imputation``, ``s07_pair_redraw`` and
  ``s09_nonresidential_year`` (``imputed``), and ``s06_residential_year``,
  ``s08_structure`` and ``s10_nonresidential_structure`` (``drawn``).
- Records: ``s04_dropped``, the dropped footprints, and the records of
  withheld values ``s03_withheld``, ``s04_withheld``, ``s07_withheld`` and the
  six frozen right after ``s10``.
- Views: ``s05_view``, ``s08_view`` and ``s15_output``, the compiled
  inventory.
- Computed stages, whose values no merge takes: ``s05_donors``, the donors
  named from the log of ``s05``, which ``s13`` reads; and ``s12_gross_area``,
  ``s13_replacement_cost`` and ``s14_building_type``, the gross areas,
  replacement costs and letters that ``s12`` to ``s14`` give, which the run
  freezes after ``s15`` and which no step reads.

Every part other than the footprints, and every record and computed stage other
than ``s05_donors``, holds each footprint's centroid as its coordinates. The
footprints, the views and ``s05_donors`` hold the footprint polygons. The run
ends with the manifest, which checks that no frozen stage changed after it was
frozen.

Possible worlds
---------------

A run computes the worlds 0 to ``n_worlds - 1``, numbered as the section
"Possible worlds" of ``brails/compilers/README.rst`` describes. The
footprints, the NSI values, the local values and the story counts of ``s03``
are the same in every world. The values that the steps from ``s04`` on draw
or copy can differ between worlds:

- the occupancy class and unit count that the budget gives;
- the values the imputer copies;
- the residential status of a target in a block group without B25024
  counts, and with it whether ``s06`` or ``s09`` gives the building its
  year;
- the construction year, and with it the design level, the structure type
  and the height class;
- the values of a re-drawn pair;
- the gross area, the replacement cost and the general building type that
  follow from these values.

A building can be excluded from some worlds and not others.

The draws are keyed by the building, the seed and the world, and a world's
values do not depend on how many worlds the run computes: world 2 of a run of
three worlds equals world 2 of a run of ten, and so do its exclusions. A run
always computes the worlds from 0. A single world ``w`` is therefore written
from a run of at least ``w + 1`` worlds, with ``write`` and its ``world``
argument.

Local inputs
------------

A local inventory holds values known locally, keyed by the ids of the
footprints, in one possible world. ``compile`` refuses a local inventory that
holds an asset the footprints lack, more than one possible world, or a
feature of ``RESERVED_LOCAL_FEATURES``, which the run writes for itself and
reads back. In each merged inventory the sources take precedence in the
order of ``PRECEDENCE``: local values first, then the footprints, NSI, the
derived values (the NSI join's sums and the values later steps compute from a
building's other features), the imputed values and the drawn values. A local
value therefore stays in every merged inventory, and no step replaces one
with a value of its own. There are two exceptions. From ``s11`` on, the
merged inventory leaves a local ``GrossArea`` out, the output's
``GrossArea`` is the area of ``s12``, and the local one is kept as
``LocalGrossArea``. And ``s15`` writes ``Excluded`` for every building in
every world, over a local one.

Some local values act on more than their own feature:

- A local ``GrossArea`` is read by the unit budget, which can change the
  occupancy classes, unit counts and draws of ``s04`` and of the steps after
  it.
- A local ``ReplacementCost`` is kept, and it gives the other buildings a rate
  only where the building's NSI records state a ``sqft``
  (``s13_replacement_cost``).
- A local ``BuildingType`` restricts the structure type draw to its letter and
  is written as it is, and a local structure type is never re-drawn.

These limits hold:

- A local ``BuildingType`` that names no Hazus general building type, such as
  ``RM`` or ``URM``, leaves the draw no structure type to choose. A building
  whose structure type would be drawn is then excluded as ``missing_input``
  under ``s08_structure_type`` or ``s10_structure_type``, in each world from
  which no earlier step excluded it.
- A local ``StructureType`` that is not a Hazus structure type, such as
  ``W1X``, is never drawn, so no draw refuses it. Where the building holds no
  height class, the height class operation of ``s08`` excludes it as
  ``missing_input``, with a construction year or without. Where the building
  also holds a local ``HeightClass``, ``s11_schema_check`` excludes it as
  ``fails_schema``. Either way, it is excluded in each world from which no
  earlier step excluded it.
- A local ``BuildingType`` that is not the letter of the building's structure
  type, a local one or one its footprint carries, excludes the building as
  ``conflict`` under ``s14_building_type``, in each world from which no
  earlier step excluded it.
- The compiler refuses a local inventory when it is built with
  ``merge_stacked``: the local ids are those of the footprints, which the
  stacked-footprint merge changes.
- A local ``PlanArea`` is written as the output's ``PlanArea``, while ``s12``
  computes the gross area from the footprint's outline. A local ``Excluded``
  is overwritten by ``s15``. A local ``SourceString`` stops the run at the
  first merged inventory with ``ValueError``.

Block groups
------------

``CompilerInputs.block_groups`` holds one polygon per census block group, with a
``GEOID`` column, as a TIGER/Line file holds them. Without it, ``TIGERLineScraper``
reads the block groups that intersect the footprints' bounding box from the
TIGER/Line files of the compiler's ``block_group_vintage``, before
``s01_footprints``. The read needs network access. The report's settings record the
release, a notice names the states read, and the report times the reading as
``s01_block_groups``. The tables of an ACS release count the block groups of its
decade's census, and a GEOID can carry over from one decade to the next with a
changed boundary. Where the run reads both the census tables and the block groups, it
therefore refuses a ``block_group_vintage`` whose decade differs from that of
``acs_vintage``. Give the polygons to avoid the download, or give nothing. Either
way, every footprint must lie in a block group.

A boundary
----------

``CompilerInputs.boundary`` is a place to compile inside the area the footprints
cover, a Shapely Polygon or MultiPolygon in longitude and latitude whose holes are
kept. A building's values depend on the other buildings of its block group and on
donors within the imputer's ``cap_m``, so the footprints are given for an area around
the place, and the buildings around it are compiled for the sake of those inside.
``buffer_m`` states how far the area reaches past the place, and ``boundary_record``
where the boundary comes from. The report's settings record both, with the boundary
as a GeoJSON geometry.

Right after ``s01_footprints``, ``compile`` marks the footprints whose centroid, as
``s01_footprints`` computes it, lies strictly inside the boundary. ``find_inside`` in
``brails/utils/region_filter.py`` holds this rule, which ``filter_to_region`` applies
too. A run with no footprint inside stops there with ``ValueError``. The report
records the footprints inside, and a notice is added where ``buffer_m`` is not given
or is less than the donor cap. Every step then runs on every footprint, and the
frozen stages and the manifest cover them all.

The result's ``inventory`` holds the buildings inside the boundary, its
``area_inventory`` every building, and its ``boundary_assets`` the ids of the
footprints inside. Its report's summary counts the actions, exclusions and schema
refusals of the buildings inside, holds the number of footprints inside and outside
as ``boundary``, and says in a notice which of its counts are of the inside.
``write`` saves the buildings inside, with their report rows and that summary, and
refuses a file for a missing default property of those buildings only. With
``keep_outside=True`` it saves every building and every row. The logs that
``detailed=True`` saves hold every building either way, since several of them are
keyed by block group.

A compilation from a place
--------------------------

``CompilerInputs.from_place`` gathers the inputs of a place, and the compiler's
``compile_place`` compiles them in one call. The place is a name with its state,
looked up among the census places and counties by ``CensusBoundary``, or in
OpenStreetMap with ``place_source='osm'``; a ``RegionBoundary``; or a Shapely
polygon. ``from_place`` takes these steps, the quick ones and the likely failures
first:

1. It asks the Census Data API whether it accepts the key, before any download, since
   the compiler reads the census tables with it.
2. It reads the boundary, from the TIGER/Line files of ``block_group_vintage`` for a
   census name, and finds its state. A census boundary lies in the state its record
   names; another boundary lies in each state that holds at least
   ``STATE_AREA_SHARE`` of its area. It first refuses a boundary that encloses no
   area, or that lies in no state or in two. Then, without ``compiler``, it refuses a
   state the compiler cannot take, and with it, what the compiler's ``check_state``
   refuses, in this order: a state other than the compiler's, a state it cannot take,
   a missing zone.
3. It widens the place, together with the block groups with land that overlap it, by
   ``buffer_m``, and refuses that area read where it intersects more than
   ``max_block_groups`` block groups.
4. It reads the Overture footprints of the area read once, keyed by geohash. The
   place and the block groups that hold a footprint inside it, widened by
   ``buffer_m``, make the area compiled. Footprints outside it are left out, and so
   are those in no block group, with a notice. A notice also names a building inside
   the place whose block group has no land and may reach past the area read.
5. It reads the NSI points of the area compiled, widened by ``DEFAULT_JOIN_CAP_M``.

The inputs carry the boundary, ``buffer_m``, the record of where the boundary comes
from, the state, ``merge_stacked`` set on, and the notices, which the report lists.
The census tables and the park points are left for the compiler to read, and the
block groups are those of the area read. A compiler built without a state or
``merge_stacked`` applies the inputs'. The docstring of ``compile_place`` states what
it adds to the two calls.

The written file
----------------

``CompilationResult.write`` writes the compiled inventory as GeoJSON, one
feature per building with its asset id and its footprint. Without
``world``, it writes every world the run computed, and a property whose
value differs between worlds holds a list of values, one per world in world
order. Given ``world``, it writes that one world alone. A one-world file
leaves out a property whose value is ``None`` in that world, so its buildings
and properties are the same whatever number of worlds the run computed. A
building excluded from every written world is not written. A building
excluded from some of them is written only with ``keep_partly_excluded``,
which also writes ``Excluded``. Such a building may lack its default
properties in the worlds it is excluded from. With ``integer_ids`` the file
takes the form the R2D tools read: the features are numbered within each
value of ``type``, which each footprint must carry, and each building holds
its asset id as ``unique_id``. After a compilation with a boundary, the file
holds the buildings inside it unless ``keep_outside`` is given (see "A boundary"
above).

Each building holds the properties of ``DEFAULT_PROPERTIES`` that apply to
it, the features listed under "The inventory and its inputs" above.
``extra_properties`` adds other features of the compiled inventory, or, with
``ALL_PROPERTIES``, every one of them except ``Excluded``, which only
``keep_partly_excluded`` writes. Among them are the footprint's own features
and its provenance, such as ``footprint_provider`` and ``height``; the values
the NSI join adds, such as ``NSIOccupancyClass``, ``NSIRecords``,
``NSIReach`` and ``NSIGrossArea``; ``StoriesFromHeight``; and
``LocalGrossArea``. An extra property changes no value of a default one. A
building's ``ReplacementCost`` divided by its ``GrossArea`` is the rate per
square foot that ``s13_replacement_cost`` gave it, except for a building that
keeps a local ``ReplacementCost``, whose rate the step computes from
``NSIGrossArea``.

The files of a write
~~~~~~~~~~~~~~~~~~~~

``write`` names each file it saves beside the GeoJSON after the stem of the GeoJSON's
path, its file name without the extension: given ``out/berkeley.geojson``, it saves
the summary as ``out/berkeley_report_summary.json``. A call whose ``world`` or
``extra_properties`` is refused writes nothing. Otherwise a call writes these files:

The path given (``<stem>.geojson`` in the example)
    The buildings, as stated above. Every call writes it, unless it refuses the file
    (see "A refused file" below). It is written under a temporary name that then
    replaces the file.
``<stem>_report_summary.json``
    The report's summary, whose keys the section "The summary" of
    ``brails/compilers/README.rst`` lists. Its counts of actions and exclusions are
    those of the written worlds, and its notices, timings and schema check those of
    every world the run computed. Every call writes it.
``<stem>_report_exclusions.parquet``
    The report's exclusion rows of the written worlds: each excluded building with
    its reason in each written world, whether or not the GeoJSON holds the building.
    Every call writes it.
``<stem>_manifest.json``
    The manifest: the digest of each frozen stage, in the order the stages were
    frozen. Every call writes it last. A call that finds a frozen stage changed after
    it was frozen raises ``RuntimeError`` in its place, once the other files are
    written.
``<stem>_report.parquet``
    The report's action rows of the written worlds, as the section "Rows" of
    ``brails/compilers/README.rst`` states. A call writes it with ``detailed=True``,
    and a refused call without it too.
``<stem>_<name>.parquet``
    The log ``<name>`` of the result's ``logs``, one file for each log the result
    holds, such as ``<stem>_gross_area.parquet``. A log with a ``world`` column keeps
    the rows of the written worlds, and a log without one is written whole. A call
    writes the logs with ``detailed=True``, and a refused call without it too. The
    result holds every log in memory either way.

After a compilation with a boundary, the GeoJSON, the summary and the report rows
hold the buildings inside it unless ``keep_outside`` is given, and the logs hold
every building (see "A boundary" above).

What a call removes
~~~~~~~~~~~~~~~~~~~

Before it writes anything, a call removes the files of action rows and of logs at the
stem that it will not write, whoever wrote them. Every call removes
``<stem>_<name>.parquet`` for each name of ``LOG_NAMES``, every log a run can hold,
that its result does not hold. A call without ``detailed`` that is not refused also
removes ``<stem>_report.parquet`` and the files of every other name of ``LOG_NAMES``.
The stem then holds no action rows and no log beside the files of that call. A call
removes no other file, and it replaces the GeoJSON, the summary, the exclusion rows
and the manifest as it writes them.

A file that cannot be removed stops the call with ``OSError``, before it writes
anything, and the files removed before it stay removed. A call that fails after the
removal leaves the files it has written so far beside those an earlier call wrote,
less the removed files.

A refused file
~~~~~~~~~~~~~~

``write`` refuses the GeoJSON when the log ``MISSING_DEFAULTS_LOG`` names a default
property that a compiled building lacks in a written world (see ``s15_output``).
After a compilation with a boundary, only the buildings inside it count, unless
``keep_outside`` is given. A refused call writes the summary, the exclusion rows, the
action rows, every log the result holds and the manifest, whatever ``detailed`` says,
since the action rows and the logs show how a building came to lack the property. It
then raises ``RuntimeError``, whose message counts the missing values by property and
names up to ten of them, and it writes no GeoJSON. A GeoJSON that an earlier call
wrote at the path stays there.

Scope
-----

The compiler draws structure types from the completed West Coast tables, so
its constructor accepts only states of that region with a RES1 distribution.
It checks the buildings against the one input schema ``DLML_DATASET``, and
the W1 carve-out of ``s08`` applies only in ``CARVE_OUT_STATE`` under
``CARVE_OUT_METHOD``.
``merge_stacked`` takes Overture footprints only, and a compiler built with
it takes no local inventory. A local inventory holds one possible world. The
implausible pairs of occupancy class and story count are those of the West
Coast tables, as the section "Pair re-draw" states.

Pair re-draw
------------

``PairRedraw`` reads a merged inventory and finds the assets whose occupancy
class and story count form an implausible pair, such as a ten-story
agricultural building. For each, it decides from the sources of the two
values which one gives way, and it replaces or re-draws that value. The
re-draw takes an inventory of one possible world. The names of its actions
and the columns of its plan and log are constants in ``pair_redraw.py``
beside this page. This section states which pairs are implausible, which
value gives way, and how the re-draw runs.

Pairs and their sources
~~~~~~~~~~~~~~~~~~~~~~~

The re-draw reads the occupancy class from ``OccupancyClass`` and the story
count from ``NumberOfStories``, the default feature names. The count is read
by ``get_hazus_story_count``, as the structure type inferer reads it, so a
count stored as ``'10'`` reads as 10. An asset whose occupancy class is not
a string, or whose count that function refuses, forms no pair. The re-draw
leaves such an asset unchanged, and the structure type inferer refuses it.

Each count falls in one of the three story bands of the Hazus Appendix A
tables, which ``get_hazus_sampling_height_bands`` returns and the plan's
``band`` column prints: ``low_rise`` for 1 to 3 stories, ``mid_rise`` for 4
to 7 and ``high_rise`` for 8 or more. An occupancy class and a count form an
implausible pair when ``is_implausible`` finds the occupancy class and the
count's band in ``IMPLAUSIBLE_PAIRS``. That list holds RES2 and AGR1 in
``mid_rise`` and ``high_rise``, and IND1 to IND4, IND6, EDU1, GOV2 and REL1 in
``high_rise``. ``get_hazus_story_count`` and
``get_hazus_sampling_height_bands`` are in ``hazus_rulesets.py``, and
``is_implausible`` and ``IMPLAUSIBLE_PAIRS`` in ``hazus_table_completion.py``,
both in ``brails/inferers/hazus_inferer_earthquake/``.

The count the footprint height gives, the height count, comes from
``StoriesFromHeightInferer.stories_of``. An NSI story count is corroborated
when the height count falls in the same story band. The asset has no height
count wherever ``StoriesFromHeightInferer.check`` would not return ``OK``:
among other cases, where the height is missing, not a positive finite number or
one value per world, where the inferer's table of meters per story has no scale
for the footprint provider or the height's source, and where the count exceeds
the most stories the inferer trusts from that source.

The re-draw reads the codes of the occupancy class and the story count from
the asset's source string (see the section "The source string" of
``brails/compilers/README.rst``). It never changes a value from a local or
footprint source, the sources of ``FIXED_SOURCES``. A pair with a local or
footprint occupancy class is only logged, whatever the source of its count.
A derived story count is the one ``StoriesFromHeightInferer`` gives from the
footprint height, so it is kept like a local one. An imputed value is left
to the validation that ``ObservedDonorImputer.impute_stage`` runs, when
given ``validate``, on the values it copies. That validation can re-draw a
target it rejects, so a pair with an imputed member is only logged. That
leaves the pairs with an NSI occupancy class. Of these, an NSI story count
gives way where the height count falls in another story band, and the
occupancy class is then re-drawn only if the pair stays implausible at that
count. Where the height gives no count, the occupancy class and the NSI
count are re-drawn together. In every other case the occupancy class gives
way.

The action table
~~~~~~~~~~~~~~~~

Each row names the sources of the occupancy class and the story count and the
action the plan records for them, as its ``action`` column prints it. In the
last row the occupancy class may come from NSI, a local or footprint source, or
imputation.

=========  ============================  ====================================
Occupancy  Story count                   Action
class
=========  ============================  ====================================
NSI        local, footprint or derived   ``redraw_occupancy``
NSI        NSI, corroborated             ``redraw_occupancy``
NSI        NSI, not corroborated, with   ``replace_stories``
           a height count that makes
           the pair plausible
NSI        NSI, not corroborated, with   ``replace_stories_redraw_occupancy``
           a height count that leaves
           the pair implausible
NSI        NSI, with no height count     ``redraw_pair``
local      local, footprint, NSI or      ``log_local``
           derived
footprint  local, footprint, NSI or      ``log_local``
           derived
imputed    local, footprint, NSI or      ``log_imputed``
           derived
any        imputed                       ``log_imputed``
=========  ============================  ====================================

``redraw_occupancy`` re-draws the occupancy class against the story count the
asset keeps. ``redraw_pair`` re-draws the occupancy class and the story count
together, from one donor. ``replace_stories`` puts the height count in place of
the NSI count and draws nothing, and ``replace_stories_redraw_occupancy`` does
the same and then re-draws the occupancy class against the height count.
``log_local`` and ``log_imputed`` change nothing.

Any other combination of codes on an implausible pair raises ``ValueError``: an
occupancy class that is derived, drawn or held by no source (``R``, ``D`` or
``-``), or a story count that is drawn or held by no source (``D`` or ``-``).

The procedure
~~~~~~~~~~~~~

The re-draw runs in two passes. ``plan`` lists the assets with an
implausible pair and the action the table above gives each, and ``redraw``
carries the actions out: it puts the height count in place of an NSI story
count, or draws new values for the asset from one observed donor (see the
section "Donors" of ``brails/compilers/README.rst``). ``redraw`` returns the
new values, the values they replace and a log, and leaves the inventory
unchanged.

``plan`` reads the inventory and changes nothing. It refuses an inventory of
more than one possible world, and it visits the assets in ascending order of
the asset id as a string. Every asset must carry a source string with one
code for each source feature, whether or not its pair is implausible. For
each asset with an implausible pair, the plan records one row of
``PLAN_COLUMNS``: the occupancy class and the story count as read, the count's
story band, the two source codes, the height count, whether the height count
corroborates the story count, and the action.

``redraw`` runs the plan and takes its rows in order. Where the action
replaces the story count, the height count goes into the result's
``derived`` values and the NSI count is withheld. Where the action re-draws,
the re-draw calls ``ObservedDonorImputer.draw`` once for the asset, and
``draw`` copies the re-drawn values from one observed donor. The imputer's
``pool`` builds the donor pool from the donors given to ``PairRedraw`` and
from the inventory, once for each call of ``redraw`` and only if some asset
is re-drawn. The draw runs in the world given to ``PairRedraw``, from
iteration 0, under purpose labels prefixed with ``PURPOSE_PAIR_REDRAW``, so
that its random numbers differ from those of the imputation step. The draw
is given the asset's residential status, and it may not change the asset's
local and footprint features. Once its attempts are spent, or no untried
candidate is left, it scans the nearest donors.

An asset is residential when its occupancy class is one of the residential
occupancy classes that "The inventory and its inputs" names. A candidate
donor passes when its values hold every re-drawn feature, its occupancy
class is a string with the same residential status as the asset, and its
occupancy class and the story count form a plausible pair. The story count
is the donor's own where the count is re-drawn, and the donor fails if
``get_hazus_story_count`` refuses it. Otherwise the count is the one the
asset keeps: its own, or the height count that replaces it. For a
residential asset, the donor's occupancy class must also agree with the
asset's unit count (``NumberOfUnits`` by default) under the count rule that
the NSI footprint join applies, ``point_class`` in
``brails/utils/unit_count_rules.py``. Under that rule a RES1 or RES3
occupancy class must be the occupancy class of the unit-count band the unit
count falls in, and a missing or non-positive count agrees with every
occupancy class.

A draw that finds a donor gives the asset new values for the re-drawn
features, and the values they replace are withheld, each with the label of
its source. A draw that finds none adds no new value, and a story count
already replaced stays replaced. If a donor's values lack a re-drawn
feature, which the test above rules out, ``RuntimeError`` is raised.

Every planned asset takes one row of ``LOG_COLUMNS`` in the log: its plan
row, which names the occupancy class it had, then the world and the draw's
outcome, attempts, iteration and donor. An asset that takes no draw is logged
with the outcome ``OUTCOME_LOGGED``, no attempts, no iteration and no donor.
The re-draw leaves the inventory unchanged, and the caller writes the result
into the sources as the ``PairRedrawResult`` docstring states. When the
sources are merged again, a re-drawn value carries the imputed code and a
height count carries the derived code.

Scope
~~~~~

The implausible pairs are those of the completed West Coast tables.
``IMPLAUSIBLE_PAIRS`` is a module constant, so a list for another region is
a change to the code, not an argument. A pair the re-draw only logs stays
implausible. The structure type inferer still draws a type for such an
asset under ``complete_tables=True``. Its tables give a distribution of
structure types for each cell, a combination of occupancy class, story band
(none for RES1) and, in the West Coast tables, era. The completed tables fill
the cells of those pairs that the Hazus manual leaves without a distribution,
which in the West Coast tables is every cell of every pair except the pre-1950
cell of ``AGR1`` in the mid-rise band.

A re-drawn asset keeps its other features, so a RES2 asset re-drawn to another
occupancy class keeps the year drawn for it as RES2.

A ``PairRedraw`` works on one world: the inventory it reads holds one
possible world, the donors are that world's observed donors, and its draws
run in the world it is given. Each world of a study takes its own
``PairRedraw``. The plan and the result do not depend on the order of the
inventory's assets.

Composed screen
---------------

``ComposedScreen`` asks the Hazus earthquake inference chain whether the
chain can accept an asset before a structure type is drawn for it. The chain
runs two checks. In the design level check, the design level inferer reads
``YearBuilt`` and writes ``DesignLevel``. In the structure type check, the
structure type inferer reads both features, with the occupancy class, the
story count and the building type. The report stages of the two checks are
``STAGE_DESIGN_LEVEL`` and ``STAGE_STRUCTURE_TYPE``, constants in
``resolution_loop.py`` beside this page. The screen reads an inventory of
one possible world, such as the realization that ``get_world_realization``
returns, and ``screen_inventory`` raises ``ValueError`` for an inventory of
any other number of worlds.

The procedure
~~~~~~~~~~~~~

The screen runs the two checks in the order the chain runs them and returns
a verdict for each asset: a pass, or the refusal of the first check that
refused it, with the features whose change would let that check produce a
value.

After checking that the inventory holds one world, ``screen_inventory`` passes
it to the design level inferer's ``check``. An asset the design level inferer
refuses keeps that refusal and is not screened further. Each other asset is
copied into a new inventory under its own id. Every copy is placed at
``SCREEN_COORDINATES``, since neither inferer reads a location. Where a copy
holds a construction year and no design level, the design level inferer's
``draw`` writes on the copy the level that the year implies. The structure
type inferer's ``check`` then runs over the copies, and an asset it refuses
receives that refusal. The screen writes on the copies only, so the
inventory it is given does not change. Each inferer's ``check`` runs once
per call. ``screen`` runs the same procedure on an inventory holding the one
asset under the id ``SCREEN_ASSET_ID``.

A refusal names each feature whose change alone would let its check produce
a value. These features are the ``resolvable_by`` of the inferer's diagnosis,
and the screen returns them as the verdict's ``candidates``, in the order
the inferer named them. Neither inferer of the chain names the design level.
If one did, the screen would return the construction year in its place,
since the level is derived from the year, listing the year once even where
the inferer also named it.

The exclusion category
~~~~~~~~~~~~~~~~~~~~~~

A refused asset that the caller excludes is filed under one of the categories
in ``EXCLUSION_CATEGORIES`` of ``hazus_eq_inventory_compiler.py``. The section
"Exclusion categories" of ``brails/compilers/README.rst`` defines each one. A
refusal with the diagnosis ``unsupported``, which an inferer gives for a
feature holding a list of values, is filed as ``unsupported``, and one with
the diagnosis ``invalid`` as ``missing_input``. An infeasible refusal, at
either check, is filed as ``conflict`` where the caller acted on a feature
the refusal named and the asset was refused again. The resolution loop below
is such a caller, but with the structure type inferer in BRAILS it never
reaches this case.

Any other infeasible refusal at the structure type check is classified by the
asset's cell: the combination of occupancy class, story band and, in the West
Coast tables, era that selects a distribution of structure types. The Hazus
manual prints that distribution for some cells, and BRAILS's completed tables
fill others. The structure type inferer's ``cell_fill`` and
``cell_is_unpublished`` report which applies. The refusal is
``no_plausible_value`` where the cell has neither a printed nor a filled
distribution. It is ``conflict`` where the cell has one, since the asset's own
features then rule out every structure type in it.

Any other infeasible refusal at the design level check is ``undetermined``.
The design level operation of ``s08_structure``, which runs the same inferer
outside the screen, files an infeasible refusal as ``missing_input``
instead. The design level inferer in BRAILS never refuses an asset as
infeasible.

Resolution loop
---------------

``ResolutionLoop`` decides, for each asset of one world's merged inventory,
whether the asset goes on to the structure type draw or is excluded from
that world. The loop has one action by which it can admit a refused asset:
it withholds the general building type, ``BuildingType``, where NSI holds
it. The source it acts on and the report stage it records under are
``WITHHELD_SOURCE`` and ``STAGE_RESOLUTION``, constants in
``resolution_loop.py``. The loop reads the realization of one world, as the
composed screen does, and ``resolve`` raises ``ValueError`` for an inventory
of any other number of worlds. The loop records its actions and exclusions
under the world of the screen's structure type inferer. A caller with
several worlds therefore builds a screen and a loop for each world, as the
compiler does. With its ``report_stages`` argument, the loop files the rows
of each of its report stages under another report stage. The compiler uses
this argument to file the rows of ``s10`` under that step's own report
stages.

The procedure
~~~~~~~~~~~~~

The loop screens the inventory once and admits every asset the screen
passes. Where a refusal other than an unsupported one names ``BuildingType``
among its candidates, and NSI gave the asset that type, the loop withholds the
type and screens the asset again, admitting it if it then passes. Every other
refused asset, and one still refused without its type, is excluded from the
world, and the report records each withholding and each exclusion.

``resolve`` first checks that the inventory holds one world and that
``building_type_sources`` gives a source for each asset that holds a building
type and for no other. It then screens the inventory once, through
``screen_inventory``. An asset the screen passes is admitted as it is, and
an asset refused as unsupported is excluded without action. Where another
refusal names ``BuildingType`` among its candidates and ``WITHHELD_SOURCE``
holds the type, the report records the type with the action ``withheld``,
and the asset is screened again without it. An asset that then passes is
admitted without the type. A type held by any other source is never
withheld, nor is one the refusal does not name.

A refused asset the loop did not act on is excluded under the report stage of
the check that refused it, in the category that the section "The exclusion
category" gives. An asset still refused once its building type is withheld
is the case the section "Scope" below describes. The report records its
building type again, with the action ``restored``. The asset's exclusion is
recorded at ``STAGE_RESOLUTION``, with a detail that holds both refusals, and
its category is ``conflict`` where the first refusal was infeasible. Every
exclusion records the world and the diagnosis and candidates of the first
refusal. Every ``withheld`` and ``restored`` row of the report holds the
world and the source.

Scope
~~~~~

The structure type inferer in BRAILS names ``BuildingType`` exactly when the
asset passes without it. With that inferer, withholding the type therefore
always admits the asset, and the ``restored`` path is not reached. The path
serves a structure type inferer for which the building type is a required
input, and BRAILS holds no such inferer. When the inferer cannot read a
building type, it refuses the asset as invalid and names no feature. The
type is then not withheld, and the asset is filed as ``missing_input``.
