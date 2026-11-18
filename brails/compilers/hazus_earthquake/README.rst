Hazus earthquake inventory compiler
===================================

The modules of ``brails.compilers.hazus_earthquake`` compile a Hazus
earthquake building inventory. ``HazusEQInventoryCompiler`` runs the
compilation, and ``PairRedraw``, ``ComposedScreen`` and ``ResolutionLoop``
carry out three of its steps. This page describes each of them in a section
of its own. The concepts they share with the imputers and inferers they call
are explained in the section "Concepts" of ``brails/compilers/README.rst``,
and the composition of a source from frozen stages in its section
"Composition and assembly".

Pair re-draw
------------

``PairRedraw`` reads a merged inventory and finds the assets whose occupancy
class and story count form an implausible pair, such as a ten-story
agricultural building. For each, it decides from the sources of the two
values which one gives way, and it replaces or re-draws that value. The
re-draw takes an inventory of one possible world (see the section "Possible
worlds" of ``brails/compilers/README.rst``). The names of its actions and the
columns of its plan and log are constants in ``pair_redraw.py`` beside this
page. This section states which pairs are implausible, which value gives way,
and how the re-draw runs.

Pairs and their sources
~~~~~~~~~~~~~~~~~~~~~~~

The re-draw reads the class from ``OccupancyClass`` and the story count from
``NumberOfStories``, the default feature names. The count is read by
``get_hazus_story_count``, as the structure type inferer reads it, so a
count stored as ``'10'`` reads as 10. An asset whose class is not a string,
or whose count that function refuses, forms no pair. The re-draw leaves it,
and the structure type inferer refuses it.

Each count falls in one of the three story bands of the Hazus Appendix A
tables, which ``get_hazus_sampling_height_bands`` returns and the plan's
``band`` column prints: ``low_rise`` for 1 to 3 stories, ``mid_rise`` for 4
to 7 and ``high_rise`` for 8 or more. A class and a count form an
implausible pair when ``is_implausible`` finds the class and the count's band
in ``IMPLAUSIBLE_PAIRS``. That list holds RES2 and AGR1 in ``mid_rise`` and
``high_rise``, and IND1 to IND4, IND6, EDU1, GOV2 and REL1 in ``high_rise``.
``get_hazus_story_count`` and ``get_hazus_sampling_height_bands`` are in
``hazus_rulesets.py``, and ``is_implausible`` and ``IMPLAUSIBLE_PAIRS`` in
``hazus_table_completion.py``, both in
``brails/inferers/hazus_inferer_earthquake/``.

The re-draw reads the codes of the occupancy class and the story count from
the asset's source string (see the section "The source string" of
``brails/compilers/README.rst``). It never changes a local
or footprint value, the sources of ``FIXED_SOURCES``, and a pair with a local
or footprint occupancy class is only logged, whatever the source of its count.
A derived story count is the one ``StoriesFromHeightInferer`` gives from the
footprint height, so it is kept like a local one. An imputed value is left to
the validation that ``ObservedDonorImputer.impute_stage`` runs on the values
it copies when given ``validate``, which can re-draw a target it rejects, so a
pair with an imputed member is only logged. That leaves the pairs with an NSI
class. Of these, an NSI story count gives way where the footprint height gives
a count in another story band, and the class is then re-drawn only if the pair
stays implausible at that count. Where the height gives no count, the class
and the NSI count are re-drawn together. In every other case the class gives
way.

The count the footprint height gives, the height count, comes from
``StoriesFromHeightInferer.stories_of``. An NSI story count is corroborated
when the height count falls in the same story band. The asset has no
height count wherever ``StoriesFromHeightInferer.check`` would not give
``OK``: among other cases, where the height is missing, not a positive
finite number or one value per world, where the footprint provider or the
height's source has no scale, and where the count exceeds the most stories
the inferer trusts from that source.

The action table
~~~~~~~~~~~~~~~~

Each row names the sources of the class and the story count and the action
the plan records for them, as its ``action`` column prints it. In the last
row the class may come from NSI, a local or footprint source, or imputation.

=========  ============================  ====================================
Class      Story count                   Action
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

``redraw_occupancy`` re-draws the class against the story count the asset
keeps. ``redraw_pair`` re-draws the class and the story count together, from
one donor. ``replace_stories`` puts the height count in place of the NSI
count and draws nothing, and ``replace_stories_redraw_occupancy`` does the
same and then re-draws the class against the height count. ``log_local`` and
``log_imputed`` change nothing.

Any other combination of codes on an implausible pair raises ``ValueError``:
a class that is derived, drawn or held by no source (``R``, ``D`` or ``-``),
or a story count that is drawn or held by no source (``D`` or ``-``).

The procedure
~~~~~~~~~~~~~

``plan`` reads the inventory and changes nothing. It refuses an inventory of
more than one possible world, and it visits the assets in ascending order of
the asset id as a string. Every asset must carry a source string with one
code for each source feature, whether or not its pair is implausible. For
each asset with an implausible pair, the plan records one row of
``PLAN_COLUMNS``: the class and the story count as read, the count's story
band, the two source codes, the height count, whether the height count
corroborates the story count, and the action.

``redraw`` runs the plan and takes its rows in order. Where the action
replaces the story count, the height count goes into the result's
``derived`` values and the NSI count is withheld. Where the action re-draws,
the re-draw calls ``ObservedDonorImputer.draw`` once for the asset, which
copies the re-drawn values from one observed donor. The donor pool is built
once for each call of ``redraw``, by the imputer's ``pool`` from the donors
given to ``PairRedraw`` and the inventory, and only if some asset is
re-drawn. The draw runs in the world given to ``PairRedraw``, from
iteration 0, under purpose labels prefixed with ``PURPOSE_PAIR_REDRAW``
(``'pair_redraw'``), so that its random numbers differ from those of the
imputation stage. It is given the asset's residential status, it may not
change the asset's local and footprint features, and once its attempts are
spent or no untried candidate is left it scans the nearest donors.

An asset is residential when its class is one of ``UNIT_COUNT_CLASSES``:
RES1, RES2 and RES3A to RES3F. A candidate donor passes when its values hold
every re-drawn feature, its class is a string with the asset's residential
status, and its class and the story count form a plausible pair. The story
count is the donor's own where the count is re-drawn, and the donor fails if
``get_hazus_story_count`` refuses it. Otherwise the count is the one the
asset keeps: its own, or the height count that replaces it. For a
residential asset, the class must also agree with the asset's unit count
(``NumberOfUnits`` by default) under the count rule that the NSI footprint
join applies, ``point_class`` in ``brails/utils/unit_count_rules.py``. Under
that rule a RES1 or RES3 class must be the class of the band the unit count
falls in, and a missing or non-positive count agrees with every class.

A draw that finds a donor gives the asset new values for the re-drawn
features, and the values they replace are withheld, each with the label of
its source. A draw that finds none adds no new value, and a story count
already replaced stays replaced. A donor whose values lack a re-drawn
feature, which the test above excludes, raises ``RuntimeError``.

Every planned asset takes one row of ``LOG_COLUMNS`` in the log: its plan
row, which names the class it had, then the world and the draw's outcome,
attempts, iteration and donor. An asset that takes no draw is logged with
the outcome ``OUTCOME_LOGGED`` (``'logged'``), no attempts, no iteration and
no donor. The re-draw leaves the inventory unchanged, and the caller writes
the result into the per-source inventories as the ``PairRedrawResult``
docstring states. Merged again, a re-drawn value carries the imputed code
and a height count the derived one.

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

A re-drawn asset keeps its other features, so a RES2 asset re-drawn to
another class keeps the year drawn for it as RES2.

A ``PairRedraw`` serves one world: the inventory it reads holds one possible
world, the donors are that world's observed donors, and its draws run in the
world it is given. Each world of a study takes its own. The plan and the
result do not depend on the order of the inventory's assets.

Composed screen
---------------

``ComposedScreen`` asks the Hazus earthquake inference chain whether the
chain can take an asset before a structure type is drawn for it. The chain
has two stages: the design level inferer reads ``YearBuilt`` and writes
``DesignLevel``, and the structure type inferer reads both, with the
occupancy class, the story count and the building type. The stage names
are ``STAGE_DESIGN_LEVEL`` and ``STAGE_STRUCTURE_TYPE``, constants in
``resolution_loop.py`` beside this page. The screen reads an inventory of
one possible world, such as the realization that ``get_world_realization``
returns. Both inferers refuse as unsupported a feature that holds one value
per world. Given an inventory of several worlds, the screen therefore
refuses each asset with a feature that the chain reads and whose value
differs between worlds. This section states how the screen runs and how
the category of a refused asset is settled.

The procedure
~~~~~~~~~~~~~

``screen_inventory`` first passes the inventory to the design level
inferer's ``check``. An asset the design level inferer refuses keeps that
refusal and is not screened further. Each other asset is copied into a new
inventory under its own id. Every copy stands at ``SCREEN_COORDINATES``,
since neither inferer reads a location. Where a copy holds a construction
year and no design level, the design level inferer's ``draw`` writes on the
copy the level that the year implies. The structure type inferer's
``check`` then runs over the copies, and an asset it refuses takes that
refusal. The screen writes on the copies only, so the inventory it is given
does not change. Each inferer's ``check`` runs once per call. ``screen``
runs the same procedure on an inventory holding the one asset under the id
``SCREEN_ASSET_ID``.

A refusal names each feature whose change alone would let its stage produce
a value. These features are the ``resolvable_by`` of the stage's diagnosis,
and the screen returns them as the verdict's ``candidates``, in the order
the stage named them. Neither inferer of the chain names the design level.
If a stage did, the screen would return the construction year in its place,
since the level is derived from the year, listing the year once even where
the stage also named it.

The exclusion category
~~~~~~~~~~~~~~~~~~~~~~

A refused asset that the caller excludes is filed under one of the categories
in ``EXCLUSION_CATEGORIES`` of ``hazus_eq_inventory_compiler.py``. The section
"Exclusion categories" of ``brails/compilers/README.rst`` defines each one. A
refusal with the diagnosis ``unsupported`` is filed as ``unsupported``, and one
with the diagnosis ``invalid`` as ``missing_input``.
An infeasible refusal is ``conflict`` where the caller acted on a feature the
refusal named and the asset was refused again. The resolution loop below is
such a caller, but with the structure type inferer in BRAILS it never reaches
this case. Any other infeasible refusal at the structure type stage is
classified by the asset's cell: the combination of occupancy class, story band
and, in the West Coast tables, era that selects a distribution of structure
types. The Hazus manual prints that
distribution for some cells and BRAILS's completed tables fill others, and
the structure type inferer's ``cell_fill`` and ``cell_is_unpublished``
report which applies. The refusal is ``no_plausible_value`` where the cell has
neither a printed nor a filled distribution. It is ``conflict`` where the
cell has one, since the asset's own features then rule out every structure
type in it. An infeasible refusal at the design level stage is
``undetermined``. The design level inferer in BRAILS never refuses an asset
as infeasible.

Resolution loop
---------------

``ResolutionLoop`` decides, for each asset of one world's merged
inventory, whether the structure type draw takes the asset or the asset is
excluded from that world. The loop has one action by which it can admit a
refused asset: it withholds the general building type, ``BuildingType``,
where NSI holds it. The source it acts on and the stage it records under
are ``WITHHELD_SOURCE`` and ``STAGE_RESOLUTION``, constants in
``resolution_loop.py``. The loop reads the realization of one world, as the
composed screen does, and records its actions and exclusions under the
world of the screen's structure type inferer. A caller with several worlds
therefore builds a screen and a loop for each world, as the compiler below
does. This section states how the loop runs and what it records.

The procedure
~~~~~~~~~~~~~

``resolve`` first checks ``building_type_sources`` for every asset, then
screens the inventory once, through ``screen_inventory``. An asset the
screen passes is admitted as it is, and an unsupported asset is excluded
without action. Where another refusal names ``BuildingType`` among its
candidates and ``WITHHELD_SOURCE`` holds the type, the report records the
type with the action ``withheld``, and the asset is screened again without
it. An asset that then passes is admitted without the type. A type held by
any other source is never withheld, nor is one the refusal does not name.

A refused asset the loop did not act on is excluded at the stage that refused
it, in the category of the section "The exclusion category". An asset still
refused once its building type is withheld is the case the section "Scope"
below describes. The report records its building type again, with the action
``restored``. The asset's exclusion is recorded at ``STAGE_RESOLUTION``, with a
detail that holds both refusals, and its category is ``conflict`` where the
first refusal was infeasible. Every exclusion records the world and the
diagnosis and candidates of the first refusal. Every ``withheld`` and
``restored`` row of the report holds the world and the source.

Scope
~~~~~

The structure type inferer in BRAILS names ``BuildingType`` exactly when the
asset passes without it, so with that inferer a withheld type always admits
the asset and the ``restored`` path is not reached. The path serves a
structure type inferer for which the building type is a required driver,
which BRAILS does not hold. A building type the inferer cannot read refuses
the asset as invalid and names no feature, so it is not withheld, and the
asset is filed as ``missing_input``.

Hazus earthquake inventory compiler
-----------------------------------

``HazusEQInventoryCompiler`` builds a Hazus earthquake building inventory of
one study area from its building footprints, its NSI points and census
tables, over several possible worlds. The caller gathers the data in a
``CompilerInputs`` and receives a ``CompilationResult``, whose ``write``
saves the inventory as GeoJSON beside its report, the manifest of its stages
and the logs of its components. Both classes are in
``hazus_eq_inventory_compiler.py`` beside this page. This section states the
run in the order the code runs it, the stages it freezes, its worlds, the
file it writes and the settings it gives its components.

The run
~~~~~~~

Step 1 takes the footprints as given, keyed by the caller's asset ids, unless
``merge_stacked`` is set. With ``merge_stacked=True``,
``StackedFootprintMerge`` first merges the stacked Overture footprints of
one building, as the section "The method" of
``brails/aggregators/stacked_footprints/README.rst`` states, and every later
step reads the merged set under its geohash keys. A merged asset takes the
height of its tallest part over the union of the parts, as that section
states. Where its story count comes from that height, the run values a podium
under a tower as the tower's story count times the merged plan area, and
its ``GrossArea`` is that product. The
merge's parts table and id log are written as the logs ``stacked_parts`` and
``stacked_asset_ids``, and a notice in the report counts the merges. The
option is off by default, since the merge takes Overture footprints only.
With it, the compiler refuses a local inventory, whose ids are those of the
footprints before the merge.

Each footprint receives its census block group from the block-group polygons,
its plan area in square feet as ``PlanArea`` and, if it is an Overture
footprint, the release and the datasets of its sources, which with the caller's
``footprint_provider`` make up ``FOOTPRINT_PROVENANCE_FEATURES``. Steps 2 and 3
join the NSI points to the footprints with ``NSIFootprintJoin`` and derive
their unit counts and occupancy classes: the join's observed values form the
NSI source, and the values it derives, such as an occupancy class set by the
count of units, the derived source.

A local inventory, which takes precedence over every other source, may hold no
feature of ``RESERVED_LOCAL_FEATURES`` (``BLOCKGROUP_GEOID`` and
``NSIGrossArea``), which the run writes for itself and reads back, and
``compile`` refuses one that does.

Step 4 completes the story counts, the unit counts and the occupancy classes,
on the merge of the parts so far. ``StoriesFromHeightInferer`` gives each
footprint with a usable height the count that height implies, which fills a
story count no source holds. For an occupancy class other than
``NSI_STORIES_CLASS`` (RES1) that count also replaces NSI's, which is withheld.
A RES1 building keeps NSI's count. ``ACSUnitBudget`` then allocates each
block group's census housing units to the footprints that no source gives an
occupancy class, as ``brails/inferers/acs_unit_budget/README.rst`` states. It
reads a footprint's ``GrossArea`` feature first as its gross area, a local one
included, so a local inventory's gross areas can change the occupancy classes,
unit counts and draws of this step and of the steps after it. It makes some
footprints near mobile-home park points mobile homes, relabeling only an
occupancy class that NSI or the join gave, and drops the small footprints it
leaves unused. The report lists the dropped footprints in every world.
``ObservedDonorImputer`` fills what is still missing from observed donors, as
the section "Donors" of ``brails/compilers/README.rst`` describes, and gives a
target whose copy forms an implausible pair of occupancy class and story count
another donor. The
residential occupancy classes of this section are those of
``UNIT_COUNT_CLASSES`` (RES1, RES2 and the RES3 classes). A reached footprint
outside them is a target for its story count alone, since the join gives units
to housing records only.

Steps 5 and 6 run over every world in three passes, each over all worlds before
the next. Step 5 (``s06_residential_year``) reads each world's realization of
the frozen view ``STAGE_STEP4``, the pair re-draw of step 6
(``s07_pair_redraw``) reads each world merged after step 5, and the rest of
step 6 reads each world merged after the re-draw. Step 5 gives construction
years to residential buildings: ``ACSYearBuiltSampler``, fitted on that
realization, draws a year for each residential building that has no
construction year, from the census tables of the building's block group and
tract. Step 6 then runs, in this order, the pair re-draw over every building of
the world and the design level, structure type and height class components over
the buildings each one takes. ``PairRedraw`` first replaces the NSI members of
implausible pairs of occupancy class and story count, as the section "Pair
re-draw" states, with the observed donors of the world's realization of
``STAGE_STEP4``. ``HazusEQDesignLevelInferer`` then derives a design level from
the construction year for each building that has a year and no design level.
Each building that has a year and no structure type then passes through
``ComposedScreen`` and ``ResolutionLoop``, as the sections "Composed screen"
and "Resolution loop" state, and ``HazusEQStructureTypeInferer`` draws a
structure type for each building the loop admits. Last,
``HazusEQHeightClassInferer`` gives a height class to each building that holds
a structure type, drawn in this step or its own, and no height class. A
building without a construction year therefore takes part in step 6 only
through the pair re-draw and, where it holds its own structure type, the height
class. Step 7 then takes up such a building if it is not excluded: it gives the
building a year, and then a design level, a structure type and a height class
as step 6 does. A building one of these components refuses is excluded from
that world. The types are drawn from the completed West Coast tables, so the
compiler takes states of that region only. In ``CARVE_OUT_STATE`` (California)
under ``CARVE_OUT_METHOD`` (5.1), a drawn W1 building whose derived level is
Pre-Code is raised to Moderate-Code, as Table C-10 of the Hazus manual assumes.

Step 7 gives a construction year, in each world, to each building without a
year whose occupancy class is missing or is not one of the residential
occupancy classes. It reads that world's realization of the frozen view
``STAGE_STEP6``. ``ResidentialNeighborYearSampler`` copies to each such
building the year of a nearby residential building of that world, as
``brails/imputers/residential_neighbor_year_sampler/README.rst`` states. The
design level, structure type and height class of the building then follow as a
residential building's do. A building the sampler finds no donor for, or one
without an occupancy class, is excluded from that world.

The last five steps follow. ``s11_gross_area`` computes, in each world, the
gross area of every building of the view ``STAGE_STEP6``, excluded buildings
included: its plan area times its story count in that world, or its plan area
alone without a positive story count, from ``footprint_gross_area`` of the
value imputer's module, so it differs between worlds where the story count
does. It is frozen as ``STAGE_S11_GROSS_AREA``, and the log ``gross_area``
names for each building and world whether the area came from the story count
(``stories``) or the plan area alone (``plan_area``). The run then withholds
from every later merge the values that the last steps replace: the local and
NSI ``GrossArea``, the join's ``ReplacementCost`` where no local value
outranks it, NSI's ``BuildingType``, and a ``GrossArea`` or ``BuildingType``
that the footprints carry, in six records named by the step whose value takes
their place. The views ``STAGE_STEP4`` and ``STAGE_STEP6`` are frozen before
the records, so the unit budget, the structure type draw and the rates of
``s13_replacement_cost`` still read these values. A ``GrossArea`` that a local
inventory gives is kept as ``LocalGrossArea``, which the first step writes on
the local values, and NSI's as ``NSIGrossArea``, which the compiler writes on
the join's values.

Steps ``s12`` to ``s15`` then run world by world, on one merge of each
world's sources in the precedence of ``PRECEDENCE``: local values first, then
the footprint, NSI, derived, imputed and drawn values. ``AssetInventory.merge``
writes the source string, as the section "The source string" of
``brails/compilers/README.rst`` describes.

``s12_schema_check`` checks every building not yet excluded from the world
against the Hazus 5.1 input schema of the damage and loss model library,
``DLML_DATASET``, and a building it refuses is excluded from that world. The
schema reads none of ``ReplacementCost``, ``GrossArea`` and ``BuildingType``,
which ``s15_output`` writes after the check.

``s13_replacement_cost`` values every building of ``STAGE_STEP6`` in the
world, excluded buildings included. ``ReplacementCostImputer`` reads that
world's realization of the view and the donors that
``imputation_donors_from_log`` names from step 4's log, as the section
"Donors" of ``brails/compilers/README.rst`` describes, and gives each building
a rate per square foot times its gross area of ``s11_gross_area``, as
``brails/imputers/replacement_cost/README.rst`` states. A rate divides a
value, NSI's, the join's sum of NSI's or a local one, by ``NSIGrossArea``,
NSI's ``sqft``, which the compiler gives the imputer in place of
``GrossArea``. A local value therefore gives a rate only on a building whose
NSI records give a ``sqft``, and a local ``GrossArea`` enters neither a rate
nor the area a rate multiplies. A building not yet excluded for which the
imputer finds no donor is excluded from that world as ``no_plausible_value``.
A building not yet excluded has an occupancy class, which the schema requires,
and a positive area, so the step files no other outcome. A building it left
without a value for another reason would be excluded as ``missing_input``.
A building the schema refused is not filed again. The step
keeps a local value as it is. Where that value gives a rate, the rate also
values each building that names the local building as its donor or draws it
as a neighbor. The step's values are frozen, over the worlds, as
``STAGE_S13_REPLACEMENT_COST``.

``s14_building_type`` gives every building without a local ``BuildingType``
the letter of its structure type in the world, excluded buildings included, by
the pairing of ``get_hazus_general_type_membership`` in
``brails/inferers/hazus_inferer_earthquake/hazus_rulesets.py`` that the
structure type inferer draws with. NSI's letter, which the draw reads, is not
written. Where a local letter differs from the letter of the building's
structure type, the file carries the local letter, and a notice in the report
counts such compiled buildings in each world. The letters are frozen, over
the worlds, as ``STAGE_S14_BUILDING_TYPE``.

``s15_output`` writes the results into the merge. Each building receives
``Excluded``, true in a world from which the report lists it as excluded, and
the letter of ``s14_building_type``. Each compiled building without a local
value takes its value of ``s13_replacement_cost`` in that world, coded by its
outcome through ``OUTCOME_VALUE_CODES``: ``R`` for its own rate and ``I`` for
another building's. Every compiled building, one with a local value included,
takes its area of ``s11_gross_area`` as its ``GrossArea``. A building excluded
from the world keeps a local value and its ``LocalGrossArea`` and takes no
computed value or area.

Last, every compiled building must hold every property of
``DEFAULT_PROPERTIES`` that applies to it, ``NumberOfUnits`` applying to the
residential occupancy classes only and ``HeightClass`` to the structure types
for which Hazus defines one. In a world a building is excluded from, it may
lack them. A building that lacks one in a world it is compiled in is a defect
of the run: the log ``MISSING_DEFAULTS_LOG`` names each such building, world
and property, a notice in the report counts them, and ``write`` saves the
report, the logs and the manifest and then refuses the GeoJSON. The worlds are
assembled into one inventory, frozen as ``STAGE_S15_OUTPUT``.

Frozen stages
~~~~~~~~~~~~~

The run is a sequence of stages, each frozen when it ends, as the section
"Stages and freezing" of ``brails/compilers/README.rst`` describes. The stage
names are the module's ``STAGE_`` constants and the value imputer's
``DONOR_STAGE``. Three kinds of stage are frozen. A part holds the values one
step adds, such as the join's NSI values. A record holds what a step withholds
from a source or drops from every source. A view holds the merge of every part
and record frozen so far, over the worlds, and the last view, the output, is
the compiled inventory. Four stages are not sources: the donors that
``imputation_donors_from_log`` names from step 4's log, under
``DONOR_STAGE``, and the areas, values and letters of ``s11_gross_area``,
``s13_replacement_cost`` and ``s14_building_type``, which ``s15_output``
writes over its merge. Every part other than the footprints, and every
record, holds each footprint's centroid as its coordinates. The footprints and
the views hold the footprint polygons. A world's source is composed from its
parts and records by
``compose_source_parts``, and the worlds of a view are added to it by
``AssetInventory.add_world``, as the section "Composition and assembly" of
``brails/compilers/README.rst`` states. The run ends with the manifest.

Worlds
~~~~~~

A run computes the compiler's ``n_worlds`` worlds, numbered as the section
"Possible worlds" of ``brails/compilers/README.rst`` describes, from 0 to
``n_worlds - 1``, by the run and by
every component. A world's values do not depend on how many worlds the run
computes, so world 3 of a run of four worlds equals world 3 of a run of ten.

The written file
~~~~~~~~~~~~~~~~

``write`` writes every world the run computed, or only the world that its
``world`` argument names. The file holds one GeoJSON feature per building, with
its id and footprint. Where every world is written, a property whose worlds
differ holds a list of values, one per world. A file of one world leaves out a
property whose value is ``None`` in that world, so the file is the same
whatever the number of worlds the run computed. Which buildings and properties
the file holds is stated on ``CompilationResult.write``. The properties of
``DEFAULT_PROPERTIES`` are those a Hazus earthquake assessment reads. The
report lists each excluded building with its reason in each world, whether or
not the file holds it. A building's ``ReplacementCost`` divided by its
``GrossArea`` is the rate per square foot that ``s13_replacement_cost`` gave
it, except for a building that keeps a local ``ReplacementCost``. The step
computes the rate of such a building as the local value divided by
``NSIGrossArea``, the ``sqft`` of the building's NSI records (their sum where
it has several), not by its ``GrossArea``, and gives that rate to the
buildings that take their rate from it.

The imputer's settings
~~~~~~~~~~~~~~~~~~~~~~

The compiler gives ``ObservedDonorImputer`` its own values of three settings in
place of the imputer's defaults: the number of nearest donors picked among
(``k``), the meters of separation counted as one unit of donor distance
(``location_scale_m``) and the weight of the difference of log plan areas
(``area_weight``). ``DEFAULT_IMPUTER_OPTIONS`` holds ``k`` of 10, a
``location_scale_m`` of 500 and an ``area_weight`` of 2. They come from a
SimCenter masking test on the buildings NSI reaches in San Francisco and
Richmond, California, in which the occupancy class, unit count and story count
of about a fifth of them were hidden and imputed again from the others. Against
the test's starting values (``k`` of 5, 100 m and a weight of 1), these values
gave better overall occupancy class, story and unit scores in both cities and a
lower exact-label agreement for non-residential buildings. The test measured
donor distance on footprint heights, where the imputer compares story counts,
so it compares settings, and the three values are not calibrated yet.
``imputer_options`` overrides them one by one. The imputer runs without batch
feedback, since the compiler gives it the donors directly, so ``n_batches`` is
refused. ``ResidentialNeighborYearSampler`` takes the imputer's ``k`` and
``cap_m``, and ``ReplacementCostImputer`` takes those and the imputer's
``location_scale_m``, ``area_weight`` and ``stories_weight``.
