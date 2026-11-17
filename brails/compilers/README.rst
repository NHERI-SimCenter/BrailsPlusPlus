Inventory compilers
===================

The classes and functions of ``brails.compilers`` build an inventory out of the
sources a method needs. This page first explains the concepts they share with
the imputers and inferers they call. It then describes the procedures they run,
where their constants come from and the scope of their rules, in a section for
each one that needs it. The docstrings state what a caller passes and receives.

Concepts
--------

Four ideas recur on this page and on the pages of the imputers and inferers.
An inventory can hold several possible worlds, one realization of its
uncertain features in each. The output of a step can be frozen as a stage,
which later steps read by name. A merge of inventories from several sources
records the source of each value in a source string. An imputer can fill a
missing value from another building, its donor. Each subsection explains one
of them.

Possible worlds
~~~~~~~~~~~~~~~

Some features of a building are not observed and are drawn instead, such as
a construction year sampled around a tract median. An inventory can hold
several possible worlds, each one realization of the features whose values
are uncertain. ``AssetInventory`` records their number as ``n_pw``, which is
1 for an inventory of one world. A feature holds either one value, shared
by every world, or a list of exactly ``n_pw`` values, one per world in
world order, and ``validate_possible_worlds`` checks that rule. A feature
whose values are themselves lists, such as ``HousingUnits``
(``DEFAULT_VECTOR_FEATURES``), holds one value per world only as a list of
lists. ``collapse_realizations`` in ``brails/utils/world_values.py`` stores
a feature's per-world values by the same rule: one value when every world
holds it, and the list otherwise.

Worlds are numbered from 0, and this number is the world index.
``get_world_realization`` returns the inventory of one world, with every
feature reduced to its value in that world. A component that draws per world
is told the world it draws. Its random numbers come from ``keyed_uniforms`` in
``brails/utils/deterministic_sampling.py``, which keys each draw by the asset,
the seed, the world, a purpose label and an iteration, so a world's draws can
be repeated and do not depend on the order of the assets. Some components
compute every world of an inventory, and others serve the one world they are
given, each world taking its own instance. Instances serve worlds of one run
only when they share the run's seed, so ``require_run_seed`` refuses a sampler
of a world other than 0 that is given no seed.

Stages and freezing
~~~~~~~~~~~~~~~~~~~

A stage is one step of a longer computation, such as an imputation, and its
output is an inventory. ``StageFreeze``, in ``brails/utils/stage_freeze.py``,
holds the output of each stage under the stage's name and records the order
in which the stages were frozen. A stage is frozen once, and freezing it
records a digest of its output. The freeze holds the output itself, not a
copy, so a stage must not change a frozen output, and a stage that needs to
change one copies it first. ``manifest`` recomputes every digest and raises
``RuntimeError`` if a frozen output no longer matches the digest recorded
when it was frozen. Frozen outputs are held in memory only.

A later step reads an earlier stage from the freeze, by its name, and not
from an inventory that other steps go on changing. It reads the output as
the earlier stage froze it, and the manifest shows any change made to that
output since. The imputers that work in stages take the freeze and the
names of the stages they read. ``ObservedDonorImputer.impute_stage`` reads
its observed donors as a frozen stage and freezes the targets it fills in
batches, and each batch draws from the observed donors and from the batches
frozen before it. ``ReplacementCostImputer.impute_stage`` values the
buildings of one frozen stage with the donors of another and changes
neither.

The source string
~~~~~~~~~~~~~~~~~

``AssetInventory.merge`` combines inventories of the same assets, one per
source, each under a label such as ``'local'`` or ``'nsi'``. Each feature
of each asset takes its value from the first source, in a stated
precedence, that holds one. The merge then records where each value came
from in the feature ``SOURCE_STRING_KEY``, ``SourceString``. The string
holds one character for each feature of ``source_features``, by default
``HAZUS_EQ_SOURCE_FEATURES``: ``OccupancyClass``, ``NumberOfUnits``,
``NumberOfStories``, ``YearBuilt``, ``DesignLevel``, ``StructureType``,
``HeightClass`` and ``ReplacementCost``, in that order. Each character is
the code that ``SOURCE_CODES`` gives the label of the value's source: ``L``
for local, ``F`` for footprint, ``N`` for NSI, ``R`` for derived, ``I`` for
imputed and ``D`` for drawn. Where no source holds a value, the character is
``SOURCE_CODE_NONE``, ``-``.

A merge records which source holds a value, never the value, so the string it
writes is the same in every world. A later step that writes a code per world
can leave one string per world: ``ReplacementCostImputer.impute`` writes the
code of the value it gives, so a building valued in one world and not in
another holds a list of strings, one per world. A merge refuses a source that
already carries a source string, so the string describes one merge.
``SOURCE_CODES``, ``SOURCE_CODE_NONE`` and ``HAZUS_EQ_SOURCE_FEATURES`` are
defined in ``brails/types/asset_inventory.py``, and ``SOURCE_STRING_KEY`` in
``brails/constants.py``. A later step reads a value's code to decide what it
may change, as the pair re-draw below does.

Donors
~~~~~~

A donor is a building whose values fill the missing values of another, the
target. ``ObservedDonorImputer``, in
``brails/imputers/observed_donor_imputer/``, copies from observed donors,
buildings whose values were observed. For each target and world it ranks
the donors near the target that match it, picks one, and copies every
missing feature from that one donor. Its class docstring states the
ranking, the pick and the fallbacks. ``impute_stage`` takes the observed
donors as an inventory, or reads them from a freeze as the stage its
``donor_stage`` argument names.

The imputer's log holds one row for each draw, with the target, the world, the
round, the donor's id and the values copied. ``imputation_donors_from_log``
reads that log and builds an inventory with one asset per target. Its feature
``IMPUTATION_DONOR_KEY``, ``ImputationDonor``, holds, per world, the id of the
donor whose values the target kept, and ``None`` in a world where it kept
none. This inventory is the donor stage that
``ReplacementCostImputer.impute_stage`` reads as its ``donor_stage``, under
the name the caller gives it (``impute`` uses ``DONOR_STAGE``,
``'imputation_donors'``), so that a target can take the rate of its donor. The
two ``donor_stage`` arguments name different stages: the observed donors for
the observed-donor imputer, and the donors named from its log for the
replacement cost imputer.

Pair re-draw
------------

``PairRedraw`` reads a merged inventory and finds the assets whose occupancy
class and story count form an implausible pair, such as a ten-story
agricultural building. For each, it decides from the sources of the two
values which one gives way, and it replaces or re-draws that value. The
re-draw takes an inventory of one possible world (see `Possible worlds`_).
The names of its actions and the columns of its plan and log are
constants in ``pair_redraw.py`` beside this page. This section states which
pairs are implausible, which value gives way, and how the re-draw runs.

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
the asset's source string (see `The source string`_). It never changes a local
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
asset under ``complete_tables=True``, since the completion fills the table
cells of those pairs.

A re-drawn asset keeps its other features, so a RES2 asset re-drawn to
another class keeps the year drawn for it as RES2.

A ``PairRedraw`` serves one world: the inventory it reads holds one possible
world, the donors are that world's observed donors, and its draws run in the
world it is given. Each world of a study takes its own. The plan and the
result do not depend on the order of the inventory's assets.

Composed screen
---------------

``ComposedScreen`` asks the Hazus earthquake inference chain whether it can
take an asset before a structure type is drawn for it. The chain has two
stages: the design level inferer reads ``YearBuilt`` and writes
``DesignLevel``, and the structure type inferer reads both, with the
occupancy class, the story count and the building type. The stage names
are ``STAGE_DESIGN_LEVEL`` and ``STAGE_STRUCTURE_TYPE``, constants in
``resolution_loop.py`` beside this page. This section states how the screen
runs and how the category of a refused asset is settled.

The procedure
~~~~~~~~~~~~~

``screen_inventory`` passes the inventory to the design level inferer's
``check``. An asset that stage refuses takes its refusal and goes no
further. The other assets are copied, under their own ids and at
``SCREEN_COORDINATES``, since neither stage reads a location, and where a
copy holds a year and no design level, the design level inferer's ``draw``
writes the level the year implies on the copy. The structure type
inferer's ``check`` then runs over the copies, and an asset it refuses
takes that refusal. Each stage's ``check`` runs once per call. ``screen``
runs the same procedure on a one-asset inventory, under
``SCREEN_ASSET_ID``.

A refusal names the features whose change alone would let its stage produce
a value, the ``resolvable_by`` of the stage's diagnosis, and the screen
returns them as the verdict's ``candidates``, in the order the stage named
them. Neither inferer of the chain names the design level. A stage that did
would have it returned as the year, from which the level is derived, and
named once.

The exclusion category
~~~~~~~~~~~~~~~~~~~~~~

A refused asset that the caller excludes is filed under one of the categories
of ``EXCLUSION_CATEGORIES`` in ``compilation_report.py``, whose comment defines
each. An unsupported refusal is ``unsupported`` and an invalid one
``missing_driver``. An infeasible refusal is ``conflict`` where the caller
acted on a feature the refusal named and the asset was refused again. The
resolution loop below is such a caller, and with the structure type inferer in
BRAILS it does not reach this case. Otherwise an infeasible refusal at the
structure type stage is ``empty_cell`` where the asset's cell has neither a
printed nor a filled row, which the structure type inferer's ``cell_fill`` and
``cell_is_unpublished`` tell, and ``conflict`` where it has one, since the
asset's own features then emptied the row. An infeasible refusal at the design
level stage, which the design level inferer does not give, is ``undetermined``.

Resolution loop
---------------

``ResolutionLoop`` admits each asset of one world's merged inventory to the
structure type draw, or excludes it from that world, and its one action is
to withhold the general building type, ``BuildingType``, that NSI holds.
The source it acts on and the stage it records under are
``WITHHELD_SOURCE`` and ``STAGE_RESOLUTION``, constants in
``resolution_loop.py``. This section states how the loop runs and what it
records.

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

A refused asset the loop did not act on is excluded at the stage that
refused it, in the category of the section "The exclusion category". An
asset still refused once its type is withheld, the case of the section
"Scope" below, has the type recorded again with the action ``restored``,
and its exclusion is recorded at ``STAGE_RESOLUTION``, as ``conflict``
where the refusal was infeasible, with a detail holding both refusals.
Every exclusion keeps the world and the first refusal's diagnosis and
candidates, and every ``withheld`` and ``restored`` record the world, the
source and the asset's block group, read from ``block_group_key``
(``BLOCK_GROUP_KEY`` by default).

Scope
~~~~~

The structure type inferer in BRAILS names ``BuildingType`` exactly when the
asset passes without it, so with that inferer a withheld type always admits
the asset and the ``restored`` path is not reached. The path serves a
structure type inferer for which the building type is a required driver,
which BRAILS does not hold. A building type the inferer cannot read refuses
the asset as invalid and names no feature, so it is not withheld, and the
asset is filed as ``missing_driver``.

Composition and assembly
------------------------

``compose_source`` builds one source of one world from stages frozen in a
``StageFreeze``, and ``WorldAssembler`` assembles one-world inventories into
one inventory over the worlds. Both are in ``world_assembly.py`` beside this
page. This section states how they run.

Composition
~~~~~~~~~~~

``compose_source`` checks its arguments and reads every stage from the
freeze by name. A stage with one possible world is read as it is held, and
a stage with several through its realization ``world``.

The parts are then united in the order given. An asset held by any part is
in the result, and an asset the ``dropped`` record holds, in any world, is
left out. An asset in ``excluded`` takes no value from the parts named in
``excluded_parts`` and keeps the values of the others. Each feature takes
the value of the one part that holds a value for it, a value being anything
``AssetInventory.merge`` does not count as missing. A feature no part holds
a value for keeps a missing value, as ``merge`` keeps a source's marker,
where ``union_disjoint`` in ``brails/utils/inventory_union.py`` drops such a
feature. Two parts holding a value for one feature of one asset, and an id
in ``excluded`` that no part and no ``dropped`` record holds, are refused
once the union is built. Last, each feature of an asset whose value in a
``withheld`` record, in ``world``, is the source label ``label`` is removed.

The result shares a stage's asset record wherever no step changes the asset
and the stage has one possible world or holds no list for the asset.
Elsewhere it copies the record and its feature dict, but not the values or
the coordinates.

Assembly
~~~~~~~~

``WorldAssembler.add`` copies the realization at position 0 asset by asset,
the record and its feature dict but not the values. At a later position it
first checks the asset ids, the coordinates and the vector features against
those held, and then compares each feature of each asset. A feature stays
one value while every position so far agrees. When a position first
differs, the feature becomes a list, the earlier value repeated for the
earlier positions, and later positions append to it. A missing marker such
as ``'NA'`` is a value like any other.

Values are compared with ``==``. Two float NaN are equal, two arrays are
equal where ``numpy.array_equal`` finds them so, and values ``==`` cannot
compare are not equal. A list is one value, compared whole. A vector
feature, declared by any realization, whose value at position 0 is not a
list cannot take another value later, since the result could not tell its
list of positions from one vector value. The rule differs from
``collapse_realizations`` in ``brails/utils/world_values.py``, which keeps a
list where every world holds ``None`` and compares with ``==`` alone.

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

Step 1 takes the footprints as given, keyed by the caller's asset ids. Each
receives its census block group from the block-group polygons, its plan area in
square feet as ``PlanArea`` and, if it is an Overture footprint, the release
and the datasets of its sources, which with the caller's ``footprint_provider``
make up ``FOOTPRINT_PROVENANCE_FEATURES``. Steps 2 and 3 join the NSI points to
the footprints with ``NSIFootprintJoin`` and derive their unit counts and
occupancy classes: the join's observed values form the NSI source, and the
values it derives, such as an occupancy class set by the count of units, the
derived source.

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
the section "Donors" describes, and gives a target whose copy forms an
implausible pair of occupancy class and story count another donor. The
residential occupancy classes of this section are those of
``UNIT_COUNT_CLASSES`` (RES1, RES2 and the RES3 classes). A reached footprint
outside them is a target for its story count alone, since the join gives units
to housing records only.

Steps 5 and 6 run in each world, as one stage, on that world's realization of
the frozen view ``STAGE_STEP4``. Step 5 dates the residential buildings:
``ACSYearBuiltSampler``, fitted on that realization, draws the construction
year of each residential building without one from the census tables of its
block group and tract. Step 6 types the dated buildings. ``PairRedraw`` first
replaces the NSI members of implausible pairs of occupancy class and story
count, as the section "Pair re-draw" states, with the observed donors of that
realization. ``HazusEQDesignLevelInferer`` derives the design level of every
building with a year and none. Each building with a year and no structure type
then passes through ``ComposedScreen`` and ``ResolutionLoop``, as the sections
"Composed screen" and "Resolution loop" state, and
``HazusEQStructureTypeInferer`` draws the structure type of each building the
loop admits. ``HazusEQHeightClassInferer`` gives every building that holds a
structure type, drawn or its own, its height class. A building one of these
components refuses is excluded from that world. The types are drawn from the
completed West Coast tables, so the compiler takes states of that region only.
In ``CARVE_OUT_STATE`` (California) under ``CARVE_OUT_METHOD`` (5.1), a drawn
W1 building whose derived level is Pre-Code is raised to Moderate-Code, as
Table C-10 of the Hazus manual assumes. A non-residential building without a
year is excluded, since no step of the run dates one.

The output step merges each world's sources in the precedence of
``PRECEDENCE``: local values first, then the footprint, NSI, derived, imputed
and drawn values. ``AssetInventory.merge`` writes the source string, as the
section "The source string" describes. NSI's gross area, the ``GrossArea``
feature the join writes, is kept as ``NSIGrossArea``, which the compiler writes
on the join's values, and a ``GrossArea`` that a local inventory gives as
``LocalGrossArea``. Every building not yet excluded from the world is checked
against the Hazus 5.1 input schema of the damage and loss model library,
``DLML_DATASET``, and a building it refuses is excluded from that world. Each
building receives ``Excluded``, true in a world from which the report lists it
as excluded. Its ``BuildingType`` is the letter a local inventory gives it,
where one does, and otherwise the letter of its structure type in that world,
by the pairing of ``get_hazus_general_type_membership`` in
``brails/inferers/hazus_inferer_earthquake/hazus_rulesets.py`` that the
structure type inferer draws with. NSI's letter, which the draw reads, is not
written. Where a local letter differs from the letter of the building's
structure type, the file carries the local letter, and a notice in the report
counts such compiled buildings in each world. The worlds are then assembled
into one inventory.

Frozen stages
~~~~~~~~~~~~~

The run is a sequence of stages, each frozen when it ends, as the section
"Stages and freezing" describes. The stage names are the module's ``STAGE_``
constants. Three kinds of stage are frozen. A part holds the values one step
adds, such as the join's NSI values. A record holds what a step withholds from
a source or drops from every source. A view holds the merge of every part and
record frozen so far, over the worlds, and the last view, the output, is the
compiled inventory. Every part other than the footprints, and every record,
holds each footprint's centroid as its coordinates. The footprints and the
views hold the footprint polygons. A world's source is composed from its parts
and records by ``compose_source``, and the worlds of a view by
``WorldAssembler``, as the section "Composition and assembly" states. The run
ends with the manifest.

Worlds
~~~~~~

Worlds are numbered as the section "Possible worlds" describes, by the run and
by every component. A world does not depend on how many worlds the run
compiles, so world 3 of a run of four worlds equals world 3 of a run of ten. A
run asked for some worlds computes every world up to the largest asked, and the
result lists the worlds asked, in ascending order, as ``worlds``.

The written file
~~~~~~~~~~~~~~~~

``write`` writes the worlds a result lists, one GeoJSON feature per building
with its id and footprint and, for a property whose worlds differ, a list of
values, one per world written. Which buildings and properties the file holds
is stated on ``CompilationResult.write``. The properties of
``DEFAULT_PROPERTIES`` are those a Hazus earthquake assessment reads. The
report lists each excluded building with its reason in each world, whether
or not the file holds it.

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
refused.
