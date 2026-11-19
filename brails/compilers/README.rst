Inventory compilers
===================

A compiler of ``brails.compilers`` builds an inventory out of the data its
method needs, such as building footprints, the National Structure Inventory
and census tables. It runs a sequence of steps, and each step adds values to
the inventory. This page states what every compiler uses: the concepts the
compilers share with the imputers and inferers they call, the composition of
a source from the steps' outputs and the assembly of an inventory over
several possible worlds, and the report a compiler keeps of its run. The
Hazus earthquake inventory compiler and the steps it runs are described in
``brails/compilers/hazus_earthquake/README.rst``. The docstrings state what
a caller passes and receives.

Concepts
--------

Four ideas recur on this page and on the pages of the imputers and
inferers. An inventory can hold several possible worlds, one realization of
its uncertain features in each. Each step's output, and what the compiler
builds between steps, is frozen as a stage, which later steps read by name.
A stage is a part, a record, a view or a computed stage. A merge of the
inventories of several sources records the source of each value in a source
string. An imputer can fill a missing value from another building, its
donor. Each of the four subsections below explains one of these ideas.

Possible worlds
~~~~~~~~~~~~~~~

Some features of a building are not observed and are drawn at random
instead, such as a construction year sampled around a tract median. An
inventory can hold several possible worlds, each one realization of the
features whose values are uncertain. ``AssetInventory`` records their number
as ``n_pw``, which is 1 for an inventory of one world. A feature holds
either one value, shared by every world, or a list of exactly ``n_pw``
values, one per world in world order, and ``validate_possible_worlds``
checks that rule. A vector feature is one that an inventory declares in its
``vector_features``, by default those of ``DEFAULT_VECTOR_FEATURES`` such as
``HousingUnits``. Its value in one world may itself be a list, so it holds
one value per world only as a list of lists. ``collapse_realizations`` in
``brails/utils/world_values.py`` stores a feature's per-world values by the
same rule: one value when every world holds the same value other than
``None``, and the list otherwise.

Worlds are numbered from 0, and this number is the world index.
``get_world_realization`` returns the realization of one world: the
inventory of that world alone, with every feature reduced to its value in
that world. A component that draws values separately in each world is told
which world it draws for. Its random numbers come from ``keyed_uniforms`` in
``brails/utils/deterministic_sampling.py``, which keys each draw by the
asset, the seed, the world, a purpose label and an iteration. A world's
draws can therefore be repeated, and they do not depend on the order of the
assets. Some components compute every world of an inventory. Others work on
only the one world they are given, and each world then takes its own
instance of the component. Such instances produce the worlds of one run only
when they share the run's seed, so ``require_run_seed`` refuses a sampler
for a world other than 0 that is given no seed.

Stages and freezing
~~~~~~~~~~~~~~~~~~~

A compiler builds its inventory in steps, and the terms below name what the
steps hand to one another. Every stage a compiler freezes is a part, a
record, a view or a computed stage.

A **stage** is an inventory frozen under a name, such as ``s02_nsi_join``, by
one step or by the compiler between steps. Once frozen, a stage does not
change, and every later step that needs it reads it by that name.

A **source** is a provider of values, named by a label such as ``local``,
``nsi``, ``imputed`` or ``drawn``. The labels and their one-letter codes are
those of ``SOURCE_CODES`` (see "The source string" below). A building's
values in one world are the merge of the values of its sources in that world.

A **part** is a stage that holds the values one step adds to one source. A
source can have several parts: the ``imputed`` source of the Hazus
earthquake compiler, for example, gathers the values of four steps.

A **record** is a stage that acts on the sources instead of adding values to
them. A record of withheld values names, for each asset and feature, the
source from which the value is withheld, so that later merges leave that
value out. A record of dropped footprints holds the footprints that a step
removed from every source and every world.

A **view** is a stage that holds the merged inventory of every world, the
merge of every part and record frozen so far. A step that needs the inventory
as it stands reads a view. The last view of a compiler is its compiled
inventory, into which the last steps write their values before it is
frozen.

A **computed stage** is a stage that holds values a step computed outside the
sources, so that no merge takes them. A later step may read it directly, as the
replacement cost step of the Hazus earthquake compiler reads a table of gross
areas. Otherwise no step reads it, and it keeps a step's values for the record.
A step that needs such values in a merged inventory writes them in itself.

``StageFreeze``, in ``brails/utils/stage_freeze.py``, holds the frozen stages
under their names and records the order in which they were frozen. This page
calls a ``StageFreeze`` a freeze. A stage is frozen once, and freezing it
records a digest, a hash of its content. The freeze holds the stage itself,
not a copy. A step must therefore not change a frozen stage, and a step that
needs to change one copies it first. ``manifest`` recomputes every digest and
raises ``RuntimeError`` if a stage no longer matches the digest recorded when
it was frozen. A manifest returned without that error therefore also shows
that no step changed a stage after it was frozen. Frozen stages are held in
memory only.

The imputers whose ``impute_stage`` reads frozen stages take the freeze and the
names of the stages they read. ``ObservedDonorImputer.impute_stage`` takes its
observed donors (see "Donors" below) as an inventory or, given a freeze, as a
frozen stage. In the second form it fills its targets in batches and freezes
each batch, and each batch draws its donors from the observed donors and from
the batches frozen before it. ``ReplacementCostImputer.impute_stage`` computes
the replacement costs of the buildings of one frozen stage with the donors held
in another, and it changes neither stage.

The source string
~~~~~~~~~~~~~~~~~

``AssetInventory.merge`` combines inventories of the same assets, one per
source, each under a label such as ``'local'`` or ``'nsi'``. The sources are
taken in a stated order of precedence, and each feature of each asset takes
its value from the first source that holds one. The merge then records where
each value came from in the feature ``SOURCE_STRING_KEY``, ``SourceString``.
The string holds one character for each feature of ``source_features``, by
default ``HAZUS_EQ_SOURCE_FEATURES``: ``OccupancyClass``, ``NumberOfUnits``,
``NumberOfStories``, ``YearBuilt``, ``DesignLevel``, ``StructureType``,
``HeightClass`` and ``ReplacementCost``, in that order. Each character is
the code that ``SOURCE_CODES`` gives the label of the value's source: ``L``
for local, ``F`` for footprint, ``N`` for NSI, ``R`` for derived, ``I`` for
imputed and ``D`` for drawn. Where no source holds a value, the character is
``SOURCE_CODE_NONE``, ``-``.

A merge records which source holds a value, never the value, and writes one
string per asset. A later step reads a value's code to decide what it may
change, as the pair re-draw of the Hazus earthquake compiler does. A merge
refuses a source that already carries a source string, so the string
describes one merge.

The Hazus earthquake compiler merges the sources of each world separately
and adds each world's merged inventory to one inventory with
``AssetInventory.add_world``. Its views and its output are inventories built
that way. A building whose sources differ between worlds then holds a list
of strings, one per world. ``ReplacementCostImputer.impute`` writes the code
of the value it gives in each world, so a building that receives a
replacement cost in one world and not in another also holds a list.
``read_source_string`` in ``brails/types/asset_inventory.py`` reads the
string of one world and refuses such a list, so a step reads a building's
codes from the realization of one world. ``SOURCE_CODES``,
``SOURCE_CODE_NONE`` and ``HAZUS_EQ_SOURCE_FEATURES`` are defined in
``brails/types/asset_inventory.py``, and ``SOURCE_STRING_KEY`` in
``brails/constants.py``.

Donors
~~~~~~

A donor is a building whose values, or a quantity computed from them, fill the
missing values of another, the target. ``ObservedDonorImputer``, in
``brails/imputers/observed_donor_imputer/``, copies from observed donors,
buildings whose values were observed. For each target and world, it ranks
the donors near the target that match it and picks one. From that one donor
it copies each of the features it fills that the target lacks. By default
it fills the occupancy class, the unit count and the story count.
Its README, ``brails/imputers/observed_donor_imputer/README.rst``, states the
ranking, the pick and the fallbacks in its sections "The distance and the
nearest donors", "The pick" and "When no donor lies within the cap".
``impute_stage`` takes the observed donors as an inventory, or reads them
from a freeze as the stage its ``donor_stage`` argument names.

The imputer's log holds one row for each draw, with the target, the world,
the round, the donor's id and the values copied. The round is
``ROUND_STATUS`` for a first draw that gives a target of unknown residential
status its status, ``ROUND_DONOR`` for the draw of the values and
``ROUND_REPAIR`` for the repair draw of a target whose copy was rejected.
``imputation_donors_from_log`` reads that log and builds an inventory with
one asset per target. Its feature ``IMPUTATION_DONOR_KEY``,
``ImputationDonor``, holds, per world, the id of the donor whose values the
target kept, and ``None`` in a world where the target kept none. This
inventory is the donor stage that ``ReplacementCostImputer.impute_stage``
reads as its ``donor_stage``, under the name the caller gives it (``impute``
uses ``DONOR_STAGE``). A target can thus take the rate per square foot of
its donor. The two ``donor_stage`` arguments name different stages: the
observed donors for the observed-donor imputer, and the donors taken from
the observed-donor imputer's log for the replacement cost imputer.

Composition and assembly
------------------------

A compiler's steps freeze their values in parts, and a later step needs the
inventory those parts make up in one world, or over every world. Two
functions build it. ``compose_source_parts``, in ``source_composition.py``
beside this page, puts one source of one world together from its parts and
from the records that act on it. The compiler then merges the sources of
the world with ``AssetInventory.merge`` (see "The source string" above).
``AssetInventory.add_world``, in ``brails/types/asset_inventory.py``, adds
the merged world to an inventory that holds the worlds computed so far. This
section states how the two functions work.

Composition
~~~~~~~~~~~

``compose_source_parts`` builds the inventory of one source in one world,
for example everything the ``imputed`` source holds in world 3. Its
arguments say which source and world, and which frozen stages to read:

- ``freeze`` is the ``StageFreeze`` that holds the stages. Every stage is
  read from it by name, so the function sees each stage as it was frozen.
- ``parts`` names the stages that hold the source's values, each once, such
  as the parts of the ``imputed`` source.
- ``world`` is the number of the world to compose. A stage that holds
  several worlds is read through its realization of that world. A stage of
  one world is read as it is, since its values hold in every world. A part or
  a record of withheld values that holds several worlds but has no such world
  is refused. The record of dropped footprints is read whole, whatever
  worlds it holds.
- ``label`` is the label of the source being composed, such as
  ``'imputed'``. The function uses it only to find, in the records of
  withheld values, the values withheld from this source.
- ``withheld`` names the records of withheld values. Each holds, for an
  asset and a feature, the label of the source a value is withheld from.
  Where a record holds this source's label for a feature of an asset in the
  world, the result has no value of that feature for that asset, whichever
  part gave it. A record that names another source leaves the value alone.
  Without ``withheld``, nothing is withheld.
- ``dropped`` names the record of dropped footprints. An asset it holds, in
  any world, is left out of the result altogether. Without ``dropped``, no
  asset is left out.

The function first checks these arguments and reads every stage. It then
takes the parts in the order given. Every asset that any part holds is in
the result, at the coordinates of the first part that holds it. Each of its
features takes the value of the one part that holds a value for it. Here,
as in ``AssetInventory.merge``, a value counts as missing where it is
``None``, NaN or another value ``pandas.isna`` reports as missing, the empty
string or the missing marker ``MISSING_VALUE_MARKER``. Anything else counts
as a value. Where no part holds a value for a feature, the feature
keeps the missing value of the first part that has the feature, much as
``merge`` keeps a source's marker. Two parts that both hold a value for one
feature of one asset in the world are an error, since each value of a
source comes from one step. Once every part has been read, the function
raises ``ValueError`` naming the assets and features. Last, the values that
the withheld records name for this source are removed.

The result shares what it can with the frozen stages. Where no later part and
no withheld record changes an asset's features, and the first part that holds
the asset holds one world or holds no list for it, the result holds that part's
own asset, not a copy. Elsewhere it holds a copy of the asset with a feature
dictionary of its own, and the copy still shares the values and the coordinates
with the stages. The result must therefore not be changed. A change to a shared
asset would show as a changed stage in the freeze's manifest.

Assembly
~~~~~~~~

``AssetInventory.add_world`` builds an inventory over several worlds, one
world at a time. It is called on the inventory being built and given a
realization, an inventory of one world, which becomes the next world. A
feature on which every world agrees keeps one value, and a feature whose
worlds differ holds a list of values, one per world.

Called on an inventory without assets, ``add_world`` takes the realization
as world 0. It copies each ``Asset`` of the realization, in ascending order
of the asset id as a string. Each copy has a feature dictionary of its own,
but the values in it are not copied. Each later realization is the next
world, and ``n_pw`` counts the worlds added. A later realization must hold
the same asset ids at the same coordinates as world 0. Its vector features
are checked as stated below, and then each feature of each asset is
compared with the inventory's. A feature stays one value while every world
so far agrees. When a world first differs, the feature becomes a list, with
the earlier value repeated for the earlier worlds, and each later world
appends its value to the list. A feature that a realization lacks counts as
``None`` in that world, and the missing marker ``MISSING_VALUE_MARKER`` is a
value like any other. A realization that ``add_world`` refuses leaves the
inventory unchanged. Realizations without assets leave the inventory without
assets and at one world. The inventory shares coordinates and feature values
with the realizations, so a caller must not change a realization after
adding it.

Values are compared with ``==``. Two float NaN are equal, two arrays are
equal where ``numpy.array_equal`` finds them so, and values that ``==``
cannot compare are not equal. In a vector feature a list is one value,
compared whole. A realization may hold a list only as the value of a vector
feature, because the inventory holds one value per world as a list. For
world 0 the vector features are those the realization declares, and for a
later world those that the inventory or the realization declares.
``add_world`` refuses a realization that holds a list in any other feature,
or a list of lists in a vector feature.
A vector feature such as ``HousingUnits`` holds one value per world only as
a list of lists, and ``add_world`` recognizes a list of lists by its first
item. Where a vector feature's value in world 0 is not a list, for example
``None``, a later world cannot give it a different value, and ``add_world``
raises ``ValueError``. This rule holds whether the inventory or the
realization declares the feature a vector. ``add_world`` differs here from
``collapse_realizations`` in ``brails/utils/world_values.py``, which keeps a
list where every world holds ``None`` and compares values with ``==`` alone.

The compilation report
----------------------

A compiler records what it did in a ``CompilationReport``, in
``compilation_report.py`` beside this page. The report holds a row for each
change a step made to a feature and a row for each building a step excluded
from a world. A value or a missing building of the compiled inventory can
thus be traced to the step responsible. A step excludes a building from a
world when it cannot complete the building there or finds it inconsistent.
How the compiled inventory shows an exclusion is each compiler's own: the
Hazus earthquake compiler keeps the building and sets its ``Excluded`` true
in that world. This section states what the rows
hold, the categories exclusions are filed under, and what the summary
counts.

Rows
~~~~

Every row carries the asset id, the world number and a report stage: the name
of the step, or of the operation of a step, that wrote it. A report stage is a
name that labels rows, not a frozen stage. The report stages are the names the
compiler gives the report, ``REPORT_STAGES`` in the Hazus earthquake compiler,
and the report keeps them in its ``stages`` attribute. Many of them are also
the names of frozen stages, such as ``s04_unit_budget``. A step of several
operations has one report stage for each operation, such as
``s08_design_level`` and ``s08_structure_type``, and no stage is frozen under
those names.

An action row records what one step did to one feature of one asset: the
report stage, the action, the feature, the values before and after, the
source the action concerns, and the reason in words. The action is one of
``ACTIONS``:

- ``withheld``, a value kept out of a draw or removed from its source, such as
  an NSI story count that the footprint height replaces;
- ``restored``, a withheld value put back after a failed attempt;
- ``redrawn``, a value that a new draw replaced, such as a member of an
  implausible pair of occupancy class and story count;
- ``derived``, a value computed from other features of the asset, such as a
  design level that the W1 carve-out of the Hazus earthquake compiler
  changed.

The values are recorded as text, and ``None`` as no value. The source is a
label of ``SOURCE_CODES``, as in the source string. The report holds the
action rows as its ``records``. These rows are not records in the sense of
"Stages and freezing".

An exclusion row records that a step excluded an asset from one world: the
report stage, the category, the diagnosis of the component that refused the
asset, a detail in words, and ``resolvable_by``, the features whose change
alone would let that component accept the asset. An asset excluded from
several worlds has one row in each.

A report accepts the report stages and categories its compiler gives it when
the report is created, and refuses a row that names any other. The Hazus
earthquake compiler gives ``REPORT_STAGES`` and ``EXCLUSION_CATEGORIES`` of
``brails/compilers/hazus_earthquake/hazus_eq_inventory_compiler.py``.
``summary``, ``save`` and ``exclusion_groups`` take a selection of worlds
and read the rows of those worlds only. Without a selection, ``summary`` and
``save`` read every row, and ``exclusion_groups`` reads worlds 0 to
``count - 1``, for the count given to ``set_world_count``.

Exclusion categories
~~~~~~~~~~~~~~~~~~~~

A category says why a step excluded a building, in terms that do not depend
on the step. Each category below has a general definition, followed by
examples from the Hazus earthquake compiler.

``missing_input``
    A value the step reads is absent, or holds a value the step cannot use,
    and no earlier step supplied a usable one. Examples: a residential
    building without a usable unit count, or in a block group the census
    tables of the year sampler do not cover; a building whose construction
    year is copied from residential neighbors and that has no occupancy
    class, or no residential neighbor with a year; a construction year that
    is not a number; a local structure type that is not a Hazus structure
    type, on a building without a height class; and a local building type
    letter that names no Hazus general building type.

``unsupported``
    A value the step reads is present in a form the step does not handle.
    Example: a feature held as a list of values, one per world, read by an
    inferer that draws in one world. The Hazus earthquake compiler gives its
    inferers the values of one world, so it files no building in this
    category.

``no_plausible_value``
    The step's model has no value for the building's combination of inputs,
    whatever else the building holds. Examples: a building whose occupancy
    class, story band and, in the West Coast tables, era select no
    distribution in the structure type table, and a building for which the
    replacement cost imputer finds no donor.

``conflict``
    The model has values for the building's combination of inputs, and other
    values the building holds rule out all of them. A building is also filed
    here when its first refusal found its values infeasible, the compiler
    set one of its values aside to admit it, and the building was refused
    again. Examples: a local building type letter S on a single-family
    house, for which the structure type table holds no steel type, and a
    local building type letter that is not the letter of the building's
    structure type, such as W beside C1.

``undetermined``
    The step refused the building, and the reason fits none of the other
    categories. Example: in the composed screen of the Hazus earthquake
    compiler, the design level inferer refuses a building as infeasible,
    meaning that no design level fits its other values. The Hazus
    earthquake compiler files the same refusal as ``missing_input`` where
    its design level operation, which runs outside the screen, receives it.
    The design level inferer in BRAILS never refuses a building this way, so
    with BRAILS's own inferers this category stays empty.

``fails_schema``
    The compiled building fails the check against the schema of the output.
    Example: a building that the input schema ``DLML_DATASET`` of the damage
    and loss model library refuses, such as an unreinforced masonry building
    with a design level the schema does not accept for that structure type.

``dropped``
    A step's rule removes the footprint from every world. Example: a small
    footprint to which the unit budget gives no housing units. Every report
    accepts this category, ``DROPPED_CATEGORY``, besides those of its
    compiler. The report counts these footprints in a group of their own,
    apart from both the buildings excluded in every world and those
    excluded in some.

The summary
~~~~~~~~~~~

``summary`` returns a dictionary, which ``save`` writes as JSON beside the
rows. Its keys are these:

- ``seed`` and ``settings``, as the report was given them.
- ``worlds``: ``computed``, the count given to ``set_world_count`` or
  ``None``, and ``written``, the selected worlds in ascending order, so that
  position ``i`` of a file holding them is world ``written[i]``. Without a
  selection, ``written`` lists every computed world, and it is ``None`` when
  the count is not known.
- ``manifest_stages``: the stages of the run's manifest, in its order.
- ``timings``: one entry of ``stage``, ``world`` and ``seconds`` for each
  timed portion of the run and each world, in the order first recorded.
- ``actions``: the action rows counted ``by_action`` and ``by_feature``.
- ``exclusions``: the exclusion rows counted ``by_category``,
  ``by_diagnosis`` and ``by_world`` (by world, then by category), and their
  ``total``.
- ``assets_excluded``: the number of distinct excluded assets as ``total``,
  and the number in each group of ``exclusion_groups`` as ``dropped``,
  ``every`` and ``some``. The last two are ``None`` when the worlds are not
  known.
- ``notices``: the notices the compiler recorded about the run, such as a
  count of missing default values.
- ``coverage``: the fraction of assets that state each input, as the
  compiler recorded it with ``set_coverage``. The Hazus earthquake compiler
  records none, so its ``coverage`` is empty.
- ``schema_check``: the result of the schema check of the compiled
  buildings, as the compiler recorded it.

Counts are keyed in sorted order. ``save`` writes a setting or a schema
check result that JSON cannot hold as its ``str`` text, and JSON writes the
world keys of ``by_world`` as text.
