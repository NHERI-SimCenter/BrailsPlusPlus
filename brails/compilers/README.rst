Inventory compilers
===================

The classes and functions of ``brails.compilers`` build an inventory out of the
sources a method needs. This page first explains the concepts they share with
the imputers and inferers they call. It then describes the procedures they run,
where their constants come from and the scope of their rules, in a section for
each one that needs it. The docstrings state what a caller passes and receives.
The Hazus earthquake inventory compiler and the steps it runs are described in
``brails/compilers/hazus_earthquake/README.rst``.

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

A merge records which source holds a value, never the value, and writes one
string per asset. The Hazus earthquake compiler merges the sources of each
world separately and adds each world's merged inventory to one inventory with
``AssetInventory.add_world``. Its views and its output are inventories built
that way. A building whose sources differ between worlds then holds a list of
strings, one per world. ``ReplacementCostImputer.impute`` writes the code of
the value it gives in each world, so a building valued in one world and not in
another also holds a list. ``read_source_string`` in
``brails/types/asset_inventory.py`` reads the string of one world and refuses
such a list, so a step reads a building's codes from the realization of one
world. A merge refuses a source that already carries a source string, so the
string describes one merge. ``SOURCE_CODES``, ``SOURCE_CODE_NONE`` and
``HAZUS_EQ_SOURCE_FEATURES`` are defined in
``brails/types/asset_inventory.py``, and ``SOURCE_STRING_KEY`` in
``brails/constants.py``. A later step reads a value's code to decide what it
may change, as the pair re-draw of the Hazus earthquake compiler does.

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

Composition and assembly
------------------------

``compose_source_parts`` builds one source of one world from stages frozen in a
``StageFreeze``, and ``AssetInventory.add_world`` adds one-world inventories,
one world at a time, to an inventory over the worlds. ``compose_source_parts``
is in ``world_assembly.py`` beside this page, and ``add_world`` in
``brails/types/asset_inventory.py``. This section states how they run.

Composition
~~~~~~~~~~~

``compose_source_parts`` checks its arguments and reads every stage from the
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

``add_world`` takes an inventory of one world, a realization, and adds it to
the inventory it is called on. An inventory without assets takes the
realization as its world 0, asset by asset, with a copy of each ``Asset`` and
of its feature dict, but not of the values. Each later realization is the next
world, and ``n_pw`` counts the worlds added. ``add_world`` first checks the
realization's asset ids, coordinates and vector features against those held,
and then compares each feature of each asset. A feature stays one value while
every world so far agrees. When a world first differs, the feature becomes a
list, the earlier value repeated for the earlier worlds, and later worlds
append to it. A key a realization lacks counts as ``None`` in that world, and
a missing marker such as ``'NA'`` is a value like any other.

Values are compared with ``==``. Two float NaN are equal, two arrays are
equal where ``numpy.array_equal`` finds them so, and values ``==`` cannot
compare are not equal. In a vector feature a list is one value, compared
whole. A realization may hold a list only as the value of a vector feature,
because the inventory holds one value per world as a list, and ``add_world``
refuses a realization that holds one elsewhere. A vector feature such as
``HousingUnits`` holds one value per world only as a list of lists, which is
recognized by its first item. Where a vector feature's value in world 0 is not
a list, for example ``None``, a later world cannot give it a different value,
and ``add_world`` raises ``ValueError``. This holds whether the inventory or
the realization declares the feature a vector. The rule differs from
``collapse_realizations`` in ``brails/utils/world_values.py``, which keeps a
list where every world holds ``None`` and compares with ``==`` alone.

The compilation report
----------------------

A compiler records what it did in a ``CompilationReport``, in
``compilation_report.py`` beside this page. The report holds a row for each
change a step made to a feature and a row for each building a step excluded
from a world, so that a value or a missing building of the compiled inventory
can be traced to the step responsible. This section states what the rows hold,
the categories exclusions are filed under, and what the summary counts.

Rows
~~~~

Every row carries the asset id and the world number, and the name of the step
of the compiler that wrote it. These names are those the compiler gives the
report, ``REPORT_STAGES`` in the Hazus earthquake compiler. They name steps of
the run, not the frozen stages of "Stages and freezing", and the report keeps
them in its ``stages`` attribute.

An action row records what one step did to one feature of one asset: the
step's name, the action, the feature, the values before and after, the source
the action concerns, and the reason in words. The action is one of
``ACTIONS``:

- ``withheld``, a value kept out of a draw or removed from its source, such as
  an NSI story count that the footprint height replaces;
- ``restored``, a withheld value put back after a failed attempt;
- ``redrawn``, a value that a new draw replaced, such as a member of an
  implausible pair of occupancy class and story count;
- ``derived``, a value computed from other features of the asset, such as a
  design level the W1 carve-out changed.

The values are recorded as text, and ``None`` as no value. The source is a
label of ``SOURCE_CODES``, as in the source string.

An exclusion row records that a step excluded an asset from one world: the
step's name, the category, the diagnosis of the component that refused the
asset, a detail in words, and ``resolvable_by``, the features whose change
alone would let that component accept the asset. An asset excluded from
several worlds has one row in each.

A report accepts the step names and categories its compiler gives it when the
report is created, and refuses a row that names any other. The Hazus
earthquake compiler gives ``REPORT_STAGES`` and ``EXCLUSION_CATEGORIES`` of
``brails/compilers/hazus_earthquake/hazus_eq_inventory_compiler.py``.
``summary``, ``save`` and ``exclusion_groups`` take a selection of worlds and
read the rows of those worlds only. Without one, ``summary`` and ``save`` read
every row, and ``exclusion_groups`` reads worlds 0 to ``count - 1``, for the
count given to ``set_world_count``.

Exclusion categories
~~~~~~~~~~~~~~~~~~~~

A category says why a step excluded a building, in terms that do not depend on
the step. Each category below has a general definition, followed by examples
from the Hazus earthquake compiler.

``missing_input``
    A value the step reads is absent, or holds a value the step cannot use,
    and no earlier step supplied a usable one. Examples: a building whose block
    group has no ACS counts and a building whose local building type letter is
    not W, S, C, M or MH.

``unsupported``
    A value the step reads is present in a form the step does not handle.
    Example: a feature held as one value per world, where the step reads the
    value of one world.

``no_plausible_value``
    The step's model has no value for the building's combination of inputs,
    whatever else the building holds. Examples: a building whose occupancy
    class, story band and, in the West Coast tables, era select no
    distribution in the structure type table, and a building for which the
    replacement cost imputer finds no donor.

``conflict``
    The model has values for the building's combination of inputs, and other
    values the building holds rule out all of them. A building is also filed
    here when the compiler set one of its values aside to admit it and the
    building was refused again. Example: a local building type letter S on a
    single-family house, for which the structure type table holds no steel
    type.

``undetermined``
    The step refused the building, and the reason fits none of the other
    categories. Example: in the composed screen, the design level inferer
    refuses a building as infeasible, meaning that no design level fits its
    other values. The design level inferer in BRAILS never refuses a building
    this way, so with BRAILS's own inferers this category stays empty.

``fails_schema``
    The compiled building fails the check against the schema of the output.
    Example: a building that the Hazus 5.1 input schema of the damage and loss
    model library refuses.

``dropped``
    A step's rule removes the footprint from every world. Example: a small
    footprint to which the unit budget gives no housing units. Every report
    accepts this category, ``DROPPED_CATEGORY``, besides those of its
    compiler, and counts these footprints in a group of their own, apart from
    both the buildings excluded in every world and those excluded in some.

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
  timed part of the run and each world, in the order first recorded.
- ``actions``: the action rows counted ``by_action`` and ``by_feature``.
- ``exclusions``: the exclusion rows counted ``by_category``,
  ``by_diagnosis`` and ``by_world`` (by world, then by category), and their
  ``total``.
- ``assets_excluded``: the number of distinct excluded assets as ``total``,
  and the number in each group of ``exclusion_groups`` as ``dropped``,
  ``every`` and ``some``. The last two are ``None`` when the worlds are not
  known.
- ``notices``: the notices the compiler recorded about the run, such as a
  contradiction it met.
- ``coverage``: for each input the compiler reads, the fraction of assets
  that state it.
- ``certificate``: the result of the schema check of the compiled buildings,
  as the compiler recorded it.

Counts are keyed in sorted order. ``save`` writes a setting or a schema check
result that JSON cannot hold as its ``str`` text, and JSON writes the world
keys of ``by_world`` as text.
