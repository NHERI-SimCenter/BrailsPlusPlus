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
string per asset. The Hazus earthquake compiler merges each world separately
and assembles the
merges into one inventory with ``WorldAssembler``, for its views and its
output. A building whose sources differ between worlds then holds a list of
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

