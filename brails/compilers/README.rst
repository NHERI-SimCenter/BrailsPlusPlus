Inventory compilers
===================

The classes of ``brails.compilers`` build an inventory out of the sources a
method needs. This page describes the procedures they run, where their
constants come from and the scope of their rules, in a section for each
class that needs one. The docstrings state what a caller passes and
receives.

Pair re-draw
------------

``PairRedraw`` reads a merged inventory and finds the assets whose occupancy
class and story count form an implausible pair, such as a ten-story
agricultural building. For each, it decides from the sources of the two
values which one gives way, and it replaces or re-draws that value. An
inventory can hold several possible worlds, each one realization of the
features whose values are uncertain, and the re-draw takes an inventory of
one world. The names of its actions and the columns of its plan and log are
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

``AssetInventory.merge`` records where each merged value came from in the
asset's source string, the feature ``SourceString``. It holds one character
for each feature of ``HAZUS_EQ_SOURCE_FEATURES``: the code ``SOURCE_CODES``
gives the value's source, or ``SOURCE_CODE_NONE``, ``-``, where no source
holds one. The codes of ``SOURCE_CODES`` are ``L`` for local, ``F`` for
footprint, ``N`` for NSI, ``R`` for derived, ``I`` for imputed and ``D`` for
drawn.

The re-draw reads the codes of the class and the story count. It never
changes a local or footprint value, the sources of ``FIXED_SOURCES``, and a
pair with a local or footprint class is only logged, whatever the source of
its count. A derived story count is the one ``StoriesFromHeightInferer``
gives from the footprint height, so it is kept like a local one. An imputed
value is left to the validation that ``ObservedDonorImputer.impute_stage``
runs on the values it copies when given ``validate``, which can re-draw a
target it rejects, so a pair with an imputed member is only logged. That
leaves the pairs with an NSI class. Of these, an NSI story count gives way
where the footprint height gives a count in another story band, and the
class is then re-drawn only if the pair stays implausible at that count.
Where the height gives no count, the class and the NSI count are re-drawn
together. In every other case the class gives way.

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
