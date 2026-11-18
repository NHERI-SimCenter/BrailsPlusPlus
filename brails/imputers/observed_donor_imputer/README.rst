Observed-donor imputer
======================

``ObservedDonorImputer`` fills the features a building lacks by copying them
from one nearby building that holds them, its donor. By default it fills the
occupancy class, the number of units and the number of stories
(``OCCUPANCY_UNITS_AND_STORIES``). The class and its defaults are in
``observed_donor_imputer.py`` beside this page. ``DonorPool`` and
``TREE_RADIUS_MARGIN`` are in ``donor_pool.py``, the distance, the pick and
``MIN_WEIGHT_DISTANCE`` in ``donor_search.py``, and the outcomes, rounds and
log columns in ``donor_records.py``. This page states what the imputer does,
the method in the order the code runs it, where its defaults come from and its
scope. The docstrings state what a caller passes and receives. Possible worlds,
frozen stages and donors are introduced in the section "Concepts" of
``brails/compilers/README.rst``.

What the imputer does
---------------------

The imputer is for an inventory in which some buildings carry observed values
and others carry none, such as footprints that no NSI point is joined to. A
building missing a feature it fills is a target, and the buildings with
observed values are the donors. For each target and each possible world, the
imputer finds the donors near the target that resemble it, picks one of them at
random, and copies every missing feature from that one donor, so the copied
values come from one building. A target with no donor nearby takes one from its
census block group.

``impute_stage`` runs the imputer over a set of targets. ``impute`` is the
short form on one inventory, whose buildings serve as both targets and
donors. ``draw`` makes the pick for one building in one world, which other
components use: the pair re-draw (described in
``brails/compilers/hazus_earthquake/README.rst``), the residential neighbor
year sampler and the replacement cost imputer's same-class neighbor draw,
which draws a nearby building of the same occupancy class.
``imputation_donors_from_log`` reads which donor each target kept.

The method
----------

A stage run takes the targets, the donors and the number of worlds. It
builds a pool of the donors. For each target and world, in ascending order of
the target's id as a string, it first settles the target's status, by
default whether it is residential, then ranks the donors within the cap by a
distance, picks one of the nearest ``k`` and copies the missing features.
With ``validate``, a copy is checked and, if rejected, drawn once more. With
batches, the targets run in groups, and each group can draw from the groups
before it. Every pick is logged. The subsections below follow that order.

The donor pool
~~~~~~~~~~~~~~

``DonorPool`` holds every donor, in ascending order of its id as a string,
with the features the imputer reads: those it fills, the soft filters, the
``required`` features and the story count. A donor's site is the centroid of
its footprint, the natural logarithm of its plan area in square meters, its
story count under ``stories_key`` and its block group under
``BLOCK_GROUP_KEY``. A building whose footprint encloses no area has no site.
It is never a donor, and as a target it is logged ``OUTCOME_NO_AREA``. A
donor that lacks a feature of ``required`` (``REQUIRED_DONOR_FEATURES`` by
default: occupancy class and story count) stays in the pool but is never
drawn, and neither is a donor without a value of ``status_key``. The target
itself is never its own donor.

The status
~~~~~~~~~~

A building has the status when its value of ``status_key`` is among
``status_values``: by default when its occupancy class is one of
``UNIT_COUNT_CLASSES``, RES1, RES2 and RES3A to RES3F, in
``brails/utils/unit_count_rules.py``. A donor must have the target's status,
and this filter is never relaxed. A caller can give a target's status in
``status``, which takes precedence over the target's own value. A target
whose status is neither given nor known, because it lacks ``status_key`` or
that feature is to be filled, first takes the status of a donor picked from
all eligible donors within the cap, or of its block group without one
there, without soft filters and under the purpose labels of the status round
(``PURPOSE_RESIDENTIAL`` by default). That pick is logged as round
``ROUND_STATUS`` and copies nothing. The pick of the donor that follows is
logged as round ``ROUND_DONOR``.

``status_probability`` gives, for a building of the pool, the chance that
this round gives it the status: the share of the inverse-distance weight
held by donors with the status among the ``k`` nearest within the cap, or,
without one there, the share of its block group's donors with the status.
It is computed in world 0 at iteration 0. With batches, a later batch's
status round reads more donors than the observed ones, so its probability can
differ from this one.

The distance and the nearest donors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The candidates are the eligible donors within ``cap_m`` meters, measured
along the great circle between centroids. Each is ranked by the donor
distance::

    sqrt((g / location_scale_m)**2 + area_weight * (ln A_t - ln A_d)**2
         + stories_weight * (n_t - n_d)**2)

where g is the great-circle distance in meters, A_t and A_d are the plan
areas of the target and the donor, and n_t and n_d their story counts in the
world drawn. The distance uses a building's one story count in every world
where it holds one value for every world, and the count of the world drawn
where it holds a list of one value per world. The story term is zero where
either side has no story count, and ``stories_key=None`` drops it. Equal
distances are ordered by donor id as a string.

A donor must also match the target on each soft filter the target holds,
by default the occupancy class. Where fewer than ``k`` donors match, the
filters are relaxed one at a time from the last. The pick is made at the
first level that holds ``k`` candidates, logged ``OUTCOME_DONOR`` at the
first level and ``OUTCOME_RELAXED`` at a later one. Where no level holds
``k``, it is made among the candidates of the loosest level, logged
``OUTCOME_FEWER_THAN_K``.

The pick
~~~~~~~~

Among the first ``k`` candidates of the level, one is picked with weights
proportional to one over its distance, which ``MIN_WEIGHT_DISTANCE`` keeps
finite. The pick uses one uniform random number keyed by the target's id,
the seed, the world, a purpose label (``PURPOSE_DONOR``) and the iteration,
through ``keyed_uniforms`` in ``brails/utils/deterministic_sampling.py``.
Every feature the target is missing is copied from the donor picked. A
feature the donor does not hold stays missing.

When no donor lies within the cap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A target with no eligible candidate within the cap draws one donor of its
own block group, read from ``BLOCK_GROUP_KEY``, with equal weights, under
``PURPOSE_FALLBACK``. The soft filters are relaxed from the last until the
block group holds a match. The donor can lie beyond the cap, in a rural
block group that covers a large area. The draw is logged
``OUTCOME_BLOCK_GROUP``. A target without a block group, or whose block
group holds no eligible donor, is logged ``OUTCOME_NO_DONOR`` and receives
nothing.

Validation and the repair draw
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``validate``, a target to which the run copied a feature of
``validated`` (by default those of ``VALIDATED_FEATURES``, occupancy class
and story count, that the imputer fills) is checked after the copy. A
rejected target takes one repair draw at iteration 1, logged as round
``ROUND_REPAIR``. The repair re-draws the copied features only, excludes the
rejected donor, keeps the status of the first draw, and tries at most
``max_attempts`` donors, each judged by ``validate``. Once no untried donor
within the cap is left, it can fall back to the block group. A target still
rejected keeps its first values, and the repair is logged
``OUTCOME_EXHAUSTED``.

``draw`` offers the same judged draw to other callers through ``admit``. A
rejected donor counts as tried, and when the attempts are spent or no
untried candidate is left, the draw by default scans the donors within the cap
nearest first and takes the first one admitted, logged
``OUTCOME_NEAREST_ADMITTED``. A draw that finds no admitted donor is logged
``OUTCOME_EXHAUSTED`` and copies nothing.

Batches
~~~~~~~

Without batches, the copies of a run never serve as donors, so every value
is copied from an observed donor. With ``n_batches`` above 1, which needs a
``StageFreeze`` holding the observed donors, each target is placed in a
batch by its uniform under ``PURPOSE_BATCH`` in world 0, so its batch is the
same in every world. Batch ``j`` draws from the observed donors and from
the batches 1 to ``j - 1``. A target of an earlier batch donates, in each
world, its own features with the values copied to it in that world. Each
batch is validated, then frozen as ``f'{batch_prefix}_batch{j}'``.

The log and the donors kept
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The log holds one row per target, world and round run, with the columns of
``LOG_COLUMNS``: among them the donor, the values copied, the filter level,
the number of candidates, the donor distance and the distance in meters,
the attempts, the outcome and the round. A target missing nothing in a
world has no row for it. ``imputation_donors_from_log`` reads the log and
gives, per target and world, the donor of the last round that copied values,
under ``IMPUTATION_DONOR_KEY``. The replacement cost imputer reads that
inventory to give a target the rate of its donor.

Defaults
--------

``DEFAULT_DONOR_K``, 20 donors, and ``DEFAULT_DONOR_CAP_M``, 2,000 meters,
are not calibrated yet. The cap is one value for every target and is never
widened, so a target beyond it reaches only its block group.
``DEFAULT_LOCATION_SCALE_M``, 100 meters counted as one unit of distance,
and ``DEFAULT_AREA_WEIGHT`` and ``DEFAULT_STORIES_WEIGHT``, both 1, are not
calibrated yet either. At these values, one plan area twice another adds
as much to the distance as 69 meters of separation, and one story of
difference as much as 100 meters. The residential neighbor year sampler
takes ``DEFAULT_DONOR_K`` and ``DEFAULT_DONOR_CAP_M`` and ranks by location
alone, and the same-class neighbor draw takes all five, unless a caller
passes others.

``REQUIRED_DONOR_FEATURES`` makes every donor hold an occupancy class and a
story count. ``max_attempts`` defaults to 5. ``TREE_RADIUS_MARGIN``, a relative
1e-6 by which the search for candidates exceeds the cap before distances are
compared with it, and ``MIN_WEIGHT_DISTANCE``, 1e-9, are settings of the
numerical method.

Scope
-----

A run serves the number of worlds it is given. The observed donors hold one
world, and the targets hold one or that number. ``impute`` takes an
inventory of one world, ``pool`` takes observed donors of one world, and
``draw`` makes the pick of one world.

A stage run compares, in each world, the story counts that the target and
the donor hold in that world, including a donor that is a target of an
earlier batch, as "The distance and the nearest donors" states. A story count
that is not a finite number, such as ``'10'`` stored as a string, is read as
none, so the story term is left out of that building's distance in that
world.

The imputer reads the block group from ``BLOCK_GROUP_KEY`` and does not
assign it. A target without one has no fallback beyond the cap. A block group
held as a list of one value per world is read as none in every world.

The status, the soft filters and the defaults suit residential occupancy
classes, units and story counts, the features the imputer fills by default.
A caller fills other features by passing ``features``, ``soft_filters``,
``required``, ``status_key`` and ``status_values``.
