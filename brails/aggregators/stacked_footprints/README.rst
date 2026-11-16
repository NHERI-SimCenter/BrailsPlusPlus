Stacked-footprint merge
=======================

``StackedFootprintMerge`` merges the Overture outlines of one building, such
as a podium and the towers drawn on it, into one asset. Its thresholds are
constants in ``stacked_footprints.py`` beside this page. This page states
where the thresholds come from, the method the merge runs, and what it
leaves unmerged. The docstrings state what a caller passes and receives.

The thresholds
--------------

``NEST_MIN`` (0.9), ``COPY_IOU`` (0.75) and ``PART_TALLER_M`` (3.0 m) are the
boundaries by which SimCenter's building inventories of eleven San Francisco
Bay Area cities, built from assessor parcels, local footprints and Overture
footprints, classify overlapping footprints. In those inventories two
footprints whose intersection over union reaches ``COPY_IOU`` are two
drawings of one building, and a footprint at least ``NEST_MIN`` inside a
larger one is nested in it. A nested pair is taken as parts of one building
when the two footprints share a parcel, or when they show the Overture
stacking signal that "The method" states. ``StackedFootprintMerge`` reads
no parcels and keeps the signal alone.

Of the 793,016 footprints of the inventories built from local footprints,
1,282 pairs overlap by more than a tenth of the smaller footprint. Their
intersection over union has two modes: 858 pairs lie below 0.1 and 126
between 0.80 and 0.95, while the bins from 0.50 to 0.75 hold 28 pairs in
all. ``COPY_IOU`` lies at the low end of that gap. The share of the smaller
footprint inside the larger has no such gap, with 159 pairs between 0.75 and
0.90, 258 between 0.90 and 0.95 and 753 at 0.95 or more, so ``NEST_MIN`` is
not set by one. No derivation of ``PART_TALLER_M`` is on record. None of the
three values is fitted to buildings known to be stacked or taken from a
published source.

The merge moves few footprints, and its counts change little near the
packaged values. Of the 157,778 Overture buildings of San Francisco at or
above 450 sq ft of plan area in release 2026-10-21.0, the merge joins 54
members into 10 heads (the terms of "The method"), so 0.034% of the
buildings leave the inventory. With one constant changed at a time, the
counts are:

=====================  =======  =======  =====
Constant               Value    Members  Heads
=====================  =======  =======  =====
``NEST_MIN``           0.85     54       10
``NEST_MIN``           0.95     50       10
``PART_TALLER_M``      2.0 m    54       10
``PART_TALLER_M``      5.0 m    53       9
=====================  =======  =======  =====

The same buildings hold no copies at a ``COPY_IOU`` of 0.7, 0.75 or 0.8.

The method
----------

``merge`` runs in six steps, in this order.

It first finds the pairs. Two footprints are copies when their intersection
over union is at least ``COPY_IOU``, and nested when their intersection
covers at least ``NEST_MIN`` of the smaller one's area. The areas are planar
in EPSG:4326, of the exterior rings a BRAILS asset stores. The larger
footprint of a pair is its container, a tie of area going to the lower
``overture_id``.

Each connected set of copies then keeps one footprint, its head: one with a
``height`` before one without, then the larger, then the lower
``overture_id``. The other copies are dropped.

Among the footprints that remain, each nested footprint goes to its largest
container, a tie going to the lower ``overture_id``, and from there up to
the top of the nest, the container that lies in no other. A footprint
merges into the top of its nest when it is taller than the top by more than
``PART_TALLER_M``, converted from meters to the merge's length unit, when it
has a ``height`` where the top has none, or when either of the two has the
``class`` ``ROOF_CLASS`` (``'roof'``). Otherwise it stays an asset of its
own.

The top of a nest, its head, and the footprints merging into it, its
members, become one asset. Its coordinates are the exterior rings of the
union of their polygons, and its ``height`` and ``num_floors`` are the
largest over the parts, absent where no part has one. The height is taken
from the first of the tallest parts, the head before its members and the
members in ascending order of ``overture_id``. ``PART_IDS_FEATURE``
(``'overture_part_ids'``) holds the members' ``overture_id`` values, sorted
and joined by ``PART_IDS_SEPARATOR`` (``';'``). ``sources`` holds the
entries of the part whose height was taken, then the head's, then the other
members', less the entries of the other parts whose property is the
height, so that ``StoriesFromHeightInferer`` reads the height's source from
the tallest part's entries wherever those name one. Where no part has a
height, ``sources`` keeps no entry whose property is the height. Every
other feature is the head's.

Last, the footprints that remain are keyed by ``assign_asset_ids`` with the
``overture_id`` as tiebreak.

Scope
-----

The signal reads Overture's ``height`` and ``class``, so the merge takes
Overture footprints only and refuses an inventory that holds others. A
nested pair without the signal stays two assets. Of the San Francisco
buildings above, 69 lie at least ``NEST_MIN`` inside a larger one, and 15
of them stay. The city inventories, which compare parcels, merge some of
these.

Since footprints are compared by their exterior rings, a footprint in the
courtyard of another lies inside it. One of the 10 San Francisco heads
holds its member this way.

A merged asset takes the tallest part's height over the whole union, with
the tallest part's source of that height, and the story count and height
class inferred from the height then apply to all of it. A gross area taken
as plan area times story count overstates the gross area of a podium with
a tower on part of it. The ``parts`` table of ``StackedMergeResult`` keeps
each part's plan area and height.
