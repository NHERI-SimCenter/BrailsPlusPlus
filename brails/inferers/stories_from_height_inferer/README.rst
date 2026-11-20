Stories from footprint height
=============================

``StoriesFromHeightInferer`` converts a footprint height to a story count
with the meters per story of the height's source. The packaged scales are
in ``height_scales.json`` beside the inferer. This page states how they were
obtained and where they apply.

The packaged scales
-------------------

The rule that turns a height into a count is stated in the docstring of
``StoriesFromHeightInferer``. ``height_scales.json`` holds the
``length_unit`` of its scales, meters for the packaged file, and per
footprint provider either ``scale`` and ``max_stories`` or, for ``overture``
alone, ``scale`` and ``max_stories`` per source of the height under
``sources``. The packaged file holds one provider, ``overture``, with an
entry for each of three sources. A scale whose entry sets ``max_stories``
gives no count above that number, and a provider or source not listed gives
no count.

The scales were fitted on buildings of eight cities of the San Francisco Bay
Area: Alameda, Berkeley, Fremont, Hayward, Oakland, San Francisco, San Jose
and Sunnyvale. The footprints and heights are those of the Overture Maps
buildings theme, release ``2026-10-21.0``. The reference story counts come
from county and city assessor parcel records, attached to the Overture
footprints on each parcel.

The fit used only buildings on exactly one parcel that are the only main
(non-accessory) footprint on it, share the footprint with no other parcel,
have assessor records that give no conflicting story counts, and have 1 to
70 stories. Of the 467,081 buildings selected, 419,862 have a height and a
whole story count, and the fit used those.

The source of a height is the dataset of the Overture ``sources`` entry that
names the height property, ``/properties/height``, and when no entry does,
the dataset of the first entry without a property, the source of the
footprint itself. For each source, pooled over the eight cities, the scale
is the value on a grid from 2.00 to 6.00 m in steps of 0.01 m that maximizes
exact agreement of the rule with the assessor count. The fits used 131,116
buildings whose height comes from Microsoft ML Buildings, 260,410 from
OpenStreetMap and 28,336 from USGS Lidar.

Microsoft heights are trusted up to two stories. Of the 3,536 buildings with
a Microsoft height and three or more assessor stories, 17.2% are predicted at
three or more stories from that height.

Two fits are flat. USGS Lidar agreement stays between 89.6% and 90.0% for
every scale from 4.01 to 4.33 m, and Microsoft agreement between 72.2% and
72.7% from 3.70 to 4.00 m. The OpenStreetMap scale sits at a step: 99.98% of
the San Francisco OpenStreetMap heights are whole meters, and 36,725 of them
are exactly 6 m, which gives one story at a scale of 4.01 m and two at
4.00 m. Without San Francisco, the OpenStreetMap fit is 3.73 m.

Scope
-----

The scales are a Bay Area calibration of Overture's heights. The packaged
default applies them unchanged to Overture buildings elsewhere. A caller with
a calibration of its own passes it as ``scales``. Passing ``scales``
replaces the packaged file, and footprints of two providers whose heights
differ in key or unit take two inferers.

The scales are keyed by ``footprint_provider``, which every footprint scraper
writes. The Microsoft and OpenStreetMap footprint scrapers write
``microsoft`` and ``osm`` there, and heights in feet under
``buildingheight`` by default. The packaged scales hold no entry for those
providers, so their footprints give no count until a caller passes
``scales`` with entries under ``microsoft`` and ``osm``, each with a
``scale`` and ``max_stories`` of its own, and sets ``height_key`` and
``length_unit`` to match those footprints.
