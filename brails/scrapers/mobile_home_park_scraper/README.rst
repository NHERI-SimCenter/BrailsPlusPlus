Mobile-home park scraper
========================

``MobileHomeParkScraper`` reads the mobile-home park points of the Homeland
Infrastructure Foundation-Level Data (HIFLD) that lie inside a region and returns
them as an asset inventory. It lives in ``mobile_home_park_scraper.py`` beside this
page. This page states where the points come from, what a read returns, and the
scraper's limits. The docstrings state the arguments, the return values and the
errors.

The source
----------

HIFLD holds a layer of the mobile-home parks of the United States, one point per
park. SeerAI keeps an archive of HIFLD layers on Source Cooperative, and the scraper
reads the park layer from it anonymously, over HTTPS: the bucket ``PARK_BUCKET_URL``,
folder ``PARK_PREFIX``.

The archive holds the layer as one Parquet file of points in longitude and latitude,
written in parts. Each complete write leaves in the folder a marker named
``_committed_<id>`` that lists the parts it added, and a finished file holds a
``_SUCCESS`` marker. The scraper lists the folder, takes the newest ``_committed_``
marker by its time of modification and reads the parts it lists. ``marker`` and
``written`` record that marker's name and time, so a result can be traced to the
write it came from.

What a read returns
-------------------

``get_assets`` returns one point asset per park strictly inside the region; a point
on the boundary is outside. Each asset is keyed by the park's ``MHPID``,
``PARK_ID_COLUMN``, as an integer, and carries the HIFLD columns as features under
their HIFLD names. The scraper reads only the region it is given;
``HazusEQInventoryCompiler`` reads the points within ``mh_park_radius_m`` of the
footprints' bounding box.

Limits
------

- Every call downloads the whole layer for the United States and filters it in
  memory. Nothing is written to disk or kept between calls.
- Only the parts of the newest marker are read, so the read assumes that the last
  write rewrote the whole file. A listing that is truncated, or that shows no
  complete write, raises ``RuntimeError`` whose message asks to try again later,
  since the archive may be in the middle of a write.
- The points are as current as the archive's copy of HIFLD, which the scraper does
  not check against any other source.
- A park whose ``MHPID`` is not a whole number, or is shared with another park inside
  the region, raises ``ValueError``.
