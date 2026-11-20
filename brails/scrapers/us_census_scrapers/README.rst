U.S. Census scrapers
====================

The modules of this folder read data of the U.S. Census Bureau: tables of the
American Community Survey (ACS) in ``acs_table_scraper.py``, the TIGER/Line files
and the block groups in them in ``tiger_files.py`` and ``tiger_line_scraper.py``,
census tracts in ``census_tract_scraper.py``, and ZIP Code Tabulation Areas in
``census_zcta_scraper.py``. This page covers the ACS table scraper only.

ACS table scraper
-----------------

``ACSTableScraper`` reads tables of the ACS from the Census Data API. This section
states what the scraper reads and returns, where it finds the Census API key, what it
does when a request fails, and its limits. The docstrings state the arguments, the
return values and the errors.

What the scraper reads
~~~~~~~~~~~~~~~~~~~~~~

``get_table`` reads one table of the ACS five-year detailed tables, such as
``B25024``, for every block group or every tract of the counties given, with one
request per county. The release is named by its last year, ``vintage``, which
defaults to ``DEFAULT_ACS_VINTAGE``; the 2024 release pools the surveys of 2020 to
2024.

The result is a data frame with one row per block group or tract and one column of
floats per estimate and per margin of error, under the Census Bureau's variable
names, such as ``B25024_002E`` and ``B25024_002M``. The annotation columns, whose
names end in ``EA`` and ``MA``, are dropped. The index, named ``GEOID_INDEX``, holds
the GEOIDs as strings of 12 digits for block groups and 11 for tracts. Rows and
columns are in ascending order. A null value, and a value below
``CODED_VALUE_LIMIT``, which marks one of the Census Bureau's coded values such as
-666666666, is NaN.

The Census API key
~~~~~~~~~~~~~~~~~~

Every request carries a Census API key. The Census Bureau issues one free of charge
at https://api.census.gov/data/key_signup.html, and it works only after it is
activated from the email the Bureau sends.

The scraper takes the key once, when it is created, from its ``api_key`` argument or
else from the first of these stores that holds one:

- the environment variable ``CENSUS_API_KEY``;
- in Google Colab, the notebook secret ``CENSUS_API_KEY``;
- the file ``~/.brails/census_api_key``.

To store the key in that file, run ``save_census_api_key()`` from
``brails/utils/census_api_key.py``. It asks for the key at a prompt that does not
show what is typed, and on POSIX systems it makes the file readable by its owner
only. ``key_source`` records where the key came from, the argument or a store, and
the messages about the key name it. ``check_key`` sends one small request and raises
if the API refuses the key.

The key appears in no exception the scraper raises, in none of the frames such an
exception carries, and in no request line the HTTP library logs. Where it would
appear, the text shows ``'<redacted>'``, the constant ``REDACTED`` of
``brails.constants``. The ``HTTPError`` the scraper raises carries the status in its
message and no response, since the response's URL holds the key.

When a request fails
~~~~~~~~~~~~~~~~~~~~

A request that fails with a connection error, or with HTTP 429, 500, 502, 503 or 504
(``RETRIED_STATUSES``), is tried again ``retries`` times, through
``create_retrying_session``. What the scraper raises then:

- A refused key: the API answers with an HTML page instead of a table. The scraper
  raises ``RuntimeError``, naming where the key came from and reminding that a new
  key must be activated.
- An answer of HTTP 4xx or 5xx, after the retries for a retried status:
  ``requests.exceptions.HTTPError``, whose message quotes the start of the API's
  answer. For HTTP 404 it asks to check the vintage.
- A county the API holds no geography for (HTTP 204): ``ValueError``.
- A connection that keeps failing, or an answer cut short:
  ``requests.exceptions.ConnectionError``.
- Any other answer that is not the table asked, such as one without the expected
  columns or with rows of another county, and an error that is not retried, such as
  too many redirects: ``RuntimeError``.

Limits
~~~~~~

- Only the five-year detailed tables are read, one whole table at a time; a single
  variable cannot be asked for.
- The geographies are block groups and tracts, selected by county. From the 2022
  release on, the ACS publishes Connecticut by its planning regions (09110 to 09190),
  so the API holds no geography for the county codes 09001 to 09015 in those
  releases.
- Nothing is cached: each call sends its requests again.
