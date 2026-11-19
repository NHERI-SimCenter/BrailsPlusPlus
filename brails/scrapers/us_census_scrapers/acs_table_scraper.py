# Copyright (c) 2026 The Regents of the University of California
#
# This file is part of BRAILS++.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.


"""
Read American Community Survey tables from the Census Data API.

.. autosummary::

      ACSTableScraper
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import pandas as pd
import requests
from requests import exceptions

from brails.scrapers.us_census_scrapers.census_tract_scraper import (
    LEVEL_BLOCK_GROUP,
    LEVEL_TRACT,
)
from brails.utils.census_api_key import CensusApiKey, resolve_census_api_key
from brails.utils.http_session import (
    RETRIED_STATUSES,
    address_without_query,
    create_retrying_session,
    get_whole_body,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# The ACS five-year release read by default, named by its last year.
DEFAULT_ACS_VINTAGE = 2024

# The Census Data API endpoint of the ACS five-year detailed tables, for the release
# whose last year is vintage.
ACS5_URL = 'https://api.census.gov/data/{vintage}/acs/acs5'

# The name of the GEOID index of a table get_table returns.
GEOID_INDEX = 'GEOID'

# The Census Bureau's coded values, such as -666666666, which the API returns in
# place of an estimate or a margin, lie below this limit.
CODED_VALUE_LIMIT = -100_000_000

# What replaces the key in the HTTP library's log records and in answer text quoted
# in messages.
_REDACTED = '<census-api-key>'

# The geography of each level as the API names it, and the response columns whose
# values, joined in this order, form a GEOID.
_API_GEOGRAPHY = {LEVEL_BLOCK_GROUP: 'block group', LEVEL_TRACT: 'tract'}
_GEOID_PARTS = {
    LEVEL_BLOCK_GROUP: ('state', 'county', 'tract', 'block group'),
    LEVEL_TRACT: ('state', 'county', 'tract'),
}

# Characters of a county FIPS code: state 2, county 3.
_COUNTY_FIPS_LENGTH = 5

# The loggers urllib3 writes request lines to, query string included: each request,
# and each retry.
_HTTP_LOGGERS = ('urllib3.connectionpool', 'urllib3.util.retry')

# The request of check_key, for the name of one state: a small answer.
_KEY_CHECK_PARAMS = {'get': 'NAME', 'for': 'state:01'}


class _KeyRedaction(logging.Filter):
    """Replace the API key in the message of a log record."""

    def __init__(self, key: str) -> None:
        """
        Hold the key to replace.

        Args:
            key (str):
                The text replaced in every message that holds it.
        """
        super().__init__()
        self._key = key

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Rewrite a record whose message holds the key, and keep every record.

        Args:
            record (logging.LogRecord):
                The record to rewrite before a handler formats it.

        Returns:
            bool:
                Always ``True``.
        """
        message = record.getMessage()
        if self._key in message:
            record.msg = message.replace(self._key, _REDACTED)
            record.args = None
        return True


class ACSTableScraper:
    """
    Read American Community Survey tables from the Census Data API.

    The scraper reads one table of the ACS five-year detailed tables for every block
    group or every tract of the counties given, with one request per county. The
    result holds the estimate and the margin of error of each line of the table as
    numbers, under the Census Bureau's variable names, such as ``B25024_002E`` and
    ``B25024_002M``. The annotation columns, whose names end in ``EA`` and ``MA``,
    are dropped. From the 2022 release on, the ACS publishes Connecticut by its
    planning regions (09110 to 09190), so the API holds no geography for the county
    codes 09001 to 09015 in those releases.

    The key is resolved once, when the scraper is created, by
    :func:`~brails.utils.census_api_key.resolve_census_api_key`, from the argument or
    else from the stores it reads, and :meth:`check_key` asks the API whether it
    accepts the key. The key is sent with every request. It appears in no exception
    the scraper raises, in none of the frames such an exception carries, and in no
    record the HTTP library writes through :mod:`logging`
    for the request.

    Args:
        api_key (str, optional):
            The Census Data API key. Defaults to ``None``, which reads the stores
            :func:`~brails.utils.census_api_key.find_census_api_key` names.
        timeout (float, optional):
            The seconds to wait for the connection and for each part of the answer,
            as ``requests`` applies a timeout. Defaults to 120.
        retries (int, optional):
            How many times a failed request is tried again, as
            :func:`~brails.utils.http_session.create_retrying_session` does it.
            Defaults to 3.

    Attributes:
        key_source (str):
            Where the key came from, in words that name the argument, the variable,
            the secret or the file, for the messages about the key.

    Raises:
        ValueError:
            If the key holds a character other than an ASCII letter or digit.
        LookupError:
            If ``api_key`` is not given or empty and no store holds a key.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 120,
        retries: int = 3,
    ) -> None:
        """Resolve the key and keep the request settings."""
        # Held as a CensusApiKey from here on, so that no frame shows its text.
        if isinstance(api_key, str):
            api_key = CensusApiKey(api_key)
        self._api_key, self.key_source = resolve_census_api_key(api_key)
        self.timeout = timeout
        self.retries = retries

    def get_table(
        self,
        table: str,
        level: str,
        counties: str | Sequence[str],
        vintage: int = DEFAULT_ACS_VINTAGE,
    ) -> pd.DataFrame:
        """
        Read one ACS five-year table for every block group or tract of the counties.

        Args:
            table (str):
                The table id, such as ``'B25024'``, in either case.
            level (str):
                :data:`LEVEL_BLOCK_GROUP` or :data:`LEVEL_TRACT`, the levels of
                :class:`~brails.scrapers.us_census_scrapers.census_tract_scraper.CensusTractScraper`.
            counties (str or Sequence[str]):
                The counties as five-digit FIPS codes, such as ``'06001'``, or one
                such code.
            vintage (int, optional):
                The last year of the five-year release. Defaults to
                :data:`DEFAULT_ACS_VINTAGE`.

        Returns:
            pandas.DataFrame:
                One row per block group or tract and one column of floats per
                estimate and per margin, the columns in ascending order of name. The
                index, named :data:`GEOID_INDEX` and in ascending order, holds the
                GEOIDs as strings of :data:`~brails.constants.GEOID_LENGTH` digits
                for block groups or :data:`~brails.constants.TRACT_GEOID_LENGTH`
                digits for tracts. A null value and a value below
                :data:`CODED_VALUE_LIMIT`, one of the Census Bureau's coded values,
                are NaN.

        Raises:
            ValueError:
                If ``table`` is not a table id of letters and digits that starts with
                a letter, ``level`` is not one of the two levels, ``counties`` is
                empty or holds a code that is not five digits or a code twice; or if
                the API holds no geography of a county; or if ``retries`` is not an
                integer of at least 0.
            TypeError:
                If ``vintage`` is not an integer.
            requests.exceptions.HTTPError:
                If the API answers with a 4xx status, or with 5xx or 429 after the
                retries. The message quotes the API's answer, and a 404's asks to
                check the vintage. The exception carries the status in its message
                and no response.
            requests.exceptions.ConnectionError:
                If the connection keeps failing after the retries, or an answer is
                cut short.
            RuntimeError:
                If the API answers with something other than the table asked: the
                page it sends for a key that is not valid or not activated, another
                page, a body without the expected columns, a value that is not a
                number or rows of another county; or if a request fails with an error
                that is not retried, such as too many redirects.
        """
        table = table.upper() if isinstance(table, str) else table
        if not isinstance(table, str) or not re.fullmatch(r'[A-Z][A-Z0-9]*', table):
            msg = f'table must be an ACS table id such as B25024, not {table!r}.'
            raise ValueError(msg)
        if level not in _API_GEOGRAPHY:
            msg = (
                f'level must be {LEVEL_BLOCK_GROUP!r} or {LEVEL_TRACT!r}, '
                f'not {level!r}.'
            )
            raise ValueError(msg)
        counties = [counties] if isinstance(counties, str) else list(counties)
        bad = [
            county
            for county in counties
            if not (
                isinstance(county, str)
                and len(county) == _COUNTY_FIPS_LENGTH
                and county.isascii()
                and county.isdigit()
            )
        ]
        if not counties or bad:
            msg = (
                'counties must hold one or more five-digit county FIPS codes '
                f'as strings, such as "06001"; got {bad[:3] or counties!r}.'
            )
            raise ValueError(msg)
        twice = [county for county in counties if counties.count(county) > 1]
        if twice:
            msg = f'counties holds {twice[0]!r} twice; give each county once.'
            raise ValueError(msg)

        pattern = re.compile(rf'{table}_\d+[EM]')
        parts = _GEOID_PARTS[level]
        frames = []
        for county in counties:
            params = {
                'get': f'group({table})',
                'for': f'{_API_GEOGRAPHY[level]}:*',
                'in': f'state:{county[:2]} county:{county[2:]}',
            }
            what = f'table {table} of {vintage} in county {county}'
            response = self._send_request(vintage, params, what)
            header, *rows = self._read_rows(response, what)
            frame = pd.DataFrame(rows, columns=header)
            columns = sorted(c for c in header if pattern.fullmatch(c))
            if not columns or not set(parts) <= set(header):
                msg = (
                    f'The Census Data API answered the request for {what} without '
                    'the estimate, margin and geography columns of a table.'
                )
                raise RuntimeError(msg)
            geoids = frame[list(parts)].astype(str).agg(''.join, axis=1)
            if not geoids.str.startswith(county).all():
                msg = (
                    f'The Census Data API answered the request for {what} with '
                    'rows of another county.'
                )
                raise RuntimeError(msg)
            try:
                values = frame[columns].apply(pd.to_numeric).astype(float)
            except ValueError:
                values = None
            if values is None:
                msg = (
                    f'The Census Data API answered the request for {what} with a '
                    'value that is not a number.'
                )
                raise RuntimeError(msg)
            values.index = pd.Index(geoids.tolist(), dtype=object)
            frames.append(values.mask(values < CODED_VALUE_LIMIT))

        result = pd.concat(frames).sort_index()
        result.index.name = GEOID_INDEX
        return result

    def check_key(self, vintage: int = DEFAULT_ACS_VINTAGE) -> None:
        """
        Ask the Census Data API whether it accepts the key.

        Send one request, for the name of one state, with the key, and raise if the
        API refuses the key, such as one not activated yet.

        Args:
            vintage (int, optional):
                The last year of the five-year release asked. Defaults to
                :data:`DEFAULT_ACS_VINTAGE`.

        Raises:
            TypeError:
                If ``vintage`` is not an integer.
            RuntimeError:
                If the API answers with the page it sends for a key that is not valid
                or not activated, whose message names :attr:`key_source`, or with
                something other than a table.
            requests.exceptions.HTTPError:
                If the API answers with a 4xx status, or with 5xx or 429 after the
                retries.
            requests.exceptions.ConnectionError:
                If the connection keeps failing after the retries, or the answer is
                cut short as :func:`~brails.utils.http_session.get_whole_body`
                finds it.
        """
        what = f'the key check against the {vintage} release'
        self._read_rows(self._send_request(vintage, _KEY_CHECK_PARAMS, what), what)

    def _send_request(
        self, vintage: int, params: dict[str, str], what: str
    ) -> requests.Response:
        """
        Send a request in a session that tries it again after a failure.

        The key is added to the query. While the request is under way, the frames of
        ``requests`` and urllib3 hold the URL with the key as plain text, so every
        exception that leaves them is raised again here, afresh: no exception raised
        here carries those frames, a context or the key. The request lines urllib3
        logs show the key replaced.

        Args:
            vintage (int):
                The last year of the five-year release asked.
            params (dict[str, str]):
                The query, without the key.
            what (str):
                What was requested, in words, for the messages.

        Returns:
            requests.Response:
                The answer, which is 5xx or 429 only after the retries.

        Raises:
            TypeError:
                If ``vintage`` is not an integer.
            ValueError:
                If ``retries`` is not an integer of at least 0.
            requests.exceptions.ConnectionError:
                If the connection keeps failing after the retries, or the answer is
                cut short as :func:`~brails.utils.http_session.get_whole_body`
                finds it, with the session's message and what was asked.
            RuntimeError:
                If the request fails with another error of ``requests``, such as too
                many redirects, which only its name describes.
            BaseException:
                Any other exception, such as ``KeyboardInterrupt``, raised again as
                one of its type with a message, or as ``RuntimeError`` where its type
                takes no message.
        """
        if isinstance(vintage, bool) or not isinstance(vintage, int):
            msg = f'vintage must be an integer year, not {vintage!r}.'
            raise TypeError(msg)
        url = ACS5_URL.format(vintage=vintage)
        params = {**params, 'key': self._api_key}
        redaction = _KeyRedaction(self._api_key)
        for name in _HTTP_LOGGERS:
            logging.getLogger(name).addFilter(redaction)
        # Not safe_get_json, which re-raises the HTTPError of a 4xx answer, whose
        # message holds the URL with the key.
        try:
            with create_retrying_session(self.retries) as session:
                response = get_whole_body(session, url, self.timeout, params=params)
        except exceptions.ConnectionError as error:
            # The session's message and get_whole_body's name no key.
            fresh = exceptions.ConnectionError(f'{error} It asked for {what}.')
        except exceptions.RequestException as error:
            # Its name alone, since its message can hold the URL with the key.
            fresh = RuntimeError(
                f'The request to the Census Data API for {what} failed '
                f'({type(error).__name__}). The API may be down; try again later.'
            )
        except BaseException as error:  # noqa: BLE001
            msg = (
                f'{type(error).__name__} during the request to the Census Data API '
                f'for {what}. Its traceback is left out, since the frames of '
                'requests and urllib3 hold the URL with the key.'
            )
            try:
                fresh = type(error)(msg)
            except Exception:  # noqa: BLE001
                fresh = RuntimeError(msg)
        else:
            return response
        finally:
            for name in _HTTP_LOGGERS:
                logging.getLogger(name).removeFilter(redaction)
        # Raised outside the except clauses, so that the exception caught, with its
        # frames, is not attached as the context.
        raise fresh

    def _read_rows(self, response: requests.Response, what: str) -> list[list]:
        """
        Check an answer of the API and return its rows.

        Args:
            response (requests.Response):
                An answer of the API.
            what (str):
                What was requested, in words, for the messages.

        Returns:
            list[list]:
                The rows of the answer, header first, with at least one row of data.

        Raises:
            ValueError:
                If the API holds no geography of the county (an empty answer, HTTP
                204).
            requests.exceptions.HTTPError:
                On a 4xx or 5xx answer; for 404 the message asks to check the
                vintage.
            RuntimeError:
                If the answer is not a table. For the page the API sends for a key
                that is not valid or not activated, the message names where the key
                came from.
        """
        body = response.text.replace(self._api_key, _REDACTED).strip()
        if response.status_code == 204:  # noqa: PLR2004
            msg = (
                f'The Census Data API holds no geography for {what}; check the '
                'county FIPS code and the vintage.'
            )
            raise ValueError(msg)
        if response.status_code >= 400:  # noqa: PLR2004
            tries = (
                f' after {self.retries} retries'
                if response.status_code in RETRIED_STATUSES
                else ''
            )
            msg = (
                f'The Census Data API answered the request for {what} with '
                f'HTTP {response.status_code}{tries}: {body[:300]}'
            )
            if response.status_code == 404:  # noqa: PLR2004
                msg += ' Check the vintage.'
            raise exceptions.HTTPError(msg)
        if 'html' in response.headers.get('Content-Type', '').lower():
            page = address_without_query(response.url)
            msg = (
                f'The Census Data API answered the request for {what} with the '
                f'page {page} instead of a table. It sends such a page when the '
                f'key is not valid or not activated: check the key from '
                f'{self.key_source}. A new key works only after it is '
                'activated from the email the Census Bureau sends.'
            )
            raise RuntimeError(msg)
        try:
            rows = response.json()
        except ValueError:
            rows = None
        if not (
            isinstance(rows, list)
            and len(rows) > 1
            and all(
                isinstance(row, list) and len(row) == len(rows[0]) for row in rows
            )
        ):
            msg = (
                f'The Census Data API answered the request for {what} with a '
                f'body that is not a table: {body[:300]!r}'
            )
            raise RuntimeError(msg)
        return rows
