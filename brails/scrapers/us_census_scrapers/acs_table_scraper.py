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

from brails.constants import REDACTED
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
    """Replace the Census API key in the message of a log record."""

    def __init__(self, key: str) -> None:
        """
        Hold the Census API key to replace.

        Args:
            key (str):
                The text replaced in every message that holds it.
        """
        super().__init__()
        self._key = key

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Rewrite a record whose message holds the Census API key; keep every record.

        Args:
            record (logging.LogRecord):
                The record to rewrite before a handler formats it.

        Returns:
            bool:
                Always ``True``.
        """
        message = record.getMessage()
        if self._key in message:
            record.msg = message.replace(self._key, REDACTED)
            record.args = None
        return True


class ACSTableScraper:
    """
    Read American Community Survey tables from the Census Data API.

    Each call of :meth:`get_table` reads one ACS five-year table for the block groups
    or tracts of a list of counties. The README beside this module states what a
    table holds, where the Census API key comes from and how the scraper keeps it out
    of messages and logs.

    Args:
        api_key (str, optional):
            The Census API key. Defaults to ``None``, which reads the stores
            :func:`~brails.utils.census_api_key.find_census_api_key` names.
        timeout (float, optional):
            Seconds to wait for the connection and for each part of the answer.
            Defaults to 120.
        retries (int, optional):
            How many times a failed request is tried again. Defaults to 3.

    Attributes:
        key_source (str):
            Where the Census API key came from, in words, for the messages about it.

    Raises:
        ValueError:
            If the Census API key holds a character other than an ASCII letter or
            digit.
        LookupError:
            If ``api_key`` is not given or empty and no store holds a Census API key.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 120,
        retries: int = 3,
    ) -> None:
        """Resolve the Census API key and keep the request settings."""
        # The Census API key is held as a CensusApiKey from here on, so that no frame
        # shows its text.
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
                One row per block group or tract, indexed by GEOID under
                :data:`GEOID_INDEX`, and one column of floats per estimate and per
                margin. The Census Bureau's coded values are NaN.

        Raises:
            ValueError:
                If ``table``, ``level`` or ``counties`` is not of the form above, a
                county is given twice, the API holds no geography of a county, or
                ``retries`` is not an integer of at least 0.
            TypeError:
                If ``vintage`` is not an integer.
            requests.exceptions.HTTPError:
                If the API answers with a 4xx status, or with 5xx or 429 after the
                retries.
            requests.exceptions.ConnectionError:
                If the connection keeps failing after the retries, or an answer is
                cut short.
            RuntimeError:
                If the API answers with something other than the table asked, such as
                the page it sends for a Census API key that is not valid or not
                activated, or a request fails with an error that is not retried.
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
        Ask the Census Data API whether it accepts the Census API key.

        Args:
            vintage (int, optional):
                The last year of the five-year release asked. Defaults to
                :data:`DEFAULT_ACS_VINTAGE`.

        Raises:
            TypeError:
                If ``vintage`` is not an integer.
            RuntimeError:
                If the API refuses the Census API key, with a message that names
                :attr:`key_source`, or answers with something other than a table.
            requests.exceptions.HTTPError:
                If the API answers with a 4xx status, or with 5xx or 429 after the
                retries.
            requests.exceptions.ConnectionError:
                If the connection keeps failing after the retries, or the answer is
                cut short.
        """
        what = f'the Census API key check against the {vintage} release'
        self._read_rows(self._send_request(vintage, _KEY_CHECK_PARAMS, what), what)

    def _send_request(
        self, vintage: int, params: dict[str, str], what: str
    ) -> requests.Response:
        """
        Send a request with the Census API key, retried after a failure.

        Every exception is raised again afresh, without the frames of ``requests``
        and urllib3, which hold the URL with the Census API key.

        Args:
            vintage (int):
                The last year of the five-year release asked.
            params (dict[str, str]):
                The query, without the Census API key.
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
                cut short.
            RuntimeError:
                If the request fails with another error of ``requests``, such as too
                many redirects.
            BaseException:
                Any other exception, such as ``KeyboardInterrupt``, as one of its
                type or as ``RuntimeError``.
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
        # message holds the URL with the Census API key.
        try:
            with create_retrying_session(self.retries) as session:
                response = get_whole_body(session, url, self.timeout, params=params)
        except exceptions.ConnectionError as error:
            # The session's and get_whole_body's messages hold no Census API key.
            fresh = exceptions.ConnectionError(f'{error} It asked for {what}.')
        except exceptions.RequestException as error:
            # Its name alone: its message can hold the URL with the Census API key.
            fresh = RuntimeError(
                f'The request to the Census Data API for {what} failed '
                f'({type(error).__name__}). The API may be down; try again later.'
            )
        except BaseException as error:  # noqa: BLE001
            msg = (
                f'{type(error).__name__} during the request to the Census Data API '
                f'for {what}. Its traceback is left out, since the frames of '
                'requests and urllib3 hold the URL with the Census API key.'
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
                If the API holds no geography of the county (HTTP 204).
            requests.exceptions.HTTPError:
                On a 4xx or 5xx answer.
            RuntimeError:
                If the answer is not a table, such as the page the API sends for a
                Census API key that is not valid or not activated.
        """
        body = response.text.replace(self._api_key, REDACTED).strip()
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
                'Census API key is not valid or not activated: check the key from '
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
