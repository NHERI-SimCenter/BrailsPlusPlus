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

"""Tests for ACSTableScraper in brails.scrapers.us_census_scrapers."""

from __future__ import annotations

import json
import logging
import traceback
from typing import TYPE_CHECKING, Any

import pytest
import requests
from requests import Response, exceptions
from urllib3.connectionpool import HTTPConnectionPool
from urllib3.exceptions import ReadTimeoutError
from urllib3.util.retry import Retry

from brails.scrapers import ACSTableScraper as ExportedScraper
from brails.scrapers.us_census_scrapers.acs_table_scraper import (
    ACS5_URL,
    DEFAULT_ACS_VINTAGE,
    GEOID_INDEX,
    ACSTableScraper,
)
from brails.scrapers.us_census_scrapers.census_tract_scraper import (
    LEVEL_BLOCK_GROUP,
    LEVEL_TRACT,
)
from brails.utils.census_api_key import CENSUS_API_KEY_NAME
from brails.utils.http_session import RETRIED_STATUSES

if TYPE_CHECKING:
    from collections.abc import Callable

KEY = 'notarealcensuskey0123456789abcdef'

# A block-group answer of the API for B25024 in Alameda County, columns in the API's
# order. The row of the second block group, listed first, holds a null estimate and
# two of the Bureau's negative codes.
BLOCK_GROUP_ROWS = [
    [
        'B25024_002E',
        'B25024_002EA',
        'B25024_002M',
        'B25024_002MA',
        'B25024_001E',
        'B25024_001EA',
        'B25024_001M',
        'B25024_001MA',
        'GEO_ID',
        'NAME',
        'state',
        'county',
        'tract',
        'block group',
    ],
    [
        '-666666666',
        None,
        '-222222222',
        '**',
        None,
        None,
        '12',
        None,
        '1500000US060014001002',
        'Block Group 2; Census Tract 4001; Alameda County; California',
        '06',
        '001',
        '400100',
        '2',
    ],
    [
        '788',
        None,
        '121',
        None,
        '870',
        None,
        '136',
        None,
        '1500000US060014001001',
        'Block Group 1; Census Tract 4001; Alameda County; California',
        '06',
        '001',
        '400100',
        '1',
    ],
]


pytestmark = pytest.mark.usefixtures('no_network')


def tract_rows(county: str, tract: str, total: str) -> list[list[str | None]]:
    """
    Return an API answer for B25034 holding one tract.

    Args:
        county (str):
            The county's three-digit code.
        tract (str):
            The tract's six-digit code.
        total (str):
            The estimate of line 1.

    Returns:
        list[list[str | None]]:
            The header and the one row.
    """
    header = ['B25034_001E', 'B25034_001EA', 'B25034_001M', 'B25034_001MA']
    geography = ['GEO_ID', 'NAME', 'state', 'county', 'tract']
    row = [total, None, '9', None, f'1400000US06{county}{tract}', 'Tract']
    return [[*header, *geography], [*row, '06', county, tract]]


def response(
    status: int = 200,
    body: object = BLOCK_GROUP_ROWS,
    content_type: str = 'application/json;charset=utf-8',
    url: str = 'https://api.census.gov/data/2024/acs/acs5',
) -> Response:
    """
    Build an HTTP response.

    Args:
        status (int, optional):
            The HTTP status the answer reports.
        body (object, optional):
            A string sent as it is, or anything else sent as JSON.
        content_type (str, optional):
            The Content-Type header.
        url (str, optional):
            The URL the response came from.

    Returns:
        requests.Response:
            The response.
    """
    answer = Response()
    answer.status_code = status
    text = body if isinstance(body, str) else json.dumps(body)
    answer._content = text.encode()
    answer.headers['Content-Type'] = content_type
    answer.url = url
    answer.encoding = 'utf-8'
    return answer


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[dict[str, Any]]]:
    """
    Return a function that makes requests.Session.get answer from a list.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Replaces requests.Session.get for the test.

    Returns:
        Callable[..., list[dict[str, Any]]]:
            A callable taking the outcomes in order, each a response to return, an
            exception to raise or a callable run with the request's URL and
            parameters, and returning the list the requests are recorded in.
    """

    def install(*outcomes: object) -> list[dict[str, Any]]:
        """
        Answer the requests with the outcomes, in order.

        Args:
            *outcomes (object):
                One outcome per request.

        Returns:
            list[dict[str, Any]]:
                The URL, parameters and timeout of each request made.
        """
        calls: list[dict[str, Any]] = []
        pending = list(outcomes)

        def get(
            _session: requests.Session,
            url: str,
            params: dict[str, str],
            timeout: float,
        ) -> Response:
            """
            Record the request and return or raise the next outcome.

            Args:
                _session (requests.Session):
                    The session the scraper sends the request in.
                url (str):
                    The endpoint.
                params (dict[str, str]):
                    The query.
                timeout (float):
                    The seconds the scraper allows for an answer.

            Returns:
                requests.Response:
                    The next outcome.

            Raises:
                Exception:
                    The next outcome, when it is an exception.
            """
            calls.append({'url': url, 'params': dict(params), 'timeout': timeout})
            outcome = pending.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            if callable(outcome):
                return outcome(url, params)
            return outcome

        monkeypatch.setattr(requests.Session, 'get', get)
        return calls

    return install


def test_class_is_exported_from_brails_scrapers() -> None:
    """The scraper is importable from brails.scrapers."""
    assert ExportedScraper is ACSTableScraper


def test_block_groups_hold_estimates_and_margins_as_numbers(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """Annotations are dropped, codes and nulls read as NaN, rows sorted by GEOID."""
    serve(response())
    table = ACSTableScraper(KEY).get_table('B25024', LEVEL_BLOCK_GROUP, ['06001'])
    assert list(table.columns) == [
        'B25024_001E',
        'B25024_001M',
        'B25024_002E',
        'B25024_002M',
    ]
    assert table.index.name == GEOID_INDEX
    assert list(table.index) == ['060014001001', '060014001002']
    assert all(isinstance(geoid, str) for geoid in table.index)
    assert (table.dtypes == 'float64').all()
    assert table.loc['060014001001', 'B25024_002E'] == 788.0
    assert table.loc['060014001002'].isna().tolist() == [True, False, True, True]


def test_request_asks_for_the_group_of_the_county(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """The default release is asked for the group of the county, with the key."""
    calls = serve(response())
    ACSTableScraper(KEY, timeout=7).get_table('B25024', LEVEL_BLOCK_GROUP, '06001')
    assert calls == [
        {
            'url': ACS5_URL.format(vintage=DEFAULT_ACS_VINTAGE),
            'params': {
                'get': 'group(B25024)',
                'for': 'block group:*',
                'in': 'state:06 county:001',
                'key': KEY,
            },
            'timeout': 7,
        }
    ]


def test_tracts_of_several_counties_are_concatenated(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """One request per county; tract GEOIDs at their full length, in GEOID order."""
    calls = serve(
        response(body=tract_rows('013', '310000', '50')),
        response(body=tract_rows('001', '400100', '1488')),
    )
    table = ACSTableScraper(KEY).get_table(
        'B25034', LEVEL_TRACT, ['06013', '06001'], vintage=2023
    )
    assert [call['params']['in'] for call in calls] == [
        'state:06 county:013',
        'state:06 county:001',
    ]
    assert all(call['params']['for'] == 'tract:*' for call in calls)
    assert all(call['url'] == ACS5_URL.format(vintage=2023) for call in calls)
    assert list(table.index) == ['06001400100', '06013310000']
    assert table['B25034_001E'].tolist() == [1488.0, 50.0]
    assert list(table.columns) == ['B25034_001E', 'B25034_001M']


def test_page_for_a_bad_key_raises_runtime_error(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """An HTML page in place of a table raises an error naming the page."""
    serve(
        response(
            body='<html><title>Invalid Key</title></html>',
            content_type='text/html',
            url='https://api.census.gov/data/invalid_key.html',
        )
    )
    with pytest.raises(
        RuntimeError,
        match=r'invalid_key\.html.*key is not valid.*the api_key argument',
    ):
        ACSTableScraper(KEY).get_table('B25024', LEVEL_TRACT, ['06001'])


def test_refusal_raises_http_error_with_the_api_explanation(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """A 4xx answer raises HTTPError holding the API's text."""
    serve(
        response(
            status=400,
            body="error: Group ' B2502X ' does not exist.",
            content_type='text/plain;charset=utf-8',
        )
    )
    with pytest.raises(exceptions.HTTPError, match='HTTP 400.*does not exist'):
        ACSTableScraper(KEY).get_table('B2502X', LEVEL_TRACT, ['06001'])


def test_empty_answer_raises_value_error(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """A 204 answer, for a county the API does not hold, raises ValueError."""
    serve(response(status=204, body=''))
    with pytest.raises(ValueError, match='holds no geography.*county 06999'):
        ACSTableScraper(KEY).get_table('B25024', LEVEL_TRACT, ['06999'])


@pytest.mark.parametrize(
    'body',
    ['not json', {'error': 'unexpected'}, [['B25024_001E', 'state']]],
    ids=['text', 'object', 'header_only'],
)
def test_body_that_is_not_a_table_raises_runtime_error(
    serve: Callable[..., list[dict[str, Any]]], body: object
) -> None:
    """A JSON answer that is not a header with rows raises RuntimeError."""
    serve(response(body=body))
    with pytest.raises(RuntimeError, match='not a table'):
        ACSTableScraper(KEY).get_table('B25024', LEVEL_TRACT, ['06001'])


def test_table_without_geography_columns_raises_runtime_error(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """A table whose answer lacks the block-group column raises RuntimeError."""
    serve(response(body=tract_rows('001', '400100', '1')))
    with pytest.raises(RuntimeError, match='geography columns'):
        ACSTableScraper(KEY).get_table('B25034', LEVEL_BLOCK_GROUP, ['06001'])


def test_requests_go_through_the_shared_retrying_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scraper's session is create_retrying_session's, with its retries."""
    adapters = []

    def get(session: requests.Session, url: str, **_kwargs: object) -> Response:
        adapters.append(session.get_adapter(url))
        return response()

    monkeypatch.setattr(requests.Session, 'get', get)
    ACSTableScraper(KEY, retries=2).get_table('B25024', LEVEL_BLOCK_GROUP, ['06001'])
    retry = adapters[0].max_retries
    assert retry.total == 2
    assert tuple(retry.status_forcelist) == RETRIED_STATUSES


def test_answer_cut_short_raises_connection_error(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """An answer cut short raises ConnectionError naming what was asked."""
    serve(exceptions.ChunkedEncodingError('cut'))
    match = 'cut short.*try again. It asked for table B25024 of 2024 in county 06001'
    with pytest.raises(exceptions.ConnectionError, match=match):
        ACSTableScraper(KEY).get_table('B25024', LEVEL_TRACT, ['06001'])


# Each failure carries the key the way requests and the API would show it: the URL
# with its query string, or a body echoing the query.
URL_WITH_KEY = (
    'https://api.census.gov/data/2024/acs/acs5?get=group%28B25024%29'
    f'&for=tract%3A%2A&in=state%3A06+county%3A001&key={KEY}'
)
LEAKING_FAILURES = {
    'cut_short': [exceptions.ChunkedEncodingError(f'Broken: {URL_WITH_KEY}')],
    'other_request_error': [exceptions.InvalidURL(f'Invalid URL {URL_WITH_KEY}')],
    'server_error': [
        response(status=500, body=f'failed {URL_WITH_KEY}', url=URL_WITH_KEY)
    ],
    'refusal': [
        response(status=400, body=f'error: bad key {KEY}', url=URL_WITH_KEY)
    ],
    'key_page': [
        response(
            body=f'<html>{KEY}</html>',
            content_type='text/html',
            url=f'https://api.census.gov/data/invalid_key.html?key={KEY}',
        )
    ],
    'not_a_table': [response(body=f'oops {KEY}', url=URL_WITH_KEY)],
}


@pytest.mark.parametrize('name', list(LEAKING_FAILURES))
def test_key_is_in_no_message_traceback_warning_or_log(
    serve: Callable[..., list[dict[str, Any]]],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    name: str,
) -> None:
    """No failure shows the key in a traceback with its frames' values, or a log."""
    caplog.set_level(logging.DEBUG)
    serve(*LEAKING_FAILURES[name])
    with pytest.raises(Exception) as raised:  # noqa: PT011
        ACSTableScraper(KEY).get_table('B25024', LEVEL_TRACT, ['06001'])
    error = raised.value
    assert not isinstance(error, (AssertionError, IndexError))
    shown = ''.join(
        traceback.TracebackException.from_exception(
            error, capture_locals=True
        ).format()
    )
    assert KEY not in shown
    assert error.__context__ is None
    assert error.__cause__ is None
    scraper = ACSTableScraper(KEY)
    for holder in (scraper, vars(scraper)):
        assert KEY not in repr(holder)
    printed = capsys.readouterr()
    assert KEY not in printed.out + printed.err
    assert KEY not in caplog.text


def test_connection_failure_keeps_the_key_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection failing after the retries raises the session's error, keyless."""

    def fail(*_args: object, **_kwargs: object) -> None:
        msg = f'Max retries exceeded with url: {URL_WITH_KEY}'
        raise exceptions.ConnectionError(msg)

    # In place of the request below the retrying session, which wraps the error.
    monkeypatch.setattr(requests.Session, 'request', fail)
    with pytest.raises(exceptions.ConnectionError, match='acs5 failed') as raised:
        ACSTableScraper(KEY).get_table('B25024', LEVEL_TRACT, ['06001'])
    error = raised.value
    shown = ''.join(
        traceback.TracebackException.from_exception(
            error, capture_locals=True
        ).format()
    )
    assert KEY not in shown
    assert error.__context__ is None


def test_request_lines_logged_by_the_http_library_are_redacted(
    serve: Callable[..., list[dict[str, Any]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """urllib3's request lines hide the key, and the filters are then removed."""
    # The stand-in writes the connection pool's line in its format and has Retry
    # write its own record; it sends nothing.
    pool_logger = logging.getLogger('urllib3.connectionpool')

    def log_and_answer(url: str, params: dict[str, str]) -> Response:
        """
        Log the request line as urllib3 does, then answer with the table.

        Args:
            url (str):
                The address of the release requested.
            params (dict[str, str]):
                The query parameters, key included.

        Returns:
            requests.Response:
                The table.
        """
        path = url.removeprefix('https://api.census.gov') + '?key=' + params['key']
        pool_logger.debug(
            '%s://%s:%s "%s %s %s" %s %s',
            'https',
            'api.census.gov',
            443,
            'GET',
            path,
            'HTTP/1.1',
            200,
            None,
        )
        Retry(total=1).increment('GET', path)
        return response()

    caplog.set_level(logging.DEBUG, logger='urllib3')
    serve(log_and_answer)
    ACSTableScraper(KEY).get_table('B25024', LEVEL_BLOCK_GROUP, ['06001'])
    assert caplog.text.count('/data/2024/acs/acs5?key=') == 2
    assert KEY not in caplog.text
    assert not pool_logger.filters
    assert not logging.getLogger('urllib3.util.retry').filters


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        (('25024', LEVEL_TRACT, ['06001']), 'table must be'),
        (('B25024', 'county', ['06001']), 'level must be'),
        (('B25024', LEVEL_TRACT, []), 'counties must hold'),
        (('B25024', LEVEL_TRACT, ['6001']), 'counties must hold'),
        (('B25024', LEVEL_TRACT, [6001]), 'counties must hold'),
        (('B25024', LEVEL_TRACT, ['06001', '06001']), "'06001' twice"),
    ],
)
def test_bad_arguments_raise_before_any_request(
    serve: Callable[..., list[dict[str, Any]]],
    arguments: tuple,
    match: str,
) -> None:
    """A malformed table, level or county list raises ValueError."""
    calls = serve()
    with pytest.raises(ValueError, match=match):
        ACSTableScraper(KEY).get_table(*arguments)
    assert calls == []


@pytest.mark.parametrize('vintage', ['2024', True, 2024.0])
def test_vintage_that_is_not_an_integer_raises_type_error(
    serve: Callable[..., list[dict[str, Any]]], vintage: object
) -> None:
    """A vintage given as a string, a bool or a float raises TypeError."""
    calls = serve()
    with pytest.raises(TypeError, match='vintage must be an integer year'):
        ACSTableScraper(KEY).get_table('B25024', LEVEL_TRACT, ['06001'], vintage)
    assert calls == []


def test_table_id_in_lower_case_is_read(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """A table id in lower case is read as the upper-case id."""
    calls = serve(response())
    ACSTableScraper(KEY).get_table('b25024', LEVEL_BLOCK_GROUP, ['06001'])
    assert calls[0]['params']['get'] == 'group(B25024)'


def test_check_key_sends_one_small_request_with_the_key(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """check_key asks the default release for the name of one state, with the key."""
    calls = serve(response(body=[['NAME', 'state'], ['Alabama', '01']]))
    ACSTableScraper(KEY).check_key()
    assert calls[0]['params'] == {'get': 'NAME', 'for': 'state:01', 'key': KEY}
    assert calls[0]['url'] == ACS5_URL.format(vintage=DEFAULT_ACS_VINTAGE)


def test_server_error_left_after_the_retries_raises_http_error(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """A 5xx answer left after the retries raises HTTPError naming the retries."""
    serve(response(status=503, body='busy'))
    with pytest.raises(exceptions.HTTPError, match='HTTP 503 after 3 retries'):
        ACSTableScraper(KEY).get_table('B25024', LEVEL_BLOCK_GROUP, ['06001'])


def test_missing_release_names_the_vintage(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """A 404 answer names the vintage asked and asks to check it."""
    serve(
        response(
            status=404,
            body='<html>not found</html>',
            url=f'https://api.census.gov/data/2030/acs/acs5?key={KEY}',
        )
    )
    with pytest.raises(exceptions.HTTPError, match='Check the vintage') as raised:
        ACSTableScraper(KEY).get_table('B25024', LEVEL_BLOCK_GROUP, ['06001'], 2030)
    assert 'of 2030' in str(raised.value)
    assert KEY not in str(raised.value)


def test_answer_that_is_not_the_table_asked_is_refused(
    serve: Callable[..., list[dict[str, Any]]],
) -> None:
    """A text cell or a row of another county raises RuntimeError."""
    header, second, first = BLOCK_GROUP_ROWS
    text = [header, second, [*first[:4], 'N', *first[5:]]]
    elsewhere = [header, second, [*first[:10], '32', '003', *first[12:]]]
    for body, match in (
        (text, 'not a number'),
        (elsewhere, 'another county'),
    ):
        serve(response(body=body))
        with pytest.raises(RuntimeError, match=match):
            ACSTableScraper(KEY).get_table('B25024', LEVEL_BLOCK_GROUP, ['06001'])


# A made-up key with a character no key holds, so that it is refused.
MALFORMED = 'fakeCensusKey000111222333444555666777a-'


def render_with_locals(error: BaseException) -> str:
    """
    Return an exception as a traceback shows it with every frame's values.

    Args:
        error (BaseException):
            The exception, whose cause and context are followed.

    Returns:
        str:
            The traceback, with the local values of each frame and the chain.
    """
    return ''.join(
        traceback.TracebackException.from_exception(
            error, capture_locals=True
        ).format()
    )


@pytest.mark.parametrize(
    ('failure', 'raised', 'match'),
    [
        (RuntimeError('socket blocked'), RuntimeError, 'RuntimeError during'),
        (KeyboardInterrupt(), KeyboardInterrupt, 'KeyboardInterrupt during'),
        (
            ReadTimeoutError(None, '/', 'Read timed out.'),
            exceptions.ConnectionError,
            r'\(ReadTimeoutError\).*pass a longer timeout',
        ),
    ],
    ids=['runtime_error', 'keyboard_interrupt', 'timeout'],
)
def test_failure_inside_urllib3_leaves_its_frames_behind(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    raised: type,
    match: str,
) -> None:
    """A failure below requests is raised afresh, its frames holding the key gone."""

    def make_request(
        _pool: HTTPConnectionPool,
        _conn: object,
        _method: str,
        url: str,
        **_kw: object,
    ) -> None:
        # The real stack of requests and urllib3 ran above, with the key in url.
        assert KEY in url
        raise failure

    monkeypatch.setattr(HTTPConnectionPool, '_make_request', make_request)
    with pytest.raises(raised, match=match) as error:
        ACSTableScraper(KEY, retries=0).get_table('B25024', LEVEL_TRACT, ['06001'])
    assert KEY not in render_with_locals(error.value)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None


def test_refused_key_hidden_in_every_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused key, given or from the environment, shows in no frame."""
    with pytest.raises(ValueError, match='api_key argument') as error:
        ACSTableScraper(MALFORMED)
    assert MALFORMED not in render_with_locals(error.value)
    monkeypatch.setenv(CENSUS_API_KEY_NAME, MALFORMED)
    with pytest.raises(ValueError, match='environment variable') as error:
        ACSTableScraper()
    assert MALFORMED not in render_with_locals(error.value)
