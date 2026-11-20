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


"""Tests for the requests session that tries a failed request again."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import MaxRetryError, NewConnectionError, ReadTimeoutError

from brails.utils.http_session import (
    MAX_RETRY_AFTER_S,
    RETRIED_STATUSES,
    address_without_query,
    create_retrying_session,
    get_whole_body,
)

pytestmark = pytest.mark.usefixtures('no_network')


def test_https_requests_retried() -> None:
    """GET over HTTPS is retried after those answers, the last one returned."""
    with create_retrying_session(4) as session:
        retry = session.get_adapter('https://www2.census.gov/').max_retries
    assert retry.total == 4
    assert retry.backoff_factor == 1
    assert set(retry.status_forcelist) == set(RETRIED_STATUSES)
    assert set(retry.allowed_methods) == {'GET'}
    assert retry.raise_on_status is False
    assert retry.respect_retry_after_header is True


def test_retry_after_bounded() -> None:
    """A Retry-After wait longer than the bound is cut to it, a shorter one kept."""
    with create_retrying_session(1) as session:
        retry = session.get_adapter('https://www2.census.gov/').max_retries
    assert retry.parse_retry_after('21600') == MAX_RETRY_AFTER_S
    assert retry.parse_retry_after('5') == 5
    assert retry.new(total=0).parse_retry_after('900') == MAX_RETRY_AFTER_S
    answer = SimpleNamespace(headers={'Retry-After': '21600'})
    assert retry.get_retry_after(answer) == MAX_RETRY_AFTER_S


@pytest.mark.parametrize('retries', [None, -1, True, 2.0, '3'])
def test_retries_checked(retries: object) -> None:
    """A count of retries that is not an integer of at least 0 is refused."""
    with pytest.raises(ValueError, match='retries must be an integer of at least 0'):
        create_retrying_session(retries)


@pytest.mark.parametrize(
    ('reason', 'retried', 'expected'),
    [
        (
            NewConnectionError(None, 'refused'),
            True,
            'failed after 2 retries (NewConnectionError). Check the connection and '
            'try again.',
        ),
        (
            ReadTimeoutError(None, '/data', 'Read timed out.'),
            True,
            'failed after 2 retries (ReadTimeoutError). The server did not '
            'answer in time; try again or pass a longer timeout.',
        ),
        (
            ReadTimeoutError(None, '/data', 'Read timed out.'),
            False,
            'failed (ReadTimeoutError). The server did not answer in time; try '
            'again or pass a longer timeout.',
        ),
    ],
    ids=['refused', 'timed_out', 'body_timed_out'],
)
def test_connection_error_names_its_reason_without_the_query(
    monkeypatch: pytest.MonkeyPatch,
    reason: Exception,
    retried: bool,  # noqa: FBT001
    expected: str,
) -> None:
    """A failed connection names its reason and the retries, never the query."""

    def send(_adapter: HTTPAdapter, request: object, **_kwargs: object) -> None:
        cause = MaxRetryError(None, request.path_url, reason) if retried else reason
        raise requests.ConnectionError(cause)

    monkeypatch.setattr(HTTPAdapter, 'send', send)
    with (
        create_retrying_session(2) as session,
        pytest.raises(requests.ConnectionError) as error,
    ):
        session.get('https://api.example.org/data?get=NAME', params={'key': 'K7'})
    assert str(error.value) == (
        f'The GET request to https://api.example.org/data {expected}'
    )
    assert error.value.__context__ is None
    assert error.value.__cause__ is None


def test_address_without_query() -> None:
    """The query, the fragment and the user information are left out."""
    url = 'https://user:K7@api.example.org:8443/data/x?get=NAME&key=K7#part'
    assert address_without_query(url) == 'https://api.example.org:8443/data/x'


def answer_of(body: bytes, headers: dict[str, str]) -> requests.Response:
    """
    Return an answer with a body and headers.

    Args:
        body (bytes):
            The body.
        headers (dict[str, str]):
            The headers.

    Returns:
        requests.Response:
            The answer, with status 200.
    """
    answer = requests.Response()
    answer.status_code = 200
    answer._content = body
    answer.headers.update(headers)
    return answer


@pytest.mark.parametrize(
    ('headers', 'cut_short'),
    [
        ({'Content-Length': '10'}, '4 of 10 bytes'),
        ({'Content-Length': '10', 'Content-Encoding': 'gzip'}, ''),
        ({'Content-Length': '4'}, ''),
        ({}, ''),
    ],
)
def test_body_shorter_than_its_length_is_cut_short(
    headers: dict[str, str], cut_short: str
) -> None:
    """An unencoded body shorter than its Content-Length is reported cut short."""
    session = SimpleNamespace(get=lambda *_a, **_k: answer_of(b'abcd', headers))
    if not cut_short:
        assert get_whole_body(session, 'https://x.org/f.zip', 9).content == b'abcd'
        return
    with pytest.raises(requests.ConnectionError) as error:
        get_whole_body(session, 'https://x.org/f.zip', 9)
    assert str(error.value) == (
        f'The answer from https://x.org/f.zip was cut short ({cut_short}). Check '
        'the connection and try again.'
    )
    assert error.value.__context__ is None


@pytest.mark.parametrize(
    'failure',
    [
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ContentDecodingError,
    ],
)
def test_body_that_breaks_off_is_cut_short(failure: type) -> None:
    """A body requests cannot finish is cut short, named without the query."""

    def get(*_args: object, **_kwargs: object) -> None:
        raise failure

    with pytest.raises(requests.ConnectionError, match=failure.__name__) as error:
        get_whole_body(
            SimpleNamespace(get=get),
            'https://x.org/f?get=NAME',
            9,
            params={'key': 'K7'},
        )
    assert str(error.value).startswith('The answer from https://x.org/f was cut')
    assert error.value.__context__ is None
