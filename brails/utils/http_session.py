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
Open a ``requests`` session that tries a failed request again.

.. autosummary::

      address_without_query
      create_retrying_session
      get_whole_body
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter, Retry
from urllib3.exceptions import MaxRetryError, NewConnectionError
from urllib3.exceptions import TimeoutError as Urllib3Timeout

if TYPE_CHECKING:
    from collections.abc import Mapping

# Answers a request is tried again after: too many requests and server errors.
RETRIED_STATUSES = (429, 500, 502, 503, 504)

# The longest wait, in seconds, a ``Retry-After`` header can ask for.
MAX_RETRY_AFTER_S = 120


def address_without_query(url: str) -> str:
    """
    Return a URL without its query string, fragment and user information.

    A message about a request names its URL this way, so that a key sent in the query
    does not appear in it.

    Args:
        url (str):
            The URL, which may carry a key in its query string.

    Returns:
        str:
            The scheme, host, port and path of ``url``.
    """
    parts = urlsplit(url)
    host = parts.netloc.rpartition('@')[2]
    return parts._replace(netloc=host, query='', fragment='').geturl()


class _BoundedRetry(Retry):
    """urllib3's ``Retry``, with a ``Retry-After`` wait cut to the bound."""

    def parse_retry_after(self, retry_after: str) -> float:
        """
        Return the wait a ``Retry-After`` header asks for, at most the bound.

        Args:
            retry_after (str):
                The header's value, seconds or an HTTP date.

        Returns:
            float:
                The wait in seconds, at most :data:`MAX_RETRY_AFTER_S`.
        """
        return min(super().parse_retry_after(retry_after), MAX_RETRY_AFTER_S)


class _RetryingSession(requests.Session):
    """A session whose ``ConnectionError`` names the URL without its query string."""

    def __init__(self, retries: int) -> None:
        """
        Mount urllib3's ``Retry`` for GET over HTTPS.

        Args:
            retries (int):
                How many times a failed request is tried again.
        """
        super().__init__()
        self.retries = retries
        retry = _BoundedRetry(
            total=retries,
            backoff_factor=1,
            status_forcelist=RETRIED_STATUSES,
            allowed_methods=['GET'],
            raise_on_status=False,
        )
        self.mount('https://', HTTPAdapter(max_retries=retry))

    def request(
        self, method: str, url: str, *args: object, **kwargs: object
    ) -> requests.Response:
        """
        Send a request as ``requests.Session.request`` does.

        Args:
            method (str):
                The HTTP method.
            url (str):
                The URL, whose query string the error message leaves out.
            *args (object):
                Further arguments of ``requests.Session.request``.
            **kwargs (object):
                Further keyword arguments of ``requests.Session.request``.

        Returns:
            requests.Response:
                The last answer.

        Raises:
            requests.ConnectionError:
                If no answer came, after the retries where urllib3 ran out of them,
                with neither the query string nor the exception of ``requests``
                attached. The message names the reason, such as ``ReadTimeoutError``,
                and for a timeout advises a longer one.
        """
        try:
            response = super().request(method, url, *args, **kwargs)
        except requests.ConnectionError as error:
            # requests wraps urllib3's error, whose reason says what happened.
            cause = error.args[0] if error.args else None
            retried = isinstance(cause, MaxRetryError)
            reason = cause.reason if retried else cause
            if not isinstance(reason, BaseException):
                reason = error
            failure = type(reason).__name__
            # urllib3 derives a refused connection from its connect timeout.
            timed_out = isinstance(
                reason, (Urllib3Timeout, requests.Timeout)
            ) and not isinstance(reason, NewConnectionError)
            # urllib3's error names the URL with its query string; no frame keeps it.
            del cause, reason
        else:
            return response
        # Raised outside the except clause, so that the exception of requests, whose
        # message holds the query string, is not attached as the context.
        after = f' after {self.retries} retries' if retried else ''
        advice = (
            'The server did not answer in time; try again or pass a longer timeout.'
            if timed_out
            else 'Check the connection and try again.'
        )
        msg = (
            f'The {method} request to {address_without_query(url)} failed{after} '
            f'({failure}). {advice}'
        )
        raise requests.ConnectionError(msg)


def create_retrying_session(retries: int) -> requests.Session:
    """
    Return a session whose GET requests over HTTPS are tried again after a failure.

    It serves a download or a request whose answer the caller judges itself, and
    retries silently, where :func:`~brails.utils.safe_get_json.safe_get_json` parses
    a JSON answer and prints a line for each try. A request that fails to connect,
    times out, or gets one of :data:`RETRIED_STATUSES` is tried again up to
    ``retries`` times, after the wait the answer's ``Retry-After`` header asks for,
    up to :data:`MAX_RETRY_AFTER_S`, or else after urllib3's exponential backoff with
    a factor of 1. urllib3 also retries a 413 answer that carries ``Retry-After``.
    When the tries run out, the last answer is returned whatever its status, for the
    caller to judge. A connection that keeps failing raises
    ``requests.ConnectionError``, whose message gives the URL as
    :func:`address_without_query` does and names urllib3's reason, and to which the
    exception of ``requests`` is not attached. A body cut short is not tried again
    and is the caller's to detect.

    Args:
        retries (int):
            How many times a failed request is tried again, 0 or more.

    Returns:
        requests.Session:
            The session, which the caller closes, for example with ``with``.

    Raises:
        ValueError:
            If ``retries`` is not an integer of at least 0.
    """
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        msg = f'retries must be an integer of at least 0, not {retries!r}.'
        raise ValueError(msg)
    return _RetryingSession(retries)


def get_whole_body(
    session: requests.Session,
    url: str,
    timeout: float,
    *,
    params: Mapping[str, str] | None = None,
) -> requests.Response:
    """
    Send a GET request and return the answer once its whole body has arrived.

    A body is cut short when ``requests`` raises ``ChunkedEncodingError`` or
    ``ContentDecodingError``, or when an answer without ``Content-Encoding`` holds
    fewer bytes than its ``Content-Length`` states, which urllib3 1.26 does not
    check. The message names the URL as :func:`address_without_query` gives it, so a
    key in the query stays out of it. The answer's status is the caller's to judge.

    Args:
        session (requests.Session):
            The session the request is sent through, such as one
            :func:`create_retrying_session` opens.
        url (str):
            The URL requested, without the query given as ``params``.
        timeout (float):
            Seconds to wait for the server to answer.
        params (Mapping[str, str], optional):
            The query, as ``requests`` takes it. Defaults to none.

    Returns:
        requests.Response:
            The answer, with its whole body.

    Raises:
        requests.ConnectionError:
            If the body is cut short, with advice to try again and no exception of
            ``requests`` attached, or as the session raises it.
    """
    try:
        response = session.get(url, params=params, timeout=timeout)
    except (
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ContentDecodingError,
    ) as error:
        cut_short = type(error).__name__
    else:
        stated = response.headers.get('Content-Length', '')
        cut_short = (
            f'{len(response.content)} of {stated} bytes'
            if stated.isdigit()
            and 'Content-Encoding' not in response.headers
            and len(response.content) < int(stated)
            else ''
        )
    # Raised outside the except clause, so that no exception of requests is attached
    # as the context.
    if cut_short:
        msg = (
            f'The answer from {address_without_query(url)} was cut short '
            f'({cut_short}). Check the connection and try again.'
        )
        raise requests.ConnectionError(msg)
    return response
