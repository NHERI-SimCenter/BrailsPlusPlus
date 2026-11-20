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
Find and store the key of the Census Bureau's Data API.

No routine here writes the key to a message, a warning or a log record, and the key
is returned as a :class:`CensusApiKey`, whose ``repr`` hides it.

.. autosummary::

      CensusApiKey
      resolve_census_api_key
      find_census_api_key
      save_census_api_key
"""

from __future__ import annotations

import contextlib
import getpass
import importlib
import os
import warnings
from pathlib import Path

from brails.constants import REDACTED

# The name of the environment variable, and of the Google Colab secret, that holds
# the key.
CENSUS_API_KEY_NAME = 'CENSUS_API_KEY'

# The file that holds the key, relative to the user's home directory.
CENSUS_API_KEY_FILE = Path('.brails') / 'census_api_key'

# The page where the Census Bureau issues keys.
CENSUS_API_KEY_SIGNUP_URL = 'https://api.census.gov/data/key_signup.html'

# The names of the Colab exceptions that mean the notebook gets no secret: it does
# not exist, the notebook has no access to it, or it was requested outside the Colab
# interface or not granted in time.
_COLAB_NO_SECRET = ('SecretNotFoundError', 'NotebookAccessError', 'TimeoutException')


class CensusApiKey(str):
    """
    A Census Data API key, which its ``repr`` does not show.

    It is the key's text in every other use, so ``requests`` sends it as given. A
    traceback that shows the values of a frame holding it, or a container holding it
    printed whole, such as the query of a request, shows
    :data:`~brails.constants.REDACTED` in its place. The routines here turn a key's
    text into one where they receive it, also a text that is then refused.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """
        Return a placeholder in place of the key.

        Returns:
            str:
                :data:`~brails.constants.REDACTED`.
        """
        return REDACTED


def _clean_key(value: str, source: str) -> CensusApiKey:
    """
    Return a key without surrounding whitespace and quotes, or refuse it.

    Args:
        value (str):
            The text read from a store, given as an argument or typed at the prompt.
        source (str):
            Where the text came from, in words, for the message.

    Returns:
        CensusApiKey:
            The key, empty where ``value`` holds none.

    Raises:
        ValueError:
            If what remains holds a character other than an ASCII letter or digit,
            which no Census Data API key holds. The message names the source and not
            the value.
    """
    # Every holder of the text is a CensusApiKey, so a frame shown with its values
    # hides it, a text that is refused below included.
    value = CensusApiKey(value)
    key = CensusApiKey(value.strip())
    if len(key) > 1 and key[0] == key[-1] and key[0] in '\'"':
        key = CensusApiKey(key[1:-1].strip())
    if key and not (key.isascii() and key.isalnum()):
        msg = (
            f'The value from {source} cannot be a Census Data API key, which '
            'holds letters and digits only.'
        )
        raise ValueError(msg)
    return CensusApiKey(key)


def resolve_census_api_key(
    api_key: str | None = None,
) -> tuple[CensusApiKey, str]:
    """
    Return the Census Data API key and its source, from the argument or a store.

    A key given as ``api_key`` is returned. Otherwise the key is the one
    :func:`find_census_api_key` finds. Surrounding whitespace and quotes are removed
    from the argument, and an empty argument reads the stores.

    Args:
        api_key (str, optional):
            A key that takes precedence over every store. Defaults to ``None``, which
            reads the stores.

    Returns:
        tuple[CensusApiKey, str]:
            The key, and its source in words for messages: the argument, or the store
            :func:`find_census_api_key` names.

    Raises:
        TypeError:
            If ``api_key`` is neither a string nor ``None``.
        ValueError:
            If the argument or a store holds a value that cannot be a key.
        LookupError:
            If ``api_key`` is not given or empty and no store holds a key, as
            :func:`find_census_api_key` raises it.
    """
    if api_key is not None and not isinstance(api_key, str):
        msg = f'api_key must be a string or None, not {type(api_key).__name__}.'
        raise TypeError(msg)
    if api_key is not None:
        api_key = CensusApiKey(api_key)
    source = 'the api_key argument'
    key = _clean_key(api_key or '', source)
    return (key, source) if key else find_census_api_key()


def find_census_api_key() -> tuple[CensusApiKey, str]:
    """
    Return the Census Data API key of the first store that holds one.

    The stores are read in this order: the environment variable
    :data:`CENSUS_API_KEY_NAME`; in Google Colab, the notebook secret of that name;
    and the file :data:`CENSUS_API_KEY_FILE` under the home directory, which
    :func:`save_census_api_key` writes. Surrounding whitespace and quotes are removed
    from a value, and an empty value counts as no key. A Colab secret that is
    missing, not granted or timed out counts as no key, and outside Colab no secret
    is read. Without a home directory no file is read.

    Returns:
        tuple[CensusApiKey, str]:
            The key, and its source in words for messages, naming the variable, the
            secret or the file.

    Raises:
        ValueError:
            If a store holds a value that cannot be a key.
        LookupError:
            If no store holds a key. The message names the stores and the page where
            the Census Bureau issues keys.
    """
    source = f'the environment variable {CENSUS_API_KEY_NAME}'
    key = _clean_key(os.environ.get(CENSUS_API_KEY_NAME, ''), source)
    if key:
        return key, source

    # Exists only in a Colab runtime.
    try:
        userdata = importlib.import_module('google.colab.userdata')
    except ImportError:
        userdata = None
    if userdata is not None:
        source = f'the Google Colab secret {CENSUS_API_KEY_NAME}'
        no_secret = tuple(
            getattr(userdata, name)
            for name in _COLAB_NO_SECRET
            if hasattr(userdata, name)
        )
        try:
            value = CensusApiKey(userdata.get(CENSUS_API_KEY_NAME) or '')
        except no_secret:
            value = ''
        key = _clean_key(value, source)
        if key:
            return key, source

    try:
        path = Path.home() / CENSUS_API_KEY_FILE
    except RuntimeError:
        path = None
    if path is not None and path.is_file():
        source = f'the file {path}'
        # utf-8-sig drops the byte-order mark some editors write.
        key = _clean_key(path.read_text(encoding='utf-8-sig'), source)
        if key:
            return key, source

    location = (
        f'the file {path}'
        if path
        else f'the file {CENSUS_API_KEY_FILE} under the home directory'
    )
    msg = (
        'No Census Data API key was found. Request one at '
        f'{CENSUS_API_KEY_SIGNUP_URL} and activate it from the email the Census '
        f'Bureau sends. Then set the environment variable {CENSUS_API_KEY_NAME}, '
        f'add a Google Colab secret named {CENSUS_API_KEY_NAME} that the notebook '
        f'can read, or run save_census_api_key() to write it to {location}.'
    )
    raise LookupError(msg)


def save_census_api_key() -> Path:
    """
    Ask for the Census Data API key and write it to the key file.

    The prompt does not show what is typed. The file is :data:`CENSUS_API_KEY_FILE`
    under the home directory, read by :func:`find_census_api_key` when neither the
    environment nor a Colab secret holds a key. When one of them holds a different
    key, or a value that cannot be a key, a warning says that it is read before the
    file. An existing file is replaced in one step, so a failure leaves the old file
    or the new one. On systems with POSIX permissions the file is readable and
    writable by its owner only (mode 600), and a folder that does not exist is
    created with mode 700. On Windows these modes are not applied, and the file
    inherits the permissions of its folder.

    Returns:
        pathlib.Path:
            The path of the file written.

    Raises:
        RuntimeError:
            If the console cannot hide what is typed. Nothing is read then.
        ValueError:
            If the answer is empty or holds a character other than an ASCII letter or
            digit. The message does not show the answer.
    """
    # Where getpass cannot turn echo off, it warns and then reads with echo. Raising
    # the warning as an error stops it before the read.
    with warnings.catch_warnings():
        warnings.simplefilter('error', getpass.GetPassWarning)
        try:
            answer = CensusApiKey(getpass.getpass('Census API key: '))
        except getpass.GetPassWarning:
            msg = (
                'This console cannot hide what is typed, so no key was asked for. '
                'Run save_census_api_key in a terminal or a notebook, or set the '
                f'environment variable {CENSUS_API_KEY_NAME}.'
            )
            raise RuntimeError(msg) from None
    key = _clean_key(answer, 'the prompt')
    if not key:
        msg = (
            'No key was typed, so none was saved. Run save_census_api_key() again '
            'and type the key.'
        )
        raise ValueError(msg)

    path = Path.home() / CENSUS_API_KEY_FILE
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # The key goes to a file of mode 600 beside the target, which then replaces the
    # target. O_BINARY and newline='\n' write a bare newline on every system.
    partial = path.with_name(f'{path.name}.partial')
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_BINARY', 0)
    try:
        descriptor = os.open(partial, flags, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as file:
            partial.chmod(0o600)
            file.write(key + '\n')
        partial.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
        raise

    # The file just written holds a key, so the stores raise no LookupError.
    try:
        found, source = find_census_api_key()
    except ValueError as error:
        warnings.warn(
            f'{error} It is read before {path}, so correct or remove it.',
            stacklevel=2,
        )
        return path
    if found != key:
        warnings.warn(
            f'A different key is in {source}, which is read before {path}. Remove '
            'that key or replace it with the new one.',
            stacklevel=2,
        )
    return path
