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

"""Tests for brails.utils.census_api_key."""

from __future__ import annotations

import getpass
import os
import stat
import sys
import traceback
import types
import warnings
from typing import TYPE_CHECKING

import pytest
import requests

from brails.utils.census_api_key import (
    CENSUS_API_KEY_FILE,
    CENSUS_API_KEY_NAME,
    CENSUS_API_KEY_SIGNUP_URL,
    CensusApiKey,
    find_census_api_key,
    resolve_census_api_key,
    save_census_api_key,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

COLAB_MODULE = 'google.colab.userdata'


pytestmark = pytest.mark.usefixtures('no_network')


def save_typed(monkeypatch: pytest.MonkeyPatch, answer: str) -> Path:
    """
    Run save_census_api_key with an answer typed at its prompt.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Replaces the prompt for the test.
        answer (str):
            The text the prompt returns.

    Returns:
        pathlib.Path:
            The path save_census_api_key returns.
    """
    monkeypatch.setattr(getpass, 'getpass', lambda _prompt: answer)
    return save_census_api_key()


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """
    Give the test an empty home directory, no key variable and no Colab.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Patches the environment and the module table.
        tmp_path (pathlib.Path):
            The folder used as the home directory.

    Returns:
        pathlib.Path:
            The home directory.
    """
    # Path.home reads HOME on POSIX systems and USERPROFILE on Windows.
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    monkeypatch.delenv(CENSUS_API_KEY_NAME, raising=False)
    # A None entry makes the import of the Colab module fail.
    monkeypatch.setitem(sys.modules, COLAB_MODULE, None)
    return tmp_path


@pytest.fixture
def colab(monkeypatch: pytest.MonkeyPatch) -> Callable[..., types.ModuleType]:
    """
    Return a factory that installs a stand-in for ``google.colab.userdata``.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Puts the stand-in in the module table for the test.

    Returns:
        Callable[..., types.ModuleType]:
            A callable taking the value ``get`` returns, or the name of the exception
            it raises, and returning the installed module.
    """

    def install(
        value: str | None = None, error: str | None = None
    ) -> types.ModuleType:
        """
        Install a Colab secrets module whose ``get`` returns or raises.

        Args:
            value (str, optional):
                The secret's value, which ``get`` returns.
            error (str, optional):
                The name of the module's exception ``get`` raises instead.

        Returns:
            types.ModuleType:
                The stand-in module.
        """
        module = types.ModuleType(COLAB_MODULE)
        module.SecretNotFoundError = type('SecretNotFoundError', (Exception,), {})
        module.NotebookAccessError = type('NotebookAccessError', (Exception,), {})
        module.TimeoutException = type('TimeoutException', (Exception,), {})

        def get(name: str) -> str | None:
            """
            Return the secret, or raise the exception chosen.

            Args:
                name (str):
                    The secret's name.

            Returns:
                str | None:
                    The secret's value.

            Raises:
                Exception:
                    The module's exception named by ``error``.
            """
            assert name == CENSUS_API_KEY_NAME
            if error is not None:
                raise getattr(module, error)
            return value

        module.get = get
        monkeypatch.setitem(sys.modules, COLAB_MODULE, module)
        return module

    return install


@pytest.mark.usefixtures('home')
def test_argument_takes_precedence_over_every_store(
    colab: Callable[..., types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key given as the argument wins over the variable, Colab and the file."""
    save_typed(monkeypatch, 'fromfile')
    monkeypatch.setenv(CENSUS_API_KEY_NAME, 'fromenvironment')
    colab(value='fromcolab')
    assert resolve_census_api_key(' fromargument ') == (
        'fromargument',
        'the api_key argument',
    )


@pytest.mark.usefixtures('home')
def test_environment_takes_precedence_over_colab_and_file(
    colab: Callable[..., types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an argument the environment variable wins over Colab and the file."""
    save_typed(monkeypatch, 'fromfile')
    monkeypatch.setenv(CENSUS_API_KEY_NAME, 'fromenvironment\n')
    colab(value='fromcolab')
    assert resolve_census_api_key()[0] == 'fromenvironment'


@pytest.mark.usefixtures('home')
def test_colab_secret_takes_precedence_over_file(
    colab: Callable[..., types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Colab secret wins over the file, and saving another key warns of it."""
    colab(value='fromcolab')
    with pytest.warns(UserWarning, match='Google Colab secret'):
        save_typed(monkeypatch, 'fromfile')
    assert resolve_census_api_key()[0] == 'fromcolab'


@pytest.mark.usefixtures('home')
@pytest.mark.parametrize(
    'error', ['SecretNotFoundError', 'NotebookAccessError', 'TimeoutException']
)
def test_colab_without_secret_or_access_falls_back_to_file(
    colab: Callable[..., types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    error: str,
) -> None:
    """A Colab secret that is missing, not granted or timed out counts as no key."""
    colab(error=error)
    save_typed(monkeypatch, 'fromfile')
    assert resolve_census_api_key()[0] == 'fromfile'


def test_file_is_read_without_colab(home: Path) -> None:
    """Outside Colab and without a variable the file gives the key, after its BOM."""
    (home / CENSUS_API_KEY_FILE).parent.mkdir()
    (home / CENSUS_API_KEY_FILE).write_text('\ufeff  fromfile\n', encoding='utf-8')
    assert resolve_census_api_key()[0] == 'fromfile'


def test_empty_values_count_as_no_key(
    home: Path,
    colab: Callable[..., types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty argument, variable and secret pass on, and the file is named."""
    monkeypatch.setenv(CENSUS_API_KEY_NAME, '  ')
    colab(value='')
    save_typed(monkeypatch, 'fromfile')
    key, source = resolve_census_api_key('')
    assert key == 'fromfile'
    assert source == f'the file {home / CENSUS_API_KEY_FILE}'


def test_no_key_names_the_stores_and_the_signup_page(
    home: Path, colab: Callable[..., types.ModuleType]
) -> None:
    """With no key anywhere the error names every store and the signup page."""
    colab(error='SecretNotFoundError')
    (home / CENSUS_API_KEY_FILE).parent.mkdir()
    (home / CENSUS_API_KEY_FILE).write_text('\n', encoding='utf-8')
    with pytest.raises(LookupError) as raised:
        resolve_census_api_key()
    message = str(raised.value)
    assert CENSUS_API_KEY_SIGNUP_URL in message
    assert f'environment variable {CENSUS_API_KEY_NAME}' in message
    assert 'Google Colab secret' in message
    assert str(home / CENSUS_API_KEY_FILE) in message


@pytest.mark.usefixtures('home')
def test_argument_of_another_type_is_refused() -> None:
    """An argument that is not a string or None raises TypeError."""
    with pytest.raises(TypeError, match='api_key must be a string or None'):
        resolve_census_api_key(12345)


def test_save_creates_the_folder_and_returns_the_path(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving creates the folder and writes the stripped key with a newline."""
    path = save_typed(monkeypatch, ' "savedkey" ')
    assert path == home / CENSUS_API_KEY_FILE
    assert path.read_bytes() == b'savedkey\n'
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]


@pytest.mark.usefixtures('home')
def test_save_replaces_an_existing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second save replaces the key the first one wrote."""
    save_typed(monkeypatch, 'firstkey')
    save_typed(monkeypatch, 'secondkey')
    assert resolve_census_api_key()[0] == 'secondkey'


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permissions only')
def test_saved_file_is_private_to_its_owner(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file has mode 600, also when it existed with a wider mode."""
    path = home / CENSUS_API_KEY_FILE
    path.parent.mkdir()
    path.write_text('oldkey\n', encoding='utf-8')
    path.chmod(0o644)
    save_typed(monkeypatch, 'newkey')
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.usefixtures('home')
@pytest.mark.skipif(os.name == 'nt', reason='POSIX permissions only')
def test_saved_folder_is_private_to_its_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A folder the save creates has mode 700."""
    path = save_typed(monkeypatch, 'newkey')
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ('value', 'match'),
    [
        ('', 'No key was typed'),
        ('   ', 'No key was typed'),
        ('two words', 'letters and digits only'),
        ('key+with/symbols', 'letters and digits only'),
    ],
)
def test_save_refuses_a_value_that_is_not_a_key(
    home: Path, monkeypatch: pytest.MonkeyPatch, value: str, match: str
) -> None:
    """An empty or impossible answer raises ValueError without showing it."""
    with pytest.raises(ValueError, match=match) as raised:
        save_typed(monkeypatch, value)
    if value.strip():
        assert value not in str(raised.value)
    assert not (home / CENSUS_API_KEY_FILE).exists()


@pytest.mark.usefixtures('home')
def test_save_warns_when_the_environment_holds_another_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different key in the variable, which is read first, gives a warning."""
    monkeypatch.setenv(CENSUS_API_KEY_NAME, 'environmentkey')
    with pytest.warns(UserWarning, match=CENSUS_API_KEY_NAME) as caught:
        save_typed(monkeypatch, 'filekey')
    message = str(caught[0].message)
    assert 'environmentkey' not in message
    assert 'filekey' not in message
    monkeypatch.setenv(CENSUS_API_KEY_NAME, 'filekey')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        save_typed(monkeypatch, 'filekey')


@pytest.mark.usefixtures('home')
def test_find_names_the_store(
    colab: Callable[..., types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """find_census_api_key names the store of the key it returns."""
    save_typed(monkeypatch, 'fromfile')
    assert find_census_api_key()[1].startswith('the file ')
    colab(value='fromcolab')
    assert find_census_api_key() == (
        'fromcolab',
        f'the Google Colab secret {CENSUS_API_KEY_NAME}',
    )
    monkeypatch.setenv(CENSUS_API_KEY_NAME, "'fromenvironment'")
    assert find_census_api_key() == (
        'fromenvironment',
        f'the environment variable {CENSUS_API_KEY_NAME}',
    )


@pytest.mark.usefixtures('home')
def test_a_value_no_key_can_be_is_refused_naming_its_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored value with characters no key holds raises, naming the store only."""
    monkeypatch.setenv(CENSUS_API_KEY_NAME, 'not a/key')
    with pytest.raises(ValueError, match='environment variable') as raised:
        resolve_census_api_key()
    assert 'not a/key' not in str(raised.value)
    with pytest.raises(ValueError, match='api_key argument'):
        resolve_census_api_key('key=with=signs')


@pytest.mark.usefixtures('home')
def test_no_home_directory_skips_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Where no home directory can be found, the file store is skipped."""

    def no_home() -> Path:
        """
        Raise as Path.home does where no home directory can be found.

        Raises:
            RuntimeError:
                Always.
        """
        raise RuntimeError

    monkeypatch.setattr('pathlib.Path.home', no_home)
    with pytest.raises(LookupError, match='No Census Data API key'):
        resolve_census_api_key()


def test_save_reads_the_key_with_getpass_under_its_prompt(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key is read with getpass under the prompt 'Census API key: '."""
    prompts = []

    def answer(prompt: str) -> str:
        """
        Record the prompt and return a key.

        Args:
            prompt (str):
                The text shown before the answer.

        Returns:
            str:
                The key typed.
        """
        prompts.append(prompt)
        return 'typedkey'

    monkeypatch.setattr(getpass, 'getpass', answer)
    save_census_api_key()
    assert prompts == ['Census API key: ']
    assert (home / CENSUS_API_KEY_FILE).read_bytes() == b'typedkey\n'


def test_save_refuses_a_console_that_echoes(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where echo cannot be turned off, RuntimeError is raised before any reading."""
    read = []

    def echoing(_prompt: str) -> str:
        """
        Warn as getpass does where it cannot hide the answer, then read it.

        Args:
            _prompt (str):
                The text shown before the answer.

        Returns:
            str:
                The key typed, read with echo.
        """
        warnings.warn('Can not control echo.', getpass.GetPassWarning, stacklevel=2)
        read.append(True)
        return 'echoedkey'

    monkeypatch.setattr(getpass, 'getpass', echoing)
    with pytest.raises(RuntimeError, match='cannot hide what is typed'):
        save_census_api_key()
    assert read == []
    assert not (home / CENSUS_API_KEY_FILE).exists()


def test_failed_save_leaves_no_partial_file(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A save that cannot replace the file removes its partial file."""

    def refuse(*_args: object) -> None:
        raise PermissionError

    monkeypatch.setattr('pathlib.Path.replace', refuse)
    with pytest.raises(PermissionError):
        save_typed(monkeypatch, 'newkey')
    assert list((home / CENSUS_API_KEY_FILE).parent.iterdir()) == []


def test_save_warns_of_a_variable_that_cannot_be_a_key(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A variable no key can be, read before the file, gives a warning."""
    monkeypatch.setenv(CENSUS_API_KEY_NAME, 'not a/key')
    with pytest.warns(UserWarning, match='environment variable.*correct or remove'):
        save_typed(monkeypatch, 'filekey')
    assert (home / CENSUS_API_KEY_FILE).read_bytes() == b'filekey\n'


def test_key_hidden_by_repr_and_sent_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key found hides from repr and containers, and requests sends it whole."""
    made_up = 'notarealcensuskey0123456789abcdef'
    monkeypatch.setenv(CENSUS_API_KEY_NAME, made_up)
    key, _ = find_census_api_key()
    assert isinstance(key, CensusApiKey)
    assert key == made_up
    assert isinstance(resolve_census_api_key(key)[0], CensusApiKey)
    for shown in (repr(key), repr([key]), str({'key': key}), f'{key!r}'):
        assert made_up not in shown
    request = requests.Request(
        'GET', 'https://api.census.gov/data', params={'key': key}
    ).prepare()
    assert request.url == f'https://api.census.gov/data?key={made_up}'


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


@pytest.mark.usefixtures('home')
def test_refused_key_hidden_in_every_frame(
    colab: Callable[..., types.ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused key, as argument, variable, secret or answer, shows in no frame."""
    # The made-up key stays a global, so that no frame of the test holds it.
    monkeypatch.setattr(getpass, 'getpass', lambda _prompt: MALFORMED)
    for call in (lambda: resolve_census_api_key(MALFORMED), save_census_api_key):
        with pytest.raises(
            ValueError, match='cannot be a Census Data API key'
        ) as error:
            call()
        assert MALFORMED not in render_with_locals(error.value)
    monkeypatch.setenv(CENSUS_API_KEY_NAME, MALFORMED)
    with pytest.raises(ValueError, match='environment variable') as error:
        find_census_api_key()
    assert MALFORMED not in render_with_locals(error.value)
    monkeypatch.delenv(CENSUS_API_KEY_NAME)
    colab(value=MALFORMED)
    with pytest.raises(ValueError, match='Colab secret') as error:
        find_census_api_key()
    assert MALFORMED not in render_with_locals(error.value)
