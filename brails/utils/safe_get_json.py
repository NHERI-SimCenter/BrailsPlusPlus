# Copyright (c) 2025 The Regents of the University of California
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
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
#    Written fmk, based on code by Adam Zsarnoczay in us_census/census_tract_scraper
#

import time
import sys
import requests
from requests import exceptions
from json import JSONDecodeError

def safe_get_json(
    url: str,
    params: dict = None,
    headers: dict = None,
    timeout: float = 10.0,
    retries: int = 3,
    backoff_factor: float = 2.0,
    valid_key: str = None,
):
    """
    A function to safely get JSON return from a http request
      - does retries, checks http connection and other errors, checks return valid JSON

    Args:
        URL (str): the url
        params (dict): params to call
        headers (dict): header to pass
        timeout (float): timeout
        retries (int): retry attempts before failure return
        backoff_factor (float): change retry delay each time by this factor
        valid_key: (str): a valid key to look for in response, default None
    
    Returns:
       JSON - 

    Raises:
       RuntimeError or HTTPError (or RequestException) on unrecoverable failure.
    """
    delay = 1.0  # initial delay in seconds
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            raw = response.text.strip()
            if not raw:
                raise RuntimeError(f"Empty response body from {url}")

            try:
                data = response.json()
            except JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON from {url}. Response: {raw}"
                ) from e

            if valid_key is not None:
                val = data.get(valid_key)
                if not val:
                    raise RuntimeError(
                        f"JSON from {url} missing or empty '{valid_key}'"
                    )

            return data  # success

        except (exceptions.ConnectionError, exceptions.Timeout) as e:
            print(
                f"WARNING: Network error on attempt {attempt}/{retries} for {url}: {type(e).__name__}. "
                f"Retrying after {delay:.1f}s ...",
                file=sys.stderr
            )
        except exceptions.HTTPError as e:
            status = e.response.status_code
            if 500 <= status < 600:
                print(
                    f"WARNING: Server error {status} on attempt {attempt}/{retries} for {url}. "
                    f"Retrying after {delay:.1f}s ...",
                    file=sys.stderr
                )
            else:
                print(
                    f"ERROR: Client error {status} for {url}. Aborting.",
                    file=sys.stderr
                )
                raise
        except RuntimeError as e:
            print(
                f"WARNING: Bad response on attempt {attempt}/{retries} for {url}: {e}",
                file=sys.stderr
            )

        time.sleep(delay)
        delay *= backoff_factor

    # All retries failed
    raise RuntimeError(f"Failed to get valid JSON from {url} after {retries} attempts")
