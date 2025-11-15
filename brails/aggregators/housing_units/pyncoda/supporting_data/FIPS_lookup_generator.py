# noqa: INP001
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
# Adam Zsarnoczay

"""
This script generates the 'fips_lookup.json' file.

It fetches a national FIPS code file from a URL, reads it using the csv module,
filters it for county and state-level codes, and saves the result
as a clean JSON lookup.
"""

import csv
import json
from pathlib import Path


def create_fips_lookup(csv_filepath: str, json_filepath: str) -> None:
    """
    Reads Census FIPS geocodes and creates a lookup table for states and counties.

    The input file can be obtained from the Census Bureau's official website:
    https://www2.census.gov/programs-surveys/popest/geographies/2020/all-geocodes-v2020.xlsx

    Args:
        csv_filepath (str): The path to the input CSV file.
        json_filepath (str): The path where the output JSON file will be saved.
    """
    fips_lookup = {}

    print(f"Reading data from '{csv_filepath}'...")

    try:
        with Path(csv_filepath).open(encoding='utf-8-sig') as infile:
            # Use DictReader to easily access columns by name
            reader = csv.DictReader(infile)

            print('Extracting data...')

            for row in reader:
                summary_level = row.get('Summary Level', '').strip()
                state_code = row.get('State Code (FIPS)', '').strip()
                county_code = row.get('County Code (FIPS)', '').strip()
                area_name = row.get(
                    'Area Name (including legal/statistical area description)', ''
                ).strip()

                # Rule for State: Summary Level is '040'
                if summary_level == '040':
                    print(f'State: {area_name}')
                    # Ensure state code is valid before adding
                    if state_code and state_code != '00':
                        fips_lookup[state_code] = area_name

                # Rule for County: Summary Level is '050'
                elif summary_level == '050':
                    # Ensure state and county codes are valid before adding
                    if (
                        state_code
                        and state_code != '00'
                        and county_code
                        and county_code != '000'
                    ):
                        # Create the combined 5-digit FIPS code for the county
                        county_fips_key = f'{state_code}{county_code}'
                        fips_lookup[county_fips_key] = area_name

    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        return
    except (csv.Error, OSError) as e:
        print(f'An error occurred: {e}')
        return

    print(json.dumps(fips_lookup, indent=2, ensure_ascii=False))

    # Write the extracted data to a JSON file
    try:
        with Path(json_filepath).open(mode='w', encoding='utf-8') as outfile:
            # Use indent for a readable, pretty-printed JSON output
            json.dump(fips_lookup, outfile, indent=2, ensure_ascii=False)
        print(f"Successfully created JSON lookup table at '{json_filepath}'")
    except (OSError, TypeError) as e:
        print(f'An error occurred while writing the JSON file: {e}')


if __name__ == '__main__':
    # --- Configuration ---
    # Change these file paths to match your input and desired output locations
    input_csv_file = 'all-geocodes-v2020.csv'
    output_json_file = 'fips_lookup.json'

    # Run the function
    create_fips_lookup(input_csv_file, output_json_file)
