import csv
import json

def create_fips_lookup(csv_filepath, json_filepath):
    """
    Reads a Census FIPS geocodes CSV and creates a JSON lookup table
    for states and counties.

    The input file can be obtained from the Census Bureau's official website:
    https://www2.census.gov/programs-surveys/popest/geographies/2020/all-geocodes-v2020.xlsx

    Args:
        csv_filepath (str): The path to the input CSV file.
        json_filepath (str): The path where the output JSON file will be saved.
    """
    fips_lookup = {}

    print(f"Reading data from '{csv_filepath}'...")

    try:
        with open(csv_filepath, mode='r', encoding='utf-8-sig') as infile:
            # Use DictReader to easily access columns by name
            reader = csv.DictReader(infile)

            print("Extracting data...")

            for row in reader:
                summary_level = row.get('Summary Level', '').strip()
                state_code = row.get('State Code (FIPS)', '').strip()
                county_code = row.get('County Code (FIPS)', '').strip()
                area_name = row.get('Area Name (including legal/statistical area description)', '').strip()

                # Rule for State: Summary Level is '040'
                if summary_level == '040':
                    print(f"State: {area_name}")
                    # Ensure state code is valid before adding
                    if state_code and state_code != '00':
                        fips_lookup[state_code] = area_name

                # Rule for County: Summary Level is '050'
                elif summary_level == '050':
                    # Ensure state and county codes are valid before adding
                    if state_code and state_code != '00' and county_code and county_code != '000':
                        # Create the combined 5-digit FIPS code for the county
                        county_fips_key = f"{state_code}{county_code}"
                        fips_lookup[county_fips_key] = area_name

    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print(json.dumps(fips_lookup, indent=2, ensure_ascii=False))

    # Write the extracted data to a JSON file
    try:
        with open(json_filepath, mode='w', encoding='utf-8') as outfile:
            # Use indent for a readable, pretty-printed JSON output
            json.dump(fips_lookup, outfile, indent=2, ensure_ascii=False)
        print(f"Successfully created JSON lookup table at '{json_filepath}'")
    except Exception as e:
        print(f"An error occurred while writing the JSON file: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # Change these file paths to match your input and desired output locations
    input_csv_file = 'all-geocodes-v2020.csv'
    output_json_file = 'fips_lookup.json'

    # Run the function
    create_fips_lookup(input_csv_file, output_json_file)