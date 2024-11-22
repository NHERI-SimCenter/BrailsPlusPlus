#!/bin/bash
# written: fmk 11/24

#
# Function to run a Python script and check if it was successful
#

# Initialize a counter variable
counter=1

run_python_script() {
    
    script_name=$1
    shift  # Remove the first argument so "$@" contains only the arguments for the Python script
    echo "Running $script_name with arguments: $@"

    stdout_log="output_${counter}.out"
    stderr_log="error_${counter}.out"
    
    python3 "$script_name" "$@" >"$stdout_log" 2>"$stderr_log"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: $script_name $@  STD_ERROR saved in $stderr_log file"
    else
        echo "SUCCESS: $script_name $@."
    fi
    
    # Increment the counter for the next script
    ((counter++))
}

#
# add now run through the tests
#

cd examples

echo "TESTING IMPORTER"
cd importer
run_python_script classic.py OSM_FootprintScraper  "Tiburon, CA" 
run_python_script importer.py OSM_FootprintScraper "Tiburon, CA" 
cd ..

echo "TESTING FOOTPRINT"
cd footprint
run_python_script brails_footprint.py "Tiburon, CA" 
cd ..

echo "TESTING IMAGE_DOWNLOADS"
cd image_downloads
run_python_script brails_download_images.py OSM_FootprintScraper "Tiburon, CA" > runTest.out
cd ..

echo "TESTING NSI_INTEGRATION"
cd nsi_integration
run_python_script brails_nsi.py OSM_FootprintScraper "Tiburon, CA" 
run_python_script brails_nsi.py USA_FootprintScraper "Tiburon, CA" 
cd ..

echo "TESTING FILTERS"
cd image_filters
run_python_script brails_filters.py OSM_FootprintScraper "Tiburon, CA" 
cd ..

echo "TESTING PROCESSORS"
cd image_processor
run_python_script brails_classifier.py USA_FootprintScraper "Larkspur, CA" NFloorDetector
cd ..

echo "TESTING IMPUTATION"
cd imputation
run_python_script imputation.py OSM_FootprintScraper "Tiburon, CA" 
cd ..

cd ..



