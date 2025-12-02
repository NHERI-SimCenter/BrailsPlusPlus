import argparse
from pathlib import Path


def main(location: str):
    """
    # Inventory Generation with Probabilistic Housing Units

    This example demonstrates the **complete, end-to-end workflow** for generating a building
    inventory enriched with household-level demographic data.

    **Goal:**
    Starting from a simple location name (e.g., "Tiburon, CA"), we will:
    1.  Build a geometric inventory of building footprints.
    2.  Enrich it with physical attributes (e.g., year built, stories).
    3.  Populate residential buildings with probabilistic households (income, race, size)
        statistically matched to the local US Census demographics.

    **Scope & Limitations:**
    - **USA Only:** This workflow relies on US-specific datasets (NSI, US Census).
      It is not robust for international locations.
    - **Service Stability:** Public data APIs (NSI, Overture, Census) occasionally experience
      downtime. If you encounter connection errors, please wait a few minutes and try again.
    - **Not Hazus-Ready:** While rich, this inventory is not yet formatted for a full
      Hazus damage assessment. For Hazus-specific inferencing, please see the scripts
      in `examples/inventory_creation`.

    **Flexibility:**
    While this script builds an inventory from scratch using BRAILS tools, you can bring
    your own! If you already have a GeoJSON building inventory with the required columns
    (Occupancy, PlanArea, Stories), you can simply load it using `AssetInventory.read_from_geojson()`
    and skip directly to **Step 6** to generate households.
    """

    """
    ## 1. Define the Region of Interest

    First, we define the geographic area for our analysis.
    The `RegionBoundary` class allows you to specify a region by its name (e.g., "Berkeley, CA")
    or by a bounding box coordinate tuple.

    **Options:**
    - To use a specific bounding box, change `type` to `'locationPolygon'` and provide
      a tuple of `(min_longitude, min_latitude, max_longitude, max_latitude)` as `data`.
    """
    from brails.types import RegionBoundary

    region_boundary_object = RegionBoundary(
        {'type': 'locationName', 'data': location}
    )

    geometry, description, osm_id = region_boundary_object.get_boundary()

    """
    ## 2. Download Building Footprints (Geometry)

    We fetch the physical shapes of **building footprints** (polygons) from Overture Maps.
    Overture Maps combines data from open sources (OpenStreetMap) and commercial donors
    (Microsoft, Google, Meta) to provide high-quality building footprints.

    **Why we do this:**
    We strictly use this for the *geometry* of the buildings. We remove all other
    metadata to demonstrate how to build a rich inventory from scratch using
    just these raw shapes.

    **Alternative Providers:**
    There is no single "best" footprint provider for every location. Depending on the
    region, one source may be more complete or recent than another. We recommend
    inspecting the results from different providers:
    - `OSM_FootprintScraper`: Fetches data from OpenStreetMap.
    - `USA_FootprintScraper`: Fetches data from the **FEMA USA Structures** database.
    """
    from brails.scrapers import OvertureMapsFootprintScraper

    bldg_inventory = OvertureMapsFootprintScraper({'length': 'ft'}).get_footprints(
        region_boundary_object
    )

    # only use the footprint geometry and remove all other metadata for clarity
    bldg_inventory.remove_features(bldg_inventory.get_all_asset_features())

    """
    ## 3. Download Building Attributes (Data)

    Next, we fetch detailed building attributes from the **National Structure Inventory (NSI)**.
    NSI is a US Army Corps of Engineers dataset designed for hazard modeling (e.g., flood damage).
    It integrates data from tax assessors, CoreLogic, and Census demographics.

    **Important Caveats:**
    - **Point Data:** NSI data is point-based, meaning each building is represented by a single
      coordinate, not a shape.
    - **Inferred Data:** Many attributes (like foundation type or number of stories) are
      often **imputed or inferred** based on regional averages rather than direct observation.
      It is a statistical model, not ground truth.

    **Feature Selection:**
    We map NSI attributes to standard SimCenter naming conventions:
    - `found_type` -> `FoundationType`: (e.g., Slab, Crawlspace) Critical for flood vulnerability.
    - `found_ht` -> `FirstFloorElevation`: Height of the first floor above ground.
    - `sqft` -> `PlanArea`: The footprint area (often derived from tax records).
    - `num_story` -> `NumberOfStories`: Inferred from height or tax data.
    - `occtype` -> `OccupancyClass`: (e.g., RES1, COM1) The primary use of the building.
    - `pop2*` -> `Population`: Estimated day/night occupancy based on Census blocks.
    - `med_yr_blt` -> `YearBuilt`: Median year built for the area.
    - `students` -> `StudentPopulation`: Estimated student counts for schools.
    - `bldgtype` -> `BuildingType`: (e.g., Wood, Masonry) Structural material type.
    """
    from brails.scrapers import NSI_Parser

    nsi_points = NSI_Parser().get_raw_data(region_boundary_object)

    # Rename the features we need later using standard SimCenter labels
    feature_rename_map = {
        'fd_id': 'fd_id',
        'type': 'type',
        'bldgtype': 'BuildingType',
        'found_type': 'FoundationType',
        'found_ht': 'FirstFloorElevation',
        'pop2amu65': 'NightPopulationUnder65',
        'pop2amo65': 'NightPopulationOver65',
        'pop2pmu65': 'DayPopulationUnder65',
        'pop2pmo65': 'DayPopulationOver65',
        'x': 'Longitude',
        'y': 'Latitude',
        'sqft': 'PlanArea',
        'num_story': 'NumberOfStories',
        'students': 'StudentPopulation',
        'med_yr_blt': 'YearBuilt',
        'occtype': 'OccupancyClass',
    }
    nsi_points.change_feature_names(feature_rename_map)

    # remove all other features for clarity
    features_to_remove = nsi_points.get_all_asset_features()
    features_to_remove.difference_update(feature_rename_map.values())
    nsi_points.remove_features(features_to_remove)

    """
    ## 4. Merge Attributes with Geometry

    We now have two datasets:
    1. **Polygons** (Shapes) from Overture Maps.
    2. **Points** (Data) from NSI.

    The `BasicPointsToPolygonsAllocator` spatially joins these sets. It assigns the attributes
    from an NSI point to the Overture footprint that contains it.

    **How it Works:**
    1. **Strict Inclusion:** By default, a point must fall strictly inside a polygon.
    2. **Convex Hull (`use_convex_hull=True`):** Complex building shapes (like U-shapes)
       can cause valid centroids to fall "outside" the polygon. We use the convex hull
       (the "shrink wrap" shape) to capture these points.
    3. **Buffer (`buffer_dist=10.0`):** We add a **10-meter** tolerance buffer to catch points that
       might be slightly offset due to GPS errors or different data vintages.

    **Caveat:**
    This is a strict spatial join. NSI points that do not land within a footprint (even
    with the buffer) will be **dropped**. Conversely, footprints that contain no NSI points
    will remain in the inventory but will have no attribute data (missing features).
    """
    from brails.aggregators import BasicPointsToPolygonsAllocator

    BasicPointsToPolygonsAllocator(
        polygon_inventory=bldg_inventory,
        point_inventory=nsi_points
    ).allocate(
        use_convex_hull=True,
        buffer_dist=10.0,
    )

    """
    ## 5. Impute Missing Data

    After the merge, our inventory is incomplete. Some footprints had no matching NSI point,
    leaving them with missing attributes (NaN). Even successfully matched buildings might
    have gaps in the original NSI data.

    The `KnnImputer` fills these gaps using a **K-Nearest Neighbors** approach. It assumes that
    nearby buildings are likely to be similar (e.g., neighbors often have similar
    year-built dates or story counts).

    **Key Options:**
    - `n_possible_worlds`: How many different imputed "versions" of the inventory to generate.
      We use `1` for a single deterministic best-guess.
    - `exclude_features`: Lists columns (like `Latitude`, `Longitude`, `id`) that should *not*
      be imputed because they are unique or spatial constants.
    - `k_nn`: The number of neighbors to consider (default is 5).
    - `create_correlation`: When True, imputation is performed sequentially in batches.
      The first batch of imputed buildings becomes part of the "truth" for the next batch,
      ensuring that neighborhoods develop consistent, spatially correlated characteristics
      rather than random noise.
    """
    from brails.imputers import KnnImputer

    bldg_inventory_imputed = KnnImputer(
        bldg_inventory,
        n_possible_worlds=1,
        exclude_features=['Latitude', 'Longitude', 'fd_id', 'id'],
    ).impute()

    """
    ## 6. Generate Probabilistic Households

    This step populates residential buildings with probabilistic households.
    The `PyncodaHousingUnitAllocator` acts as a bridge to the **`pyncoda`** library.

    **What does `pyncoda` do?**
    It downloads US Census demographic data and generates a probabilistic population
    that statistically matches the local census block. It then assigns those households
    to the residential buildings in our inventory.

    **Configuration:**
    - `vintage`: Which Census year to use ('2010' or '2020').
    - `key_features`: Tells `pyncoda` which columns in *our* inventory correspond to the
      building traits it needs (Occupancy, Area, Stories) to make intelligent assignments.
    - `seed`: Ensures the random generation is reproducible.
    - `clean_work_dir`: When `True`, creates a fresh working environment. Set to `False`
      to reuse downloaded Census data for faster re-runs on the same location.
    """
    from brails.aggregators import PyncodaHousingUnitAllocator

    # Define a working directory for pyncoda intermediate files
    work_dir = Path.cwd() / 'pyncoda_working_dir'

    PyncodaHousingUnitAllocator(
        inventory=bldg_inventory_imputed,
        vintage='2020',
        seed=9878,
        key_features=dict(
            occupancy_col='OccupancyClass',
            plan_area_col='PlanArea',
            story_count_col='NumberOfStories',
            length_unit='ft',
        ),
        work_dir=str(work_dir),
        clean_work_dir=True,
    ).allocate()

    """
    ## 7. Export Results

    Finally, we save our work to local files:
    - **`bldg_inventory.geojson`**: The physical building inventory. Buildings with
      assigned households will contain a `HousingUnits` field with a list of IDs.
    - **`housing_units.json`**: The demographic inventory. This file contains detailed
      attributes (e.g., `Ownership`, `Race`, `IncomeGroup`) for every household ID.

    These two files are linked relationally by these IDs.
    """
    # Buildings
    bldg_inventory_imputed.write_to_geojson('bldg_inventory.geojson')

    # Housing Units
    bldg_inventory_imputed.housing_unit_inventory.to_json('housing_units.json')

    print(f'SUCCESS. Files saved to: {Path.cwd()}')


if __name__ == '__main__':
    # Set up command-line arguments:
    parser = argparse.ArgumentParser(
        description='Get a household inventory for the specified location.'
    )
    parser.add_argument(
        'location',
        type=str,
        nargs='?',
        default='Tiburon, CA',
        help='Name of the location to analyze.',
    )

    # Parse the command-line arguments:
    args = parser.parse_args()
    print(args)

    # Run the main function
    main(args.location)
