# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2026-11-20

This release adds a compiler for Hazus earthquake inventories. The compiler builds
the building inventory that a Hazus earthquake assessment reads, over several
possible worlds. The release also adds the scrapers, inferers and imputers that the
compiler runs. The README beside each module describes that module in full.

### Added
- **Hazus earthquake inventory compiler** (`HazusEQInventoryCompiler` and
  `CompilerInputs` in `brails.compilers`). It compiles a building inventory over
  several possible worlds from footprints, NSI records, census tables and mobile-home
  parks. A local inventory takes precedence over every other source. `compile`
  returns a `CompilationResult`, and `brails/compilers/hazus_earthquake/README.rst`
  states the run step by step. Where the inputs leave the block groups, the ACS
  tables or the mobile-home parks as `None`, the compiler reads them: the block
  groups from the TIGER/Line files, the ACS tables from the Census Data API and the
  parks from HIFLD. `seismic_zone`, the 1997 UBC zone of the study area, is required.
  The compiler draws structure types from the West Coast tables of Hazus. It
  therefore accepts only states of that region, within the limits that the section
  "Scope" of its README states. Unless the compiler or the inputs name the state, the
  compiler takes it from the block groups. In that case, it refuses inputs whose
  block groups lie in two states.
- **A compilation from a place name** (`CompilerInputs.from_place`,
  `HazusEQInventoryCompiler.compile_place`). The place can be a census place or
  county named with its state, an OpenStreetMap name, a region or a polygon. It gives
  every input of the area, with one Overture read. `CompilerInputs.boundary` keeps
  the buildings inside the place in `CompilationResult.inventory`, and
  `area_inventory` holds every building compiled.
- **The compiler's file and report.** `CompilationResult.write` writes every world,
  or the one that `world` names. With `integer_ids=True`, it writes the integer-id
  form that R2D reads. Beside the GeoJSON, it writes the report's summary, its
  exclusion rows and the manifest of the frozen stages. `detailed=True` adds the
  action rows and every log. A write removes the log files of an earlier write that
  it does not write again. In the report (`CompilationReport`), every building
  excluded from a world carries a diagnosis.
- **Pieces of the compiler that other compilers can use.** `StageFreeze`
  (`brails/utils/stage_freeze.py`) freezes each stage's output with a digest, and it
  raises when a frozen output changes. `brails.compilers` holds `PairRedraw`, which
  re-draws implausible pairs of occupancy class and story count, and `ComposedScreen`
  and `ResolutionLoop`.
- **Hazus earthquake inferers** (`brails/inferers/hazus_inferer_earthquake/`).
  `HazusEQStructureTypeInferer`, `HazusEQDesignLevelInferer` (with
  `assign_design_levels` and `apply_w1_carveout`) and `HazusEQHeightClassInferer`
  draw from tables of the Hazus Inventory Technical Manual. The tables do not print
  71 West Coast mid-rise and high-rise cells, and `complete_tables=True` fills them.
  A state is given by its full name.
- **Census and footprint inferers.** `ACSUnitBudget` places the ACS B25024 housing
  units of a block group on its footprints. `ACSYearBuiltSampler` and
  `YearBuiltSampler` draw construction years from ACS tables or around a tract
  median. `StoriesFromHeightInferer` converts a footprint height to a story count,
  with packaged scales for Overture's height sources.
- **Per-asset diagnoses** (`brails/inferers/inference_diagnosis.py`). Every inferer
  the compiler runs, except `ACSUnitBudget`, has `check()`, `draw()` and `infer()`.
  `check()` classifies each asset and names the features that would let the inferer
  produce a value. `infer()` runs both `check()` and `draw()`, and it names every
  failing asset.
- **Imputers.** `ObservedDonorImputer` copies a building's missing occupancy class,
  unit count and story count from one nearby observed donor per world.
  `ReplacementCostImputer` values every building as a rate per square foot times its
  gross area, and it keeps a local value. `ResidentialNeighborYearSampler` copies the
  year of a nearby residential building to a non-residential one.
- **U.S. Census scrapers and the Census API key.** `ACSTableScraper` scrapes ACS
  five-year tables per block group or tract, and `TIGERLineScraper` scrapes the block
  groups that overlap a region. `read_tiger_file` is also new.
  `CensusTractScraper.get_census_geographies` writes each asset's tract and
  block-group GEOIDs. `save_census_api_key` stores the key from a prompt that does
  not echo. `find_census_api_key` and `resolve_census_api_key` find the key in the
  environment variable `CENSUS_API_KEY`, a Google Colab secret or the file
  `~/.brails/census_api_key`.
- **Other scrapers.** `MobileHomeParkScraper` scrapes HIFLD mobile-home parks.
  `NSIFileReader` reads NSI records from a parquet table, such as one that
  `NSI_Parser.get_raw_data(raw_parquet_path=...)` writes. The new `simcenter_format`
  on `get_raw_data` maps NSI records to SimCenter names, among them `sqft` to
  `GrossArea`, `fullrep` to `ReplacementCost` and `med_yr_blt` to
  `YearBuiltTractMedian`. The legacy map is unchanged. `OvertureMapsFootprintScraper`
  gains `'local_path'` and `download`.
- **Footprints and asset ids.** `assign_asset_ids` keys footprints by the geohash of
  their centroid. The USA Structures, Microsoft, OpenStreetMap and Overture scrapers
  use it with `'id_scheme': 'geohash'`, but the default stays `'index'`. The geohash
  scheme also drops footprints under `min_area`, 450 sq ft by default.
  `StackedFootprintMerge` merges the stacked Overture footprints of one building.
  `NSIFootprintJoin` joins NSI points to footprints, and it drops none.
- **`AssetInventory`.** `merge` takes each feature from the first source, in an order
  of precedence, that holds a value. It records the source of each feature in
  `SourceString`, which `read_source_string` and `SOURCE_LABELS` read. The inventory
  also gains `validate_possible_worlds`, `add_world`, `has_area` and
  `centroid_and_plan_area`. For R2D's integer ids, `write_to_geojson` takes
  `integer_ids=True` and `read_from_geojson` takes `key_by_unique_id=True`.
  `write_to_geojson(asset_type=...)` writes assets without a `type`, in either id
  form. A new file member, `n_worlds`, gives the number of possible worlds.
  `write_to_geojson` writes it, and `read_from_geojson` reads it.
- **Regions.** `filter_to_region` (`brails/utils/region_filter.py`) keeps the assets
  inside a place name, a region or a polygon, with their possible worlds. It takes an
  inventory, a GeoJSON file or several files. `CensusBoundary` is a census place or
  county of the TIGER/Line files, looked up by name with its state. `GeometryRegion`
  is a region made from a Shapely polygon.
- **Inventory validation against DLML input schemas** (`brails/validators/`). A
  validator reports whether an inventory meets the input requirements of a method. It
  changes nothing.
- **Reproducible sampling** (`brails/utils/deterministic_sampling.py`). Per-asset
  uniforms are keyed, so a draw survives the reordering and subsetting of an
  inventory.
- **Example notebooks** in `examples/hazus_eq_compiler/`.
  `compile_city_inventory.ipynb` stores the Census API key and compiles Berkeley,
  California, with `compile_place`. It then writes the result and maps it.
  `filter_county_file_for_r2d.ipynb` cuts a county file of ten possible worlds to
  Berkeley with `filter_to_region` and takes one world. It writes that world for R2D
  with integer ids, the asset type `building` and the class `RES3` in place of
  `RES3A` to `RES3F`.

### Changed
- **Python 3.9 is no longer supported.** BRAILS++ requires Python 3.10, 3.11 or 3.12.
- **`simcenter-dlml` and `scipy` are core dependencies.**
- **BRAILS reads the records of the NSI 2026 Base only.**
- **`HazusInfererEarthquake`, `HazusInferer` and `SimCenterInferer` take
  `grossArea_key` in place of `planArea_key`**, because the replacement cost needs a
  gross area. The new key defaults to `GrossArea`, and passing `planArea_key` raises
  `TypeError`. An inventory that holds only `PlanArea` therefore gets no
  `ReplacementCost` and a warning, unless the caller passes
  `grossArea_key='PlanArea'`. `HazusInfererWind` keeps `planArea_key`.
- **The Hazus earthquake rulesets moved** from
  `brails/inferers/hazus_inferer/hazus_rulesets.py` to
  `brails/inferers/hazus_inferer_earthquake/hazus_rulesets.py`. Code that imports
  them from the old module needs the new path.
- **`is_inventory` moved to `InputValidator`** and is no longer on
  `InventoryValidator`. Call `InputValidator.is_inventory`.
- **Footprint assets carry their provider.** Every footprint scraper writes
  `footprint_provider`. USA Structures assets also carry `usa_build_id` and
  `usa_uuid`, and OpenStreetMap assets carry `osm_id`. `KnnImputer` leaves
  `footprint_provider`, these three ids and `overture_id` out of its predictors. The
  predictors for an existing Overture inventory therefore change, and
  `exclude_provider_ids=False` restores them.
- **`OvertureMapsFootprintScraper` reads Overture's `building` type only.** Building
  parts therefore no longer become assets, and `building_id` no longer appears. The
  scraper derives from `FootprintScraper`, as the other footprint scrapers do.
- **GeoJSON files of several possible worlds.** `write_to_geojson` writes `n_worlds`.
  It raises `ValueError` where `n_pw` is not an integer of at least 1, and it refuses
  a list that holds a number of values other than `n_pw`. On reading,
  `read_from_geojson` refuses a list whose length differs from the file's `n_worlds`.
  In a file without `n_worlds`, it refuses lists of different lengths. An inventory
  that already holds assets refuses a file of another number of worlds.
  `all_elevations` is a vector feature by default (`DEFAULT_VECTOR_FEATURES`), so its
  list is one value, not one per world.
- **`write_to_geojson` and `HousingUnitInventory.to_json` write NaN as `null`**, and
  they raise on an infinite value. `write_to_geojson` also writes a temporary file
  and then replaces the output with it. The output is therefore a new file, not the
  old one written over, and its directory must be writable.
- **`get_world_realization`** returns the world of a single-world inventory for any
  index, instead of raising. It copies coordinates and the housing-unit inventory,
  and it copies list and dict values one level deep. Other values are shared with the
  source inventory. A realization has `n_pw` 1, so `get_dataframe` accepts one that
  holds a vector feature such as `HousingUnits`.
- **`CensusTractScraper.get_census_tracts` raises** for an asset outside every tract,
  a client error or exhausted retries. It previously printed an error and returned
  the tracts fetched so far. When it raises, it writes nothing. A footprint is
  located by the centroid of its exterior rings. An asset on an edge shared by two
  tracts takes the lower GEOID.

### Fixed
- **`read_from_geojson` reads the possible worlds of a file without `n_worlds`** from
  its lists. It previously left `n_pw` at 1. When it takes a number above 1 this way,
  it prints a line naming that number.
- **NSI points stand at their records' `x` and `y`**, which are not rounded. They
  previously stood at the point NSI serves, which is rounded to six decimals and
  could fall across a footprint's edge. `GetNSIData` and
  `get_filtered_data_given_inventory` also match footprints by `x` and `y`. A record
  without `x` or `y` keeps the served point.
- **NSI records sharing a coordinate are kept.** Records are keyed by `fd_id`, so a
  mixed-use structure keeps one record per occupancy. A pull can therefore return
  more assets than before, several of them at one coordinate.
- **`BasicPointsToPolygonsAllocator` buffers in ground meters**, in a projected CRS
  estimated from the polygons. Before, a `buffer_dist` of 10 m reached 7.91 m in San
  Francisco. Candidates are ranked before they are taken, so matching does not depend
  on order.
- **Overture releases resolve against the S3 bucket** instead of the published list,
  because that list names releases the bucket no longer holds.
- **`OvertureMapsFootprintScraper.get_footprints` returns an empty inventory** for a
  region without buildings. It previously raised `KeyError`.
- **NSI merge coordinates.** Before the fix, every matched building recorded the
  coordinates of the last footprint processed.
- The NSI basement flag decode was inverted.
- The Hazus state region mapping lacked Washington and the Northern Mariana Islands.
  Its key `Virgin Islands` is now `U.S. Virgin Islands`, as in the weight tables.

---

## [4.3.0] - 2026-05-23

### Added
- **Housing unit summarizer** (`PyncodaHousingUnitSummarizer`), which aggregates household-level data produced by pyncoda into building-level attributes: population, vacancy, tenure, and demographics such as income, race, and social characteristics. Vacant units are excluded from statistics and missing values serialize cleanly to JSON.
- **Inventory validators framework** with the first concrete implementation, `HazusEarthquakeValidator`, for checking and repairing building inventories against Hazus earthquake requirements. `InventoryValidator` became an abstract base with a method for fixing inventories.
- **User-supplied housing unit counts:** the pyncoda allocator accepts an optional `unit_count_col`, so per-building counts can be given directly instead of inferred from occupancy type.
- Mirror fallback and a `BRAILSv2` User-Agent for the Overpass API, plus a `safe_overpass_json` helper and a `get_safe_json` helper now used by `region_boundary` and the OSM footprint scraper.

### Changed
- Python is capped at 3.12 for torch compatibility, `torchaudio` is pinned to fix Google Colab installs, and the required NumPy version was updated.
- `is_inventory` became a static method, with clearer `TypeError` reporting in the validator base class.
- Improved handling of varied and multi-polygon footprint geometries, and of missing `input_dict` values during initialization.
- Updated DOI and contributor list.

### Fixed
- `parse_geojson_geometry` strips altitude and extra dimensions, so GeoJSON files with 3D positions no longer fail coordinate validation and lose geometry.
- A vertex-order bug in `is_box`, and a regression where `None` headers were rejected.
- A bug in the `gpt_base_model` utilities.

---

## [4.2.0] - 2025-12-03

### Added
- **Housing unit allocation** (`PyncodaHousingUnitAllocator`), a model that assigns households to building inventories, built on pyncoda by Nathanael Rosenheim (Texas A&M). Python scripts and notebooks are in `examples/housing_units/`.
- Methods for merging asset inventories and for spatially aggregating point features onto them (`brails/aggregators/points_to_polygons`).
- Reading an `AssetInventory` from an existing GeoJSON file.
- `AssetInventory.get_all_asset_features`, which simplifies data retrieval.

### Changed
- Type hints across types, scrapers, and aggregators modernized to Python 3.9 syntax.
- `ruff format` applied to selected files.
- The legacy `runTests` workflow was deprecated in favor of the `tests` workflow, and project imports were modernized.

### Fixed
- Error handling in the NSI scraper, and multi-geometry support in `get_geojson`.
- GeoJSON attribute extraction in `read_from_geojson`, and MultiPolygon handling in `extract_aerial_imagery`.

---

## [4.1.4] - 2025-09-25

### Added
- Scrapers for Land Cover, Land Use, and FIRM datasets.
- BRAILS++ examples and API documentation in the public-facing documentation.

### Changed
- Updated the supported Python versions.

---

## [4.1.3] - 2025-08-20

### Added
- A USGS elevation service scraper, and RAPIDTools functionality integrated into BRAILS++.
- Overture Maps enhancements, including footprint scrapers.
- An `ImageSet` plotting method, notebook examples for image downloads, and examples for extracting aerial imagery from orthomosaic data.
- Centroid calculation for `Asset` objects, a `bbox2poly` method, and Shapely-to-BRAILS++ geometry conversion utilities.
- Power network inventory support with sample data.

### Changed
- Constants and unit defaults refactored, with added precision options.
- `remove_features` made more flexible, and `ImageSet.set_directory` more stable.
- `rasterio` added to the requirements.
- Extensive docstring work for Sphinx, doctest examples, better example organization, and updated Sphinx settings and Autosummary references.

### Fixed
- The NTI API endpoint, the geometry validators, and plan area unit conversion.

---

## [4.1.1] - 2025-07-16

### Fixed
- Bug fixes for the inferers.

---

## [4.1.0] - 2025-06-06

### Added
- Scrapers for ASCE hazard data, OSM power network inventories, and Overture Maps footprints.
- Expanded multi-hazard inference for hurricane wind, flood, and earthquake features.
- An image-based module for assessing structural damage from panoramic street-level imagery.
- Spatial join strategies for building inventories from diverse geospatial sources.
- Example notebooks for the image classifiers (roof shape, year built, foundation elevation), the GPT and VLM classifiers, `ChimneyDetector`, `FacadeParser`, street-level damage detection, and inventory creation pipelines.

### Changed
- PEP 561 compliance for static type checking, and more consistent internal data structures.

---

## [4.0.0] - 2024-10-01

### Added
- A new modular architecture.
- Attribute prediction modules built on CLIP, GroundingDINO, SAM, and GPT-4 vision language models.
- Imputers that fill data gaps in generated inventories.
