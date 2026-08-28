# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Work toward Hazus earthquake inventory compilation: a validator subsystem that
checks an inventory against published input schemas, the first pieces of the
earthquake inference chain, and the reproducible sampling the chain draws from.

### Added
- **Inventory validation against DLML input schemas** (`brails/validators/`). A validator answers whether an inventory satisfies the input requirements of a method, and changes nothing; repairing an inventory is a separate concern. `simcenter-dlml` is now a core dependency.
- **Possible-world validation** (`AssetInventory.validate_possible_worlds`), which checks that every feature is either a scalar shared by all worlds or a list holding exactly `n_pw` realizations, and honors vector-valued features.
- **SimCenter conventions for raw NSI data** (`brails/scrapers/nsi_parser/nsi_conventions.py`). Maps raw NSI records to SimCenter feature names and decodes the packed occupancy code. `get_raw_data` gains a keyword-only `simcenter_format` flag, off by default.
- **Hazus earthquake design levels** (`HazusEQDesignLevelInferer`, with `assign_design_levels` and `apply_w1_carveout`). The W1 carve-out is a separate function because the manual applies it after the structure-type draw.
- **Earthquake ruleset getters** for the Hazus Inventory Manual Appendix A sampling bands, the Table 5-1 height class of each structure type, and general-type membership, each carrying the source table in its docstring.
- **Reproducible sampling helpers** (`brails/utils/deterministic_sampling.py`): keyed per-asset uniforms and an exact truncated normal. A draw is a function of the asset's identity, so it survives reordering and subsetting of an inventory.
- **`YearBuiltSampler`** (`brails/inferers/year_built_sampler/`), which draws a per-building construction year around a tract median. Each sampler generates its own seed unless given one and records it, so runs differ but any run can be repeated.

### Changed
- **`med_yr_blt` from NSI now maps to `YearBuiltTractMedian`.** It is the median year built of the structures in a Census tract, so every building in a tract carries the same value. The converted path no longer emits `YearBuilt` at all. The legacy `nsi2brailsmap` path is unchanged and still maps `med_yr_blt` to `YearBuilt`, so the two conventions coexist for now.
- The Hazus earthquake rulesets moved into `brails/inferers/hazus_inferer_earthquake/`. Legacy callers keep working through updated imports.
- `is_inventory` moved to `InputValidator`, joining the family of checks it belongs to.
- `scipy` is now a declared dependency. It was arriving only transitively through scikit-learn.

### Fixed
- **Point-to-polygon allocation** (`BasicPointsToPolygonsAllocator`). `buffer_dist` was applied in EPSG:3857, whose units exceed ground meters by 1/cos(latitude), so a nominal 10 m buffer reached 7.91 m in San Francisco. Buffering now happens in a projected CRS estimated from the polygons. Candidates are also ranked before they are taken, by containment and true distance with ids breaking ties, which makes matching deterministic under reordering.
- **Overture release resolution.** The default release came from the published release YAML, which names releases the S3 bucket no longer holds and typically trails it by months. Releases now resolve against the bucket.
- **NSI merge coordinates.** The merge loop rebuilt every matched footprint's coordinate list from a variable left over by an earlier loop, so every matched building recorded the coordinates of the last footprint processed.
- The NSI basement flag decode was inverted, now it is parsed correctly.
- The Hazus state region mapping was missing Washington and the Northern Mariana Islands; now it is comprehensive.

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
