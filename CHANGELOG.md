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
- **Hazus earthquake design levels** (`HazusEQDesignLevelInferer`, with `assign_design_levels` and `apply_w1_carveout`). `check()` diagnoses every asset, `draw()` assumes they all passed, and `infer()` runs both and names every failing asset. The W1 carve-out is a separate function because the manual applies it after the structure-type draw. The state is given by its full name, as in `'California'`, which the Hazus region table validates; an unrecognized name raises.
- **Earthquake ruleset getters** for the Hazus Inventory Manual Appendix A sampling bands, the Table 5-1 height class of each structure type, and general-type membership, each carrying the source table in its docstring.
- **Reproducible sampling helpers** (`brails/utils/deterministic_sampling.py`): keyed per-asset uniforms and an exact truncated normal. A draw is a function of the asset's identity, so it survives reordering and subsetting of an inventory.
- **`YearBuiltSampler`** (`brails/inferers/year_built_sampler/`), which draws a per-building construction year around a tract median. Each sampler generates its own seed unless given one and records it, so runs differ but any run can be repeated.
- **`HazusEQDesignLevelInferer` refuses a boolean year and a year beyond the integer cast.** A digit string is read as the year it spells, and a whole float as the year it equals.
- **`HazusEQStructureTypeInferer`** (`brails/inferers/hazus_inferer_earthquake/`), which draws a Hazus structure type from the Inventory Manual Appendix A distributions. An observed general building type narrows the candidate types before the draw, as does a design level outside the set its region admits for URM, C3 and S5, and the remaining weights are renormalized. Constructing one for American Samoa, Guam or the Northern Mariana Islands raises, since the tables carry no single-family distribution for them.
- **Story readers in the earthquake rulesets** (`hazus_rulesets.py`). `get_hazus_story_count` reads a story count and `get_hazus_sampling_height_band` gives the Appendix A sampling band of one, and `HazusEQStructureTypeInferer` reads stories through both. The rulesets also take `UNREINFORCED_MASONRY_TYPES` from the inferer module, which still exports it, and gain `WEST_COAST`, the region group whose tables are stratified by era. A `README.rst` beside them states which manual editions the tables come from.
- **Completed West Coast structure type tables** (`hazus_table_completion.py` in `brails/inferers/hazus_inferer_earthquake/`). The Appendix A tables print no row for 71 West Coast mid-rise and high-rise cells. The module fills each from printed rows and lists the occupancy and story band pairs it treats as implausible. The README states what the completion rests on. `HazusEQStructureTypeInferer` draws from it under `complete_tables=True`, off by default. Its `cell_is_unpublished` says whether the tables print a row for an asset's cell, and `cell_fill` how a completed cell was filled.
- **Re-draw of implausible occupancy and story pairs** (`PairRedraw` in `brails/compilers/hazus_earthquake/pair_redraw.py`, exported from `brails.compilers`). Where an asset's class and story count form a pair the completed tables list as implausible and the class is NSI's, the class is re-drawn from one observed donor through `ObservedDonorImputer.draw`. An NSI story count that the footprint height does not corroborate is replaced by the count `StoriesFromHeightInferer` gives from that height, as a derived value, and the class is re-drawn only if the pair stays implausible; without a height count, the class and the count are re-drawn together. A pair holding a local, footprint or imputed class, or an imputed story count, is logged and left.
- **`HazusEQHeightClassInferer`**, which assigns the Table 5-1 height class of a structure type and story count. W1, W2, S3, PC1 and MH are written with a null height class, since their damage models ignore height.
- **Per-asset diagnoses** (`brails/inferers/inference_diagnosis.py`). `check()` classifies each asset as OK, invalid, unsupported or infeasible, says what happened in words, and names every feature that would, changed on its own, let the inferer produce a value. The features carry no order of preference: which to give up is the caller's decision. An unsupported asset names none. A value in a shape the inferer cannot read yet is its own category. Every inferer returns the same frame, so one caller screens them all alike.
- **Grouped categorical draws** (`grouped_categorical` in `brails/utils/deterministic_sampling.py`), which invert keyed uniforms through per-group weight vectors in one pass.
- **Geometry-based asset ids for footprints** (`brails/utils/asset_id.py`). `assign_asset_ids` gives each footprint the 10-character geohash of the planar centroid of its exterior rings in EPSG:4326. Footprints sharing a geohash within one call are suffixed with their plan area in square meters. Footprints whose areas are also equal receive a rank computed from the geometry and a tiebreak, and from input order only between members identical in both. Every collision is logged. The module uses the standard library only.
- **`id_scheme` and `min_area` options for the footprint scrapers.** The USA Structures, Microsoft, OpenStreetMap and Overture scrapers accept `'id_scheme': 'geohash'` in their `input_dict` to key assets by the geohash id, which does not change when an inventory is reordered or subset, nor, for a footprint outside every collision set, when it is pulled over another region. The default, `'index'`, keeps positional integer keys. `min_area` drops footprints below a plan area, in the square of the scraper's length unit and measured by `AssetInventory.centroid_and_plan_area`, before keys are assigned. It defaults to 450 sq ft under `'geohash'` and to no floor under `'index'`. The dropped footprints and the collisions are kept on the scraper as `asset_id_skipped` and `asset_id_log`.
- **Provider ids on footprint assets.** Every footprint asset records the provenance feature `footprint_provider` (`usa_structures`, `microsoft`, `osm` or `overture`). USA Structures assets also carry `usa_build_id` and `usa_uuid`, and OpenStreetMap assets carry `osm_id`, beside Overture's existing `overture_id`.
- **Per-source merge with a source string** (`AssetInventory.merge`). `merge` takes one inventory per source, each keyed by asset ids of the inventory it is called on, possibly a subset, and returns a new inventory in which each feature of each asset comes from the first source in a stated precedence that holds a value. `None`, NaN and the other values `pandas.isna` reports as missing, the empty string and the `'NA'` that BRAILS readers write for an absent provider value are not values, as in `KnnImputer`. The inventory it is called on supplies the asset set and the geometry, and `merge` raises `ValueError` for an asset id that only another source holds. Each asset also receives `SourceString`, one character per declared feature giving the code of the source its value came from (`L` local, `F` footprint, `N` NSI, `R` derived, `I` imputed, `D` drawn) or `-` where no source holds one. The default features are `OccupancyClass`, `NumberOfUnits`, `NumberOfStories`, `YearBuilt`, `DesignLevel`, `StructureType`, `HeightClass` and `ReplacementCost`. The string records which source holds a value, never the value, so it is the same in every possible world. The inputs are not modified, and the result does not depend on the order of `others` or of the assets in any inventory.
- **`NSIFileReader`** (`brails/scrapers/nsi_parser/nsi_file_reader.py`), which reads a parquet table of NSI records, such as one combining several counties, as a point inventory keyed by `fd_id`. Every non-null cell becomes a feature under its NSI name. The assets follow ascending `fd_id` whatever the row order, and a repeated `fd_id` raises `ValueError`. It takes `simcenter_format` as `get_raw_data` does, and an optional `region`.
- **`raw_parquet_path` on `get_raw_data`.** `NSI_Parser.get_raw_data` gains a keyword-only `raw_parquet_path`. When given, the records inside the region are written to that parquet file under their NSI names and the field types declared in `NSI_RAW_SCHEMA`, before any SimCenter conversion, and `NSIFileReader` reads them back.
- **`NSIFootprintJoin`** (`brails/aggregators/points_to_polygons/nsi_footprint_join.py`), which joins NSI points to building footprints and drops none of them. A point inside a footprint, its boundary included, belongs to it. The remaining points are taken in an order drawn from their `fd_id` and the `seed` argument, drawn and recorded when omitted, the same in every possible world. Each takes the nearest footprint within a cap (`cap_m`, 15 m by default) that accepts it: an empty footprint accepts any point, a footprint whose own class (below) is a single-dwelling class accepts nothing more, and multi-family and non-residential points share any other footprint. A point no footprint accepts is returned as unlinked, with its units and the reason. Each footprint that receives a housing record gets a unit count. Each receiving footprint gets an occupancy class, the count band of its units where the governing record is RES1 or RES3 and that record's class otherwise, with NSI's own class kept as `NSIOccupancyClass`. The governing record is ranked among every record the footprint receives, inside or by the walk: housing first, then the largest `sqft`, then the lowest `fd_id`. Values NSI states and values the join derives are returned in two inventories under the same feature names. `BasicPointsToPolygonsAllocator` is unchanged.
- **Freeze of stage outputs** (`StageFreeze` in `brails/utils/`). A pipeline stage's output is frozen once under a name and held by reference, and `read` returns the held object, so a stage must not change a frozen output and copies one it needs to change. The freeze records the stage order and a BLAKE2b-128 digest of each output over a canonical form independent of asset and feature order. The digest covers an attached housing-unit inventory. `manifest` recomputes every digest and raises `RuntimeError` naming a stage whose output changed after it was frozen, and `save_manifest` writes the order and the digests to JSON. A value without a canonical form, such as `pd.NA`, a `Decimal` or a `datetime`, raises `TypeError` at `freeze`. Outputs are held in memory and are not written to disk.
- **Integer-id GeoJSON export for the R2D tool stack** (`AssetInventory.write_to_geojson(..., integer_ids=True)`). R2D parses the feature id as an integer. With the option, the features are numbered from 1 within each `type` in asset-id order, numeric when every asset id, as a string, is digits without a leading zero (`'0'` included), and carry the asset id in the property `unique_id`. An asset that already carries `unique_id` or `AIM_id`, or has no `type`, raises `ValueError`. `read_from_geojson(..., key_by_unique_id=True)` reads such a file back keyed by `unique_id`, as ints under the same rule, so export, read and re-export number alike for ids that are plain digits.
- **`ObservedDonorImputer`** (`brails/imputers/observed_donor_imputer/`), which fills a building's missing occupancy class, unit count and story count by copying them together from one observed donor per possible world. The donor is picked by the target's keyed uniform among the nearest donors within a distance cap that share its status, nearness combining location, plan area and story count. The status (residential occupancy by default) and the features whose copy triggers validation are constructor arguments, so other compilers can reuse the class, and the donor pool, search and result records sit in modules of their own.
- **Batch feedback in `ObservedDonorImputer`** (`n_batches`, default 1). Given a `StageFreeze`, `impute_stage` reads the observed donors from it, gives each target one of `n_batches` batches by its keyed uniform, and lets batch `j` draw from the observed donors and the frozen batches before it. Each batch is validated, repaired and frozen.
- **Census block groups beside tracts** (`CensusTractScraper.get_census_geographies`), which writes each asset's tract and block-group GEOIDs, from TIGERweb or from given polygons such as a TIGER/Line file, into a copy of the inventory. A footprint is located by the centroid its asset id uses, an asset on a shared edge takes the lowest GEOID whatever the order of the assets, and an asset outside every polygon raises. The feature keys are in `brails.constants`.
- **`StoriesFromHeightInferer`** (`brails/inferers/stories_from_height_inferer/`), which converts a footprint height to a story count by a scale in length per story. The scales are keyed by `footprint_provider`, and Overture's entry may give one scale per source of the height, which the footprint's `sources` names. The packaged `height_scales.json`, described in the README beside it, calibrates Overture's three height sources. The inferer writes the count under one feature, `StoriesFromHeight` by default, replacing any value held there.
- **`meter_crs`** (`brails/utils/metric_crs.py`), the check that a projected CRS measures ground meters, which `NSIFootprintJoin` applies.
- **Footprint geometry on `AssetInventory`.** The static methods `has_area` and `centroid_and_plan_area` give a footprint's area test, centroid and plan area as its geohash asset id reads them. They replace the public `has_area`, `centroid`, `plan_area_m2` and `centroid_geohash` of `brails/utils/asset_id.py`, whose routines are now private.
- **Shared feature names** (`brails/constants.py`) and **`collapse_realizations`** (`brails/utils/world_values.py`). The SimCenter feature names that several modules share are constants in `brails/constants.py`. `collapse_realizations` is the rule by which `ObservedDonorImputer` stores per-world values. The imputer module imports both, so they stay importable from it.
- **`is_missing_value`** (`brails/types/asset_inventory.py`), the public name of the test `merge` applies to missing values.
- **`ObservedDonorImputer.status_probability`**, which returns the chance that the imputer's status round gives an asset the status, residential by default.
- **`apply_to_sources`** (`brails/utils/source_updates.py`), which withholds replaced values from their per-source inventories and writes the replacements into the imputed one, so that a second merge codes them `I`.
- **`ACSUnitBudget`** (`brails/inferers/acs_unit_budget/`), which places the ACS B25024 housing units that a block group's footprints with observed records do not account for on its other footprints, as mobile homes near mobile-home park points and as dwellings given a structure size in each possible world. No dwelling holds more units than its gross area allows at `min_unit_area_sqft` per unit, and units that find no room are reported in `gaps`. The size bins take Hazus occupancy classes, a choice its README explains.
- **`union_disjoint`** (`brails/utils/inventory_union.py`), which joins inventories holding disjoint features of their assets into one, raising where two hold the same feature of an asset.
- **`ACSYearBuiltSampler`** (`brails/inferers/acs_year_built_sampler/acs_year_built_sampler.py`), which draws a construction year for each residential asset from the ACS tables B25034 and B25127 of its block group and tract. `fit` reads one world's inventory and fits one distribution over the B25034 year bins per block group and units-in-structure class. `check`, `draw` and `infer` follow `YearBuiltSampler`'s contract. A block group without B25034 units takes its tract's shares, and a tract missing from B25127 takes its county's pool. An asset is refused when its block group and tract both lack B25034 units, or when B25127 holds no tract of its county.
- **Named donor search defaults** (`DEFAULT_DONOR_K`, `DEFAULT_DONOR_CAP_M`, `DEFAULT_LOCATION_SCALE_M`, `DEFAULT_AREA_WEIGHT` and `DEFAULT_STORIES_WEIGHT` in `observed_donor_imputer.py`), the defaults of `ObservedDonorImputer`'s `k`, `cap_m` and distance terms, which `ReplacementCostImputer` shares.
- **`ResidentialNeighborYearSampler`** (`brails/imputers/residential_neighbor_year_sampler/`), which copies to a building outside the residential classes the year of a nearby residential building in the same world, through the observed-donor search on location alone, with the block group as fallback. It takes its donors from a frozen stage. A `README.rst` beside it states the defaults and the scope.
- **Donor stage for the replacement cost** (`imputation_donors_from_log` in `brails/imputers/observed_donor_imputer/observed_donor_imputer.py`), which names, per target and world, the observed donor whose values the target kept, from the log of `ObservedDonorImputer.impute_stage`.
- **`ReplacementCostImputer`** (`brails/imputers/replacement_cost/`), which values every building, in each world, as a rate per square foot times its plan area times its story count, or its plan area alone without a story count, and keeps a local value. The rate is the building's own NSI or local value over its `GrossArea`, else its observed donor's, else that of a building of its occupancy class drawn by location, plan area and story count (the same-class neighbor draw), and each log row's outcome gives its value's source code.
- **Local files and an export for `OvertureMapsFootprintScraper`.** `'local_path'` reads GeoParquet files of Overture `building` rows in place of the S3 bucket, with the release from `'overtureRelease'` or the file, and `download(region, path)` writes the raw rows of a region, every column included, to a file `'local_path'` reads. A `README.rst` beside the scraper states what it reads, how the release is resolved and the file `download` writes.
- **Merge of stacked Overture footprints** (`StackedFootprintMerge` in `brails/aggregators/stacked_footprints/stacked_footprints.py`, exported from `brails.aggregators`). A footprint nested in a larger one merges into it when Overture's stacking signal holds, and the merged set is re-keyed by `assign_asset_ids`.
- **`read_source_string`** (`brails/types/asset_inventory.py`), which returns an asset's source string after checking that it holds one code per feature, a source code or the code for none. `SOURCE_LABELS` beside it gives the label of each source code. `PairRedraw` and `ReplacementCostImputer` read source strings through the reader.
- **Overture sources parser** (`brails/inferers/stories_from_height_inferer/stories_from_height_inferer.py`): `parse_sources` is public, for code that reads a footprint's `sources` as the stories inferer does.
- **Compilation report** (`brails/compilers/compilation_report.py`, exported from `brails.compilers`): `CompilationReport` records, per world, what a compilation did to each asset and why an asset was excluded, with seconds per stage, and writes them with the component logs. The compiler gives it the names of its stages and exclusion categories.
- **`ComposedScreen`** (`brails/compilers/hazus_earthquake/resolution_loop.py`, exported from `brails.compilers`), which screens an asset against the Hazus earthquake design level and structure type inferers.
- **`ResolutionLoop`** (`brails/compilers/hazus_earthquake/resolution_loop.py`, exported from `brails.compilers`), which withholds an NSI building type that a refusal of the screen names and screens the asset again.
- **Composition of a source** (`brails/compilers/world_assembly.py`): `compose_source_parts` builds one world's source from frozen stages.
- **`AssetInventory.add_world`** (`brails/types/asset_inventory.py`), which adds a world to an inventory from an inventory of that world alone, holding a feature as a list only where it differs between worlds.
- **Hazus earthquake inventory compiler** (`brails/compilers/hazus_earthquake/hazus_eq_inventory_compiler.py`; `HazusEQInventoryCompiler` and `CompilerInputs` exported from `brails.compilers`): `HazusEQInventoryCompiler` joins NSI points to footprints over several possible worlds and records the run in a report, the manifest of its frozen stages and the logs of its components.
- **The Hazus earthquake inventory compiler's file** (`CompilationResult.write`): each building holds the properties of `DEFAULT_PROPERTIES` and the further features `extra_properties` names, and the `GrossArea` of NSI and that of a local inventory are kept as `NSIGrossArea` and `LocalGrossArea`.
- **Hazus earthquake inventory compiler, step 4.** It writes story counts from footprint heights, allocates census housing units and imputes missing occupancy classes, unit counts and story counts.
- **Hazus earthquake inventory compiler, steps 5 and 6.** It draws years and structure types per world, derives design levels and height classes, writes each building's type letter from its structure type unless a local inventory gives one, and checks each realization against the Hazus input schema.
- **Hazus earthquake inventory compiler, steps 7 and 8.** It dates non-residential buildings from residential neighbors and values every building as a rate times its gross area computed from the footprint, which the file writes as `GrossArea`. A building left without a value is excluded, and `write` refuses a file in which a compiled building lacks a default property. `footprint_gross_area` of `brails/imputers/replacement_cost/replacement_cost_imputer.py` is public.
- The Hazus earthquake inventory compiler merges stacked Overture footprints before assigning block groups when given `merge_stacked=True`.
- The Hazus earthquake inventory compiler computes `n_worlds` worlds, and `CompilationResult.write` writes every world or, given `world`, the one world it names. A file of one world leaves out a property whose value is `None` in that world.
- `ReplacementCostImputer.impute` and `impute_stage` take `target_area_key`, a feature holding each building's gross area, which the rate then multiplies in place of the area computed from the footprint, and `impute_stage` takes `area_stage`, a frozen stage holding that feature. A building without a positive number there gets no value, logged as `no_area`, and the log labels a feature's area `AREA_FROM_FEATURE`.

### Changed
- **Python 3.9 is no longer supported.** BRAILS++ requires Python 3.10, 3.11 or 3.12, the versions the tests run on.
- **BRAILS is compatible with the records of the NSI 2026 Base only.**
- **`med_yr_blt` from NSI now maps to `YearBuiltTractMedian`.** It is the median year built of the structures in a Census tract, so every building in a tract carries the same value. The converted path no longer emits `YearBuilt` at all. The legacy `nsi2brailsmap` path is unchanged and still maps `med_yr_blt` to `YearBuilt`, so the two conventions coexist for now.
- **`sqft` from NSI now maps to `GrossArea`, not `PlanArea`.** NSI reports the structure's estimated square footage, its gross area. `NSIFootprintJoin` writes the stated and summed areas under `GrossArea`, and no NSI field becomes `PlanArea`.
- **`ReplacementCost` from NSI is now `fullrep`, not `val_struct`.** `fullrep` is NSI's full, not depreciated, replacement value, and `NSIFootprintJoin` writes and sums it. `val_struct`, the depreciated value, passes through under its NSI name.
- **`HazusInfererEarthquake`, the legacy `HazusInferer` and its subclass `SimCenterInferer` take `grossArea_key`, default `GrossArea`, in place of `planArea_key`.** The replacement cost multiplies this area by a cost per square foot without a story factor, so it needs a gross area. A caller passing `planArea_key` gets a `TypeError`. An inventory that carries the area only as `PlanArea` now gets no `ReplacementCost` and a warning, unless the caller passes `grossArea_key='PlanArea'`. `HazusInfererWind` keeps `planArea_key` and `PlanArea`.
- The Hazus earthquake rulesets moved into `brails/inferers/hazus_inferer_earthquake/`. Legacy callers keep working through updated imports.
- **`is_inventory` moved to `InputValidator`** and is no longer available on `InventoryValidator`, which published it as a public static method in 4.3.0. Call sites need updating to `InputValidator.is_inventory`.
- `scipy` is now a declared dependency. It was arriving only transitively through scikit-learn.
- **`OvertureMapsFootprintScraper` now derives from `FootprintScraper`** as well as from `OvertureMapsScraper`, so all four footprint scrapers share one key-assignment interface.
- **`KnnImputer` no longer uses footprint provider ids or the provider name as predictors.** The provider ids (`overture_id`, `usa_build_id`, `usa_uuid`, `osm_id`) and `footprint_provider`, listed in `brails.constants.PROVIDER_FEATURES`, are dropped from the predictors by default and kept in the output. This changes the predictors for an existing Overture inventory, which already carried `overture_id`. Pass `exclude_provider_ids=False` for the previous behavior.
- **`AssetInventory.get_world_realization` on a single-world inventory** (`n_pw` 1, no feature holding more than one realization) returns that world for any index instead of raising for an index above 0. A realization now has `n_pw` 1, so `get_dataframe` accepts one holding a vector feature such as `HousingUnits`. It is built from the selected world without copying the others: coordinates and the housing-unit inventory are copied, list and dict feature values are copied one level deep, and other values are shared with the source.
- **`AssetInventory.write_to_geojson` and `HousingUnitInventory.to_json` write a NaN as `null`** and raise `ValueError` on an infinite value, outside a tuple such as a coordinate pair. `write_to_geojson` now writes a temporary file beside the output and replaces the output with it, so the output is a new file (a new inode) instead of the old one written over, and the directory must be writable: a writable file in a read-only directory is no longer written.
- **The `AssetInventory.merge` refusal of a source holding a housing-unit inventory** is reworded to name `combine` as the method that combines such inventories.
- **`CensusTractScraper.get_census_tracts` raises** where it printed an error and returned the tracts fetched so far: for an asset outside every tract, a client error or exhausted retries, and it then writes nothing. A footprint is located by the centroid of its exterior rings, and an asset on an edge shared by two tracts takes the lower GEOID, where it took the tract fetched first.
- **`OvertureMapsFootprintScraper` reads Overture's `building` type only.** Building parts, portions of a building already present, no longer become assets, and the `building_id` feature does not appear. Under `'geohash'`, a building whose id carried an area suffix only because a part shared its cell gets the bare geohash.

### Fixed
- **NSI records sharing a coordinate are no longer discarded.** The fetch keyed its records by point geometry, so where NSI serves several records on one coordinate, one occupancy of a mixed-use structure per record, only the last one survived, and which one that was depended on the order the API happened to return. Records are now keyed by `fd_id`. `get_raw_data` returns every record, so a pull covering mixed-use structures now yields more assets than before, several of them at the same coordinate.
- **Point-to-polygon allocation** (`BasicPointsToPolygonsAllocator`). `buffer_dist` was applied in EPSG:3857, whose units exceed ground meters by 1/cos(latitude), so a nominal 10 m buffer reached 7.91 m in San Francisco. Buffering now happens in a projected CRS estimated from the polygons. Candidates are also ranked before they are taken, by containment and true distance with ids breaking ties, which makes matching deterministic under reordering.
- **Overture release resolution.** The default release came from the published release YAML, which names releases the S3 bucket no longer holds and typically trails it by months. Releases now resolve against the bucket.
- **NSI merge coordinates.** The merge loop rebuilt every matched footprint's coordinate list from a variable left over by an earlier loop, so every matched building recorded the coordinates of the last footprint processed.
- The NSI basement flag decode was inverted, now it is parsed correctly.
- The Hazus state region mapping was missing Washington and the Northern Mariana Islands; now it is comprehensive.
- `grouped_categorical` no longer draws a trailing category of weight zero. Where a weight vector ended in zeros and its normalized cumulative sum stopped a fraction below one, the largest uniforms selected the last zero-weight category.
- **`OvertureMapsFootprintScraper.get_footprints` returns an empty inventory for a region holding no building**, instead of raising `KeyError`.

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
