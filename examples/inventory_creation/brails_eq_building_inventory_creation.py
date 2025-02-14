import numpy as np
from brails.utils import Importer

LOCATION_NAME = 'Alameda, Alameda County, CA'
INVENTORY_OUTPUT = 'AlamedaInventory_EQ.geojson'
NO_POSSIBLE_WORLDS = 1

importer = Importer()

region_data = {"type": "locationName", "data": LOCATION_NAME}
region_boundary_class = importer.get_class("RegionBoundary")
region_boundary_object = region_boundary_class(region_data)
nsi_class = importer.get_class('NSI_Parser')
nsi = nsi_class()
nsi_inventory = nsi.get_raw_data_given_boundary(region_boundary_object, 'ft')

scraper_class = importer.get_class('USA_FootprintScraper')
scraper = scraper_class({'length': 'ft'})
scraper_inventory = scraper.get_footprints(region_boundary_object)

nsi_inventory = nsi.get_filtered_data_given_inventory(
    scraper_inventory, "ft", get_extended_features=True)

knn_imputer_class = importer.get_class("KnnImputer")

imputer = knn_imputer_class(
    nsi_inventory, n_possible_worlds=NO_POSSIBLE_WORLDS,
    exclude_features=['lat', 'lon', 'fd_id'])
imputed_inventory = imputer.impute()

CA_AVG = 78672  # state average
CA_STD_DEV = CA_AVG*0.5  # 50% cov

# Step 1: Calculate the parameters of the underlying normal distribution
mu = np.log(CA_AVG**2 /
            np.sqrt(CA_STD_DEV**2 + CA_AVG**2))
sigma = np.sqrt(np.log(1 + (CA_STD_DEV**2 / CA_AVG**2)))

# Step 2: Generate the lognormal sample using the parameters of the normal
# distribution
for key, val in imputed_inventory.inventory.items():
    lognormal_sample = np.random.lognormal(
        mean=mu, sigma=sigma, size=NO_POSSIBLE_WORLDS)
    val.add_features({"Income": lognormal_sample[0]})

# The names of NEW keys to be inferred.
STRUCTURE_TYPE_KEY = 'StructureTypeHazus'  # instead of  "constype" from NSI
REPLACEMENT_COST_KEY = 'ReplacementCostHazus'  # instead of NSI "repaircost"

# The names of existing keys to be used as "predictors"
YEAR_BUILT_KEY = 'erabuilt'
OCCUPANCY_CLASS_KEY = 'occupancy'
INCOME_KEY = 'Income'
NUMBER_OF_STORIES_KEY = 'numstories'
PLAN_AREA_KEY = 'fpAreas'
SPLIT_LEVEL_KEY = 'splitlevel'

infer_features_for_hazusdl = importer.get_class("Infer_features_for_HazusDL")
inferer = infer_features_for_hazusdl(input_inventory=imputed_inventory,
                                     n_possible_worlds=NO_POSSIBLE_WORLDS,
                                     yearBuilt_key=YEAR_BUILT_KEY,
                                     occupancyClass_key=OCCUPANCY_CLASS_KEY,
                                     numberOfStories_key=NUMBER_OF_STORIES_KEY,
                                     income_key=INCOME_KEY,
                                     splitLevel_key=SPLIT_LEVEL_KEY,
                                     structureType_key=STRUCTURE_TYPE_KEY,
                                     replacementCost_key=REPLACEMENT_COST_KEY,
                                     planArea_key=PLAN_AREA_KEY,
                                     clean_features=False)
hazus_inferred_inventory = inferer.infer()

imputer = knn_imputer_class(hazus_inferred_inventory,
                            n_possible_worlds=NO_POSSIBLE_WORLDS)
hazus_inferred_inventory = imputer.impute()

inferer = infer_features_for_hazusdl(
    input_inventory=hazus_inferred_inventory,
    n_possible_worlds=NO_POSSIBLE_WORLDS,
    yearBuilt_key=YEAR_BUILT_KEY,
    structureType_key='StructureType',
    clean_features=True)
r2d_ready_inventory = inferer.infer()

_ = r2d_ready_inventory.write_to_geojson(
    output_file=INVENTORY_OUTPUT)
