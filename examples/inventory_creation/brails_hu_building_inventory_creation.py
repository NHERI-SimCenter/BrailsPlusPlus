from brails.utils import Importer

LOCATION_NAME = 'Fort Myers Beach, FL'
INVENTORY_OUTPUT = 'FortMyersInventory_HU.geojson'
NO_POSSIBLE_WORLDS = 1

importer = Importer()

region_data = {"type": "locationName", "data": LOCATION_NAME}
region_boundary_class = importer.get_class("RegionBoundary")
region_boundary_object = region_boundary_class(region_data)

nsi_class = importer.get_class('NSI_Parser')
nsi = nsi_class()
nsi_inventory = nsi.get_raw_data_given_boundary(region_boundary_object, 'ft')

scraper_class = importer.get_class('MS_FootprintScraper')
scraper = scraper_class({'length': 'ft'})
scraper_inventory = scraper.get_footprints(region_boundary_object)

nsi_inventory = nsi.get_filtered_data_given_inventory(
    scraper_inventory, "ft", get_extended_features=True)

knn_imputer_class = importer.get_class("KnnImputer")

imputer = knn_imputer_class(
    nsi_inventory, n_possible_worlds=NO_POSSIBLE_WORLDS,
    exclude_features=['lat', 'lon', 'fd_id'])
imputed_inventory = imputer.impute()

# Get aerial imagery using GoogleSatellite:
google_satellite_class = importer.get_class('GoogleSatellite')
google_satellite = google_satellite_class()
images_aerial = google_satellite.get_images(imputed_inventory,
                                            'tmp/satellite/')

roof_shape_classifier_class = importer.get_class('RoofShapeClassifier')
roof_shape_classifier = roof_shape_classifier_class()
predictions = roof_shape_classifier.predict(images_aerial)

for key, val in imputed_inventory.inventory.items():
    val.add_features({'DesignWindSpeed': 159,
                      'FloodZone': 'AE',
                      'RoofShape': predictions[key]})

imputed_inventory.change_feature_names({'erabuilt': 'YearBuilt',
                                        'constype': 'BuildingMaterial',
                                        'numstories': 'NumberOfStories'})
hurricaneInferer = importer.get_class("HazusHurricaneInferer")
inferer = hurricaneInferer(
    input_inventory=imputed_inventory, clean_features=True)
hazus_inferred_inventory = inferer.infer()

imputer = knn_imputer_class(hazus_inferred_inventory,
                            n_possible_worlds=NO_POSSIBLE_WORLDS)
hazus_inventory_final = imputer.impute()

_ = hazus_inventory_final.write_to_geojson(
    output_file=INVENTORY_OUTPUT)
