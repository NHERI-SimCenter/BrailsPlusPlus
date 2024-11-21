
def user_inferer(inventory_array):

	#
	# Defining my mapping following Table 5-4 in Hazus Inventory Technical Manual 6
	#

	OccToSqft = {
	  	  "RES1": {
		    "AverageSqFtPerFloor": 1500,
    		"MaximumSqFt": 5000
		  },
		  "RES2": {
		    "AverageSqFtPerFloor": 1500,
    		"MaximumSqFt": 2000
		  },
		  "RES3A": {
		    "AverageSqFtPerFloor": 1500,
    		"MaximumSqFt": 1161500
		  },
		  "RES3B": {
		    "AverageSqFtPerFloor": 1500,
    		"MaximumSqFt": 1161500
		  },
		  "RES3C": {
		    "AverageSqFtPerFloor": 3000,
    		"MaximumSqFt": 1161500
		  }
		}

	new_features = {}
	for key,bldg in inventory_array.items():

		bldgidx = key
		
		if bldg["properties"]["occupancy"] == "RES1":
			new_features[bldgidx] = {"fpAreas": OccToSqft["RES1"]["AverageSqFtPerFloor"],
									 "fpAreas_max": OccToSqft["RES1"]["MaximumSqFt"]} 
		elif bldg["properties"]["occupancy"] == "RES2":
			new_features[bldgidx] = {"fpAreas": OccToSqft["RES2"]["AverageSqFtPerFloor"],
									 "fpAreas_max": OccToSqft["RES2"]["MaximumSqFt"]} 
		elif bldg["properties"]["occupancy"] == "RES3A":
			new_features[bldgidx] = {"fpAreas": OccToSqft["RES3A"]["AverageSqFtPerFloor"],
									 "fpAreas_max": OccToSqft["RES3A"]["MaximumSqFt"]} 
		elif bldg["properties"]["occupancy"] == "RES3B":
			new_features[bldgidx] = {"fpAreas": OccToSqft["RES3B"]["AverageSqFtPerFloor"],
									 "fpAreas_max": OccToSqft["RES3B"]["MaximumSqFt"]} 
		elif bldg["properties"]["occupancy"] == "RES3C":
			new_features[bldgidx] = {"fpAreas": OccToSqft["RES3C"]["AverageSqFtPerFloor"],
									 "fpAreas_max": OccToSqft["RES3C"]["MaximumSqFt"]} 

	return new_features
