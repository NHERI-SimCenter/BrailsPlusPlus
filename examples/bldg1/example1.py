
import json
from brails import Importer

importer = Importer()

# Open the JSON file
with open('example1.json', 'r') as file:
    # Load JSON data
    data = json.load(file)

    #
    # check valid type of workflow
    #
    
    typeWorkflow = data['type']
    if typeWorkflow != 'building_inventory':
        print('FATAL: Workflow Type: ', typeWorkflow, 'is  not a building_inventory')
        exit(-1)

    #
    # get the footprints
    #
    
    foot_print_data = data.get('footprint_app')
    if foot_print_data is None:
        print('FATAL: A Building Inventory Generator needs a footprint_app key and data, None Provided')
        exit(-1)
    
    footPrintSourceClassName = foor_print_data['classType']
    pythonClass = importer.getClass(footPrintSourceClassName)
    footPrints = pythonClass(input_file.location, input_file.footPtintSource,args)
    

        
    value = data['key']
#BRAILS:

    # load input file2
    input_file = load(inputFile)

    # determine footprintSource application


    # augment footprintData
    for augmentObject in input_file.augmentFiles:
       footPrints = this.augment(footPrints,augmentObject) 

    # create filter array for street images
    filterArrayStreet=[]
    for filter_object in input_file.streetFilters:
       pythonClass = this.getClass(filter_object)
       filterArrayStreet.add(getClass(pythonClass))

    # create filter array for arial images 
    filterArrayAeriel = [self.getClass( self.getClass(filter_object)) 
                         for filter_object in input_file.aerialFilters] 


 


"""

{
    "location":{
	"specType":"cityName",
	"args":"Tiburon, CA"
    },
    "augmentFiles":[
	{"fileName":"assessor.csv",
	 "colMapping":"mapAssessor.json"
	},
	{"fileName":"myData.csv",
	 "colMapping":"None"
	}
    ],
    "footprintSource":{
	"classType":"Google",
	"appData":{
	    "gcpKey":"qg;lgqlgj"
	}
    }
    "satelliteImageSource":{
	"classType":"Google",
	"appData":{
	    "gcpKey":"qg;lgqlgj"
	}
    },
    "streetImagesSource":{
	"classType":"Google",
	"appData":{
	    "gcpKey":"qg;lgqlgj"
	}
    }
    "streetFilters":[
	"classType":"VanishingLines",
	"args":{
	},
	"classType":"Obstructed",
	"args":{
	}
	"classType":"NearbyHouses",
	"args":{
	}    
    ],
    "aerialFilters":[
	"classType":"",
	"args":{
	}
    ],
    "predictionsAerial"=[
	{
	    "classType":"BrailsRoofType",
	    "args":{
		"model"::"zenodo://blah/blah/blah"
	    },
	    "heading":"roofShape"
	},
	],
    {
	    "classType":"LLM",
	    "args":{
		"prompts"::"this is a pricture of a roof, is it a gabled roof, hiiped roof or flat roof?"
            }
	    "heading":"roofShapeLLM"
    },
    "predictionsStreet"=[
	{
	    "classType":"BrailsNOS",
	    "args":{
		"model"::"zenodo://blah/blah/blah"
	    },
	    "heading":"numberOfStories"
	},
	],
    {
	    "classType":"LLM_numberOfStories",
	    "args":{
		"prompts"::"This is a picture of a House, How many stories are there?"
            }
	    "heading":"nosLLM"
	},
	{
	    "classType":"FirstFloorElevation",
	    "args":{
            }
	    "heading":"firstFloorElevation"
	}  
}
"""