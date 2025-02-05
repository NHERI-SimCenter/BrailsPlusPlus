import numpy as np


def user_inferer(inventory_array):
    #
    # Defining my mapping following Table 6-10 in Hazus Inventory Technical Manual 6
    #
    # Baseline Hazus Contents Value as Percent of Structure Value

    contents_value_over_str_value = {
        "RES1": 0.50,
        "RES2": 0.50,
        "RES3A": 0.50,
        "RES3B": 0.50,
        "RES3C": 0.50,
        "RES3D": 0.50,
        "RES3E": 0.50,
        "RES3F": 0.50,
        "RES3": 0.50,
        "RES4": 0.50,
        "RES5": 0.50,
        "RES6": 0.50,
        "COM1": 1.00,
        "COM2": 1.00,
        "COM3": 1.00,
        "COM4": 1.00,
        "COM5": 1.00,
        "COM6": 1.50,
        "COM7": 1.50,
        "COM8": 1.00,
        "COM9": 1.00,
        "COM10": 0.50,
        "IND1": 1.50,
        "IND2": 1.50,
        "IND3": 1.50,
        "IND4": 1.50,
        "IND5": 1.50,
        "IND6": 1.00,
        "AGR1": 1.00,
        "REL1": 1.00,
        "GOV1": 1.00,
        "GOV2": 1.50,
        "EDU1": 1.00,
        "EDU2": 1.50,
    }

    new_features = {}
    for idx, bldg in inventory_array.items():
        occ_type = bldg["properties"]["occupancy"]
        contents_value_ratio = contents_value_over_str_value.get(occ_type, np.nan)
        contents_value = contents_value_ratio * bldg["properties"]["repaircost"]
        new_features[idx] = {"contentsValue": contents_value}

    return new_features
