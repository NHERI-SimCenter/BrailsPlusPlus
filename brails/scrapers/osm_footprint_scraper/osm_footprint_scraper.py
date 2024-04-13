# written Barbaros Cetiner 03/24
#license: BSD-2

from brails.scrapers.footprint_scraper import FootprintScraper
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import AssetInventory

import math
import json
import requests
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import groupby
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, box
from shapely.ops import linemerge, unary_union, polygonize
from shapely.strtree import STRtree
from brails.utils.geo_tools import *
import concurrent.futures
from requests.adapters import HTTPAdapter, Retry
import unicodedata


class OSM_FootprintScraper(FootprintScraper):
    """
    A class to generate the foorprint data utilizing Open Street Maps API

    Attributes:

    Methods:
        __init__: Constructor that just creates an empty footprint
        get_inventory(id, coordinates): to get the inventory

    """

    def _cleanstr(self, inpstr):
        return "".join(
            char
            for char in inpstr
            if not char.isalpha()
            and not char.isspace()
            and (char == "." or char.isalnum())
        )

    def _yearstr2int(self, inpstr):
        if inpstr != "NA":
            yearout = self._cleanstr(inpstr)
            yearout = yearout[:4]
            if len(yearout) == 4:
                try:
                    yearout = int(yearout)
                except:
                    yearout = None
            else:
                yearout = None
        else:
            yearout = None

        return yearout

    def _height2float(self, inpstr, lengthUnit):

        if inpstr != "NA":
            heightout = self._cleanstr(inpstr)
            try:
                if lengthUnit == "ft":
                    heightout = round(float(heightout) * 3.28084, 1)
                else:
                    heightout = round(float(heightout), 1)
            except:
                heightout = None
        else:
            heightout = None

        return heightout

    def __init__(self, input: dict):
        """
        Initialize the object

        Args
            input: a dict defining length units, if no ;length' ft is assumed
        """

        self.lengthUnit = input.get("length")
        if self.lengthUnit == None:
            self.lengthUnit = "ft"

    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        This method will be used by the caller to obtain the footprints for builings in an area.

        Args:
            region (RegionBoundary): The region of interest.

        Returns:
            BuildingInventory: A building inventory for buildings in the region.

        """

        bpoly, queryarea_printname, osmid = region.get_boundary()

        if osmid != None:

            queryarea_turboid = osmid + 3600000000
            query = f"""
            [out:json][timeout:5000][maxsize:2000000000];
            area({queryarea_turboid})->.searchArea;
            way["building"](area.searchArea);
            out body;
            >;
            out skel qt;
            """

        else:
            bpoly, queryarea_printname = self.__bbox2poly(queryarea)

            if len(queryarea) == 4:
                bbox = [
                    min(queryarea[1], queryarea[3]),
                    min(queryarea[0], queryarea[2]),
                    max(queryarea[1], queryarea[3]),
                    max(queryarea[0], queryarea[2]),
                ]
                bbox = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            elif len(queryarea) > 4:
                bbox = 'poly:"'
                for i in range(int(len(queryarea) / 2)):
                    bbox += f"{queryarea[2*i+1]} {queryarea[2*i]} "
                bbox = bbox[:-1] + '"'

            query = f"""
            [out:json][timeout:5000][maxsize:2000000000];
            way["building"]({bbox});
            out body;
            >;
            out skel qt;
            """

        url = "http://overpass-api.de/api/interpreter"
        r = requests.get(url, params={"data": query})

        datalist = r.json()["elements"]
        nodedict = {}
        for data in datalist:
            if data["type"] == "node":
                nodedict[data["id"]] = [data["lon"], data["lat"]]

        attrmap = {
            "start_date": "erabuilt",
            "building:start_date": "erabuilt",
            "construction_date": "erabuilt",
            "roof:shape": "roofshape",
            "height": "buildingheight",
        }

        levelkeys = {"building:levels", "roof:levels", "building:levels:underground"}
        otherattrkeys = set(attrmap.keys())
        datakeys = levelkeys.union(otherattrkeys)

        attrkeys = ["buildingheight", "erabuilt", "numstories", "roofshape"]
        attributes = {key: [] for key in attrkeys}
        fpcount = 0
        footprints = []
        for data in datalist:
            if data["type"] == "way":
                nodes = data["nodes"]
                footprint = []
                for node in nodes:
                    footprint.append(nodedict[node])
                footprints.append(footprint)

                fpcount += 1
                availableTags = set(data["tags"].keys()).intersection(datakeys)
                for tag in availableTags:
                    nstory = 0
                    if tag in otherattrkeys:
                        attributes[attrmap[tag]].append(data["tags"][tag])
                    elif tag in levelkeys:
                        try:
                            nstory += int(data["tags"][tag])
                        except:
                            pass

                    if nstory > 0:
                        attributes["numstories"].append(nstory)
                for attr in attrkeys:
                    if len(attributes[attr]) != fpcount:
                        attributes[attr].append("NA")

        attributes["buildingheight"] = [
            self._height2float(height, self.lengthUnit)
            for height in attributes["buildingheight"]
        ]

        attributes["erabuilt"] = [
            self._yearstr2int(year) for year in attributes["erabuilt"]
        ]

        attributes["numstories"] = [
            nstories if nstories != "NA" else None
            for nstories in attributes["numstories"]
        ]

        print(
            f"\nFound a total of {fpcount} building footprints in {queryarea_printname}"
        )

        return self._create_asset_inventory(footprints, attributes, self.lengthUnit)
