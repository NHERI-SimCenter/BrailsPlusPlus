# Written: Barbaros Cetiner(ImHandler in old BRAILS)
#          minor edits for BRAILS++ by fmk
# license: BSD-3 (see LICENSCE file: https://github.com/NHERI-SimCenter/BrailsPlusPlus)

from brails.scrapers.footprint_scraper import FootprintScraper
from brails.types.region_boundary import RegionBoundary
from brails.types.asset_inventory import AssetInventory
from brails.utils.geo_tools import *

import math
import requests
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import groupby
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, box
from shapely.ops import linemerge, unary_union, polygonize
from shapely.strtree import STRtree

import concurrent.futures
from requests.adapters import HTTPAdapter, Retry
import unicodedata


class MS_FootprintScraper(FootprintScraper):
    """
    A class to generate the foorprint data utilizing Microsofts

    Attributes:

    Methods:
        __init__: Constructor that just creates an empty footprint
        get_inventory(id, coordinates): to get the inventory

    """

    def __init__(self, input: dict):
        """
        Initialize the object

        Args
            input: a dict defining length units, if no ;length' ft is assumed
        """

        self.lengthUnit = input.get("length")
        if self.lengthUnit == None:
            self.lengthUnit = "ft"

    def _deg2num(self, lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2**zoom
        xtile = int((lon + 180) / 360 * n)
        ytile = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
        return (xtile, ytile)

    def _determine_tile_coords(self, bbox):
        xlist = []
        ylist = []
        for vert in bbox:
            (lat, lon) = (vert[1], vert[0])
            x, y = self._deg2num(lat, lon, 9)
            xlist.append(x)
            ylist.append(y)

            xlist = list(range(min(xlist), max(xlist) + 1))
            ylist = list(range(min(ylist), max(ylist) + 1))
        return (xlist, ylist)

    def _xy2quadkey(self, xtile, ytile):
        xtilebin = str(bin(xtile))
        xtilebin = xtilebin[2:]
        ytilebin = str(bin(ytile))
        ytilebin = ytilebin[2:]
        zpad = len(xtilebin) - len(ytilebin)
        if zpad < 0:
            xtilebin = xtilebin.zfill(len(xtilebin) - zpad)
        elif zpad > 0:
            ytilebin = ytilebin.zfill(len(ytilebin) + zpad)
        quadkeybin = "".join(i + j for i, j in zip(ytilebin, xtilebin))
        quadkey = ""
        for i in range(0, int(len(quadkeybin) / 2)):
            quadkey += str(int(quadkeybin[2 * i : 2 * (i + 1)], 2))
        return int(quadkey)

    def _bbox2quadkeys(self, bpoly):
        bbox = bpoly.bounds
        bbox_coords = [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ]

        (xtiles, ytiles) = self._determine_tile_coords(bbox_coords)
        quadkeys = []
        for xtile in xtiles:
            for ytile in ytiles:
                quadkeys.append(self._xy2quadkey(xtile, ytile))

        quadkeys = list(set(quadkeys))
        return quadkeys

    def _parse_file_size(self, strsize):
        strsize = strsize.lower()
        if "gb" in strsize:
            multiplier = 1e9
            sizestr = "gb"
        elif "mb" in strsize:
            multiplier = 1e6
            sizestr = "mb"
        elif "kb" in strsize:
            multiplier = 1e3
            sizestr = "kb"
        else:
            multiplier = 1
            sizestr = "b"

        return float(strsize.replace(sizestr, "")) * multiplier

    def _download_ms_tiles(self, quadkeys, bpoly):
        dftiles = pd.read_csv(
            "https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv"
        )

        # Define length unit conversion factor:
        if self.lengthUnit == "ft":
            convFactor = 3.28084
        else:
            convFactor = 1

        footprints = []
        bldgheights = []
        for quadkey in tqdm(quadkeys):
            rows = dftiles[dftiles["QuadKey"] == quadkey]
            if rows.shape[0] == 1:
                url = rows.iloc[0]["Url"]
            elif rows.shape[0] > 1:
                rows.loc[:, "Size"] = rows["Size"].apply(
                    lambda x: self._parse_file_size(x)
                )
                url = rows[rows["Size"] == rows["Size"].max()].iloc[0]["Url"]
            else:
                continue

            df_fp = pd.read_json(url, lines=True)
            for index, row in tqdm(df_fp.iterrows(), total=df_fp.shape[0]):
                fp_poly = Polygon(row["geometry"]["coordinates"][0])
                if fp_poly.intersects(bpoly):
                    footprints.append(row["geometry"]["coordinates"][0])
                    height = row["properties"]["height"]
                    if height != -1:
                        bldgheights.append(round(height * convFactor, 1))
                    else:
                        bldgheights.append(None)

        return (footprints, bldgheights)

    def get_footprints(self, region: RegionBoundary) -> AssetInventory:
        """
        This method will be used by the caller to obtain the footprints for builings in an area.

        Args:
            region (RegionBoundary): The region of interest.

        Returns:
            BuildingInventory: A building inventory for buildings in the region.

        """

        bpoly, queryarea_printname, osmid = region.get_boundary()

        quadkeys = self._bbox2quadkeys(bpoly)
        attributes = {"buildingheight": []}
        (footprints, attributes["buildingheight"]) = self._download_ms_tiles(
            quadkeys, bpoly
        )

        print(
            f"\nFound a total of {len(footprints)} building footprints in {queryarea_printname}"
        )

        return self._create_asset_inventory(footprints, attributes, self.lengthUnit)
