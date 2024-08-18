from brails.scaper.region_boundary import RegionBoundary


class BoundaryFromLocation(RegionBoundary):

    def __init__(self, input: dict):
        self.location = input["location"]
        self.outfile = None

    def get_boundary(self):

        queryarea = self.location.replace(" ", "+").replace(",", "+")

        queryarea_formatted = ""
        for i, j in groupby(queryarea):
            if i == "+":
                queryarea_formatted += i
            else:
                queryarea_formatted += "".join(list(j))

        nominatimquery = (
            "https://nominatim.openstreetmap.org/search?"
            + f"q={queryarea_formatted}&format=jsonv2"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)"
                + " AppleWebKit/537.36 (KHTML, like Gecko)"
                + " Chrome/39.0.2171.95 Safari/537.36"
            )
        }
        r = requests.get(nominatimquery, headers=headers)
        datalist = r.json()

        areafound = False
        for data in datalist:
            queryarea_osmid = data["osm_id"]
            queryarea_name = data["display_name"]
            if data["osm_type"] == "relation":
                areafound = True
                break

        if areafound == True:
            try:
                print(f"Found {queryarea_name}")
            except:
                queryareaNameUTF = unicodedata.normalize("NFKD", queryarea_name).encode(
                    "ascii", "ignore"
                )
                queryareaNameUTF = queryareaNameUTF.decode("utf-8")
                print(f"Found {queryareaNameUTF}")
        else:
            sys.exit(
                f"Could not locate an area named {queryarea}. "
                + "Please check your location query to make sure "
                + "it was entered correctly."
            )

        queryarea_printname = queryarea_name.split(",")[0]

        url = "http://overpass-api.de/api/interpreter"

        # Get the polygon boundary for the query area:
        query = f"""
         [out:json][timeout:5000];
         rel({queryarea_osmid});
         out geom;
        """

        r = requests.get(url, params={"data": query})

        datastruct = r.json()["elements"][0]
        if datastruct["tags"]["type"] in ["boundary", "multipolygon"]:
            lss = []
            for coorddict in datastruct["members"]:
                if coorddict["role"] == "outer":
                    ls = []
                    for coord in coorddict["geometry"]:
                        ls.append([coord["lon"], coord["lat"]])
                    lss.append(LineString(ls))

            merged = linemerge([*lss])
            borders = unary_union(merged)  # linestrings to a MultiLineString
            polygons = list(polygonize(borders))

            if len(polygons) == 1:
                bpoly = polygons[0]
            else:
                bpoly = MultiPolygon(polygons)

        else:
            sys.exit(
                f"Could not retrieve the boundary for {queryarea}. "
                + "Please check your location query to make sure "
                + "it was entered correctly."
            )
        if self.outfile:
            write_polygon2geojson(bpoly, outfile)

        return bpoly, queryarea_printname, queryarea_osmid
