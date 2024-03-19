from brails.utils.utils import Importer


importer = Importer()

# #importAll()
# import('RoofShape')

# roofShapeClass=getClass("RoofShape")

roof_shape_class = importer.get_class('RoofShape')

# roofShapeObject=roofShapeObject("Hello World")

roofShapeObject = roof_shape_class({})

# roofShapeObject.predict('blah')
