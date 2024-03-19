import sys
import inspect

from brails.utils.utils import Importer

importer = Importer()
my_class = importer.get_class('FacadeParser')
instance = my_class({})
instance.predict('Hello World')


