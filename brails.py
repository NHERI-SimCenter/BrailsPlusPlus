import sys
import inspect

from brails.utils.importer import import_all_modules
from brails.utils.get_class import get_class

# Import all modules in the framework directory
import_all_modules("brails/processing")

class_name = "SoftStory"
cls = get_class(class_name)

if cls:
    print(f"Class '{class_name}' found:", cls)
    obj = cls("hello")
    obj.predict("hello")
else:
    print(f"Class '{class_name}' not found.")

