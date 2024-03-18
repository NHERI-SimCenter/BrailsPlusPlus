import importlib
import os
import glob

def import_all_modules(directory):
    """
    Import all modules within a directory.
    
    Parameters:
        directory (str): The directory path containing the modules.
    """
    # Get a list of all Python files in the directory
    module_files = glob.glob(os.path.join(directory, "*.py"))

     # Iterate over each module file
    for module_file in module_files:
        # Get the module name from the file path
        module_name = os.path.splitext(os.path.basename(module_file))[0]
        
        # Import the module dynamically
        importlib.import_module(f"{directory}.{module_name}")


def get_class(class_name):
    """
    Retrieve a class object given its name as a string.
    
    Parameters:
        class_name (str): The name of the class.
    
    Returns:
        class object: The class object if found, None otherwise.
    """
    # Iterate over all loaded modules
    for module_name, module in globals().items():
        if hasattr(module, class_name):
            return getattr(module, class_name)
    
    return None
