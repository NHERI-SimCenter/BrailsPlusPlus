import importlib
import os
import sys
import inspect

def import_all_modules(directory):
    """
    Import all modules within a directory and its subdirectories.
    
    Parameters:
        directory (str): The directory path containing the modules.
    """
    # Ensure the directory is in sys.path
    sys.path.append(directory)

    # Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Get the module name from the file path
                module_name = os.path.splitext(os.path.relpath(os.path.join(root, file), directory))[0].replace(os.sep, ".")
                
                # Import the module dynamically
                print('module_name',module_name)
                importlib.import_module(module_name)
                for name, obj in inspect.getmembers(sys.modules[module_name]):
                    if inspect.isclass(obj):
                        print(obj)
                        
                for name, obj in inspect.getmembers(module_name, inspect.isclass):
                    print(f"- {name}")
                    
        for dir in dirs:
            import_all_modules(dir)

