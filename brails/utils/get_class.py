import inspect
import sys

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
        print(module_name)
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                print(obj)
                
        if hasattr(module, class_name):
            return getattr(module, class_name)
    
    return None
