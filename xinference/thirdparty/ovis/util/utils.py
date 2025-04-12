import os
from importlib import import_module


def rank0_print(*args):
    if int(os.getenv("LOCAL_PROCESS_RANK", os.getenv("LOCAL_RANK", 0))) == 0:
        print(*args)


def smart_unit(num):
    if num / 1.0e9 >= 1:
        return f'{num / 1.0e9:.2f}B'
    else:
        return f'{num / 1.0e6:.2f}M'


def import_class_from_string(full_class_string):
    # Split the path to get separate module and class names
    module_path, _, class_name = full_class_string.rpartition('.')

    # Import the module using the module path
    module = import_module(module_path)

    # Get the class from the imported module
    cls = getattr(module, class_name)
    return cls
