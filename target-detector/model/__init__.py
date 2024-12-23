import importlib


def str_to_class(module_name=".", class_name="ByteTracker"):
    """Return a class type from a string reference"""
    print(f"Dynamically loading class {class_name}\n")
    try:
        module_ = importlib.import_module(module_name + class_name.lower(), package="model")
        try:
            class_ = getattr(module_, class_name)
        except AttributeError:
            print('Class does not exist\n')
    except ImportError:
        print('Module does not exist\n')
    return class_ or None
