import os


def get_parameters(param, default=None):
    """
    This function is used to get the value of a parameter from the configuration file.
    If the parameter is not found in the configuration file, it will return the default value.
    """
    value = os.environ.get(param) or os.getenv(str(param).upper())
    return value if value else default
