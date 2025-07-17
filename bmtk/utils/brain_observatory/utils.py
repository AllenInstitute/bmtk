import ast
import numpy as np
import hashlib

class FakeTqdm:
    def __init__(self, *args, **kwargs):
        pass
    
    def update(self, *args, **kwargs):
        pass


def write_from_stream(path: str, stream):
    with open(path, "wb") as fil:
        for chunk in stream:
            fil.write(chunk)


def json_handler(obj):
    """Used by write_json convert a few non-standard types to things that the
    json package can handle."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, bool) or isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    else:
        raise TypeError(
            "Object of type %s with value of %s is not JSON serializable"
            % (type(obj), repr(obj))
        )


def infer_column_types(dataframe):
    dataframe = dataframe.copy()

    for colname in dataframe.columns:
        try:
            dataframe[colname] = dataframe[colname].apply(ast.literal_eval)
        except (ValueError, SyntaxError):
            continue
    
    dataframe = dataframe.infer_objects()
    return dataframe



def file_hash_from_path(file_path):
    """
    Return the hexadecimal file hash for a file

    Parameters
    ----------
    file_path: Union[str, Path]
        path to a file

    Returns
    -------
    str:
        The file hash (Blake2b; hexadecimal) of the file
    """
    hasher = hashlib.blake2b()
    with open(file_path, 'rb') as in_file:
        chunk = in_file.read(1000000)
        while len(chunk) > 0:
            hasher.update(chunk)
            chunk = in_file.read(1000000)
    return hasher.hexdigest()
