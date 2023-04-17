import collections

import prody
import base64
import pickle
import inspect

def calculate_rmsd(handle1, handle2):
    handle1 = handle1.copy()
    prody.calcTransformation(handle1, handle2).apply(handle1)
    return prody.calcRMSD(handle1, handle2)


def set_residue_data(structure, name, values, is_bool=False):
    resindex_to_value = dict(zip(structure.ca.getResindices(), values))
    arr = [resindex_to_value[i] for i in structure.getResindices()]
    if is_bool:
        structure.setFlags(name, arr)
    else:
        structure.setData(name, arr)


def serialize(obj):
    encoded = base64.b64encode(
        pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
    return encoded.decode("ascii")


def deserialize(data):
    decoded = base64.b64decode(data)
    return pickle.loads(decoded)


def args_from_function_signature(function, include=[], exclude=[]):
    signature = inspect.signature(function)
    result = collections.OrderedDict()
    for parameter in signature.parameters.values():
        if parameter.name == 'self' or parameter.name in exclude:
            continue
        if include and parameter.name not in include:
            continue
        d = {}
        if parameter.annotation in (str, int, float, bool):
            d['type'] = parameter.annotation
        else:
            d['type'] = object
        if parameter.default is not parameter.empty:
            d['default'] = parameter.default
        result[parameter.name] = d
    return result
