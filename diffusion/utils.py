from pprint import pprint
from collections import OrderedDict

import torch


def get_shape(obj, printout:bool=True):
    """
    Scan a nested dict/list for tensors and print out all their shapes in the
    same nested structure.
    """
    if isinstance(obj, list):
        output = [get_shape(x, False) for x in obj]

    elif isinstance(obj, dict) or isinstance(obj, OrderedDict):
        output = {k:get_shape(v, False) for k, v in obj.items()}

    elif isinstance(obj, torch.Tensor):
        output = obj.shape

    else:
        raise TypeError(f"WTF if this?? {type(obj)}")

    if printout:
        pprint(output)
    else:
        return output
