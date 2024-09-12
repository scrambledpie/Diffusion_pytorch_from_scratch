from pathlib import Path
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
        import pdb; pdb.set_trace()
        raise TypeError(f"WTF if this?? {type(obj)}")

    if printout:
        pprint(output)
    else:
        return output



def get_latest_state_dict(
    xp_id:int=None,
    epoch:int=None,
) -> dict:
    """
    Read the checkpoints folder, get the lastest checkpoint from the latest
    experiment.
    """
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"

    if xp_id is None:
        xp_ids = [int(f.stem.split("_")[0]) for f in checkpoint_dir.glob("*")]
        xp_id = max(xp_ids)

    xp_dir = list(checkpoint_dir.glob(f"{xp_id}_*"))[0]

    if epoch is None:
        epochs = [int(f.stem.split(" ")[0]) for f in xp_dir.glob("*")]
        if len(epochs) == 0:
            raise FileNotFoundError(f"Experiment {xp_id} has no saved data")
        epoch = max(epochs)

    checkpont_file = list(xp_dir.glob(f"{epoch} *"))[0]
    state_dict = torch.load(
        checkpont_file,
        weights_only=True,
        map_location=torch.device('cpu')
    )
    print(f"Loaded model: {checkpont_file}")
    return state_dict, xp_id, epoch
