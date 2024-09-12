from pathlib import Path

import torch
from tensorboardX import SummaryWriter

from diffusion.generate import generate_images
from diffusion.plotting import plot_images
from diffusion.unet import UNet


def get_latest_state_dict(
    xp_id:int=None,
    epoch:int=None,
) -> tuple[dict, int, int]:
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



def generate_and_log_on_cpu(
    xp_id:int,
    epoch:int,
    batch_size:int=25,
    height:int=109,
    width:int=89,
    LOG_DIR:Path=None,
    seed:int=4,
    num_steps:int=20,
):
    """
    Restore the model from a checkpoint, generate some new images, plot and save
    as png. This function assumes checkpoints are saved in ROOT / checkpoints /
    """
    model_state_dict, _, _ = get_latest_state_dict(xp_id=xp_id, epoch=epoch)
    model = UNet(device="cpu")
    model.load_state_dict(model_state_dict)
    x_new = generate_images(
        model=model,
        num_steps=num_steps,
        batchsize=batch_size,
        seed=seed,
        height=height,
        width=width,
    )
    x_new_reshape = x_new.reshape((-1, 3, height, width))

    plot_images(
        x_new_reshape,
        f"pics/celebA_{xp_id}_{epoch}_seed_{seed}.png",
        ncol=num_steps,
        nrow=batch_size,
    )

    if LOG_DIR is not None:
        writer = SummaryWriter(log_dir=LOG_DIR)
        for i in range(batch_size):
            writer.add_images(f"generated {i}", x_new[i, :, :, :, :], epoch)


if __name__=="__main__":
    generate_and_log_on_cpu(
        xp_id=92,
        epoch=38,
        height=109,
        width=89,
        seed=10,
    )

