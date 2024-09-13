import torch

from diffusion.generate import generate_images
from diffusion.plotting import plot_images
from diffusion.unet.unet import UNet

from folders import CHECKPOINTS_DIR, PICS_DIR


def restore_latest_model(
    xp_id:int=None,
    epoch:int=None,
    device:str="cpu"
) -> tuple[dict, int, int]:
    """
    Read the checkpoints folder, get the lastest checkpoint from the latest
    experiment.

    WARNING: this code is fragile! It assumes the checkpoint folder hasn't got
    anything other than checkpoint sub directories and epoch files!
    e.g.
        CHECKPOINTS_DIR
          - 4_20240910_193059
             - 1 epoch.pt
             - 2 epoch.pt
             - 3 epoch.pt
          - 5_20240910_1935212
             - 1 epoch.pt
             - 2 epoch.pt
             - 3 epoch.pt
    """
    if xp_id is None:
        # if no xp_id provided, just get the latest one
        xp_ids = [int(f.stem.split("_")[0]) for f in CHECKPOINTS_DIR.glob("*")]
        xp_id = max(xp_ids)

    xp_dir = list(CHECKPOINTS_DIR.glob(f"{xp_id}_*"))[0]

    if epoch is None:
        # if no epoch provided, just get the newest file
        files = [f for f in xp_dir.glob("*.pt")]
        checkpoint_file = max(files, key=lambda f: f.stat().st_mtime)
    else:
        checkpoint_file = list(xp_dir.glob(f"{epoch} *"))[0]

    state_dict = torch.load(
        checkpoint_file,
        weights_only=True,
        map_location=torch.device(device)
    )

    model = UNet(device=device)
    model.load_state_dict(state_dict)

    print(f"Loaded model: {checkpoint_file}")

    return model


def generate_and_log_on_cpu(
    xp_id:int,
    epoch:int,
    batch_size:int=25,
    height:int=109,
    width:int=89,
    seed:int=4,
    num_steps:int=20,
):
    """
    Restore the model from a checkpoint, generate some new images, plot and save
    as png. This function assumes checkpoints are saved in ROOT / checkpoints /
    """
    model = restore_latest_model(xp_id=xp_id, epoch=epoch)

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
        x=x_new_reshape,
        filename=PICS_DIR / f"{xp_id}_{epoch}_seed_{seed}.png",
        ncol=num_steps,
        nrow=batch_size,
    )


if __name__=="__main__":
    # xp 92 was trained on celebA, height=109, width=89
    generate_and_log_on_cpu(
        xp_id=92,
        epoch=38,
        height=109,
        width=89,
        seed=19,
        batch_size=2,
        num_steps=10,
    )

