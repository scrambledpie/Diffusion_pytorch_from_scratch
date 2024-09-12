import datetime
from pathlib import Path


ROOT = Path(__file__).parent.parent

CHECKPOINTS_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "tensorboard_logs"
PICS_DIR =  ROOT / "pics"


def make_new_folders() -> tuple[Path, Path]:
    """ Make brand new folders for checkpointing and logging """

    # xp_id: an integer theat increments for eacn new eXPeriment
    # get all the folder xp_id prefixed integers and go one higher
    xp_folders = CHECKPOINTS_DIR.glob("*")
    xp_id = 0
    for folder in xp_folders:
        try:
            xp_id_old = int(folder.stem.split("_")[0])
            xp_id = max(xp_id, xp_id_old + 1)
        except ValueError:
            pass

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    MODEL_NAME = f"{xp_id}_{timestamp}"

    CHECKPOINT_DIR = CHECKPOINTS_DIR / MODEL_NAME
    CHECKPOINT_DIR.mkdir()

    LOG_DIR = LOGS_DIR / MODEL_NAME
    LOG_DIR.mkdir()

    print(f"Made new folders: {CHECKPOINT_DIR}, {LOG_DIR}")

    return CHECKPOINT_DIR, LOG_DIR
