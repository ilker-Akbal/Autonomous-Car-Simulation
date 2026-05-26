from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "adas5_targeted_aug_finetune_from_old_img1024_b8_ep50"
    / "weights"
    / "best.pt"
)

TRAFFIC_LIGHT_STATE_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "traffic_light_state_resnet18_carla"
    / "best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
