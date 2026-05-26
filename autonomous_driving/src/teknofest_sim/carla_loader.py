import glob
import os
import sys


def load_carla(carla_root: str):
    """
    CARLA PythonAPI egg yolunu otomatik sys.path'e ekler.
    Ubuntu + Python 3.10 için CARLA_0.9.15 kurulumu hedeflenmiştir.
    """
    carla_root = os.path.abspath(os.path.expanduser(carla_root))

    egg_pattern = os.path.join(
        carla_root,
        "PythonAPI",
        "carla",
        "dist",
        "carla-*%d.%d-%s.egg"
        % (
            sys.version_info.major,
            sys.version_info.minor,
            "linux-x86_64",
        ),
    )

    eggs = glob.glob(egg_pattern)
    if eggs and eggs[0] not in sys.path:
        sys.path.append(eggs[0])

    python_api = os.path.join(carla_root, "PythonAPI", "carla")
    if python_api not in sys.path:
        sys.path.append(python_api)

    import carla

    return carla