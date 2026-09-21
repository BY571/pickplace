"""Stage 2.1: TD3+BC from the camera shards (see ../README.md).

Usage: python pipeline/2_1_offline_rl/td3_bc/train.py [key=value ...]   (config: config.yaml next to this file)
"""

import os

import hydra
from omegaconf import DictConfig

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path: str, name: str):
    """Load a module by file path: with cameras enabled, Isaac Sim's bundled ``cv2/utils`` shadows ``utils``."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@hydra.main(config_path="", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    from pickplace.app import launch_app

    # Cameras on: the student is evaluated online in the camera env between gradient steps.
    launch_app(headless=cfg.app.headless, enable_cameras=bool(cfg.eval.interval), device=cfg.device)

    runner = _load(os.path.join(os.path.dirname(HERE), "runner.py"), "offline_runner")
    algo = _load(os.path.join(HERE, "utils.py"), "td3_bc_utils")
    runner.train(cfg, algo.make_algo, "td3_bc")


if __name__ == "__main__":
    main()
