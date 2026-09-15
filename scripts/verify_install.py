"""Smoke-check the simulator stack: launches Isaac Sim headless, then imports torch/Isaac Lab/TorchRL."""

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch  # noqa: E402
import torchrl  # noqa: E402
import isaaclab  # noqa: E402
import isaaclab_physx  # noqa: E402,F401
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper  # noqa: E402,F401

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} torchrl={torchrl.__version__}")
print(f"isaaclab={getattr(isaaclab, '__version__', 'unknown')}")
assert torch.cuda.is_available(), "CUDA not available"
print("VERIFY_OK")
app.close()
