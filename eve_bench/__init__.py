from . import aorticarch

import gymnasium as gym


gym.register(
    id="eve_bench/arch_vmr94",
    entry_point="eve_bench.aorticarch:ArchVMR94",
    kwargs={
        "normalize_obs": True,
        "init_visual": False,
        "target_reached_threshold": 5,
        "step_limit": 150,
        "normalize_action": False,
        "obs_type": aorticarch.ObservationType.TRACKING,
    },
)
