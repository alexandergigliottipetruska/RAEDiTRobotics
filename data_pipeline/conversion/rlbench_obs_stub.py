"""Minimal stub to allow unpickling RLBench Observation objects without
installing the full rlbench package (which requires PyRep + CoppeliaSim).

Usage — call this BEFORE any pickle.load() of low_dim_obs.pkl:

    from data_pipeline.conversion.rlbench_obs_stub import register_stub
    register_stub()

How it works:
    pickle stores class references as module paths. When loading, Python
    resolves `rlbench.backend.observation.Observation`. We insert a fake
    module at that path in sys.modules with a bare Observation class.
    pickle then sets attributes directly via __dict__, so all fields
    (gripper_pose, gripper_open, joint_positions, ...) are preserved.
"""

import sys
import types


def register_stub() -> None:
    """Insert minimal rlbench stubs into sys.modules if not already present."""
    if "rlbench" in sys.modules:
        return  # real package already installed, nothing to do

    class Observation:
        """Stub — pickle sets all fields via __dict__."""
        pass

    class Demo(list):
        """Stub for rlbench.demo.Demo (a typed list of Observations).

        The real Demo stores items in `_observations` (set by pickle via
        __dict__), and delegates __len__/__getitem__ to that list.
        We replicate that so `obs_list[0]` and `len(obs_list)` work.
        """

        def __len__(self):
            if hasattr(self, "_observations"):
                return len(self._observations)
            return super().__len__()

        def __getitem__(self, idx):
            if hasattr(self, "_observations"):
                return self._observations[idx]
            return super().__getitem__(idx)

        def __iter__(self):
            if hasattr(self, "_observations"):
                return iter(self._observations)
            return super().__iter__()

    # Build fake module hierarchy
    _rlbench           = types.ModuleType("rlbench")
    _rlbench_backend   = types.ModuleType("rlbench.backend")
    _rlbench_obs       = types.ModuleType("rlbench.backend.observation")
    _rlbench_demo      = types.ModuleType("rlbench.demo")

    _rlbench_obs.Observation  = Observation
    _rlbench_demo.Demo        = Demo
    _rlbench.backend          = _rlbench_backend
    _rlbench_backend.observation = _rlbench_obs

    sys.modules["rlbench"]                    = _rlbench
    sys.modules["rlbench.backend"]            = _rlbench_backend
    sys.modules["rlbench.backend.observation"] = _rlbench_obs
    sys.modules["rlbench.demo"]               = _rlbench_demo
