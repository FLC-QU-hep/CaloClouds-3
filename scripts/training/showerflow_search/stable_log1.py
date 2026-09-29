"""
Compatibility shim: the log1_stable variant now lives in the package proper,
at pointcloud/models/stable_log1.py, so the training/generation pipeline can
import it too (it used to be importable only from this search directory).

Kept as a re-export so `import stable_log1` in train_trial.py kept working and
there is still exactly ONE definition of LOG_OFFSET and the transforms - the
module docstring over there explains why two copies that can drift apart would
be a bug, not a convenience.
"""

from pointcloud.models.stable_log1 import (  # noqa: F401
    LOG_OFFSET,
    REQUIRED_CACHE_FILES,
    ClampedSafeExpTransform,
    StableLog1Factory,
    compile_HybridTanH_log1_stable,
    compute_log_stats,
    ensure_registered,
    register,
)
