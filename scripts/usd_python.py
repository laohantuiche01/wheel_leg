"""Use the installed standalone USD libraries without starting Kit or Isaac Sim."""
import importlib.util
import os
from pathlib import Path
import sys
import sysconfig


def enable_usd():
    if importlib.util.find_spec("pxr") is not None:
        return
    cache = Path(sysconfig.get_paths()["purelib"]) / "isaacsim/extscache"
    candidates = sorted(cache.glob("omni.usd.libs-*/pxr/Usd/__init__.py"))
    if len(candidates) != 1:
        raise RuntimeError("Use the existing Isaac Python environment or set PYTHONPATH/LD_LIBRARY_PATH for pxr")
    library = candidates[0].parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(library) + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = str(library / "bin") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    os.execve(sys.executable, [sys.executable, *sys.argv], env)
