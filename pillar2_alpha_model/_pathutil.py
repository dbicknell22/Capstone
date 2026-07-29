"""Path resolution that works whether this code runs as a script (where
`__file__` is defined) or is pasted/`%load`-ed into a notebook cell (where
it isn't — notebooks never set `__file__` for cell code, only for an actual
imported module file). Every other module in this package should resolve
paths through here rather than touching `__file__` directly.
"""
from pathlib import Path


def _candidate_dirs() -> list:
    dirs = []
    try:
        dirs.append(Path(__file__).resolve().parent)
    except NameError:
        pass

    cwd = Path.cwd()
    dirs.append(cwd)
    dirs.extend(cwd.parents)
    if cwd.exists():
        dirs.extend(p for p in cwd.iterdir() if p.is_dir())

    seen, ordered = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def find_dir_containing(marker: str) -> Path:
    """The first directory (checking this file's own folder, the current
    working directory, its parents, and its immediate subdirectories, in
    that order) that contains `marker`."""
    for d in _candidate_dirs():
        if (d / marker).exists():
            return d
    raise FileNotFoundError(
        f"Could not find a directory containing '{marker}'. Searched this "
        f"module's folder (if running as a script), the current working "
        f"directory ({Path.cwd()}), its parent directories, and its "
        f"immediate subdirectories. Run this from inside the cloned repo, "
        f"or change the working directory to somewhere near it first."
    )
