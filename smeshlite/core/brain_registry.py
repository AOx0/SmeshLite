"""
brain_registry — auto-discovery of CharacterBrain subclasses from /agents.

Drop a .py file into the top-level /agents directory and any
CharacterBrain subclass *defined in that file* is registered, keyed by
its BRAIN_NAME (or class name if BRAIN_NAME is unset).  /agents is plain
user content (not an installable subpackage), so files are loaded via
importlib by path rather than by package import.

A broken agent file (syntax error, missing dependency, etc.) is skipped
with a warning rather than aborting discovery of the rest.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from smeshlite.core.brain import CharacterBrain

# smeshlite/core/brain_registry.py -> smeshlite/core -> smeshlite -> <repo root>
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENTS_DIR = _PROJECT_ROOT / "agents"


@dataclass
class BrainEntry:
    """One registered brain implementation discovered from /agents."""
    name: str
    cls: type[CharacterBrain]
    module_name: str
    source_file: Path


def discover_brains(agents_dir: Path | str | None = None) -> dict[str, BrainEntry]:
    """Scan agents_dir for CharacterBrain subclasses, keyed by display name."""
    agents_dir = Path(agents_dir) if agents_dir is not None else DEFAULT_AGENTS_DIR
    registry: dict[str, BrainEntry] = {}

    if not agents_dir.is_dir():
        return registry

    for py_file in sorted(agents_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"agents.{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            warnings.warn(
                f"brain_registry: skipping '{py_file.name}' — failed to import: {exc}",
                stacklevel=2,
            )
            sys.modules.pop(module_name, None)
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, CharacterBrain)
                and obj is not CharacterBrain
                and obj.__module__ == module.__name__
            ):
                display_name = getattr(obj, "BRAIN_NAME", None) or obj.__name__
                if display_name in registry:
                    warnings.warn(
                        f"brain_registry: duplicate brain name '{display_name}' "
                        f"in '{py_file.name}' overrides "
                        f"'{registry[display_name].source_file.name}'",
                        stacklevel=2,
                    )
                registry[display_name] = BrainEntry(
                    name=display_name,
                    cls=obj,
                    module_name=module_name,
                    source_file=py_file,
                )

    return registry


_cache: dict[str, BrainEntry] | None = None


def list_brains(agents_dir: Path | str | None = None, *, refresh: bool = False) -> list[str]:
    """Return sorted display names of all discovered brains."""
    global _cache
    if _cache is None or refresh:
        _cache = discover_brains(agents_dir)
    return sorted(_cache.keys())


def get_brain_class(
    name: str, agents_dir: Path | str | None = None, *, refresh: bool = False
) -> type[CharacterBrain]:
    """Look up a registered brain class by display name. Raises KeyError if not found."""
    global _cache
    if _cache is None or refresh:
        _cache = discover_brains(agents_dir)
    if name not in _cache:
        available = ", ".join(sorted(_cache.keys())) or "(none)"
        raise KeyError(f"Unknown brain '{name}'. Available: {available}")
    return _cache[name].cls


def create_brain(
    name: str, *args, agents_dir: Path | str | None = None, refresh: bool = False, **kwargs
) -> CharacterBrain:
    """Instantiate a registered brain by display name, passing through any args/kwargs."""
    cls = get_brain_class(name, agents_dir=agents_dir, refresh=refresh)
    return cls(*args, **kwargs)
