"""
Tests for smeshlite.core.brain_registry — discovery, collisions, broken files.
"""
import pytest

from smeshlite.core.brain import CharacterBrain, InputState
from smeshlite.core import brain_registry
from smeshlite.core.brain_registry import discover_brains


GOOD_BRAIN_SRC = '''
from smeshlite.core.brain import CharacterBrain, InputState

class GoodBot(CharacterBrain):
    BRAIN_NAME = "Good Bot"

    def think(self, context, out):
        out.right = True
'''

NO_NAME_BRAIN_SRC = '''
from smeshlite.core.brain import CharacterBrain, InputState

class UnnamedBot(CharacterBrain):
    def think(self, context, out):
        out.left = True
'''

REEXPORT_SRC = '''
# Re-exports ExternalBrain without subclassing it locally — should NOT register.
from smeshlite.core.brain import ExternalBrain, CharacterBrain

class LocalBot(CharacterBrain):
    BRAIN_NAME = "Local Only"
    def think(self, context, out):
        pass
'''

BROKEN_SRC = '''
this is not valid python (((
'''

COLLISION_SRC = '''
from smeshlite.core.brain import CharacterBrain

class AnotherGoodBot(CharacterBrain):
    BRAIN_NAME = "Good Bot"   # collides with GOOD_BRAIN_SRC's name

    def think(self, context, out):
        out.attack = True
'''


@pytest.fixture(autouse=True)
def _reset_registry_cache():
    brain_registry._cache = None
    yield
    brain_registry._cache = None


def _write(tmp_path, filename, src):
    (tmp_path / filename).write_text(src)


def test_discover_finds_brain_with_brain_name(tmp_path):
    _write(tmp_path, "good.py", GOOD_BRAIN_SRC)
    registry = discover_brains(tmp_path)
    assert "Good Bot" in registry
    assert registry["Good Bot"].cls.__name__ == "GoodBot"


def test_discover_falls_back_to_class_name(tmp_path):
    _write(tmp_path, "unnamed.py", NO_NAME_BRAIN_SRC)
    registry = discover_brains(tmp_path)
    assert "UnnamedBot" in registry


def test_discover_excludes_imported_not_subclassed(tmp_path):
    _write(tmp_path, "reexport.py", REEXPORT_SRC)
    registry = discover_brains(tmp_path)
    assert "Local Only" in registry
    assert all(name != "ExternalBrain" for name in registry)


def test_discover_skips_broken_file_with_warning(tmp_path):
    _write(tmp_path, "broken.py", BROKEN_SRC)
    _write(tmp_path, "good.py", GOOD_BRAIN_SRC)

    with pytest.warns(UserWarning, match="broken.py"):
        registry = discover_brains(tmp_path)

    assert "Good Bot" in registry
    assert len(registry) == 1


def test_discover_handles_name_collision_with_warning(tmp_path):
    _write(tmp_path, "a_good.py", GOOD_BRAIN_SRC)
    _write(tmp_path, "b_collision.py", COLLISION_SRC)

    with pytest.warns(UserWarning, match="duplicate brain name"):
        registry = discover_brains(tmp_path)

    # b_collision.py sorts after a_good.py -> last-write-wins
    assert registry["Good Bot"].cls.__name__ == "AnotherGoodBot"


def test_discover_empty_dir_returns_empty(tmp_path):
    registry = discover_brains(tmp_path)
    assert registry == {}


def test_discover_missing_dir_returns_empty(tmp_path):
    registry = discover_brains(tmp_path / "does_not_exist")
    assert registry == {}


def test_get_brain_class_and_create_brain(tmp_path):
    _write(tmp_path, "good.py", GOOD_BRAIN_SRC)
    registry = discover_brains(tmp_path)
    cls = registry["Good Bot"].cls
    instance = cls()
    assert isinstance(instance, CharacterBrain)

    out = InputState()
    instance.think(None, out)
    assert out.right is True


def test_list_get_create_brain_with_explicit_dir(tmp_path):
    _write(tmp_path, "good.py", GOOD_BRAIN_SRC)

    names = brain_registry.list_brains(tmp_path, refresh=True)
    assert names == ["Good Bot"]

    cls = brain_registry.get_brain_class("Good Bot", tmp_path)
    assert cls.__name__ == "GoodBot"

    brain = brain_registry.create_brain("Good Bot", agents_dir=tmp_path)
    assert isinstance(brain, CharacterBrain)


def test_get_brain_class_unknown_raises(tmp_path):
    _write(tmp_path, "good.py", GOOD_BRAIN_SRC)
    with pytest.raises(KeyError):
        brain_registry.get_brain_class("Nonexistent Bot", tmp_path, refresh=True)


# -----------------------------------------------------------------------
# Real /agents directory (project root) — smoke test
# -----------------------------------------------------------------------

def test_default_agents_dir_discovers_examples():
    """The shipped /agents examples should be discoverable without errors."""
    registry = discover_brains()  # uses DEFAULT_AGENTS_DIR
    names = set(registry.keys())
    assert "Random Bot" in names
    assert "Chaser Bot" in names
    # SB3 template should be discoverable even if SB3 isn't installed,
    # since the import is deferred to __init__.
    assert "SB3 Agent (template)" in names
