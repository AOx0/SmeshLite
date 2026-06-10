"""
Tests for smeshlite.core.sensors — Stage.raycast geometry and brain SENSORS wiring.
"""
import pytest

from smeshlite.core.brain import CharacterBrain
from smeshlite.core.character import Character
from smeshlite.core.sensors import RaycastSensor, RaycastResult, evaluate_sensors
from smeshlite.core.stage import Stage
from smeshlite.data.stage_data import DEFAULT_STAGE


class FakeChar:
    """Minimal stand-in exposing the attributes evaluate_sensors() reads."""

    def __init__(self, x, y, vx=0.0, vy=0.0, facing=1.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.facing = facing


@pytest.fixture
def stage():
    return Stage(DEFAULT_STAGE)


# ---------------------------------------------------------------------------
# Stage.raycast geometry
# ---------------------------------------------------------------------------

def test_raycast_hits_main_platform_below(stage):
    result = stage.raycast(0.0, 50.0, 0.0, -1.0, max_dist=100.0)
    assert result.hit
    assert result.distance == pytest.approx(50.0)
    assert result.end == pytest.approx((0.0, 0.0))


def test_raycast_misses_when_nothing_in_range(stage):
    # Straight up from above the main platform: nothing overhead within range.
    result = stage.raycast(0.0, 50.0, 0.0, 1.0, max_dist=50.0)
    assert not result.hit
    assert result.distance == pytest.approx(50.0)
    assert result.end == pytest.approx((0.0, 100.0))


def test_raycast_hits_side_platform(stage):
    # Left side platform spans x in [-200, -60] at y=85.
    result = stage.raycast(-130.0, 100.0, 0.0, -1.0, max_dist=20.0)
    assert result.hit
    assert result.distance == pytest.approx(15.0)
    assert result.end == pytest.approx((-130.0, 85.0))


def test_raycast_hits_kill_wall(stage):
    result = stage.raycast(690.0, 0.0, 1.0, 0.0, max_dist=50.0)
    assert result.hit
    assert result.distance == pytest.approx(10.0)
    assert result.end == pytest.approx((700.0, 0.0))


def test_raycast_hits_kill_floor(stage):
    result = stage.raycast(0.0, -340.0, 0.0, -1.0, max_dist=20.0)
    assert result.hit
    assert result.distance == pytest.approx(10.0)
    assert result.end == pytest.approx((0.0, -350.0))


def test_raycast_zero_direction_falls_back_to_straight_down(stage):
    result = stage.raycast(0.0, 10.0, 0.0, 0.0, max_dist=20.0)
    assert result.hit
    assert result.distance == pytest.approx(10.0)
    assert result.end == pytest.approx((0.0, 0.0))


def test_raycast_result_value_normalizes_distance(stage):
    hit = stage.raycast(0.0, 8.0, 0.0, -1.0, max_dist=10.0)
    assert hit.value == pytest.approx(0.2)

    miss = stage.raycast(0.0, 50.0, 0.0, 1.0, max_dist=50.0)
    assert miss.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_sensors: local-space resolution, facing mirror, velocity mode
# ---------------------------------------------------------------------------

def test_static_offset_mirrors_with_facing(stage):
    sensor = RaycastSensor(name="forward_down", origin=(5.0, 0.0), direction=(1.0, -1.0), max_dist=50.0)

    facing_right = FakeChar(x=0.0, y=10.0, facing=1.0)
    r_right = evaluate_sensors((sensor,), facing_right, stage)["forward_down"]
    assert r_right.origin == pytest.approx((5.0, 10.0))
    assert r_right.end[0] > r_right.origin[0]  # ray points toward +x

    facing_left = FakeChar(x=0.0, y=10.0, facing=-1.0)
    r_left = evaluate_sensors((sensor,), facing_left, stage)["forward_down"]
    assert r_left.origin == pytest.approx((-5.0, 10.0))
    assert r_left.end[0] < r_left.origin[0]  # ray points toward -x


def test_velocity_origin_and_direction(stage):
    sensor = RaycastSensor(name="predictive", origin="velocity", direction="velocity", max_dist=20.0)

    char = FakeChar(x=0.0, y=20.0, vx=4.0, vy=-3.0, facing=1.0)
    result = evaluate_sensors((sensor,), char, stage)["predictive"]

    assert result.origin == pytest.approx((4.0, 17.0))
    assert result.end[0] > result.origin[0]  # follows +vx
    assert result.end[1] < result.origin[1]  # follows -vy


def test_velocity_falls_back_to_no_offset_and_straight_down_when_idle(stage):
    sensor = RaycastSensor(name="predictive", origin="velocity", direction="velocity", max_dist=10.0)

    char = FakeChar(x=0.0, y=9.0, vx=0.0, vy=0.0, facing=1.0)
    result = evaluate_sensors((sensor,), char, stage)["predictive"]

    assert result.origin == pytest.approx((0.0, 9.0))
    assert result.hit
    assert result.distance == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Character integration
# ---------------------------------------------------------------------------

class EyeBrain(CharacterBrain):
    SENSORS = (
        RaycastSensor(name="ground_below", origin=(0.0, -2.0), direction=(0.0, -1.0), max_dist=10.0),
    )

    def think(self, context, out):
        pass


class NoEyeBrain(CharacterBrain):
    def think(self, context, out):
        pass


def test_character_with_sensors_populates_context_and_cache(stage):
    char = Character(player_id=0, spawn_x=0.0, spawn_y=10.0)
    char.set_brain(EyeBrain())

    char.gather_input([], stage)

    assert "ground_below" in char._sensor_results
    result = char._sensor_results["ground_below"]
    assert isinstance(result, RaycastResult)
    assert result.hit
    assert result.distance == pytest.approx(8.0)


def test_character_without_sensors_has_empty_results(stage):
    char = Character(player_id=0, spawn_x=0.0, spawn_y=10.0)
    char.set_brain(NoEyeBrain())

    char.gather_input([], stage)

    assert char._sensor_results == {}
