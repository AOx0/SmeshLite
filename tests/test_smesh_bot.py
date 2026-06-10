"""
Tests for agents.smesh_bot.SmeshBot — the ported original-game CPU AI.
"""
import pytest

from smeshlite.core.brain import BrainContext, InputState, OpponentContext
from smeshlite.core.character import Action
from smeshlite.core.sensors import RaycastResult
from smeshlite.core.brain_registry import discover_brains

from agents.smesh_bot import SmeshBot


def make_sensor(hit: bool = True, distance: float = 10.0, max_dist: float = 60.0) -> RaycastResult:
    return RaycastResult(origin=(0.0, 0.0), end=(0.0, 0.0), distance=distance, max_dist=max_dist, hit=hit)


def make_context(**overrides) -> BrainContext:
    defaults = dict(
        x=0.0, y=0.0, vx=0.0, vy=0.0,
        in_air=False, facing=1.0,
        damage_pct=0.0, stocks=3,
        action=Action.RUN.value, action_frame=0, attack_num=0, charge_amount=0.0,
        invincibility=0,
        opponents=[],
        sensors={"ground_ahead": make_sensor(hit=True), "ground_below": make_sensor(hit=True)},
    )
    defaults.update(overrides)
    return BrainContext(**defaults)


def make_opponent(**overrides) -> OpponentContext:
    defaults = dict(
        x=100.0, y=0.0, vx=0.0, vy=0.0,
        damage_pct=0.0, stocks=3, facing=-1.0,
        in_air=False, action=Action.NONE.value, attack_num=0, action_frame=0, charge_amount=0.0,
    )
    defaults.update(overrides)
    return OpponentContext(**defaults)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discoverable_as_smesh_bot():
    registry = discover_brains()
    assert "Smesh Bot" in registry
    assert registry["Smesh Bot"].cls.__name__ == "SmeshBot"


# ---------------------------------------------------------------------------
# Basic safety / no-op cases
# ---------------------------------------------------------------------------

def test_no_opponents_does_nothing():
    bot = SmeshBot()
    out = InputState()
    bot.think(make_context(opponents=[]), out)
    assert out == InputState()


# ---------------------------------------------------------------------------
# Pursuit
# ---------------------------------------------------------------------------

def test_pursues_opponent_to_the_right():
    bot = SmeshBot(attack_frequency=0.0, jump_frequency=0.0)
    out = InputState()
    ctx = make_context(x=0.0, opponents=[make_opponent(x=200.0, y=0.0)])
    bot.think(ctx, out)
    assert out.right is True
    assert out.left is False


def test_jumps_when_opponent_is_above():
    bot = SmeshBot(attack_frequency=0.0, jump_frequency=0.0)
    out = InputState()
    ctx = make_context(x=0.0, y=0.0, opponents=[make_opponent(x=200.0, y=100.0)])
    bot.think(ctx, out)
    assert out.up is True


def test_jumps_at_a_ledge_when_grounded():
    bot = SmeshBot(attack_frequency=0.0, jump_frequency=0.0)
    out = InputState()
    ctx = make_context(
        x=0.0, y=0.0, in_air=False,
        opponents=[make_opponent(x=200.0, y=0.0)],
        sensors={"ground_ahead": make_sensor(hit=False), "ground_below": make_sensor(hit=True)},
    )
    bot.think(ctx, out)
    assert out.up is True


# ---------------------------------------------------------------------------
# React: invulnerable opponent -> retreat
# ---------------------------------------------------------------------------

def test_retreats_from_respawning_opponent():
    bot = SmeshBot()
    out = InputState()
    ctx = make_context(x=100.0, opponents=[make_opponent(x=200.0, action=Action.RESPAWN.value)])
    bot.think(ctx, out)
    assert out.left is True
    assert out.right is False


# ---------------------------------------------------------------------------
# React: fall recovery, fired once per airborne excursion
# ---------------------------------------------------------------------------

def test_fall_recovery_fires_once_then_latches():
    bot = SmeshBot()
    falling_ctx = make_context(
        x=-400.0, y=200.0, vy=-5.0, in_air=True,
        opponents=[make_opponent()],
        sensors={"ground_ahead": make_sensor(hit=True), "ground_below": make_sensor(hit=False)},
    )

    out = InputState()
    bot.think(falling_ctx, out)
    assert out.up is True
    assert out.attack is True

    # Same falling/no-ground situation next tick: must NOT re-trigger recovery.
    out2 = InputState()
    bot.think(falling_ctx, out2)
    assert out2.up is False

    # Landing resets the latch.
    grounded_ctx = make_context(x=-400.0, y=85.0, vy=0.0, in_air=False, opponents=[make_opponent()])
    bot.think(grounded_ctx, InputState())

    out3 = InputState()
    bot.think(falling_ctx, out3)
    assert out3.up is True


# ---------------------------------------------------------------------------
# Charge attack state machine
# ---------------------------------------------------------------------------

def test_charge_attack_starts_and_holds_while_aimed_at_target():
    bot = SmeshBot(attack_frequency=1.0, aggressiveness_distance=120.0, jump_frequency=0.0)
    target = make_opponent(x=50.0, y=0.0, damage_pct=0.0)
    ctx = make_context(x=0.0, y=0.0, in_air=False, opponents=[target])

    out = InputState()
    bot.think(ctx, out)
    assert bot._charging is True
    assert out.attack is True

    # Next tick: opponent hasn't moved, charge not yet full -> keep charging.
    ctx2 = make_context(
        x=0.0, y=0.0, in_air=False,
        action=Action.CHARGING.value, charge_amount=0.1,
        opponents=[target],
    )
    out2 = InputState()
    bot.think(ctx2, out2)
    assert bot._charging is True
    assert out2.attack is True


def test_charge_attack_exits_when_charged_and_in_sweetspot_range():
    bot = SmeshBot(attack_frequency=1.0, likes_charging_up_to=60.0, jump_frequency=0.0)
    target = make_opponent(x=50.0, y=0.0, damage_pct=0.0)
    bot.think(make_context(x=0.0, y=0.0, in_air=False, opponents=[target]), InputState())
    assert bot._charging is True

    # Fully charged and within the default sweetspot distance (80) -> exit.
    ctx = make_context(
        x=0.0, y=0.0, in_air=False,
        action=Action.CHARGING.value, charge_amount=1.0, attack_num=2,
        opponents=[target],
    )
    out = InputState()
    bot.think(ctx, out)
    assert bot._charging is False
    assert out.attack is False
    assert bot._watch_hit_frames > 0


def test_charge_attack_sweetspot_updates_after_a_landed_hit():
    bot = SmeshBot(attack_frequency=1.0, jump_frequency=0.0)
    target = make_opponent(x=50.0, y=0.0, damage_pct=0.0)
    bot.think(make_context(x=0.0, y=0.0, in_air=False, opponents=[target]), InputState())

    exit_ctx = make_context(
        x=0.0, y=0.0, in_air=False,
        action=Action.CHARGING.value, charge_amount=1.0, attack_num=3,
        opponents=[target],
    )
    bot.think(exit_ctx, InputState())
    assert bot._charging is False
    assert 3 not in bot._sweetspots

    # Opponent's damage rose -> the charge attack connected; sweetspot learned.
    hit_target = make_opponent(x=50.0, y=0.0, damage_pct=15.0)
    bot.think(make_context(x=0.0, y=0.0, in_air=False, opponents=[hit_target]), InputState())
    assert 3 in bot._sweetspots


def test_charge_attack_bails_if_opponent_moves_away():
    bot = SmeshBot(attack_frequency=1.0, jump_frequency=0.0)
    target = make_opponent(x=50.0, y=0.0, damage_pct=0.0)
    bot.think(make_context(x=0.0, y=0.0, in_air=False, opponents=[target]), InputState())
    assert bot._charging is True

    # Opponent darts to the opposite side -> aim direction flips, dot < 0.5 -> bail.
    moved_target = make_opponent(x=-50.0, y=0.0, damage_pct=0.0)
    ctx = make_context(
        x=0.0, y=0.0, in_air=False,
        action=Action.CHARGING.value, charge_amount=0.2,
        opponents=[moved_target],
    )
    out = InputState()
    bot.think(ctx, out)
    assert bot._charging is False
    assert out.attack is False
