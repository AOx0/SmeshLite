"""
Basic validation tests for SmeshLiteEnv.
Run with: python -m pytest tests/ -v
"""
import numpy as np
import pytest

from smeshlite import SmeshLiteEnv
from smeshlite.core.match import Match, MatchConfig
from smeshlite.core.character import Character, Action
from smeshlite.core.brain import CharacterBrain, InputState, BrainContext, ExternalBrain
from smeshlite.data.stage_data import STAGE_0
from smeshlite.data.character_def import MINIUM_DEF


# -----------------------------------------------------------------------
# Environment smoke tests
# -----------------------------------------------------------------------

def test_env_reset_returns_correct_obs_shape():
    env = SmeshLiteEnv()
    obs, info = env.reset()
    assert obs.shape == (22,)
    assert obs.dtype == np.float32


def test_env_step_idle():
    env = SmeshLiteEnv()
    env.reset()
    idle = [0, 0, 0, 0]   # left=0, right=0, up=0, attack=0
    obs, reward, terminated, truncated, info = env.step(idle)
    assert obs.shape == (22,)
    assert isinstance(reward, float)
    assert not truncated


def test_env_random_agent_1000_steps():
    env = SmeshLiteEnv()
    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        assert obs.shape == (22,)
        assert np.isfinite(r)
        if term or trunc:
            obs, _ = env.reset()
    env.close()


def test_env_terminates_on_ko():
    """Force a KO and verify done flag."""
    env = SmeshLiteEnv(stocks=1)
    env.reset()
    char = env.match.characters[1]
    # Characters spawn in RESPAWN state; ground_update skips that state.
    # Set AERIAL so the kill-zone check runs.
    char.action = Action.AERIAL
    char.x = 800.0
    char.y = 0.0
    char.ground_update(env.match.stage)
    env.match._check_win()
    assert env.match.done


def test_action_space_is_multibinary():
    from gymnasium.spaces import MultiBinary
    env = SmeshLiteEnv()
    assert isinstance(env.action_space, MultiBinary)
    assert env.action_space.n == 4


# -----------------------------------------------------------------------
# Physics tests
# -----------------------------------------------------------------------

def test_character_falls_under_gravity():
    c = Character(0, 0.0, 100.0)
    c.action = Action.AERIAL
    c.in_air = True
    initial_y = c.y
    c.physics_step()
    assert c.y < initial_y, "Character should fall under gravity"


def test_character_variable_gravity():
    """Holding up should apply lighter gravity than not holding up."""
    c_hold = Character(0, 0.0, 100.0)
    c_hold.action = Action.AERIAL
    c_hold.in_air = True
    c_hold._input.up = True

    c_free = Character(0, 0.0, 100.0)
    c_free.action = Action.AERIAL
    c_free.in_air = True
    c_free._input.up = False

    c_hold.physics_step()
    c_free.physics_step()
    assert c_hold.y > c_free.y, "Holding up should slow descent (lighter gravity)"


def test_character_lands_on_platform():
    from smeshlite.core.stage import Stage
    from smeshlite.core.character import PHYS_HALF_H
    stage = Stage(STAGE_0)
    c = Character(0, 0.0, PHYS_HALF_H + 2.0)
    c.action = Action.AERIAL
    c.in_air = True
    c.vy = -5.0
    c.physics_step()
    c.ground_update(stage)
    assert not c.in_air, "Character should land on main platform"
    assert abs(c.y - PHYS_HALF_H) < 1.0


def test_character_dies_outside_kill_zone():
    from smeshlite.core.stage import Stage
    stage = Stage(STAGE_0)
    c = Character(0, 800.0, 0.0, stocks=2)
    c.action = Action.AERIAL
    c.in_air = True
    c.ground_update(stage)
    assert c.stocks == 1
    assert c.action == Action.RESPAWN


def test_attack_hitbox_active_window():
    c = Character(0, 0.0, 0.0)
    c.action = Action.ATTACK
    c.attack_num = 1
    c.has_hit = False

    c.action_frame = 5    # before active window (active_start=13)
    assert c.get_active_hitbox() is None

    c.action_frame = 14   # inside active window
    assert c.get_active_hitbox() is not None

    c.action_frame = 20   # after active window (active_end=15)
    assert c.get_active_hitbox() is None


# -----------------------------------------------------------------------
# Damage / knockback tests
# -----------------------------------------------------------------------

def test_damage_increases_damage_pct():
    c = Character(0, 0.0, 0.0)
    c.receive_damage(10.0, 0.0, 100.0, 0.0)
    c.apply_pending_damage()
    assert c.damage_pct == pytest.approx(10.0)


def test_knockback_sets_hitstun():
    c = Character(0, 0.0, 0.0)
    c.receive_damage(10.0, 1.0, 100.0, 0.0)
    c.apply_pending_damage()
    assert c.action == Action.HITSTUN


# -----------------------------------------------------------------------
# Brain system tests
# -----------------------------------------------------------------------

def test_character_brain_interface():
    """CharacterBrain.think is a no-op; subclasses can override."""
    brain = CharacterBrain()
    ctx = BrainContext(
        x=0, y=0, vx=0, vy=0, in_air=False, facing=1.0,
        damage_pct=0, stocks=3, action=0, action_frame=0,
        attack_num=0, charge_amount=0, invincibility=0,
    )
    out = InputState()
    brain.think(ctx, out)
    assert not out.left and not out.right and not out.up and not out.attack


def test_external_brain_injects_input():
    brain = ExternalBrain()
    brain.pending = InputState(left=True, attack=True)
    ctx = BrainContext(
        x=0, y=0, vx=0, vy=0, in_air=False, facing=1.0,
        damage_pct=0, stocks=3, action=0, action_frame=0,
        attack_num=0, charge_amount=0, invincibility=0,
    )
    out = InputState()
    brain.think(ctx, out)
    assert out.left is True
    assert out.attack is True
    assert out.right is False


def test_character_set_brain_and_gather():
    """Character.gather_input fills the _input buffer from the brain."""
    brain = ExternalBrain()
    brain.pending = InputState(right=True)

    config = MatchConfig()
    match = Match(config)
    match.reset(n_players=2)
    char = match.characters[0]
    char.set_brain(brain)
    opponents = [match.characters[1]]
    char.gather_input(opponents, match.stage)

    assert char._input.right is True
    assert char._input.left is False


# -----------------------------------------------------------------------
# Gymnasium API compliance
# -----------------------------------------------------------------------

def test_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env
    env = SmeshLiteEnv()
    check_env(env, warn=True, skip_render_check=True)
    env.close()


def test_headless_speed():
    """Informational: measure steps/sec headless."""
    import time
    env = SmeshLiteEnv()
    env.reset()
    idle = [0, 0, 0, 0]
    N = 10_000
    t0 = time.perf_counter()
    for _ in range(N):
        obs, r, term, trunc, info = env.step(idle)
        if term or trunc:
            env.reset()
    elapsed = time.perf_counter() - t0
    steps_per_sec = N / elapsed
    print(f"\n  {steps_per_sec:.0f} steps/sec (headless)")
    assert steps_per_sec > 1000, f"Too slow: {steps_per_sec:.0f} steps/sec"
    env.close()
