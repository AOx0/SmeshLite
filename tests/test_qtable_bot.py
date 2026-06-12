"""
Tests for agents.qtable_bot.QTableBot — the tabular Q-learning brain.
"""
from smeshlite.core.brain import BrainContext, InputState, OpponentContext
from smeshlite.core.character import Action
from smeshlite.core.brain_registry import discover_brains

from agents.qtable_bot import QTableBot, _encode_state


def make_context(**overrides) -> BrainContext:
    defaults = dict(
        x=0.0, y=0.0, vx=0.0, vy=0.0,
        in_air=False, facing=1.0,
        damage_pct=0.0, stocks=3,
        action=Action.RUN.value, action_frame=0, attack_num=0, charge_amount=0.0,
        invincibility=0,
        opponents=[],
        stage_platform_x1=-320.0, stage_platform_x2=320.0,
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

def test_discoverable_as_qtable_bot():
    registry = discover_brains()
    assert "Q-Table Bot" in registry
    assert registry["Q-Table Bot"].cls.__name__ == "QTableBot"


# ---------------------------------------------------------------------------
# Basic safety / no-op cases
# ---------------------------------------------------------------------------

def test_no_opponents_does_nothing():
    bot = QTableBot()
    out = InputState()
    bot.think(make_context(opponents=[]), out)
    assert out == InputState()


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def test_state_encoding_bins():
    ctx = make_context(x=0.0, in_air=False, action=Action.RUN.value)

    def state_for(opp_x, opp_y=0.0, opp_action=Action.NONE.value):
        return _encode_state(ctx, make_opponent(x=opp_x, y=opp_y, action=opp_action))

    # dx bins: <-250, -250..-100, -100..-25, -25..25, 25..100, 100..250, >=250
    assert state_for(-400.0)[0] == 0
    assert state_for(-150.0)[0] == 1
    assert state_for(-30.0)[0] == 2
    assert state_for(0.0)[0] == 3
    assert state_for(30.0)[0] == 4
    assert state_for(150.0)[0] == 5
    assert state_for(400.0)[0] == 6

    # dy bins: <-30, -30..30, >=30
    assert state_for(0.0, opp_y=-50.0)[1] == 0
    assert state_for(0.0, opp_y=0.0)[1] == 1
    assert state_for(0.0, opp_y=50.0)[1] == 2

    # in_air / hitstun flags
    air_ctx = make_context(x=0.0, in_air=True, action=Action.HITSTUN.value)
    s = _encode_state(air_ctx, make_opponent(x=0.0, y=0.0))
    assert s[2] == 1  # in_air
    assert s[5] == 1  # hitstun

    # opponent "threat" flag
    assert _encode_state(ctx, make_opponent(x=0.0, y=0.0, action=Action.SMASH.value))[3] == 1
    assert _encode_state(ctx, make_opponent(x=0.0, y=0.0, action=Action.NONE.value))[3] == 0

    # edge zone: near-left / center / near-right of stage_platform_x1..x2
    left_ctx = make_context(x=-300.0)
    center_ctx = make_context(x=0.0)
    right_ctx = make_context(x=300.0)
    assert _encode_state(left_ctx, make_opponent(x=0.0, y=0.0))[4] == 0
    assert _encode_state(center_ctx, make_opponent(x=0.0, y=0.0))[4] == 1
    assert _encode_state(right_ctx, make_opponent(x=0.0, y=0.0))[4] == 2


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def test_epsilon_zero_picks_best_action():
    bot = QTableBot(epsilon=0.0)
    ctx = make_context(x=0.0, opponents=[make_opponent(x=200.0, y=0.0)])
    state = _encode_state(ctx, ctx.opponents[0])

    bot.q_table[state] = [0.0] * QTableBot.N_ACTIONS
    bot.q_table[state][2] = 5.0  # action 2 == RIGHT

    out = InputState()
    bot.think(ctx, out)
    assert out.right is True
    assert out.left is False
    assert out.up is False
    assert out.attack is False


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def test_reward_updates_q_value_on_damage():
    bot = QTableBot(epsilon=0.0, alpha=0.5, gamma=0.9)

    ctx1 = make_context(x=0.0, opponents=[make_opponent(x=200.0, y=0.0, damage_pct=0.0)])
    bot.think(ctx1, InputState())
    state1, action1, _ = bot._pending

    # Opponent's damage rose -> positive reward -> Q[state1][action1] moves up from 0.
    ctx2 = make_context(x=0.0, opponents=[make_opponent(x=200.0, y=0.0, damage_pct=10.0)])
    bot.think(ctx2, InputState())

    assert bot.q_table[state1][action1] > 0.0


def test_new_episode_resets_pending_transition():
    bot = QTableBot(epsilon=0.0, alpha=0.5)

    ctx1 = make_context(x=0.0, opponents=[make_opponent(x=200.0, y=0.0, damage_pct=0.0)])
    bot.think(ctx1, InputState())
    assert bot._pending is not None

    bot.new_episode()
    assert bot._pending is None
    assert bot.episode_reward == 0.0
    assert bot.q_table == {}

    # A big damage jump must NOT retroactively update anything: new_episode()
    # cleared the pending (state, action) it would have been credited to.
    ctx2 = make_context(x=0.0, opponents=[make_opponent(x=200.0, y=0.0, damage_pct=99.0)])
    bot.think(ctx2, InputState())
    assert bot.q_table == {}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    bot = QTableBot()
    bot.q_table[(1, 2, 0, 1, 2, 0)] = [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 1.5]

    path = tmp_path / "qtable.json"
    bot.save(str(path))

    loaded = QTableBot()
    loaded.load(str(path))
    assert loaded.q_table == bot.q_table
