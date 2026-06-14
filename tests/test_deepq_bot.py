"""
Tests for agents.deepq_bot.DeepQBot — the DQN (PyTorch MLP) brain.
"""
import pytest

torch = pytest.importorskip("torch")

from smeshlite.core.brain import BrainContext, InputState, OpponentContext
from smeshlite.core.character import Action
from smeshlite.core.brain_registry import discover_brains

from agents.deepq_bot import DeepQBot, _encode_state, OBS_DIM


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

def test_discoverable_as_deepq_bot():
    registry = discover_brains()
    assert "Deep Q Bot" in registry
    assert registry["Deep Q Bot"].cls.__name__ == "DeepQBot"


# ---------------------------------------------------------------------------
# Basic safety / no-op cases
# ---------------------------------------------------------------------------

def test_no_opponents_does_nothing():
    bot = DeepQBot()
    out = InputState()
    bot.think(make_context(opponents=[]), out)
    assert out == InputState()


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def test_state_encoding_shape_and_edges():
    ctx = make_context(x=-220.0, opponents=[make_opponent(x=100.0, y=0.0)])
    state = _encode_state(ctx)

    assert state.shape == (OBS_DIM,)
    assert state.dtype.name == "float32"

    edge_left = (ctx.x - ctx.stage_platform_x1) / 500.0
    edge_right = (ctx.stage_platform_x2 - ctx.x) / 500.0
    assert state[-2] == pytest.approx(edge_left)
    assert state[-1] == pytest.approx(edge_right)


def test_state_encoding_relative_features():
    ctx = make_context(
        x=-220.0, y=10.0, vx=1.0, vy=0.0, facing=-1.0, action=Action.HITSTUN.value,
        opponents=[make_opponent(x=100.0, y=-5.0, vx=4.0, vy=2.0, action=Action.SMASH.value)],
    )
    target = ctx.opponents[0]
    state = _encode_state(ctx)

    dx = target.x - ctx.x
    dy = target.y - ctx.y
    dx_facing = dx * ctx.facing
    dvx = target.vx - ctx.vx
    dvy = target.vy - ctx.vy

    # indices 22-28: dx, dy, dx_facing, dvx, dvy, opponent_threat, self_hitstun
    assert state[22] == pytest.approx(dx / 500.0)
    assert state[23] == pytest.approx(dy / 500.0)
    assert state[24] == pytest.approx(dx_facing / 500.0)
    assert state[25] == pytest.approx(dvx / 20.0)
    assert state[26] == pytest.approx(dvy / 20.0)
    assert state[27] == pytest.approx(1.0)  # opponent is SMASHing -> threat
    assert state[28] == pytest.approx(1.0)  # self is in hitstun


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def test_epsilon_zero_is_greedy_over_policy_net():
    bot = DeepQBot(epsilon=0.0)
    ctx = make_context(x=0.0, opponents=[make_opponent(x=200.0, y=0.0)])
    state = _encode_state(ctx)

    with torch.no_grad():
        q = bot.policy_net(torch.from_numpy(state).unsqueeze(0).to(bot.device))
    expected = int(torch.argmax(q, dim=1).item())

    assert bot._select_action(state) == expected


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def test_learning_runs_without_error():
    bot = DeepQBot(epsilon=1.0, action_repeat=1, train_every=4, batch_size=8, buffer_size=1000)

    for i in range(40):
        ctx = make_context(
            x=0.0,
            opponents=[make_opponent(x=200.0, y=0.0, damage_pct=float(i))],
        )
        bot.think(ctx, InputState())

    assert len(bot.replay_buffer) > 0
    assert bot.last_loss is not None


# ---------------------------------------------------------------------------
# Action repeat
# ---------------------------------------------------------------------------

def test_action_repeat_holds_action_and_batches_reward():
    bot = DeepQBot(epsilon=0.0, action_repeat=3, dmg_scale=0.01)

    original_select = bot._select_action
    calls = []

    def spy(state):
        action = original_select(state)
        calls.append(action)
        return action

    bot._select_action = spy

    for i in range(7):
        ctx = make_context(
            x=0.0,
            opponents=[make_opponent(x=200.0, y=0.0, damage_pct=float(i) * 5.0)],
        )
        bot.think(ctx, InputState())

    # A new action is only chosen every action_repeat ticks (ticks 1, 4, 7).
    assert len(calls) == 3
    # Transitions are only pushed once the held action's window completes (ticks 4, 7).
    assert len(bot.replay_buffer) == 2

    for _, _, reward, _, done in bot.replay_buffer:
        # 3 ticks * 5.0 damage dealt * dmg_scale=0.01
        assert reward == pytest.approx(0.15)
        assert done is False


# ---------------------------------------------------------------------------
# Terminal reward
# ---------------------------------------------------------------------------

def test_new_episode_returns_total_reward_including_terminal():
    bot = DeepQBot(epsilon=0.0, ko_reward=1.0)

    ctx = make_context(x=0.0, opponents=[make_opponent(x=200.0, y=0.0, damage_pct=0.0, stocks=3)])
    bot.think(ctx, InputState())
    assert bot._pending is not None

    # Opponent's stock count dropped -> terminal KO reward should be credited.
    final_snapshot = (ctx.damage_pct, ctx.stocks, 50.0, 2)
    total = bot.new_episode(final_snapshot=final_snapshot)

    assert total >= 1.0
    assert bot._pending is None
    assert bot.episode_reward == 0.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    bot = DeepQBot()
    path = tmp_path / "deepq.pt"
    bot.save(str(path))

    loaded = DeepQBot()
    loaded.load(str(path))

    for k, v in bot.policy_net.state_dict().items():
        assert torch.equal(v, loaded.policy_net.state_dict()[k])
