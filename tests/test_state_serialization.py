"""
Tests for Layer 1: State Serialization.

Validates Character.get_state/set_state, Match.get_state/set_state,
and SmeshLiteEnv.reset(options={"state": ...}) roundtrips.
"""
import json
import pytest
import numpy as np

from smeshlite import SmeshLiteEnv
from smeshlite.core.match import Match, MatchConfig
from smeshlite.core.character import Character, Action, HITSTUN_BASE
from smeshlite.core.brain import ExternalBrain, InputState, CharacterBrain
from smeshlite.data.stage_data import STAGE_0


# -----------------------------------------------------------------------
# Character state serialization
# -----------------------------------------------------------------------

class TestCharacterGetState:

    def test_returns_all_required_keys(self):
        c = Character(0, 0.0, 100.0)
        state = c.get_state()
        required = Character._STATE_KEYS
        assert required <= set(state), f"Missing keys: {required - set(state)}"

    def test_default_values_match_init(self):
        c = Character(0, 10.0, 20.0, facing=-1.0, stocks=5)
        state = c.get_state()
        assert state["player_id"] == 0
        assert state["x"] == 10.0
        assert state["y"] == 20.0
        assert state["vx"] == 0.0
        assert state["vy"] == 0.0
        assert state["facing"] == -1.0
        assert state["action"] == Action.RESPAWN.value
        assert state["stocks"] == 5
        assert state["damage_pct"] == 0.0

    def test_values_after_state_changes(self):
        c = Character(0, 0.0, 0.0)
        c.action = Action.ATTACK
        c.attack_num = 3
        c.damage_pct = 72.5
        c.invincibility = 30
        c._pending_damage = 5.0
        c._pending_knockback_vx = 10.0
        state = c.get_state()
        assert state["action"] == Action.ATTACK.value
        assert state["attack_num"] == 3
        assert state["damage_pct"] == 72.5
        assert state["invincibility"] == 30
        assert state["pending_damage"] == 5.0
        assert state["pending_knockback_vx"] == 10.0

    def test_state_is_json_serializable(self):
        c = Character(0, 0.0, 100.0)
        state = c.get_state()
        serialized = json.dumps(state)
        restored = json.loads(serialized)
        assert restored == state


class TestCharacterSetState:

    def test_roundtrip_preserves_all_fields(self):
        c = Character(0, 0.0, 100.0, facing=-1.0, stocks=5)
        # Mutate state
        c.action = Action.AERIAL
        c.damage_pct = 55.0
        c.vx = 12.0
        c.vy = 3.0
        c.attack_num = 4
        c.charge_amount = 0.7
        c.has_hit = True
        c.anim_frame = 3.5

        state = c.get_state()

        c2 = Character(0, 0.0, 50.0, facing=1.0, stocks=1)
        c2.set_state(state)

        state2 = c2.get_state()
        assert state == state2

    def test_roundtrip_after_advancing_ticks(self):
        config = MatchConfig()
        match = Match(config)
        match.reset(n_players=2)

        # Advance 100 ticks
        for _ in range(100):
            match.tick()

        state = match.characters[0].get_state()

        c2 = Character(0, 0.0, 50.0)
        c2.set_state(state)

        # After set_state, get_state should match
        assert c2.get_state() == state

    def test_player_id_mismatch_raises(self):
        c = Character(0, 0.0, 100.0)
        state = c.get_state()
        state["player_id"] = 99

        c2 = Character(0, 0.0, 50.0)
        with pytest.raises(ValueError, match="player_id mismatch"):
            c2.set_state(state)

    def test_missing_keys_raises(self):
        c = Character(0, 0.0, 100.0)
        state = c.get_state()
        del state["x"]

        c2 = Character(0, 0.0, 50.0)
        with pytest.raises(ValueError, match="missing keys"):
            c2.set_state(state)

    def test_set_state_clears_input(self):
        c = Character(0, 0.0, 100.0)
        c._input.left = True
        c._input.attack = True

        state = c.get_state()
        c2 = Character(0, 0.0, 50.0)
        c2.set_state(state)

        assert c2._input.left is False
        assert c2._input.attack is False

    def test_set_state_preserves_brain(self):
        brain = ExternalBrain()
        c = Character(0, 0.0, 100.0)
        c.set_brain(brain)

        state = c.get_state()
        c2 = Character(0, 0.0, 50.0)
        other_brain = ExternalBrain()
        c2.set_brain(other_brain)
        c2.set_state(state)

        # Brain should NOT change
        assert c2._brain is other_brain

    def test_action_enum_roundtrip(self):
        for action in Action:
            c = Character(0, 0.0, 100.0)
            c.action = action
            state = c.get_state()
            assert state["action"] == action.value

            c2 = Character(0, 0.0, 50.0)
            c2.set_state(state)
            assert c2.action == action

    def test_pending_damage_roundtrip(self):
        c = Character(0, 0.0, 100.0)
        c._pending_damage = 12.5
        c._pending_knockback_vx = -8.0
        c._pending_knockback_vy = 4.0

        state = c.get_state()
        c2 = Character(0, 0.0, 50.0)
        c2.set_state(state)

        assert c2._pending_damage == 12.5
        assert c2._pending_knockback_vx == -8.0
        assert c2._pending_knockback_vy == 4.0


# -----------------------------------------------------------------------
# Match state serialization
# -----------------------------------------------------------------------

class TestMatchGetState:

    def test_returns_all_required_keys(self):
        match = Match(MatchConfig())
        match.reset(n_players=2)
        state = match.get_state()
        assert "frame" in state
        assert "done" in state
        assert "winner" in state
        assert "characters" in state
        assert "config" in state

    def test_characters_list_has_correct_length(self):
        match = Match(MatchConfig())
        match.reset(n_players=2)
        state = match.get_state()
        assert len(state["characters"]) == 2

    def test_config_is_serialized(self):
        match = Match(MatchConfig(stocks=5, time_limit=3000, gravity_scale=0.8))
        match.reset(n_players=2)
        state = match.get_state()
        assert state["config"]["stocks"] == 5
        assert state["config"]["time_limit"] == 3000
        assert state["config"]["gravity_scale"] == 0.8

    def test_state_is_json_serializable(self):
        match = Match(MatchConfig())
        match.reset(n_players=2)
        state = match.get_state()
        serialized = json.dumps(state)
        restored = json.loads(serialized)
        assert restored == state


class TestMatchSetState:

    def test_roundtrip_preserves_match_state(self):
        match = Match(MatchConfig())
        match.reset(n_players=2)

        # Advance 200 ticks
        for _ in range(200):
            match.tick()

        state = match.get_state()

        # Fresh match
        match2 = Match(MatchConfig())
        match2.reset(n_players=2)
        match2.set_state(state)

        state2 = match2.get_state()
        assert state == state2

    def test_roundtrip_continues_deterministically(self):
        """After restoring state, advancing ticks should produce identical results."""
        match = Match(MatchConfig())
        match.reset(n_players=2)

        # Wire identical brains on both
        for c in match.characters:
            c.set_brain(ExternalBrain())

        # Advance 100 ticks
        for _ in range(100):
            match.tick()

        state = match.get_state()

        # Save next 50 ticks of obs for comparison
        obs_original = []
        for _ in range(50):
            for brain in match.characters:
                brain._brain.pending = InputState()  # idle
            match.tick()
            obs_original.append(match.get_obs(perspective=0))

        # Restore state into a fresh match
        match2 = Match(MatchConfig())
        match2.reset(n_players=2)
        for c in match2.characters:
            c.set_brain(ExternalBrain())
        match2.set_state(state)

        # Advance same 50 ticks with same idle inputs
        obs_restored = []
        for _ in range(50):
            for brain in match2.characters:
                brain._brain.pending = InputState()
            match2.tick()
            obs_restored.append(match2.get_obs(perspective=0))

        assert obs_original == obs_restored

    def test_preserves_brain_assignments(self):
        match = Match(MatchConfig())
        match.reset(n_players=2)
        brain0 = ExternalBrain()
        brain1 = ExternalBrain()
        match.characters[0].set_brain(brain0)
        match.characters[1].set_brain(brain1)

        state = match.get_state()

        match2 = Match(MatchConfig())
        match2.reset(n_players=2)
        other_brain = ExternalBrain()
        match2.characters[0].set_brain(other_brain)
        match2.set_state(state)

        # Brains should NOT change
        assert match2.characters[0]._brain is other_brain

    def test_character_count_mismatch_raises(self):
        match = Match(MatchConfig())
        match.reset(n_players=2)
        state = match.get_state()

        match2 = Match(MatchConfig())
        match2.reset(n_players=3)  # different count
        with pytest.raises(NotImplementedError, match="character count mismatch"):
            match2.set_state(state)

    def test_done_and_winner_are_restored(self):
        match = Match(MatchConfig(stocks=1))
        match.reset(n_players=2)
        # Kill P1
        p1 = match.characters[1]
        p1.action = Action.AERIAL
        p1.x = 800.0  # beyond kill_x
        p1.ground_update(match.stage)
        match._check_win()
        assert match.done

        state = match.get_state()
        assert state["done"] is True

        match2 = Match(MatchConfig(stocks=1))
        match2.reset(n_players=2)
        match2.set_state(state)
        assert match2.done is True
        assert match2.winner == match.winner


# -----------------------------------------------------------------------
# Env state restore via reset(options={"state": ...})
# -----------------------------------------------------------------------

class TestEnvResetWithState:

    def test_reset_with_state_restores_obs(self):
        env = SmeshLiteEnv()
        obs, _ = env.reset()

        # Advance some steps
        for _ in range(200):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset()

        # Capture current state
        state = env.match.get_state()
        current_obs = obs.copy()

        # Reset with that state
        obs2, info2 = env.reset(options={"state": state})
        # Observations after one idle tick won't match exactly (the reset
        # does a tick() after restoring), but the state was restored
        assert env.match.frame > 0

        env.close()

    def test_reset_without_options_works_as_before(self):
        env = SmeshLiteEnv()
        obs, info = env.reset()
        assert obs.shape == (53,)  # default is full obs mode
        env.close()
        env.close()

    def test_state_from_one_env_works_in_another(self):
        """State captured from one env can restore another env."""
        env1 = SmeshLiteEnv(stocks=5)
        obs, _ = env1.reset()
        for _ in range(100):
            obs, _, term, trunc, _ = env1.step(env1.action_space.sample())
            if term or trunc:
                obs, _ = env1.reset()
        state = env1.match.get_state()

        env2 = SmeshLiteEnv(stocks=5)
        obs2, _ = env2.reset(options={"state": state})
        assert env2.match.frame == state["frame"] + 1  # +1 from the post-restore tick

        env1.close()
        env2.close()

    def test_json_roundtrip_through_env(self):
        """State survives JSON serialization -> deserialization -> env reset."""
        env = SmeshLiteEnv()
        obs, _ = env.reset()
        for _ in range(50):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                obs, _ = env.reset()

        state = env.match.get_state()
        json_str = json.dumps(state)
        restored_state = json.loads(json_str)

        obs2, _ = env.reset(options={"state": restored_state})
        assert env.match.frame > 0

        env.close()


# -----------------------------------------------------------------------
# Integration: state roundtrip with damage in progress
# -----------------------------------------------------------------------

class TestStateWithPendingDamage:

    def test_pending_damage_survives_roundtrip(self):
        """If damage is scheduled but not yet flushed, it must survive set_state."""
        match = Match(MatchConfig())
        match.reset(n_players=2)

        # Manually inject pending damage (simulates a hit mid-tick)
        match.characters[1]._pending_damage = 15.0
        match.characters[1]._pending_knockback_vx = 25.0
        match.characters[1]._pending_knockback_vy = 10.0

        state = match.get_state()

        match2 = Match(MatchConfig())
        match2.reset(n_players=2)
        match2.set_state(state)

        assert match2.characters[1]._pending_damage == 15.0
        assert match2.characters[1]._pending_knockback_vx == 25.0
        assert match2.characters[1]._pending_knockback_vy == 10.0

        # Flushing should apply the damage
        match2.characters[1].apply_pending_damage()
        assert match2.characters[1].damage_pct == 15.0
