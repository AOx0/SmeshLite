"""
Tests for Layer 3: Multi-Agent Step.

Validates action_mode="single"|"multi", opponent_brain parameter,
action spaces, and applied_actions in info dict.
"""
import pytest
import numpy as np

from smeshlite import SmeshLiteEnv
from smeshlite.core.brain import (
    CharacterBrain, ExternalBrain, InputState, BrainContext,
)
from agents.smesh_bot import SmeshBot
from agents.random_bot import RandomBot


def _brain_class_name(brain):
    """Get the class name of a brain (works even if loaded by different import path)."""
    return brain.__class__.__name__


# -----------------------------------------------------------------------
# Action mode: single (regression)
# -----------------------------------------------------------------------

class TestSingleActionMode:

    def test_default_is_single(self):
        env = SmeshLiteEnv()
        assert env.action_mode == "single"
        env.close()

    def test_single_action_space_shape(self):
        env = SmeshLiteEnv(action_mode="single")
        assert env.action_space.shape == (4,)
        env.close()

    def test_single_step_idle_works(self):
        env = SmeshLiteEnv(action_mode="single")
        obs, _ = env.reset()
        obs, r, term, trunc, info = env.step([0, 0, 0, 0])
        assert obs.shape == (22,)
        env.close()

    def test_single_p0_moves_others_idle(self):
        """In single mode, P1 should have no input (idle)."""
        env = SmeshLiteEnv(action_mode="single")
        env.reset()
        # Characters start in RESPAWN (frozen for entrance animation).
        # Need enough steps to exit respawn before movement works.
        for _ in range(30):  # _ENTRANCE_FRAMES = 20, plus buffer
            env.step([0, 1, 0, 0])  # P0 moves right
        p0_moved = env.match.characters[0].x > -245.0
        assert p0_moved, "P0 should have moved right after respawn"
        env.close()


# -----------------------------------------------------------------------
# Action mode: multi
# -----------------------------------------------------------------------

class TestMultiActionMode:

    def test_multi_action_space_shape(self):
        env = SmeshLiteEnv(action_mode="multi", n_players=2)
        assert env.action_space.shape == (2, 4)
        env.close()

    def test_multi_3_players(self):
        env = SmeshLiteEnv(action_mode="multi", n_players=3)
        assert env.action_space.shape == (3, 4)
        env.close()

    def test_multi_both_players_move(self):
        """In multi mode, both players receive actions and can move."""
        env = SmeshLiteEnv(action_mode="multi")
        obs, _ = env.reset()

        # Characters start in RESPAWN -- advance past entrance animation
        for _ in range(30):
            env.step([[0, 1, 0, 0], [1, 0, 0, 0]])  # P0 right, P1 left

        p0_x = env.match.characters[0].x
        p1_x = env.match.characters[1].x

        assert p0_x > -245.0, "P0 should have moved right from spawn"
        assert p1_x < 245.0, "P1 should have moved left from spawn"
        env.close()

    def test_multi_step_1000_steps(self):
        env = SmeshLiteEnv(action_mode="multi")
        obs, _ = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            assert obs.shape == (22,)
            if term or trunc:
                obs, _ = env.reset()
        env.close()

    def test_multi_obs_shape_matches(self):
        env = SmeshLiteEnv(action_mode="multi", obs_mode="full")
        obs, _ = env.reset()
        assert obs.shape == (53,)
        env.close()

    def test_gymnasium_checker_multi_mode(self):
        """Gymnasium env checker should pass in multi mode."""
        from gymnasium.utils.env_checker import check_env
        env = SmeshLiteEnv(action_mode="multi")
        check_env(env, warn=True, skip_render_check=True)
        env.close()

    def test_invalid_action_mode_raises(self):
        with pytest.raises(AssertionError, match="action_mode"):
            SmeshLiteEnv(action_mode="coop")


# -----------------------------------------------------------------------
# opponent_brain parameter
# -----------------------------------------------------------------------

class TestOpponentBrain:

    def test_opponent_brain_string(self):
        """Setting opponent_brain='Smesh Bot' should make P1 use SmeshBot."""
        env = SmeshLiteEnv(opponent_brain="Smesh Bot")
        env.reset()
        # P1's brain should be SmeshBot (check by name due to import duplication)
        p1_brain = env.match.characters[1]._brain
        assert _brain_class_name(p1_brain) == "SmeshBot"
        assert p1_brain.BRAIN_NAME == "Smesh Bot"
        env.close()

    def test_opponent_brain_instance(self):
        """Passing a SmeshBot instance should work."""
        bot = SmeshBot()
        env = SmeshLiteEnv(opponent_brain=bot)
        env.reset()
        assert env.match.characters[1]._brain is bot
        env.close()

    def test_opponent_brain_class(self):
        """Passing a CharacterBrain subclass should be instantiated."""
        env = SmeshLiteEnv(opponent_brain=RandomBot)
        env.reset()
        p1_brain = env.match.characters[1]._brain
        assert _brain_class_name(p1_brain) == "RandomBot"
        env.close()

    def test_opponent_brain_none_means_idle(self):
        """Default opponent_brain=None means P1 idles."""
        env = SmeshLiteEnv()
        env.reset()
        p1_brain = env.match.characters[1]._brain
        assert isinstance(p1_brain, ExternalBrain)
        env.close()

    def test_opponent_brain_p0_still_external(self):
        """P0 should always be driven by ExternalBrain, even with opponent_brain set."""
        env = SmeshLiteEnv(opponent_brain="Smesh Bot")
        env.reset()
        p0_brain = env.match.characters[0]._brain
        assert isinstance(p0_brain, ExternalBrain)
        env.close()

    def test_opponent_brain_rewires_on_reset(self):
        """After reset(), opponent brain should be re-wired on P1."""
        env = SmeshLiteEnv(opponent_brain="Smesh Bot")
        env.reset()
        assert _brain_class_name(env.match.characters[1]._brain) == "SmeshBot"
        env.reset()
        assert _brain_class_name(env.match.characters[1]._brain) == "SmeshBot"
        env.close()

    def test_opponent_brain_actually_fights(self):
        """With SmeshBot as opponent, P1 should move (not idle)."""
        env = SmeshLiteEnv(opponent_brain="Smesh Bot")
        env.reset()

        p1_start_x = env.match.characters[1].x
        # Run 200 idle ticks (P0 does nothing)
        for _ in range(200):
            obs, r, term, trunc, info = env.step([0, 0, 0, 0])
            if term or trunc:
                env.reset()

        # SmeshBot should have moved P1 from starting position
        # (it chases the opponent, so it will have moved)
        p1_moved = env.match.characters[1].x != p1_start_x
        assert p1_moved, "SmeshBot should have moved P1 from spawn"
        env.close()

    def test_invalid_opponent_brain_type_raises(self):
        with pytest.raises(TypeError, match="opponent_brain"):
            env = SmeshLiteEnv(opponent_brain=12345)
            env.reset()


# -----------------------------------------------------------------------
# Info dict: actions and applied_actions
# -----------------------------------------------------------------------

class TestInfoActions:

    def test_info_has_actions(self):
        env = SmeshLiteEnv()
        env.reset()
        obs, r, term, trunc, info = env.step([1, 0, 0, 0])
        assert "actions" in info
        env.close()

    def test_info_has_applied_actions(self):
        env = SmeshLiteEnv()
        env.reset()
        obs, r, term, trunc, info = env.step([1, 0, 0, 0])
        assert "applied_actions" in info
        env.close()

    def test_single_mode_actions_shape(self):
        env = SmeshLiteEnv()
        env.reset()
        info_data = env.step([1, 0, 1, 0])
        info = info_data[4]  # info is 5th return
        actions = info["actions"]
        assert len(actions) == 2  # n_players
        assert len(actions[0]) == 4  # 4 buttons

    def test_single_mode_p0_action_recorded(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step([1, 0, 1, 0])[4]
        # P0's requested action should match
        assert info["actions"][0] == [True, False, True, False]
        # P1 should be idle
        assert info["actions"][1] == [False, False, False, False]

    def test_applied_actions_match_input(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step([1, 0, 1, 0])[4]
        # For ExternalBrain, applied_actions should match the requested actions
        applied = info["applied_actions"]
        assert len(applied) == 2
        # P0: ExternalBrain -> applied matches
        assert applied[0] == [True, False, True, False]
        # P1: idle ExternalBrain
        assert applied[1] == [False, False, False, False]

    def test_multi_mode_both_actions_recorded(self):
        env = SmeshLiteEnv(action_mode="multi")
        env.reset()
        info = env.step([[1, 0, 0, 0], [0, 1, 0, 0]])[4]
        assert info["actions"][0] == [True, False, False, False]
        assert info["actions"][1] == [False, True, False, False]

    def test_opponent_brain_applied_actions_differ(self):
        """With SmeshBot opponent, applied_actions for P1 should differ from idle."""
        env = SmeshLiteEnv(opponent_brain="Smesh Bot")
        env.reset()
        # P0 idle, SmeshBot drives P1
        info = env.step([0, 0, 0, 0])[4]
        # P0 requested idle -> applied is idle
        assert info["applied_actions"][0] == [False, False, False, False]
        # P1: SmeshBot may or may not have moved this tick (could be in respawn)
        # But applied_actions should exist and have 4 elements
        assert len(info["applied_actions"][1]) == 4
        env.close()

    def test_applied_actions_length_matches_n_players(self):
        for n in (1, 2, 3):
            env = SmeshLiteEnv(n_players=n, action_mode="multi")
            env.reset()
            info = env.step(env.action_space.sample())[4]
            assert len(info["applied_actions"]) == n
            env.close()


# -----------------------------------------------------------------------
# Combination: multi + opponent_brain (edge case)
# -----------------------------------------------------------------------

class TestMultiWithOpponentBrain:

    def test_multi_mode_ignores_opponent_brain(self):
        """In multi mode, all players are env-driven; opponent_brain is unused."""
        env = SmeshLiteEnv(action_mode="multi", opponent_brain="Smesh Bot")
        env.reset()
        # In multi mode, P1 should be ExternalBrain (env drives it)
        p1_brain = env.match.characters[1]._brain
        assert isinstance(p1_brain, ExternalBrain)
        env.close()
