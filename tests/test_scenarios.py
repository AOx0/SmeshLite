"""
Tests for Layer 5: Scenario System.

Validates scenario registry, built-in scenarios, env integration via
reset(options={"scenario": ...}), and state roundtripping.
"""
import pytest
import numpy as np

from smeshlite import SmeshLiteEnv
from smeshlite.scenarios import (
    Scenario, SCENARIOS,
    register_scenario, get_scenario, list_scenarios, sample_scenario,
    scenario_from_match,
)


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

class TestRegistry:

    def test_list_scenarios_returns_sorted(self):
        names = list_scenarios()
        assert names == sorted(names)

    def test_list_scenarios_has_builtins(self):
        names = list_scenarios()
        assert "default" in names
        assert "high_damage" in names
        assert "off_stage_recovery" in names

    def test_get_scenario_returns_scenario(self):
        s = get_scenario("default")
        assert isinstance(s, Scenario)
        assert s.name == "default"

    def test_get_scenario_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown scenario"):
            get_scenario("nonexistent_scenario_xyz")

    def test_register_custom_scenario(self):
        from smeshlite.scenarios import _match_state, _char_state
        custom = Scenario(
            name="test_custom_12345",
            description="test",
            state=_match_state([
                _char_state(player_id=0, x=0.0, y=68.1),
                _char_state(player_id=1, x=50.0, y=68.1),
            ]),
            tags=("test",),
        )
        register_scenario(custom)
        assert "test_custom_12345" in list_scenarios()
        assert get_scenario("test_custom_12345").name == "test_custom_12345"
        # Clean up
        del SCENARIOS["test_custom_12345"]

    def test_register_overwrites(self):
        from smeshlite.scenarios import _match_state, _char_state
        s1 = Scenario(name="overwrite_test", description="v1",
                      state=_match_state([_char_state(), _char_state()]))
        s2 = Scenario(name="overwrite_test", description="v2",
                      state=_match_state([_char_state(), _char_state()]))
        register_scenario(s1)
        register_scenario(s2)
        assert get_scenario("overwrite_test").description == "v2"
        del SCENARIOS["overwrite_test"]


class TestSampleScenario:

    def test_sample_returns_scenario(self):
        s = sample_scenario()
        assert isinstance(s, Scenario)
        assert s.name in list_scenarios()

    def test_sample_with_tag_filter(self):
        s = sample_scenario(tags=("recovery",))
        assert "recovery" in s.tags

    def test_sample_with_difficulty_filter(self):
        s = sample_scenario(difficulty="easy")
        assert s.difficulty == "easy"

    def test_sample_no_match_raises(self):
        with pytest.raises(ValueError, match="No scenarios match"):
            sample_scenario(tags=("nonexistent_tag_xyz",))

    def test_sample_tag_or_logic(self):
        """Tags filter uses OR logic: any tag in the set matches."""
        # "edge_guard" scenario has tags ("edge_guard", "recovery")
        s = sample_scenario(tags=("edge_guard",))
        assert s.name == "edge_guard"

    def test_sample_difficulty_and_tags(self):
        s = sample_scenario(tags=("recovery",), difficulty="hard")
        assert "recovery" in s.tags
        assert s.difficulty == "hard"


# -----------------------------------------------------------------------
# Built-in scenarios
# -----------------------------------------------------------------------

class TestBuiltinScenarios:

    @pytest.mark.parametrize("name", list_scenarios())
    def test_scenario_loads_in_env(self, name):
        """Every built-in scenario should load without error."""
        env = SmeshLiteEnv()
        obs, info = env.reset(options={"scenario": name})
        assert obs.shape == env.observation_space.shape
        # Step a few times to verify the state doesn't crash
        for _ in range(10):
            obs, r, t, tr, info = env.step(env.action_space.sample())
            if t or tr:
                break
        env.close()

    @pytest.mark.parametrize("name", list_scenarios())
    def test_scenario_has_required_fields(self, name):
        s = get_scenario(name)
        assert s.name
        assert s.description
        assert "characters" in s.state
        assert "frame" in s.state
        assert "config" in s.state

    @pytest.mark.parametrize("name", list_scenarios())
    def test_scenario_two_characters(self, name):
        """All built-in scenarios should define 2 characters."""
        s = get_scenario(name)
        assert len(s.state["characters"]) == 2

    @pytest.mark.parametrize("name", list_scenarios())
    def test_scenario_works_with_minimal_obs(self, name):
        env = SmeshLiteEnv(obs_mode="minimal")
        obs, info = env.reset(options={"scenario": name})
        assert obs.shape == env.observation_space.shape
        env.close()


# -----------------------------------------------------------------------
# Scenario state content checks
# -----------------------------------------------------------------------

class TestScenarioStateContent:

    def test_high_damage_sets_damage(self):
        env = SmeshLiteEnv()
        env.reset(options={"scenario": "high_damage"})
        obs, r, t, tr, info = env.step(env.action_space.sample())
        assert info["damage"][0] == 120.0
        assert info["damage"][1] == 120.0
        env.close()

    def test_one_stock_left_sets_stocks(self):
        env = SmeshLiteEnv()
        env.reset(options={"scenario": "one_stock_left"})
        obs, r, t, tr, info = env.step(env.action_space.sample())
        assert info["stocks"][0] == 1
        assert info["stocks"][1] == 3
        env.close()

    def test_off_stage_recovery_p0_below_stage(self):
        env = SmeshLiteEnv()
        env.reset(options={"scenario": "off_stage_recovery"})
        # P0 should be below the main platform (y < 0)
        p0_y = env.match.characters[0].y
        assert p0_y < 0, f"P0 should be below stage, got y={p0_y}"
        assert env.match.characters[0].in_air
        env.close()

    def test_edge_guard_p1_off_stage(self):
        env = SmeshLiteEnv()
        env.reset(options={"scenario": "edge_guard"})
        # P1 should be off the stage platform (x > 320 or y < 0)
        p1 = env.match.characters[1]
        assert p1.x > 320 or p1.y < 0, f"P1 should be off-stage, got x={p1.x} y={p1.y}"
        env.close()

    def test_center_stage_close_together(self):
        env = SmeshLiteEnv()
        env.reset(options={"scenario": "center_stage"})
        p0_x = env.match.characters[0].x
        p1_x = env.match.characters[1].x
        assert abs(p0_x - p1_x) < 100, "Characters should be close together"
        env.close()

    def test_charging_smash_p0_charging(self):
        """P0 should start in CHARGING action."""
        s = get_scenario("charging_smash")
        assert s.state["characters"][0]["action"] == 6  # Action.CHARGING.value


# -----------------------------------------------------------------------
# Env integration
# -----------------------------------------------------------------------

class TestEnvScenarioIntegration:

    def test_reset_with_scenario(self):
        env = SmeshLiteEnv()
        obs, info = env.reset(options={"scenario": "high_damage"})
        # After one step, damage should be preserved
        obs, r, t, tr, info = env.step(env.action_space.sample())
        assert info["damage"][0] == 120.0
        env.close()

    def test_reset_with_state_still_works(self):
        """The existing options={'state': ...} path should still work."""
        env = SmeshLiteEnv()
        obs, _ = env.reset()
        state = env.match.get_state()
        # Modify state
        state["characters"][0]["damage_pct"] = 50.0
        obs2, _ = env.reset(options={"state": state})
        # After one step
        obs2, r, t, tr, info = env.step(env.action_space.sample())
        assert info["damage"][0] == 50.0
        env.close()

    def test_scenario_takes_precedence_over_state(self):
        """If both scenario and state are in options, scenario wins."""
        env = SmeshLiteEnv()
        # This should use scenario, not state
        obs, _ = env.reset(options={
            "scenario": "high_damage",
            "state": {"frame": 1, "done": False, "winner": None,
                      "characters": [], "config": {}},
        })
        # high_damage should be applied (120% damage), not the empty state
        obs, r, t, tr, info = env.step(env.action_space.sample())
        assert info["damage"][0] == 120.0
        env.close()

    def test_scenario_with_opponent_brain(self):
        """Scenario + opponent_brain should work together."""
        env = SmeshLiteEnv(
            action_mode="single",
            opponent_brain="Smesh Bot",
        )
        obs, _ = env.reset(options={"scenario": "high_damage"})
        # Step should work fine
        for _ in range(50):
            obs, r, t, tr, info = env.step([0, 1, 0, 0])
            if t or tr:
                obs, _ = env.reset(options={"scenario": "high_damage"})
        env.close()

    def test_scenario_state_roundtrip(self):
        """Loading a scenario, capturing state, reloading should be identical."""
        env = SmeshLiteEnv()
        env.reset(options={"scenario": "high_damage"})
        # One tick to settle
        env.step(env.action_space.sample())
        state = env.match.get_state()

        # Reload into a fresh env
        env2 = SmeshLiteEnv()
        env2.reset(options={"state": state})
        state2 = env2.match.get_state()

        assert state["characters"][0]["damage_pct"] == state2["characters"][0]["damage_pct"]
        assert state["characters"][1]["damage_pct"] == state2["characters"][1]["damage_pct"]
        env.close()
        env2.close()


# -----------------------------------------------------------------------
# scenario_from_match helper
# -----------------------------------------------------------------------

class TestScenarioFromMatch:

    def test_captures_match_state(self):
        env = SmeshLiteEnv()
        env.reset(options={"scenario": "high_damage"})
        env.step(env.action_space.sample())

        s = scenario_from_match(
            name="captured",
            description="Captured from high_damage",
            match=env.match,
            difficulty="hard",
            tags=("captured",),
        )
        assert s.name == "captured"
        assert s.difficulty == "hard"
        assert "captured" in s.tags
        assert "characters" in s.state
        env.close()


# -----------------------------------------------------------------------
# Invalid inputs
# -----------------------------------------------------------------------

class TestInvalidInputs:

    def test_invalid_scenario_name_helpful_message(self):
        with pytest.raises(KeyError, match="Unknown scenario"):
            get_scenario("definitely_not_a_scenario")

    def test_invalid_scenario_name_in_reset(self):
        env = SmeshLiteEnv()
        with pytest.raises(KeyError, match="Unknown scenario"):
            env.reset(options={"scenario": "nope"})
        env.close()
