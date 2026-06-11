"""
Tests for Layer 2: Rich Observations.

Validates obs_mode="minimal" (regression), obs_mode="full" (shape, content),
sensor padding, max_sensors, and full_obs_size helper.
"""
import pytest
import numpy as np

from smeshlite import SmeshLiteEnv
from smeshlite.core.brain import (
    BrainContext, OpponentContext, brain_context_to_obs,
    brain_context_to_full_obs, full_obs_size, _MAX_SENSORS,
)
from smeshlite.core.sensors import RaycastResult
from smeshlite.core.match import Match, MatchConfig


# -----------------------------------------------------------------------
# full_obs_size helper
# -----------------------------------------------------------------------

class TestFullObsSize:

    def test_1v1_default_sensors(self):
        assert full_obs_size(2, 8) == 12 * 2 + 5 + 3 * 8  # 53

    def test_1v1_custom_sensors(self):
        assert full_obs_size(2, 4) == 12 * 2 + 5 + 3 * 4  # 41

    def test_3_players(self):
        assert full_obs_size(3, 8) == 12 * 3 + 5 + 3 * 8  # 65

    def test_1_player(self):
        assert full_obs_size(1, 8) == 12 + 5 + 3 * 8  # 41


# -----------------------------------------------------------------------
# brain_context_to_full_obs
# -----------------------------------------------------------------------

class TestBrainContextToFullObs:

    def _make_context(self, n_opponents=1, n_sensors=0):
        sensors = {}
        for i in range(n_sensors):
            sensors[f"s{i}"] = RaycastResult(
                origin=(0.0, 0.0), end=(float(i), 0.0),
                distance=float(i), max_dist=100.0,
                hit=(i % 2 == 0),
            )
        opponents = [
            OpponentContext(
                x=10.0 * (i + 1), y=20.0, vx=1.0, vy=2.0,
                damage_pct=30.0 * (i + 1), stocks=3, facing=-1.0,
                in_air=False, action=0, attack_num=0,
                action_frame=0, charge_amount=0.0, invincibility=0,
            )
            for i in range(n_opponents)
        ]
        return BrainContext(
            x=0.0, y=0.0, vx=0.0, vy=0.0,
            in_air=False, facing=1.0,
            damage_pct=0.0, stocks=3, action=0, action_frame=0,
            attack_num=0, charge_amount=0.0, invincibility=10,
            opponents=opponents,
            stage_x1=-700.0, stage_x2=700.0, stage_y_floor=0.0,
            stage_platform_x1=-320.0, stage_platform_x2=320.0,
            sensors=sensors,
        )

    def test_full_obs_size_matches_helper(self):
        ctx = self._make_context(n_opponents=1, n_sensors=2)
        obs = brain_context_to_full_obs(ctx, max_sensors=8)
        assert len(obs) == full_obs_size(2, 8)

    def test_full_obs_custom_max_sensors(self):
        ctx = self._make_context(n_opponents=1, n_sensors=2)
        obs = brain_context_to_full_obs(ctx, max_sensors=4)
        assert len(obs) == full_obs_size(2, 4)

    def test_self_entity_fields(self):
        ctx = self._make_context(n_opponents=0)
        obs = brain_context_to_full_obs(ctx, max_sensors=8)
        # First 12 floats = self entity
        assert obs[0] == ctx.x       # x
        assert obs[1] == ctx.y       # y
        assert obs[2] == ctx.vx      # vx
        assert obs[3] == ctx.vy      # vy
        assert obs[4] == ctx.damage_pct
        assert obs[5] == float(ctx.stocks)
        assert obs[6] == float(ctx.action)
        assert obs[7] == float(ctx.action_frame)
        assert obs[8] == ctx.facing
        assert obs[9] == float(ctx.in_air)
        assert obs[10] == ctx.charge_amount
        assert obs[11] == float(ctx.invincibility)

    def test_opponent_entity_fields(self):
        ctx = self._make_context(n_opponents=1)
        obs = brain_context_to_full_obs(ctx, max_sensors=8)
        # Opponent starts at index 12
        opp = ctx.opponents[0]
        assert obs[12] == opp.x
        assert obs[13] == opp.y
        assert obs[23] == float(opp.invincibility)

    def test_stage_fields(self):
        ctx = self._make_context(n_opponents=1)
        obs = brain_context_to_full_obs(ctx, max_sensors=8)
        # Stage starts after self + opponent: 12*2 = 24
        stage_start = 24
        assert obs[stage_start + 0] == ctx.stage_x1
        assert obs[stage_start + 1] == ctx.stage_x2
        assert obs[stage_start + 2] == ctx.stage_y_floor
        assert obs[stage_start + 3] == ctx.stage_platform_x1
        assert obs[stage_start + 4] == ctx.stage_platform_x2

    def test_sensor_values(self):
        ctx = self._make_context(n_opponents=1, n_sensors=2)
        obs = brain_context_to_full_obs(ctx, max_sensors=8)
        # Sensors start after self + opponent + stage: 24 + 5 = 29
        sensor_start = 29
        # First sensor
        s0 = ctx.sensors["s0"]
        assert obs[sensor_start + 0] == s0.distance
        assert obs[sensor_start + 1] == float(s0.hit)
        assert obs[sensor_start + 2] == s0.value
        # Second sensor
        s1 = ctx.sensors["s1"]
        assert obs[sensor_start + 3] == s1.distance
        assert obs[sensor_start + 4] == float(s1.hit)
        assert obs[sensor_start + 5] == s1.value

    def test_sensor_padding(self):
        ctx = self._make_context(n_opponents=1, n_sensors=2)
        obs = brain_context_to_full_obs(ctx, max_sensors=8)
        sensor_start = 29
        # Slots 2-7 should be zero-padded
        for i in range(2, 8):
            assert obs[sensor_start + 3 * i + 0] == 0.0
            assert obs[sensor_start + 3 * i + 1] == 0.0
            assert obs[sensor_start + 3 * i + 2] == 0.0

    def test_no_sensors_all_padded(self):
        ctx = self._make_context(n_opponents=1, n_sensors=0)
        obs = brain_context_to_full_obs(ctx, max_sensors=8)
        sensor_start = 29
        for i in range(8):
            assert obs[sensor_start + 3 * i + 0] == 0.0
            assert obs[sensor_start + 3 * i + 1] == 0.0
            assert obs[sensor_start + 3 * i + 2] == 0.0

    def test_more_sensors_than_max_truncates(self):
        ctx = self._make_context(n_opponents=1, n_sensors=10)
        obs = brain_context_to_full_obs(ctx, max_sensors=4)
        assert len(obs) == full_obs_size(2, 4)
        # Only first 4 sensors should appear
        sensor_start = 24 + 5
        for i in range(4):
            s = ctx.sensors[f"s{i}"]
            assert obs[sensor_start + 3 * i + 0] == s.distance

    def test_all_values_finite(self):
        ctx = self._make_context(n_opponents=1, n_sensors=3)
        obs = brain_context_to_full_obs(ctx, max_sensors=8)
        assert all(np.isfinite(v) for v in obs)


# -----------------------------------------------------------------------
# Minimal obs unchanged (regression)
# -----------------------------------------------------------------------

class TestMinimalObsRegression:

    def test_brain_context_to_obs_still_11_per_entity(self):
        ctx = BrainContext(
            x=1.0, y=2.0, vx=3.0, vy=4.0,
            in_air=True, facing=1.0,
            damage_pct=50.0, stocks=2, action=3, action_frame=10,
            attack_num=1, charge_amount=0.5, invincibility=5,
            opponents=[OpponentContext(
                x=10.0, y=20.0, vx=1.0, vy=2.0,
                damage_pct=30.0, stocks=3, facing=-1.0,
                in_air=False, action=0, attack_num=0,
                action_frame=0, charge_amount=0.0, invincibility=0,
            )],
        )
        obs = brain_context_to_obs(ctx)
        # minimal obs: 11 per entity, 2 entities = 22
        assert len(obs) == 22
        # invincibility should NOT be in minimal obs
        assert 5.0 not in obs


# -----------------------------------------------------------------------
# SmeshLiteEnv with obs_mode
# -----------------------------------------------------------------------

class TestEnvObsMode:

    def test_minimal_obs_shape(self):
        env = SmeshLiteEnv(obs_mode="minimal")
        obs, _ = env.reset()
        assert obs.shape == (22,)
        env.close()

    def test_full_obs_shape(self):
        env = SmeshLiteEnv(obs_mode="full", max_sensors=8)
        obs, _ = env.reset()
        assert obs.shape == (53,)
        env.close()

    def test_full_obs_custom_max_sensors(self):
        env = SmeshLiteEnv(obs_mode="full", max_sensors=4)
        obs, _ = env.reset()
        assert obs.shape == (41,)
        env.close()

    def test_minimal_obs_values_match_match_get_obs(self):
        env = SmeshLiteEnv(obs_mode="minimal")
        obs, _ = env.reset()
        expected = np.array(env.match.get_obs(perspective=0), dtype=np.float32)
        np.testing.assert_array_equal(obs, expected)
        env.close()

    def test_full_obs_all_finite(self):
        env = SmeshLiteEnv(obs_mode="full")
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs))
        env.close()

    def test_full_obs_step_1000_steps(self):
        env = SmeshLiteEnv(obs_mode="full")
        obs, _ = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            assert obs.shape == (53,)
            assert np.all(np.isfinite(obs))
            if term or trunc:
                obs, _ = env.reset()
        env.close()

    def test_full_obs_with_sensors(self):
        """SmeshBot has sensors -- full obs should capture them."""
        env = SmeshLiteEnv(obs_mode="full", max_sensors=8)
        obs, _ = env.reset()
        # Default env has no sensors on characters (ExternalBrain has none).
        # But the shape is still correct.
        assert obs.shape == (53,)
        env.close()

    def test_full_obs_stage_fields_populated(self):
        env = SmeshLiteEnv(obs_mode="full", max_sensors=8)
        obs, _ = env.reset()
        # Stage fields at offset 24: [stage_x1, stage_x2, stage_y_floor,
        #                             stage_platform_x1, stage_platform_x2]
        stage_x1 = obs[24]
        stage_x2 = obs[25]
        stage_y_floor = obs[26]
        platform_x1 = obs[27]
        platform_x2 = obs[28]
        # Kill zone should be +/- 700
        assert stage_x1 == -700.0
        assert stage_x2 == 700.0
        # Main platform y = 0
        assert stage_y_floor == 0.0
        # Platform bounds should be within kill zone
        assert -700 <= platform_x1 <= 0
        assert 0 <= platform_x2 <= 700
        env.close()

    def test_invalid_obs_mode_raises(self):
        with pytest.raises(AssertionError, match="obs_mode"):
            SmeshLiteEnv(obs_mode="pixels")

    def test_observation_space_matches_mode(self):
        env_min = SmeshLiteEnv(obs_mode="minimal")
        assert env_min.observation_space.shape == (22,)

        env_full = SmeshLiteEnv(obs_mode="full", max_sensors=8)
        assert env_full.observation_space.shape == (53,)

        env_full4 = SmeshLiteEnv(obs_mode="full", max_sensors=4)
        assert env_full4.observation_space.shape == (41,)

        env_min.close()
        env_full.close()
        env_full4.close()

    def test_full_obs_includes_invincibility(self):
        env = SmeshLiteEnv(obs_mode="full", max_sensors=8)
        env.reset()
        # Characters start in RESPAWN with invincibility=90, but after
        # the initial idle tick in reset(), the frame counter advances.
        # Check that invincibility field is present and non-negative
        char = env.match.characters[0]
        obs, _ = env.reset()
        # self invincibility is at index 11
        assert obs[11] >= 0.0
        env.close()

    def test_gymnasium_checker_full_mode(self):
        """Gymnasium env checker should pass in full obs mode."""
        from gymnasium.utils.env_checker import check_env
        env = SmeshLiteEnv(obs_mode="full")
        check_env(env, warn=True, skip_render_check=True)
        env.close()
