"""
Tests for Layer 4: Rich Info Dict + Per-Agent Interface.

Validates action masks, per-agent observations, per-agent rewards,
prev_obs, and the expanded info dict.
"""
import pytest
import numpy as np

from smeshlite import SmeshLiteEnv
from smeshlite.core.character import Character, Action


# -----------------------------------------------------------------------
# Action masks
# -----------------------------------------------------------------------

class TestActionMask:

    def test_dead_all_false(self):
        """DEAD character: no actions are effective."""
        c = Character(0, 0, 0)
        c.action = Action.DEAD
        assert c.action_mask() == [False, False, False, False]

    def test_respawn_movement_only(self):
        """RESPAWN: can move (DI) but no up or attack."""
        c = Character(0, 0, 0)
        c.action = Action.RESPAWN
        c.action_frame = 5
        assert c.action_mask() == [True, True, False, False]

    def test_hitstun_movement_only(self):
        """HITSTUN: can DI (left/right) but no up or attack."""
        c = Character(0, 0, 0)
        c.action = Action.HITSTUN
        c.action_frame = 10
        assert c.action_mask() == [True, True, False, False]

    def test_attack_drift_but_no_new_attack(self):
        """ATTACK: can drift + hold up (variable gravity), no new attack."""
        c = Character(0, 0, 0)
        c.action = Action.ATTACK
        c.attack_num = 1
        c.action_frame = 5
        assert c.action_mask() == [True, True, True, False]

    def test_smash_drift_but_no_new_attack(self):
        """SMASH: can drift + hold up, no new attack."""
        c = Character(0, 0, 0)
        c.action = Action.SMASH
        c.attack_num = 2
        c.action_frame = 10
        assert c.action_mask() == [True, True, True, False]

    def test_charging_all_true(self):
        """CHARGING: can release attack, drift, hold up."""
        c = Character(0, 0, 0)
        c.action = Action.CHARGING
        c.attack_num = 4
        c.charge_amount = 0.5
        assert c.action_mask() == [True, True, True, True]

    def test_none_grounded_all_true(self):
        """NONE on ground: all actions effective."""
        c = Character(0, 0, 68.1)
        c.action = Action.NONE
        c.in_air = False
        c.recover_used = False
        assert c.action_mask() == [True, True, True, True]

    def test_run_grounded_all_true(self):
        """RUN on ground: all actions effective."""
        c = Character(0, 0, 68.1)
        c.action = Action.RUN
        c.in_air = False
        c.recover_used = False
        assert c.action_mask() == [True, True, True, True]

    def test_aerial_no_recover_up_false(self):
        """AERIAL with recover_used: up is NOT effective (no double jump)."""
        c = Character(0, 0, 200)
        c.action = Action.AERIAL
        c.in_air = True
        c.recover_used = True
        mask = c.action_mask()
        assert mask == [True, True, False, True]

    def test_aerial_with_recover_up_true(self):
        """AERIAL without recover_used: up IS effective (uair recovery)."""
        c = Character(0, 0, 200)
        c.action = Action.AERIAL
        c.in_air = True
        c.recover_used = False
        mask = c.action_mask()
        assert mask == [True, True, True, True]

    def test_jump_up_true(self):
        """JUMP in air without recover_used: up is effective."""
        c = Character(0, 0, 200)
        c.action = Action.JUMP
        c.in_air = True
        c.recover_used = False
        assert c.action_mask() == [True, True, True, True]

    def test_jump_recover_used_up_false(self):
        """JUMP in air with recover_used: up is NOT effective."""
        c = Character(0, 0, 200)
        c.action = Action.JUMP
        c.in_air = True
        c.recover_used = True
        assert c.action_mask() == [True, True, False, True]


class TestActionMaskInEnv:

    def test_action_masks_in_info(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert "action_masks" in info
        env.close()

    def test_action_masks_shape(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        masks = info["action_masks"]
        assert len(masks) == env.n_players
        for m in masks:
            assert len(m) == 4
            assert all(isinstance(v, (bool, np.bool_)) for v in m)
        env.close()

    def test_action_masks_respawn_period(self):
        """During respawn, masks should block up and attack."""
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        # Characters start in RESPAWN -- check mask
        masks = info["action_masks"]
        for m in masks:
            assert m[0] is True or m[0] == True   # left OK (DI)
            assert m[1] is True or m[1] == True   # right OK (DI)
            assert m[2] is False or m[2] == False  # up blocked
            assert m[3] is False or m[3] == False  # attack blocked
        env.close()

    def test_action_masks_after_respawn(self):
        """After respawn animation, up and attack should become available."""
        env = SmeshLiteEnv()
        env.reset()
        # Advance past entrance (20 frames) with idle actions
        for _ in range(25):
            env.step([[0, 0, 0, 0], [0, 0, 0, 0]])
        info = env.step([[0, 0, 0, 0], [0, 0, 0, 0]])[4]
        # With idle actions, characters should be in NONE/AERIAL/RUN
        # and have attack available (not stuck in attack/hitstun)
        masks = info["action_masks"]
        any_attack = any(m[3] for m in masks)
        assert any_attack, "At least one player should be able to attack after respawn (idle)"
        env.close()

    def test_action_masks_3_players(self):
        env = SmeshLiteEnv(n_players=3)
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert len(info["action_masks"]) == 3
        env.close()


# -----------------------------------------------------------------------
# Per-agent observations
# -----------------------------------------------------------------------

class TestPerAgentObs:

    def test_obs_list_in_info(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert "obs" in info
        env.close()

    def test_obs_list_length(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert len(info["obs"]) == env.n_players
        env.close()

    def test_obs_0_matches_step_return(self):
        """info['obs'][0] should match the obs returned by step()."""
        env = SmeshLiteEnv()
        env.reset()
        obs, _, _, _, info = env.step(env.action_space.sample())
        assert np.allclose(obs, info["obs"][0])
        env.close()

    def test_p1_perspective_swapped(self):
        """P1's observation should have self=P1, opponent=P0."""
        env = SmeshLiteEnv()
        obs, _ = env.reset()
        # P0 spawns at x=-245, P1 spawns at x=+245
        # P0's obs: self_x ~ -245, opp_x ~ +245
        # P1's obs: self_x ~ +245, opp_x ~ -245
        info = env.step(env.action_space.sample())[4]
        p0_obs = info["obs"][0]
        p1_obs = info["obs"][1]
        assert p0_obs[0] < p1_obs[0], "P0 should spawn left of P1"
        assert p1_obs[0] > p0_obs[0], "P1 should spawn right of P0"
        env.close()

    def test_p1_self_is_p0_opponent(self):
        """P1's self position should equal P0's opponent position."""
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        p0_obs = info["obs"][0]
        p1_obs = info["obs"][1]
        # P0's self_x == P1's opponent_x
        assert np.isclose(p0_obs[0], p1_obs[12]), \
            f"P0 self_x {p0_obs[0]} != P1 opp_x {p1_obs[12]}"
        # P1's self_x == P0's opponent_x
        assert np.isclose(p1_obs[0], p0_obs[12]), \
            f"P1 self_x {p1_obs[0]} != P0 opp_x {p0_obs[12]}"
        env.close()

    def test_per_agent_obs_minimal_mode(self):
        env = SmeshLiteEnv(obs_mode="minimal")
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert len(info["obs"]) == 2
        assert info["obs"][0].shape == (22,)
        assert info["obs"][1].shape == (22,)
        env.close()

    def test_per_agent_obs_3_players(self):
        env = SmeshLiteEnv(n_players=3)
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert len(info["obs"]) == 3
        env.close()


# -----------------------------------------------------------------------
# Per-agent rewards
# -----------------------------------------------------------------------

class TestPerAgentRewards:

    def test_rewards_in_info(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert "rewards" in info
        env.close()

    def test_rewards_length(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert len(info["rewards"]) == env.n_players
        env.close()

    def test_zero_sum_1v1(self):
        """For 1v1, P0 reward + P1 reward should be 0 (zero-sum)."""
        env = SmeshLiteEnv(action_mode="single", opponent_brain="Smesh Bot")
        env.reset()
        for _ in range(30):
            env.step([0, 0, 0, 0])

        total_p0 = 0.0
        total_p1 = 0.0
        for _ in range(1000):
            obs, r, t, tr, info = env.step([0, 1, 0, 1])
            total_p0 += info["rewards"][0]
            total_p1 += info["rewards"][1]
            if t or tr:
                obs, _ = env.reset()

        assert abs(total_p0 + total_p1) < 1e-6, \
            f"Zero-sum violated: P0={total_p0}, P1={total_p1}, sum={total_p0 + total_p1}"
        env.close()

    def test_step_reward_is_p0_reward(self):
        """The step() return reward should equal info['rewards'][0]."""
        env = SmeshLiteEnv(action_mode="single", opponent_brain="Smesh Bot")
        env.reset()
        for _ in range(30):
            env.step([0, 0, 0, 0])

        for _ in range(100):
            obs, r, t, tr, info = env.step([0, 1, 0, 0])
            assert r == info["rewards"][0], \
                f"step reward {r} != info['rewards'][0] {info['rewards'][0]}"
            if t or tr:
                obs, _ = env.reset()
        env.close()

    def test_rewards_3_players(self):
        env = SmeshLiteEnv(n_players=3)
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert len(info["rewards"]) == 3
        env.close()


# -----------------------------------------------------------------------
# Prev obs
# -----------------------------------------------------------------------

class TestPrevObs:

    def test_prev_obs_in_info(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert "prev_obs" in info
        env.close()

    def test_prev_obs_matches_previous_step(self):
        """prev_obs should be the obs from the previous step or reset."""
        env = SmeshLiteEnv()
        obs_reset, _ = env.reset()
        obs_1, _, _, _, info_1 = env.step(env.action_space.sample())
        assert np.allclose(info_1["prev_obs"], obs_reset), \
            "prev_obs on first step should match reset obs"

        obs_2, _, _, _, info_2 = env.step(env.action_space.sample())
        assert np.allclose(info_2["prev_obs"], obs_1), \
            "prev_obs should match previous step's obs"
        env.close()

    def test_prev_obs_temporal_consistency(self):
        """info['prev_obs'] + action -> info['obs'][0] is the (s_t, a_t, s_{t+1}) triple."""
        env = SmeshLiteEnv()
        obs, _ = env.reset()
        for _ in range(5):
            obs, r, t, tr, info = env.step(env.action_space.sample())
            # prev_obs + this step's obs form a valid transition
            assert info["prev_obs"].shape == obs.shape
            # They should differ (game is advancing)
            # (they could be identical on the very first tick during respawn freeze)
            if t or tr:
                obs, _ = env.reset()
        env.close()


# -----------------------------------------------------------------------
# Per-player stats (damage, stocks lists)
# -----------------------------------------------------------------------

class TestPerPlayerStats:

    def test_damage_list_in_info(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert "damage" in info
        assert len(info["damage"]) == env.n_players
        env.close()

    def test_stocks_list_in_info(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert "stocks" in info
        assert len(info["stocks"]) == env.n_players
        env.close()

    def test_damage_values_match_characters(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        for i in range(env.n_players):
            assert np.isclose(
                info["damage"][i],
                env.match.characters[i].damage_pct,
            )
        env.close()

    def test_stocks_values_match_characters(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        for i in range(env.n_players):
            assert info["stocks"][i] == env.match.characters[i].stocks
        env.close()

    def test_old_p0_keys_removed(self):
        """The p0_damage/p0_stocks/p1_damage/p1_stocks keys should be gone."""
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        assert "p0_damage" not in info
        assert "p0_stocks" not in info
        assert "p1_damage" not in info
        assert "p1_stocks" not in info
        env.close()


# -----------------------------------------------------------------------
# Info dict completeness
# -----------------------------------------------------------------------

class TestInfoCompleteness:

    def test_all_expected_keys_present(self):
        env = SmeshLiteEnv()
        env.reset()
        info = env.step(env.action_space.sample())[4]
        expected = {
            "frame", "winner", "actions", "applied_actions",
            "damage_events", "action_masks", "obs", "rewards",
            "damage", "stocks", "prev_obs",
        }
        missing = expected - set(info.keys())
        assert not missing, f"Missing info keys: {missing}"
        env.close()

    def test_gymnasium_checker_still_passes(self):
        from gymnasium.utils.env_checker import check_env
        env = SmeshLiteEnv()
        check_env(env, warn=True, skip_render_check=True)
        env.close()
