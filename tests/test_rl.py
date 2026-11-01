"""Tests for Phase 4D: Contextual Bandit, Censored Demand, Economic Reward, Safety Limits, and 5-Seed Benchmark."""

from datetime import date, timedelta
import random
import numpy as np
import pandas as pd
import pytest
from app.db import get_db
from ml.rl import (
    MULTIPLIERS,
    BASELINE_ACTION_IDX,
    ContextualBandit,
    BakeryReplayEnv,
    calculate_economic_reward,
    estimate_actual_demand,
    discretize_state,
    is_rl_enabled,
    save_policy,
    get_active_policy,
    rollback_policy,
    evaluate_rl_against_baselines,
    evaluate_rl_command,
)
from ml.predictor import predict_item_demand


def test_economic_reward_with_explicit_prices():
    """Test reward calculation when unit_price and unit_cost are provided."""
    # Price: £2.50, Cost: £0.80
    # Sold: 20, Baked: 25, Demand: 22
    # Unsold: 5, Unmet Demand: 2
    # Revenue: 2.50 * 20 = 50.00
    # Production Cost: 0.80 * 25 = 20.00
    # Waste penalty: 0.25 * 0.80 * 5 = 1.00
    # Lost sale penalty: 0.50 * (2.50 - 0.80) * 2 = 1.70
    # Net reward = 50.00 - 20.00 - 1.00 - 1.70 = 27.30
    reward, notice = calculate_economic_reward(
        sold=20,
        baked=25,
        demand=22.0,
        unit_price=2.50,
        unit_cost=0.80,
    )
    assert reward == 27.30
    assert notice is None


def test_economic_reward_missing_price_fallback_and_notice():
    """Test that missing price or cost applies default 1.0/0.5 economics with an explicit notice."""
    reward, notice = calculate_economic_reward(
        sold=10,
        baked=10,
        demand=10.0,
        unit_price=None,
        unit_cost=None,
    )
    # Default: P=1.0, C=0.5 -> 1.0*10 - 0.5*10 = 5.0
    assert reward == 5.00
    assert notice is not None
    assert "Default economics applied: unit price 1.0, unit cost 0.5" in notice


def test_censored_demand_unbiasing():
    """Test right-censored demand adjustment when sold_out is True."""
    # Not sold out: demand equals observed sales
    d_normal = estimate_actual_demand(sold=20, sold_out=False)
    assert d_normal == 20.0

    # Sold out before closing: 15% inflation adjustment (at least 1 unit)
    d_censored = estimate_actual_demand(sold=20, sold_out=True)
    # 20 + max(1, round(0.15 * 20)) = 20 + 3 = 23.0
    assert d_censored == 23.0

    # Low sales edge case: 2 units sold out -> 2 + 1 = 3.0
    d_low = estimate_actual_demand(sold=2, sold_out=True)
    assert d_low == 3.0


def test_safety_limit_constrained_to_one_step_under_30_observations():
    """Safety rule: while an item has < 30 observations, action can NEVER move >1 step from best action."""
    bandit = ContextualBandit(observations_count=10, epsilon=0.10)
    state = "2_0_0_1_1_0"

    # Best action is baseline index 3 (multiplier 1.0)
    # Allowed actions: |a - 3| <= 1 -> indices {2, 3, 4} (multipliers 0.9, 1.0, 1.1)
    allowed_indices = {2, 3, 4}

    # Test across 200 random selections (including exploration)
    for s in range(200):
        rng = random.Random(s)
        action_idx = bandit.select_action(state, is_eval=False, rng=rng)
        assert action_idx in allowed_indices, f"Action index {action_idx} violated 1-step safety boundary under 30 observations!"
        multiplier = MULTIPLIERS[action_idx]
        assert multiplier in (0.9, 1.0, 1.1)


def test_safety_limit_lifts_at_30_observations():
    """After 30 observations, policy can explore the full multiplier space."""
    bandit = ContextualBandit(observations_count=35, epsilon=0.10)
    state = "2_0_0_1_1_0"

    chosen_indices = set()
    for s in range(500):
        rng = random.Random(s)
        act = bandit.select_action(state, is_eval=False, rng=rng)
        chosen_indices.add(act)

    # Should be able to choose beyond {2, 3, 4}
    assert any(a in (0, 1, 5, 6) for a in chosen_indices)


def test_epsilon_strictly_capped_at_point_one():
    """Epsilon cannot exceed 0.10 under any configuration."""
    bandit_high = ContextualBandit(epsilon=0.50)
    assert bandit_high.epsilon == 0.10

    bandit_normal = ContextualBandit(epsilon=0.05)
    assert bandit_normal.epsilon == 0.05


def test_replay_environment_reset_and_step():
    """Test Gym-style BakeryReplayEnv reset and step API."""
    dates = [(date(2026, 9, 1) + timedelta(days=i)).isoformat() for i in range(14)]
    df = pd.DataFrame([
        {"date": d, "quantity_sold": 25.0 + (i % 5), "temp_mean_c": 18.0, "is_rainy": 0, "is_holiday": 0, "sold_out": False, "leftover": 2}
        for i, d in enumerate(dates)
    ])

    env = BakeryReplayEnv(df, unit_price=2.0, unit_cost=0.70, seed=42)
    s0, info0 = env.reset()
    assert isinstance(s0, str)
    assert info0["index"] == 0

    # Step with action 3 (multiplier 1.0)
    next_s, reward, done, truncated, step_info = env.step(BASELINE_ACTION_IDX)
    assert isinstance(next_s, str)
    assert isinstance(reward, float)
    assert done is False
    assert "baked" in step_info
    assert "sim_sold" in step_info
    assert "waste" in step_info


def test_policy_persistence_and_rollback(app):
    """Test saving policy parameters and rolling back to prior versions."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE shop_id = 1").fetchone()
        item_id = item["id"]

        bandit_v1 = ContextualBandit(observations_count=15)
        bandit_v1.q_table["test_state"] = [1.0, 1.2, 1.5, 2.0, 1.8, 1.1, 0.9]

        rec1 = save_policy(1, item_id, bandit_v1, beat_buffer_rule=False, db=db)
        assert rec1["version"] == 1
        assert rec1["active"] == 1

        bandit_v2 = ContextualBandit(observations_count=35)
        bandit_v2.q_table["test_state"] = [2.0, 2.2, 2.5, 3.0, 2.8, 2.1, 1.9]

        rec2 = save_policy(1, item_id, bandit_v2, beat_buffer_rule=True, db=db)
        assert rec2["version"] == 2
        assert rec2["active"] == 1

        # Roll back to previous version (v1)
        res_roll = rollback_policy(1, item_id, db=db)
        assert res_roll["previous_version"] == 2
        assert res_roll["active_version"] == 1

        loaded_bandit, meta = get_active_policy(1, item_id, db=db)
        assert meta["version"] == 1
        assert loaded_bandit.observations_count == 15


def test_rl_flag_toggle_in_predictor(app, monkeypatch):
    """Test that RL learned adjustments only show when RL_ENABLED=true, obs >= 30, and beat_buffer_rule=true."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id FROM items WHERE shop_id = 1").fetchone()
        item_id = item["id"]

        # 1. When RL_ENABLED is false (default): rl_active must be False
        monkeypatch.setenv("RL_ENABLED", "0")
        assert not is_rl_enabled()
        pred_off = predict_item_demand(item_id, "2026-11-20", shop_id=1)
        assert pred_off["rl_active"] is False

        # 2. When RL_ENABLED is true, but policy has < 30 observations: rl_active must be False
        monkeypatch.setenv("RL_ENABLED", "1")
        assert is_rl_enabled()
        b_low = ContextualBandit(observations_count=20)
        save_policy(1, item_id, b_low, beat_buffer_rule=True, db=db)

        pred_low = predict_item_demand(item_id, "2026-11-20", shop_id=1)
        assert pred_low["rl_active"] is False

        # 3. When RL_ENABLED is true, obs >= 30, and beat_buffer_rule is True: rl_active is True!
        b_qualified = ContextualBandit(observations_count=45)
        save_policy(1, item_id, b_qualified, beat_buffer_rule=True, db=db)

        pred_on = predict_item_demand(item_id, "2026-11-20", shop_id=1)
        assert pred_on["rl_active"] is True
        assert pred_on["rl_multiplier"] in MULTIPLIERS
        assert "Reinforcement Learning Policy" in pred_on["rl_notice"]


def test_five_seed_evaluation_benchmark_run(runner, app):
    """Test 5-seed evaluation benchmark runs and outputs comparison table with profit, waste, and stock-outs."""
    with app.app_context():
        db = get_db()
        item = db.execute("SELECT id, name FROM items WHERE shop_id = 1").fetchone()
        item_id = item["id"]

        # Seed sufficient history for 28-day holdout + 14-day train
        dates = [(date(2026, 6, 1) + timedelta(days=i)).isoformat() for i in range(80)]
        for i, d in enumerate(dates):
            db.execute(
                "INSERT INTO daily_log (shop_id, item_id, date, sold, baked, leftover) VALUES (1, ?, ?, ?, ?, 0)"
                " ON CONFLICT(shop_id, item_id, date) DO UPDATE SET sold = excluded.sold",
                (item_id, d, 25 + (i % 7), 30),
            )
        db.commit()

        eval_res = evaluate_rl_against_baselines(shop_id=1, item_id=item_id, seeds=[42, 123, 456, 789, 999], holdout_days=28, db=db)
        assert eval_res["status"] == "success"
        assert len(eval_res["seeds"]) == 5
        summary = eval_res["summary"]
        for strat in ("naive", "forecast_only", "forecast_buffer", "rl_policy"):
            assert strat in summary
            assert "profit_mean" in summary[strat]
            assert "profit_var" in summary[strat]
            assert "waste_mean" in summary[strat]
            assert "stockouts_mean" in summary[strat]

    # Test CLI execution
    result = runner.invoke(evaluate_rl_command, ["--shop-id", "1", "--item-id", str(item_id)])
    assert result.exit_code == 0
    assert "PHASE 4D: 5-SEED EVALUATION BENCHMARK" in result.output
    assert "Contextual Bandit (RL)" in result.output

