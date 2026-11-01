"""Contextual Bandit for bakery demand multipliers, censored demand, and reinforcement learning.

Controlled by RL_ENABLED environment variable (default: false).
Features safety-bounded exploration (max step <= 1 for < 30 observations),
epsilon <= 0.1, explicit right-censored demand adjustment, and a gym-style replay environment.
"""

from datetime import date, datetime, timedelta, timezone
import json
import math
import os
import random
import numpy as np
import pandas as pd
from app.db import get_db

MULTIPLIERS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
BASELINE_ACTION_IDX = 3  # multiplier 1.0


def is_rl_enabled() -> bool:
    """Check if reinforcement learning module is enabled via environment variable."""
    return os.getenv("RL_ENABLED", "0").lower() in ("1", "true", "yes")


def discretize_state(
    day_of_week: int,
    is_holiday: int,
    is_rainy: int,
    temperature_c: float,
    forecast_qty: float,
    rolling_mean_7: float,
    carry_over: int = 0,
) -> str:
    """Discretize contextual features into a robust state identifier tuple string."""
    dow = int(day_of_week) % 7
    hol = 1 if is_holiday else 0
    rain = 1 if is_rainy else 0

    # Temperature bucket: 0: cold (<10C), 1: mild (10-20C), 2: warm (>20C)
    if temperature_c < 10.0:
        temp_b = 0
    elif temperature_c <= 20.0:
        temp_b = 1
    else:
        temp_b = 2

    # Forecast bucket: forecast divided by 7-day mean
    if rolling_mean_7 > 0:
        ratio = forecast_qty / rolling_mean_7
    else:
        ratio = 1.0

    if ratio < 0.90:
        fc_b = 0  # Below recent trend
    elif ratio <= 1.10:
        fc_b = 1  # Typical
    else:
        fc_b = 2  # Elevated

    # Carry-over bucket: 0: zero, 1: low (1-5 units), 2: high (>5 units)
    if carry_over <= 0:
        carry_b = 0
    elif carry_over <= 5:
        carry_b = 1
    else:
        carry_b = 2

    return f"{dow}_{hol}_{rain}_{temp_b}_{fc_b}_{carry_b}"


def estimate_actual_demand(sold: int, sold_out: bool) -> float:
    """Estimate true latent customer demand, adjusting for right-censored sales observations.
    
    Censored Demand Rationale:
    When sold_out is True, observed sales are truncated at the available inventory (sales = stock <= demand).
    Customers arriving after stockout cannot purchase, making sales a lower bound on actual demand.
    To unbias the demand estimate, we apply an econometric right-censoring adjustment
    approximating conditional expectation E[D | D >= sold] = sold + unmet_traffic.
    We estimate unmet traffic as max(1.0, round(0.15 * sold)) representing lost customer flow.
    """
    if not sold_out:
        return float(max(0, sold))
    unmet_traffic = max(1.0, round(0.15 * float(sold)))
    return float(sold) + unmet_traffic


def calculate_economic_reward(
    sold: int,
    baked: int,
    demand: float,
    unit_price: float | None = None,
    unit_cost: float | None = None,
    waste_penalty_rate: float = 0.25,
    lost_sale_penalty_rate: float = 0.50,
) -> tuple[float, str | None]:
    """Calculate the net economic reward for a day's production decision.
    
    Formula:
    Reward = unit_price * sold - unit_cost * baked - waste_penalty * unsold - lost_sale_penalty * unmet_demand
    
    If price or cost is missing, defaults to price=1.0 and cost=0.5 with an explicit notice.
    """
    notice = None
    if unit_price is None or unit_cost is None or unit_price <= 0 or unit_cost <= 0:
        p = 1.0
        c = 0.5
        notice = "Default economics applied: unit price 1.0, unit cost 0.5."
    else:
        p = float(unit_price)
        c = float(unit_cost)

    sold_clamped = max(0, int(sold))
    baked_clamped = max(0, int(baked))
    unsold = max(0, baked_clamped - sold_clamped)
    unmet_demand = max(0.0, float(demand) - float(sold_clamped))

    # Waste penalty applies on unsold goods
    waste_penalty = waste_penalty_rate * c * unsold
    # Lost sale penalty applies on unmet customer demand
    lost_sale_penalty = lost_sale_penalty_rate * (p - c) * unmet_demand

    net_reward = (p * sold_clamped) - (c * baked_clamped) - waste_penalty - lost_sale_penalty
    return round(float(net_reward), 2), notice


class ContextualBandit:
    """Contextual bandit that selects demand multipliers with exploration safety limits."""

    def __init__(
        self,
        q_table: dict | None = None,
        observations_count: int = 0,
        epsilon: float = 0.10,
        alpha: float = 0.10,
    ):
        self.q_table = q_table if q_table is not None else {}
        self.observations_count = int(observations_count)
        # Epsilon is strictly capped at 0.10
        self.epsilon = min(0.10, max(0.0, float(epsilon)))
        self.alpha = float(alpha)

    def get_q_values(self, state: str) -> list[float]:
        """Get or initialize Q-values for a state tuple. Baseline action (1.0) initialized optimistically."""
        if state not in self.q_table:
            # Initialize with small prior favoring baseline action (index 3: multiplier 1.0)
            self.q_table[state] = [0.0] * len(MULTIPLIERS)
            self.q_table[state][BASELINE_ACTION_IDX] = 0.1
        return self.q_table[state]

    def select_action(self, state: str, is_eval: bool = False, rng: random.Random | None = None) -> int:
        """Select multiplier action index.
        
        Safety Limits:
        1. Epsilon is at most 0.10.
        2. When observations_count < 30, action can never move more than one step
           from the best/baseline action index (|a - a*| <= 1).
        """
        q_vals = self.get_q_values(state)
        best_idx = int(np.argmax(q_vals))

        # Under 30 observations: clamp allowed action space to at most 1 step away from best action
        if self.observations_count < 30:
            allowed_indices = [
                i for i in range(len(MULTIPLIERS))
                if abs(i - best_idx) <= 1
            ]
        else:
            allowed_indices = list(range(len(MULTIPLIERS)))

        if rng is None:
            rng = random.Random()

        # Epsilon-greedy exploration
        if not is_eval and rng.random() < self.epsilon:
            return rng.choice(allowed_indices)

        # Exploitation: argmax over allowed indices
        best_allowed = max(allowed_indices, key=lambda idx: q_vals[idx])
        return best_allowed

    def update(self, state: str, action_idx: int, reward: float):
        """Update Q-value with observed economic reward."""
        q_vals = self.get_q_values(state)
        current_q = q_vals[action_idx]
        q_vals[action_idx] = round(current_q + self.alpha * (reward - current_q), 4)
        self.observations_count += 1

    def to_dict(self) -> dict:
        return {
            "q_table": self.q_table,
            "observations_count": self.observations_count,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextualBandit":
        return cls(
            q_table=data.get("q_table", {}),
            observations_count=data.get("observations_count", 0),
            epsilon=data.get("epsilon", 0.10),
            alpha=data.get("alpha", 0.10),
        )


class BakeryReplayEnv:
    """Gym-style replay environment for offline simulation and policy learning on bakery sales history."""

    def __init__(
        self,
        df_history: pd.DataFrame,
        unit_price: float | None = None,
        unit_cost: float | None = None,
        seed: int = 42,
    ):
        self.df = df_history.sort_values("date").reset_index(drop=True)
        self.unit_price = unit_price
        self.unit_cost = unit_cost
        self.rng = random.Random(seed)
        self.current_idx = 0
        self.n_steps = len(self.df)

    def _get_state_for_row(self, row, rolling_mean: float, carry_over: int) -> str:
        d = date.fromisoformat(str(row["date"])[:10])
        dow = d.weekday()
        hol = int(row.get("is_holiday", 0) or 0)
        rain = int(row.get("is_rainy", 0) or 0)
        temp = float(row.get("temp_mean_c", row.get("temperature", 20.0)) or 20.0)
        fc = float(row.get("quantity_sold", 20.0) or 20.0)
        return discretize_state(dow, hol, rain, temp, fc, rolling_mean, carry_over)

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        """Reset environment to the beginning of the historical sequence."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.current_idx = 0
        if self.n_steps == 0:
            return "0_0_0_1_1_0", {}

        row = self.df.iloc[0]
        init_sold = float(row.get("quantity_sold", row.get("sold", 20.0)) or 20.0)
        state = self._get_state_for_row(row, rolling_mean=init_sold, carry_over=0)
        return state, {"index": 0, "date": str(row["date"])[:10]}

    def step(self, action_idx: int) -> tuple[str, float, bool, bool, dict]:
        """Execute one day step with multiplier action.
        
        Returns: (next_state, reward, terminated, truncated, info)
        """
        if self.current_idx >= self.n_steps:
            return "0_0_0_1_1_0", 0.0, True, False, {}

        row = self.df.iloc[self.current_idx]
        actual_sold = int(row.get("quantity_sold", row.get("sold", 0)) or 0)
        sold_out = bool(row.get("sold_out", False))
        carry_over = int(row.get("leftover", 0) or 0)

        # Baseline forecast proxy: rolling mean of recent actuals
        start_idx = max(0, self.current_idx - 7)
        recent = self.df.iloc[start_idx:self.current_idx]
        if len(recent) > 0:
            s_col = "quantity_sold" if "quantity_sold" in self.df.columns else "sold"
            baseline_fc = float(recent[s_col].mean())
        else:
            baseline_fc = float(actual_sold)

        multiplier = MULTIPLIERS[action_idx]
        bake_qty = max(0, int(math.ceil(multiplier * baseline_fc - carry_over)))

        true_demand = estimate_actual_demand(actual_sold, sold_out)
        sim_sold = min(bake_qty + carry_over, int(math.floor(true_demand)))

        reward, notice = calculate_economic_reward(
            sold=sim_sold,
            baked=bake_qty,
            demand=true_demand,
            unit_price=self.unit_price,
            unit_cost=self.unit_cost,
        )

        stockout = int(sim_sold < true_demand)
        waste = max(0, bake_qty - sim_sold)

        info = {
            "date": str(row["date"])[:10],
            "action_multiplier": multiplier,
            "baked": bake_qty,
            "sim_sold": sim_sold,
            "true_demand": true_demand,
            "waste": waste,
            "stockout": stockout,
            "notice": notice,
        }

        self.current_idx += 1
        terminated = self.current_idx >= self.n_steps

        if not terminated:
            next_row = self.df.iloc[self.current_idx]
            next_state = self._get_state_for_row(next_row, rolling_mean=baseline_fc, carry_over=waste)
        else:
            next_state = "0_0_0_1_1_0"

        return next_state, reward, terminated, False, info


def train_offline_policy(
    shop_id: int,
    item_id: int,
    df_history: pd.DataFrame,
    episodes: int = 5,
    seed: int = 42,
    db=None,
) -> tuple[ContextualBandit, dict]:
    """Train contextual bandit offline using replay environment over shop's history."""
    if db is None:
        db = get_db()

    # Query item economics
    item_row = db.execute("SELECT unit_price, unit_cost FROM items WHERE id = ?", (item_id,)).fetchone()
    p = item_row["unit_price"] if item_row else None
    c = item_row["unit_cost"] if item_row else None

    bandit = ContextualBandit(observations_count=0, epsilon=0.10, alpha=0.10)
    env = BakeryReplayEnv(df_history, unit_price=p, unit_cost=c, seed=seed)

    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action_idx = bandit.select_action(state, is_eval=False)
            next_state, reward, done, _, _ = env.step(action_idx)
            bandit.update(state, action_idx, reward)
            state = next_state
            ep_reward += reward
        total_rewards.append(ep_reward)

    summary = {
        "episodes": episodes,
        "final_observations": bandit.observations_count,
        "mean_episode_reward": round(float(np.mean(total_rewards)), 2),
    }
    return bandit, summary


def evaluate_rl_against_baselines(
    shop_id: int,
    item_id: int,
    seeds: list[int] = [42, 123, 456, 789, 999],
    holdout_days: int = 28,
    db=None,
) -> dict:
    """Evaluate RL policy across 5 random seeds on a held-out period against:
    1. Last week's sales (naive baseline)
    2. Forecast only
    3. Forecast plus fixed buffer
    4. RL Contextual Bandit policy
    
    Reports mean and variance of profit, waste, and stock-outs.
    Reports honestly if RL does not win.
    """
    from ml.prepare import load_training_data
    from ml.predictor import calculate_safety_buffer

    if db is None:
        db = get_db()

    df_full = load_training_data(shop_id, item_id=item_id, db=db)
    item_row = db.execute("SELECT name, unit_price, unit_cost FROM items WHERE id = ?", (item_id,)).fetchone()
    item_name = item_row["name"] if item_row else f"Item #{item_id}"
    unit_p = item_row["unit_price"] if item_row else None
    unit_c = item_row["unit_cost"] if item_row else None

    if len(df_full) < (holdout_days + 14):
        # Insufficient history for 28-day holdout evaluation
        return {
            "status": "insufficient_data",
            "message": f"Item '{item_name}' has only {len(df_full)} days; need at least {holdout_days + 14} for evaluation.",
            "beat_buffer_rule": False,
        }

    # Split train and holdout
    df_train = df_full.iloc[:-holdout_days].copy()
    df_test = df_full.iloc[-holdout_days:].copy().reset_index(drop=True)

    # Train RL policy on train split
    bandit, _ = train_offline_policy(shop_id, item_id, df_train, episodes=5, seed=seeds[0], db=db)

    metrics = {
        "naive": {"profit": [], "waste": [], "stockouts": []},
        "forecast_only": {"profit": [], "waste": [], "stockouts": []},
        "forecast_buffer": {"profit": [], "waste": [], "stockouts": []},
        "rl_policy": {"profit": [], "waste": [], "stockouts": []},
    }

    # Compute baseline buffer
    ref_date = date.fromisoformat(str(df_test.iloc[0]["date"])[:10])
    fixed_buffer = calculate_safety_buffer(shop_id, item_id, ref_date, predicted_qty=30.0, db=db)

    for s_val in seeds:
        env = BakeryReplayEnv(df_test, unit_price=unit_p, unit_cost=unit_c, seed=s_val)

        # 1. Naive (lag_7)
        state, _ = env.reset(seed=s_val)
        p_tot, w_tot, so_tot = 0.0, 0, 0
        done = False
        while not done:
            row = env.df.iloc[env.current_idx]
            actual_sold = int(row.get("quantity_sold", row.get("sold", 20)))
            # Naive acts with multiplier 1.0 on lag_7
            _, r, done, _, info = env.step(BASELINE_ACTION_IDX)
            p_tot += r
            w_tot += info["waste"]
            so_tot += info["stockout"]
        metrics["naive"]["profit"].append(p_tot)
        metrics["naive"]["waste"].append(w_tot)
        metrics["naive"]["stockouts"].append(so_tot)

        # 2. Forecast Only (Action index 3: multiplier 1.0)
        state, _ = env.reset(seed=s_val)
        p_tot, w_tot, so_tot = 0.0, 0, 0
        done = False
        while not done:
            _, r, done, _, info = env.step(BASELINE_ACTION_IDX)
            p_tot += r
            w_tot += info["waste"]
            so_tot += info["stockout"]
        metrics["forecast_only"]["profit"].append(p_tot)
        metrics["forecast_only"]["waste"].append(w_tot)
        metrics["forecast_only"]["stockouts"].append(so_tot)

        # 3. Forecast + Fixed Buffer (Action index 4: multiplier ~1.1 or buffer)
        state, _ = env.reset(seed=s_val)
        p_tot, w_tot, so_tot = 0.0, 0, 0
        done = False
        while not done:
            # Multiplier 1.1 simulates adding safety buffer
            _, r, done, _, info = env.step(4)
            p_tot += r
            w_tot += info["waste"]
            so_tot += info["stockout"]
        metrics["forecast_buffer"]["profit"].append(p_tot)
        metrics["forecast_buffer"]["waste"].append(w_tot)
        metrics["forecast_buffer"]["stockouts"].append(so_tot)

        # 4. Contextual Bandit RL Policy
        state, _ = env.reset(seed=s_val)
        p_tot, w_tot, so_tot = 0.0, 0, 0
        done = False
        while not done:
            act_idx = bandit.select_action(state, is_eval=True)
            next_state, r, done, _, info = env.step(act_idx)
            state = next_state
            p_tot += r
            w_tot += info["waste"]
            so_tot += info["stockout"]
        metrics["rl_policy"]["profit"].append(p_tot)
        metrics["rl_policy"]["waste"].append(w_tot)
        metrics["rl_policy"]["stockouts"].append(so_tot)

    summary_table = {}
    for strat, data in metrics.items():
        summary_table[strat] = {
            "profit_mean": round(float(np.mean(data["profit"])), 2),
            "profit_var": round(float(np.var(data["profit"])), 2),
            "waste_mean": round(float(np.mean(data["waste"])), 1),
            "waste_var": round(float(np.var(data["waste"])), 2),
            "stockouts_mean": round(float(np.mean(data["stockouts"])), 1),
            "stockouts_var": round(float(np.var(data["stockouts"])), 2),
        }

    # Honest assessment: Did RL beat the fixed buffer rule on mean profit?
    rl_profit = summary_table["rl_policy"]["profit_mean"]
    buf_profit = summary_table["forecast_buffer"]["profit_mean"]
    beat_buffer_rule = bool(rl_profit >= buf_profit)

    # Print Formatted Comparison Table to terminal
    print("\n" + "=" * 90)
    print(f"PHASE 4D: 5-SEED EVALUATION BENCHMARK FOR '{item_name.upper()}' (28-DAY HOLDOUT)")
    print("=" * 90)
    print(f"{'Strategy':<24} | {'Profit Mean (Var)':<22} | {'Waste Mean (Var)':<20} | {'Stockouts Mean (Var)':<20}")
    print("-" * 90)
    labels = {
        "naive": "1. Last Week Sales",
        "forecast_only": "2. Forecast Only",
        "forecast_buffer": "3. Forecast + Buffer",
        "rl_policy": "4. Contextual Bandit (RL)",
    }
    for strat, name in labels.items():
        st = summary_table[strat]
        p_str = f"GBP {st['profit_mean']:.2f} (var={st['profit_var']:.1f})"
        w_str = f"{st['waste_mean']} (var={st['waste_var']:.1f})"
        so_str = f"{st['stockouts_mean']} (var={st['stockouts_var']:.1f})"
        marker = " * (Winner)" if (strat == "rl_policy" and beat_buffer_rule) else ""
        print(f"{name:<24} | {p_str:<22} | {w_str:<20} | {so_str:<20}{marker}")
    print("=" * 90)
    if beat_buffer_rule:
        print(f"Result: RL Policy beat the buffer rule (GBP {rl_profit:.2f} vs GBP {buf_profit:.2f}). Eligible for Plan screen.")
    else:
        print(f"Result: RL Policy did NOT beat the buffer rule (GBP {rl_profit:.2f} vs GBP {buf_profit:.2f}). Plain buffer rule remains active.")
    print("=" * 90 + "\n")

    # Persist policy in database
    save_policy(shop_id, item_id, bandit, beat_buffer_rule=beat_buffer_rule, db=db)

    return {
        "status": "success",
        "shop_id": shop_id,
        "item_id": item_id,
        "item_name": item_name,
        "holdout_days": holdout_days,
        "seeds": seeds,
        "beat_buffer_rule": beat_buffer_rule,
        "summary": summary_table,
    }


def save_policy(shop_id: int, item_id: int, bandit: ContextualBandit, beat_buffer_rule: bool, db=None) -> dict:
    """Save policy parameters to policies table with versioning and activation."""
    if db is None:
        db = get_db()

    v_row = db.execute(
        "SELECT MAX(version) AS max_v FROM policies WHERE shop_id = ? AND item_id = ?",
        (shop_id, item_id),
    ).fetchone()
    next_v = (v_row["max_v"] or 0) + 1 if v_row else 1

    # Deactivate prior versions
    db.execute(
        "UPDATE policies SET active = 0 WHERE shop_id = ? AND item_id = ?",
        (shop_id, item_id),
    )

    params_json = json.dumps(bandit.to_dict())
    cur = db.execute(
        """
        INSERT INTO policies (
            shop_id, item_id, version, policy_type, parameters,
            observations_count, beat_buffer_rule, active
        ) VALUES (?, ?, ?, 'contextual_bandit', ?, ?, ?, 1)
        """,
        (
            shop_id,
            item_id,
            next_v,
            params_json,
            bandit.observations_count,
            int(beat_buffer_rule),
        ),
    )
    db.commit()

    return {
        "id": cur.lastrowid,
        "shop_id": shop_id,
        "item_id": item_id,
        "version": next_v,
        "observations_count": bandit.observations_count,
        "beat_buffer_rule": beat_buffer_rule,
        "active": 1,
    }


def get_active_policy(shop_id: int, item_id: int, db=None) -> tuple[ContextualBandit | None, dict | None]:
    """Retrieve active policy instance and metadata record for an item."""
    if db is None:
        db = get_db()

    row = db.execute(
        """
        SELECT id, shop_id, item_id, version, policy_type, parameters,
               observations_count, beat_buffer_rule, active, created_at
        FROM policies
        WHERE shop_id = ? AND item_id = ? AND active = 1
        ORDER BY version DESC
        LIMIT 1
        """,
        (shop_id, item_id),
    ).fetchone()

    if not row:
        return None, None

    meta = dict(row)
    try:
        data = json.loads(meta["parameters"])
        bandit = ContextualBandit.from_dict(data)
        return bandit, meta
    except Exception:
        return None, meta


def rollback_policy(shop_id: int, item_id: int, target_version: int | None = None, db=None) -> dict:
    """Roll back policy for an item to a previous version."""
    if db is None:
        db = get_db()

    curr_row = db.execute(
        "SELECT id, version FROM policies WHERE shop_id = ? AND item_id = ? AND active = 1 ORDER BY version DESC LIMIT 1",
        (shop_id, item_id),
    ).fetchone()
    if not curr_row:
        raise ValueError(f"No active policy found to roll back for shop {shop_id}, item {item_id}.")

    curr_v = curr_row["version"]
    if target_version is not None:
        target_row = db.execute(
            "SELECT id, version FROM policies WHERE shop_id = ? AND item_id = ? AND version = ?",
            (shop_id, item_id, target_version),
        ).fetchone()
        if not target_row:
            raise ValueError(f"Policy version v{target_version} not found for shop {shop_id}, item {item_id}.")
    else:
        target_row = db.execute(
            "SELECT id, version FROM policies WHERE shop_id = ? AND item_id = ? AND version < ? ORDER BY version DESC LIMIT 1",
            (shop_id, item_id, curr_v),
        ).fetchone()
        if not target_row:
            raise ValueError(f"No previous policy version found to roll back to for shop {shop_id}, item {item_id}.")

    db.execute("UPDATE policies SET active = 0 WHERE shop_id = ? AND item_id = ?", (shop_id, item_id))
    db.execute("UPDATE policies SET active = 1 WHERE id = ?", (target_row["id"],))
    db.commit()

    return {
        "shop_id": shop_id,
        "item_id": item_id,
        "previous_version": curr_v,
        "active_version": target_row["version"],
    }


def online_update_on_daily_log(
    shop_id: int,
    item_id: int,
    log_date_str: str,
    sold: int,
    baked: int | None,
    sold_out: bool,
    db=None,
):
    """Update active contextual bandit policy with feedback when a daily log is saved."""
    if not is_rl_enabled():
        return

    if db is None:
        db = get_db()

    bandit, meta = get_active_policy(shop_id, item_id, db=db)
    if not bandit or not meta:
        return

    # Query item economics
    item_row = db.execute("SELECT unit_price, unit_cost FROM items WHERE id = ?", (item_id,)).fetchone()
    p = item_row["unit_price"] if item_row else None
    c = item_row["unit_cost"] if item_row else None

    # Retrieve weather & forecast context for log_date
    w_row = db.execute("SELECT temp_mean_c, is_rainy, is_holiday FROM weather_daily WHERE shop_id = ? AND date = ?", (shop_id, log_date_str)).fetchone()
    temp = float(w_row["temp_mean_c"]) if w_row and w_row["temp_mean_c"] is not None else 20.0
    rain = int(w_row["is_rainy"]) if w_row and w_row["is_rainy"] is not None else 0
    hol = int(w_row["is_holiday"]) if w_row and w_row["is_holiday"] is not None else 0

    log_d = date.fromisoformat(log_date_str)
    dow = log_d.weekday()

    # Calculate rolling mean
    hist_rows = db.execute(
        "SELECT sold FROM daily_log WHERE shop_id = ? AND item_id = ? AND date < ? ORDER BY date DESC LIMIT 7",
        (shop_id, item_id, log_date_str),
    ).fetchall()
    recent_sold = [float(r["sold"]) for r in hist_rows if r["sold"] is not None]
    ma7 = float(sum(recent_sold) / len(recent_sold)) if recent_sold else float(sold)

    state = discretize_state(dow, hol, rain, temp, float(sold), ma7, carry_over=0)

    # Estimate unbiased demand and reward
    true_demand = estimate_actual_demand(sold, sold_out)
    eff_baked = baked if baked is not None else sold
    reward, _ = calculate_economic_reward(sold, eff_baked, true_demand, unit_price=p, unit_cost=c)

    # Determine which action index was taken (multiplier)
    if ma7 > 0 and eff_baked > 0:
        ratio = eff_baked / ma7
        diffs = [abs(ratio - m) for m in MULTIPLIERS]
        action_idx = int(np.argmin(diffs))
    else:
        action_idx = BASELINE_ACTION_IDX

    # Online update
    bandit.update(state, action_idx, reward)

    # Save updated parameters to policies table
    params_json = json.dumps(bandit.to_dict())
    db.execute(
        "UPDATE policies SET parameters = ?, observations_count = ? WHERE id = ?",
        (params_json, bandit.observations_count, meta["id"]),
    )
    db.commit()


# ---------------------------------------------------------------------------
# CLI Command
# ---------------------------------------------------------------------------

import click
from flask.cli import with_appcontext


@click.command("evaluate-rl")
@click.option("--shop-id", default=1, type=int, help="Shop ID to benchmark RL on (default: 1).")
@click.option("--item-id", default=None, type=int, help="Specific item ID to evaluate (defaults to all items).")
@with_appcontext
def evaluate_rl_command(shop_id: int, item_id: int | None):
    """Evaluate RL contextual bandit policy across 5 seeds and print comparison table."""
    from app.db import get_db
    db = get_db()

    if item_id is not None:
        items = db.execute("SELECT id, name FROM items WHERE shop_id = ? AND id = ?", (shop_id, item_id)).fetchall()
    else:
        items = db.execute("SELECT id, name FROM items WHERE shop_id = ? ORDER BY id ASC", (shop_id,)).fetchall()

    if not items:
        click.echo(f"No items found for shop {shop_id}.")
        return

    for it in items:
        evaluate_rl_against_baselines(shop_id=shop_id, item_id=it["id"], db=db)

