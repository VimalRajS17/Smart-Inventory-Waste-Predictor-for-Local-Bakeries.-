"""60-day historical replay simulation comparing ordering and baking policies."""

from datetime import date, datetime, timedelta
import math
import os
import click
from flask import current_app
from flask.cli import with_appcontext
import numpy as np

from app.db import get_db
from ml.predictor import calculate_safety_buffer, get_carry_over


POLICY_KEYS = [
    ("same_as_last_week", "1. Same as Last Week (Baseline)"),
    ("forecast_only", "2. Forecast Only"),
    ("forecast_buffer", "3. Forecast + Safety Buffer"),
    ("rl_policy", "4. Contextual Bandit (RL)"),
]


def run_historical_simulation(
    shop_id: int,
    days_count: int = 60,
    db=None,
    verbose: bool = True,
) -> dict:
    """Replays the last `days_count` days (default 60) for a given shop across policies:
    1. same_as_last_week (naive baseline lag-7)
    2. forecast_only (raw forecast rounded up)
    3. forecast_buffer (forecast + 80th percentile buffer - carry-over)
    4. rl_policy (contextual bandit bake multiplier, if enabled and available)

    Computes physical waste and unmet demand, converts to financial metrics using item
    economics (unit_price, unit_cost), records outcomes into evaluation_results,
    and returns a summary dictionary.
    """
    if db is None:
        db = get_db()

    # 1. Fetch shop info
    shop = db.execute("SELECT id, name FROM shops WHERE id = ?", (shop_id,)).fetchone()
    if not shop:
        raise ValueError(f"Shop with id {shop_id} does not exist.")
    shop_name = shop["name"]

    # 2. Determine simulation date window
    latest_row = db.execute(
        "SELECT MAX(date) AS max_d FROM daily_log WHERE shop_id = ? AND sold IS NOT NULL",
        (shop_id,),
    ).fetchone()

    if latest_row and latest_row["max_d"]:
        end_date = date.fromisoformat(latest_row["max_d"])
    else:
        end_date = date.today()

    start_date = end_date - timedelta(days=days_count - 1)
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    # 3. Fetch items catalogue
    items = db.execute(
        """
        SELECT id, name, shelf_life_days, unit_price, unit_cost
        FROM items
        WHERE shop_id = ?
        ORDER BY id ASC
        """,
        (shop_id,),
    ).fetchall()

    if not items:
        return {
            "status": "empty",
            "shop_id": shop_id,
            "shop_name": shop_name,
            "period": {"start": start_str, "end": end_str, "days": days_count},
            "policies": [],
            "items": [],
        }

    # Check RL status
    rl_enabled = False
    try:
        rl_enabled = bool(current_app.config.get("RL_ENABLED", False))
    except RuntimeError:
        rl_enabled = os.getenv("RL_ENABLED", "0").lower() in ("1", "true", "yes")

    # Load active bandit policies if RL enabled
    bandit_policies = {}
    if rl_enabled:
        from ml.rl import get_active_policy
        for it in items:
            pol = get_active_policy(shop_id, it["id"], db=db)
            if pol:
                bandit_policies[it["id"]] = pol

    # Pre-fetch all daily logs and forecasts for fast lookup
    daily_rows = db.execute(
        """
        SELECT item_id, date, sold, baked, wasted, leftover, sold_out
        FROM daily_log
        WHERE shop_id = ? AND date <= ?
        ORDER BY date ASC
        """,
        (shop_id, end_str),
    ).fetchall()

    logs_map = {}  # (item_id, date_str) -> row
    item_dates_sales = {}  # item_id -> {date_str: sold}
    for r in daily_rows:
        key = (r["item_id"], r["date"])
        logs_map[key] = r
        if r["item_id"] not in item_dates_sales:
            item_dates_sales[r["item_id"]] = {}
        if r["sold"] is not None:
            item_dates_sales[r["item_id"]][r["date"]] = int(r["sold"])

    forecast_rows = db.execute(
        """
        SELECT item_id, date, predicted_quantity
        FROM forecasts
        WHERE shop_id = ? AND date >= ? AND date <= ?
        """,
        (shop_id, start_str, end_str),
    ).fetchall()
    forecasts_map = {(r["item_id"], r["date"]): float(r["predicted_quantity"]) for r in forecast_rows}

    # Track metrics per policy per item: policy_name -> item_id -> metrics dict
    active_policies = ["same_as_last_week", "forecast_only", "forecast_buffer"]
    if rl_enabled:
        active_policies.append("rl_policy")

    per_item_results = {p: {it["id"]: {
        "baked": 0, "sold": 0, "waste_units": 0, "waste_cost": 0.0,
        "lost_sales_units": 0, "lost_sales_cost": 0.0, "net_profit": 0.0
    } for it in items} for p in active_policies}

    # Simulation loop across items and dates
    for it in items:
        i_id = it["id"]
        shelf_life = int(it["shelf_life_days"]) if it["shelf_life_days"] else 1
        price = float(it["unit_price"]) if it["unit_price"] is not None else 1.0
        cost = float(it["unit_cost"]) if it["unit_cost"] is not None else 0.5
        sales_history = item_dates_sales.get(i_id, {})

        curr_date = start_date
        while curr_date <= end_date:
            d_str = curr_date.isoformat()
            log_entry = logs_map.get((i_id, d_str))

            # If no log entry for this day, skip day
            if not log_entry or log_entry["sold"] is None:
                curr_date += timedelta(days=1)
                continue

            actual_sold = int(log_entry["sold"])
            sold_out = bool(log_entry["sold_out"])

            # Censored demand adjustment: when sold out, demand exceeded stock
            if sold_out:
                actual_demand = actual_sold + max(1, math.ceil(0.15 * actual_sold))
            else:
                actual_demand = actual_sold

            # Forecast estimate
            if (i_id, d_str) in forecasts_map:
                f_qty = forecasts_map[(i_id, d_str)]
            else:
                # 7-day rolling mean fallback from prior history
                prior_dates = [
                    sales_history[k]
                    for k in sorted(sales_history.keys())
                    if k < d_str
                ]
                f_qty = float(np.mean(prior_dates[-7:])) if len(prior_dates) >= 7 else (float(actual_sold) if actual_sold > 0 else 10.0)

            # Safety buffer & carry-over (Phase 4B)
            # Evaluate backtest errors prior to curr_date
            buffer_qty = calculate_safety_buffer(shop_id, i_id, curr_date, f_qty, percentile=80.0, db=db)
            carry_over, _ = get_carry_over(shop_id, i_id, shelf_life, curr_date, db=db)

            # Determine bake quantities per policy
            # 1. same_as_last_week (naive lag-7)
            lag7_date_str = (curr_date - timedelta(days=7)).isoformat()
            if lag7_date_str in sales_history:
                bake_naive = sales_history[lag7_date_str]
            else:
                # Fallback to rolling mean
                prior_sales = [v for k, v in sales_history.items() if k < d_str]
                bake_naive = int(round(np.mean(prior_sales[-7:]))) if prior_sales else actual_sold
            bake_naive = max(0, int(bake_naive))

            # 2. forecast_only
            bake_forecast_only = max(0, int(math.ceil(f_qty)))

            # 3. forecast_buffer
            bake_buffer = max(0, int(math.ceil(f_qty + buffer_qty - carry_over)))

            # 4. rl_policy (contextual bandit)
            bake_rl = bake_buffer  # fallback default
            if rl_enabled and i_id in bandit_policies:
                bandit = bandit_policies[i_id]
                try:
                    from ml.rl import discretize_state
                    # Approximate state features for date
                    day_of_week = curr_date.weekday()
                    is_holiday = 0
                    is_rainy = 0
                    temp_c = 20.0
                    w_row = db.execute(
                        "SELECT temp_mean_c, is_rainy, is_holiday FROM weather_daily WHERE shop_id = ? AND date = ?",
                        (shop_id, d_str),
                    ).fetchone()
                    if w_row:
                        temp_c = float(w_row["temp_mean_c"]) if w_row["temp_mean_c"] is not None else 20.0
                        is_rainy = int(w_row["is_rainy"]) if w_row["is_rainy"] is not None else 0
                        is_holiday = int(w_row["is_holiday"]) if w_row["is_holiday"] is not None else 0

                    prior_7 = [v for k, v in sales_history.items() if k < d_str][-7:]
                    mean_7 = float(np.mean(prior_7)) if prior_7 else f_qty
                    state = discretize_state(day_of_week, is_holiday, is_rainy, temp_c, f_qty, mean_7, carry_over)
                    mult = bandit.select_action(state, explore=False)
                    bake_rl = max(0, int(math.ceil(mult * (f_qty + buffer_qty) - carry_over)))
                except Exception:
                    bake_rl = bake_buffer

            policy_bakes = {
                "same_as_last_week": bake_naive,
                "forecast_only": bake_forecast_only,
                "forecast_buffer": bake_buffer,
            }
            if rl_enabled:
                policy_bakes["rl_policy"] = bake_rl

            # Compute financial outcomes for each policy on curr_date
            for p_name, b_qty in policy_bakes.items():
                s_qty = min(b_qty, actual_demand)
                w_units = max(0, b_qty - actual_demand)
                l_units = max(0, actual_demand - b_qty)

                rev = s_qty * price
                prod_cost = b_qty * cost
                w_cost = w_units * cost
                l_cost = l_units * price
                profit = rev - prod_cost

                st = per_item_results[p_name][i_id]
                st["baked"] += b_qty
                st["sold"] += s_qty
                st["waste_units"] += w_units
                st["waste_cost"] += w_cost
                st["lost_sales_units"] += l_units
                st["lost_sales_cost"] += l_cost
                st["net_profit"] += profit

            curr_date += timedelta(days=1)

    # 4. Aggregate across all items for each policy
    shop_totals = {}
    for p_name in active_policies:
        total_baked = sum(per_item_results[p_name][it["id"]]["baked"] for it in items)
        total_sold = sum(per_item_results[p_name][it["id"]]["sold"] for it in items)
        total_waste_u = sum(per_item_results[p_name][it["id"]]["waste_units"] for it in items)
        total_waste_c = sum(per_item_results[p_name][it["id"]]["waste_cost"] for it in items)
        total_lost_u = sum(per_item_results[p_name][it["id"]]["lost_sales_units"] for it in items)
        total_lost_c = sum(per_item_results[p_name][it["id"]]["lost_sales_cost"] for it in items)
        total_net_profit = sum(per_item_results[p_name][it["id"]]["net_profit"] for it in items)

        shop_totals[p_name] = {
            "total_baked": total_baked,
            "total_sold": total_sold,
            "total_waste_units": total_waste_u,
            "total_waste_cost": round(total_waste_c, 2),
            "total_lost_sales_units": total_lost_u,
            "total_lost_sales_cost": round(total_lost_c, 2),
            "net_profit": round(total_net_profit, 2),
            "savings_vs_baseline": 0.0,
        }

    # Baseline profit (same as last week)
    baseline_profit = shop_totals["same_as_last_week"]["net_profit"]
    for p_name in active_policies:
        shop_totals[p_name]["savings_vs_baseline"] = round(shop_totals[p_name]["net_profit"] - baseline_profit, 2)

    # Compute savings vs baseline per item as well
    for it in items:
        i_id = it["id"]
        base_item_profit = per_item_results["same_as_last_week"][i_id]["net_profit"]
        for p_name in active_policies:
            st = per_item_results[p_name][i_id]
            st["savings_vs_baseline"] = round(st["net_profit"] - base_item_profit, 2)
            st["waste_cost"] = round(st["waste_cost"], 2)
            st["lost_sales_cost"] = round(st["lost_sales_cost"], 2)
            st["net_profit"] = round(st["net_profit"], 2)

    # 5. Persist simulation outcomes into evaluation_results table
    db.execute(
        "DELETE FROM evaluation_results WHERE shop_id = ? AND period_start = ? AND period_end = ?",
        (shop_id, start_str, end_str),
    )

    # Insert shop aggregate records (item_id = NULL)
    for p_name in active_policies:
        agg = shop_totals[p_name]
        db.execute(
            """
            INSERT INTO evaluation_results (
                shop_id, item_id, period_start, period_end, days_count, policy_name,
                total_baked, total_sold, total_waste_units, total_waste_cost,
                total_lost_sales_units, total_lost_sales_cost, net_profit, savings_vs_baseline
            ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                shop_id, start_str, end_str, days_count, p_name,
                agg["total_baked"], agg["total_sold"], agg["total_waste_units"], agg["total_waste_cost"],
                agg["total_lost_sales_units"], agg["total_lost_sales_cost"], agg["net_profit"], agg["savings_vs_baseline"],
            ),
        )

    # Insert per-item records
    for it in items:
        i_id = it["id"]
        for p_name in active_policies:
            st = per_item_results[p_name][i_id]
            db.execute(
                """
                INSERT INTO evaluation_results (
                    shop_id, item_id, period_start, period_end, days_count, policy_name,
                    total_baked, total_sold, total_waste_units, total_waste_cost,
                    total_lost_sales_units, total_lost_sales_cost, net_profit, savings_vs_baseline
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    shop_id, i_id, start_str, end_str, days_count, p_name,
                    st["baked"], st["sold"], st["waste_units"], st["waste_cost"],
                    st["lost_sales_units"], st["lost_sales_cost"], st["net_profit"], st["savings_vs_baseline"],
                ),
            )
    db.commit()

    # 6. Formatted terminal output
    if verbose:
        policy_label_map = dict(POLICY_KEYS)
        print("\n" + "=" * 105)
        print(f"PHASE 5: 60-DAY HISTORICAL REPLAY SIMULATION FOR '{shop_name.upper()}' ({start_str} to {end_str})")
        print("SIMULATION LABEL: Simulation on past data")
        print("=" * 105)
        print(f"{'Policy':<28} | {'Baked':<7} | {'Waste Units (GBP)':<18} | {'Lost Sales (GBP)':<18} | {'Net Profit':<12} | {'Savings vs Base':<15}")
        print("-" * 105)
        for p_name in active_policies:
            label = policy_label_map.get(p_name, p_name)
            st = shop_totals[p_name]
            w_str = f"{st['total_waste_units']} (GBP {st['total_waste_cost']:.2f})"
            l_str = f"{st['total_lost_sales_units']} (GBP {st['total_lost_sales_cost']:.2f})"
            p_str = f"GBP {st['net_profit']:.2f}"
            sav_str = f"+GBP {st['savings_vs_baseline']:.2f}" if st['savings_vs_baseline'] > 0 else f"GBP {st['savings_vs_baseline']:.2f}"
            winner = " * Winner" if st['savings_vs_baseline'] == max(shop_totals[p]['savings_vs_baseline'] for p in active_policies) and st['savings_vs_baseline'] > 0 else ""
            print(f"{label:<28} | {st['total_baked']:<7} | {w_str:<18} | {l_str:<18} | {p_str:<12} | {sav_str:<15}{winner}")
        print("=" * 105 + "\n")

    # Format return dictionary for API consumers
    policy_list = []
    policy_label_map = dict(POLICY_KEYS)
    for p_name in active_policies:
        st = shop_totals[p_name]
        policy_list.append({
            "policy_name": p_name,
            "display_name": policy_label_map.get(p_name, p_name),
            "total_baked": st["total_baked"],
            "total_sold": st["total_sold"],
            "waste_units": st["total_waste_units"],
            "waste_cost": st["total_waste_cost"],
            "lost_sales_units": st["total_lost_sales_units"],
            "lost_sales_cost": st["total_lost_sales_cost"],
            "net_profit": st["net_profit"],
            "savings_vs_baseline": st["savings_vs_baseline"],
        })

    items_detail = []
    for it in items:
        i_id = it["id"]
        item_policies = []
        for p_name in active_policies:
            st = per_item_results[p_name][i_id]
            item_policies.append({
                "policy_name": p_name,
                "display_name": policy_label_map.get(p_name, p_name),
                "baked": st["baked"],
                "sold": st["sold"],
                "waste_units": st["waste_units"],
                "waste_cost": st["waste_cost"],
                "lost_sales_units": st["lost_sales_units"],
                "lost_sales_cost": st["lost_sales_cost"],
                "net_profit": st["net_profit"],
                "savings_vs_baseline": st["savings_vs_baseline"],
            })
        items_detail.append({
            "item_id": i_id,
            "item_name": it["name"],
            "policies": item_policies,
        })

    return {
        "status": "success",
        "simulation_label": "Simulation on past data",
        "shop_id": shop_id,
        "shop_name": shop_name,
        "period": {
            "start": start_str,
            "end": end_str,
            "days": days_count,
        },
        "policies": policy_list,
        "items": items_detail,
    }


@click.command("evaluate")
@click.option("--shop-id", default=None, type=int, help="Shop ID to evaluate (defaults to all shops or demo shop 1).")
@click.option("--days", default=60, type=int, help="Number of historical days to replay (default: 60).")
@with_appcontext
def evaluate_command(shop_id: int | None, days: int):
    """Replays the last 60 days with each policy and stores results in evaluation_results."""
    db = get_db()
    if shop_id is not None:
        shop_ids = [shop_id]
    else:
        rows = db.execute("SELECT id FROM shops ORDER BY id ASC").fetchall()
        shop_ids = [r["id"] for r in rows] if rows else [1]

    for s_id in shop_ids:
        try:
            run_historical_simulation(shop_id=s_id, days_count=days, db=db, verbose=True)
        except Exception as exc:
            click.echo(f"Error evaluating shop {s_id}: {exc}", err=True)
