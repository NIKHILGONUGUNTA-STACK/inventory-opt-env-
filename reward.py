# ─────────────────────────────────────────
# reward.py
# Reward shaping logic for all 3 tasks
# ─────────────────────────────────────────


class RewardShaper:
    """
    Computes shaped reward from a simulator step result.
    Provides continuous partial feedback every step —
    not just binary win/lose at episode end.
    """

    def __init__(
        self,
        revenue_per_unit: float,
        holding_cost: float,
        stockout_penalty: float,
        warehouse_capacity: int,
        capacity_violation_weight: float = 10.0,
        step_penalty: float = 0.01,
        budget_violation_weight: float = 5.0,
    ):
        self.revenue_per_unit = revenue_per_unit
        self.holding_cost = holding_cost
        self.stockout_penalty = stockout_penalty
        self.warehouse_capacity = warehouse_capacity
        self.capacity_violation_weight = capacity_violation_weight
        self.step_penalty = step_penalty
        self.budget_violation_weight = budget_violation_weight

    def compute(self, step_result: dict) -> dict:
        """
        Takes raw simulator step result dict and returns
        a detailed reward breakdown.

        Reward formula:
          total = revenue
                - holding_cost_incurred
                - stockout_cost_incurred
                - capacity_violation_penalty
                - budget_violation_penalty
                - step_penalty
        """

        units_fulfilled = step_result["units_fulfilled"]
        units_unmet     = step_result["units_unmet"]
        stock           = step_result["stock"]
        budget_remaining = step_result["budget_remaining"]

        # ── positive signal ──────────────────────────
        revenue = self.revenue_per_unit * units_fulfilled

        # ── holding cost ─────────────────────────────
        holding = self.holding_cost * stock

        # ── stockout penalty ─────────────────────────
        stockout = self.stockout_penalty * units_unmet

        # ── capacity violation ───────────────────────
        capacity_excess = max(0, stock - self.warehouse_capacity)
        cap_penalty = self.capacity_violation_weight * capacity_excess

        # ── budget violation (Task 3) ─────────────────
        # penalize if agent tries to spend more than budget
        budget_penalty = 0.0
        if budget_remaining <= 0:
            budget_penalty = self.budget_violation_weight

        # ── fixed step penalty ────────────────────────
        # discourages doing nothing / infinite loops
        step_pen = self.step_penalty

        # ── total ────────────────────────────────────
        total = revenue - holding - stockout - cap_penalty - budget_penalty - step_pen

        return {
            "total_reward":               round(total, 4),
            "revenue":                    round(revenue, 4),
            "holding_cost_incurred":      round(holding, 4),
            "stockout_cost_incurred":     round(stockout, 4),
            "capacity_violation_penalty": round(cap_penalty, 4),
            "budget_violation_penalty":   round(budget_penalty, 4),
            "step_penalty":               round(-step_pen, 4),
            "units_fulfilled":            units_fulfilled,
            "units_unmet":                units_unmet,
        }

    def compute_shaped_bonus(
        self,
        stock: int,
        demand_forecast: float,
        lead_time: int,
        in_transit: int,
        days_remaining: int,
    ) -> float:
        """
        Optional shaping bonus — rewards the agent for
        being in a 'good state' even before demand hits.

        Good state = stock + in_transit covers forecasted
        demand for at least (lead_time + 1) days ahead.

        This bonus is small (+0.1 to +0.5) so it guides
        without dominating the main reward signal.
        """
        coverage_needed = demand_forecast * (lead_time + 1)
        coverage_available = stock + in_transit

        if coverage_available >= coverage_needed:
            # well prepared — small bonus scaled by days remaining
            surplus_ratio = min(coverage_available / max(coverage_needed, 1), 2.0)
            bonus = 0.1 * surplus_ratio
        else:
            # underprepared — small nudge penalty
            gap_ratio = (coverage_needed - coverage_available) / max(coverage_needed, 1)
            bonus = -0.2 * gap_ratio

        return round(bonus, 4)


# ─────────────────────────────────────────
# FACTORY — one shaper per task config
# ─────────────────────────────────────────
def make_shaper_for_task(task_id: str) -> RewardShaper:
    """
    Returns a RewardShaper tuned to the task's
    cost parameters. Matches the simulator settings
    in tasks.py exactly.
    """
    configs = {
        "task1_easy": dict(
            revenue_per_unit=5.0,
            holding_cost=0.5,
            stockout_penalty=2.0,
            warehouse_capacity=500,
            capacity_violation_weight=10.0,
            step_penalty=0.01,
            budget_violation_weight=0.0,   # no budget in Task 1
        ),
        "task2_medium": dict(
            revenue_per_unit=6.0,
            holding_cost=0.8,
            stockout_penalty=3.0,
            warehouse_capacity=400,
            capacity_violation_weight=10.0,
            step_penalty=0.01,
            budget_violation_weight=0.0,   # no budget in Task 2
        ),
        "task3_hard": dict(
            revenue_per_unit=8.0,
            holding_cost=1.2,
            stockout_penalty=5.0,
            warehouse_capacity=350,
            capacity_violation_weight=10.0,
            step_penalty=0.01,
            budget_violation_weight=5.0,   # strict budget in Task 3
        ),
        "task4_extreme": dict(
            revenue_per_unit=10.0,
            holding_cost=2.0,              # very expensive to hold
            stockout_penalty=8.0,          # very expensive to stockout
            warehouse_capacity=300,        # tiny warehouse
            capacity_violation_weight=20.0, # harsh capacity penalties
            step_penalty=0.01,
            budget_violation_weight=10.0,  # very harsh budget penalties
        ),
    }
    if task_id not in configs:
        raise ValueError(f"Unknown task_id '{task_id}'")
    return RewardShaper(**configs[task_id])
