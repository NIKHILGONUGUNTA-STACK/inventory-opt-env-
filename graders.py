# ─────────────────────────────────────────
# graders.py
# Deterministic graders for all 3 tasks
# Each returns a score 0.0–1.0
# ─────────────────────────────────────────
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────
# EPISODE TRACKER — collects stats per step
# ─────────────────────────────────────────
@dataclass
class EpisodeTracker:
    """
    Accumulates per-step data during an episode.
    Graders read from this at episode end.
    """
    total_demand:        int   = 0
    total_fulfilled:     int   = 0
    total_unmet:         int   = 0
    total_holding_cost:  float = 0.0
    total_revenue:       float = 0.0
    total_stockout_cost: float = 0.0
    capacity_violations: int   = 0
    budget_violations:   int   = 0
    steps:               int   = 0
    stock_history:       List[int]   = field(default_factory=list)
    reward_history:      List[float] = field(default_factory=list)

    def record(self, step_result: dict, reward_breakdown: dict):
        """Record one step's outcome."""
        actual_demand = step_result.get("actual_demand", 0)
        self.total_demand        += actual_demand
        self.total_fulfilled     += step_result["units_fulfilled"]
        self.total_unmet         += step_result["units_unmet"]
        self.total_holding_cost  += reward_breakdown["holding_cost_incurred"]
        self.total_revenue       += reward_breakdown["revenue"]
        self.total_stockout_cost += reward_breakdown["stockout_cost_incurred"]
        self.steps               += 1

        stock = step_result["stock"]
        self.stock_history.append(stock)
        self.reward_history.append(reward_breakdown["total_reward"])

        if reward_breakdown["capacity_violation_penalty"] > 0:
            self.capacity_violations += 1
        if reward_breakdown.get("budget_violation_penalty", 0) > 0:
            self.budget_violations += 1

    def reset(self):
        """Clear all accumulated stats."""
        self.total_demand        = 0
        self.total_fulfilled     = 0
        self.total_unmet         = 0
        self.total_holding_cost  = 0.0
        self.total_revenue       = 0.0
        self.total_stockout_cost = 0.0
        self.capacity_violations = 0
        self.budget_violations   = 0
        self.steps               = 0
        self.stock_history       = []
        self.reward_history      = []


# ─────────────────────────────────────────
# BASE GRADER
# ─────────────────────────────────────────
class BaseGrader:
    """
    All graders inherit from this.
    Subclasses implement score().
    """
    def __init__(self, task_id: str):
        self.task_id = task_id

    def score(self, tracker: EpisodeTracker) -> dict:
        raise NotImplementedError


# ─────────────────────────────────────────
# TASK 1 GRADER — easy
# Metric: service level only
# Score = units_fulfilled / units_demanded
# ─────────────────────────────────────────
class Task1Grader(BaseGrader):
    def __init__(self):
        super().__init__("task1_easy")

    def score(self, tracker: EpisodeTracker) -> dict:
        """
        Task 1 is purely about fulfilling demand.
        Perfect score = never run out of stock.

        service_level = fulfilled / total_demand
        final_score   = service_level (already 0–1)
        """
        if tracker.total_demand == 0:
            return self._result(1.0, 1.0, 1.0, 1.0)

        service_level = tracker.total_fulfilled / tracker.total_demand
        service_level = round(min(service_level, 1.0), 4)

        # cost efficiency: penalize excessive holding cost
        # benchmark: holding 1x daily demand in stock = cost_efficiency 1.0
        avg_stock = (
            sum(tracker.stock_history) / len(tracker.stock_history)
            if tracker.stock_history else 0
        )
        avg_demand_per_day = tracker.total_demand / max(tracker.steps, 1)
        # efficiency drops if avg stock > 3x daily demand (overstocking)
        overstock_ratio = avg_stock / max(avg_demand_per_day * 3, 1)
        cost_efficiency = round(max(0.0, 1.0 - max(0, overstock_ratio - 1.0) * 0.2), 4)

        # constraint adherence: no capacity violations in Task 1
        constraint = 1.0 if tracker.capacity_violations == 0 else max(
            0.0, 1.0 - tracker.capacity_violations / tracker.steps
        )
        constraint = round(constraint, 4)

        # final score: Task 1 weights service level heavily
        final = round(
            0.80 * service_level +
            0.10 * cost_efficiency +
            0.10 * constraint,
            4
        )

        return self._result(final, service_level, cost_efficiency, constraint)

    def _result(self, score, sl, ce, ca):
        return {
            "task_id":              self.task_id,
            "score":                max(0.0, min(1.0, score)),
            "service_level":        sl,
            "cost_efficiency":      ce,
            "constraint_adherence": ca,
            "details": {
                "weight_service_level":        0.80,
                "weight_cost_efficiency":      0.10,
                "weight_constraint_adherence": 0.10,
            }
        }


# ─────────────────────────────────────────
# TASK 2 GRADER — medium
# Metrics: service level + cost efficiency
# ─────────────────────────────────────────
class Task2Grader(BaseGrader):
    def __init__(self):
        super().__init__("task2_medium")

    def score(self, tracker: EpisodeTracker) -> dict:
        """
        Task 2 balances fulfillment with cost control.
        Agent must not overstock (expensive) or understock (stockouts).

        service_level   = fulfilled / demand
        cost_efficiency = 1 - normalized_total_cost
        constraint      = penalize capacity violations
        """
        if tracker.total_demand == 0:
            return self._result(1.0, 1.0, 1.0, 1.0)

        # service level
        service_level = round(
            min(tracker.total_fulfilled / tracker.total_demand, 1.0), 4
        )

        # cost efficiency
        # total cost = holding + stockout
        total_cost = tracker.total_holding_cost + tracker.total_stockout_cost
        # benchmark: perfect agent spends ~10% of revenue on costs
        benchmark_cost = 0.10 * tracker.total_revenue
        cost_ratio = total_cost / max(benchmark_cost, 1.0)
        cost_efficiency = round(max(0.0, 1.0 - (cost_ratio - 1.0) * 0.15), 4)

        # constraint adherence
        violation_rate = tracker.capacity_violations / max(tracker.steps, 1)
        constraint = round(max(0.0, 1.0 - violation_rate * 2.0), 4)

        # final: equal weight on service + cost, small constraint bonus
        final = round(
            0.50 * service_level +
            0.35 * cost_efficiency +
            0.15 * constraint,
            4
        )

        return self._result(final, service_level, cost_efficiency, constraint)

    def _result(self, score, sl, ce, ca):
        return {
            "task_id":              self.task_id,
            "score":                max(0.0, min(1.0, score)),
            "service_level":        sl,
            "cost_efficiency":      ce,
            "constraint_adherence": ca,
            "details": {
                "weight_service_level":        0.50,
                "weight_cost_efficiency":      0.35,
                "weight_constraint_adherence": 0.15,
            }
        }


# ─────────────────────────────────────────
# TASK 3 GRADER — hard
# Metrics: service level + cost + constraints
# Full rubric, strictest scoring
# ─────────────────────────────────────────
class Task3Grader(BaseGrader):
    def __init__(self):
        super().__init__("task3_hard")

    def score(self, tracker: EpisodeTracker) -> dict:
        """
        Task 3 full rubric:
        - Service level (40%): fulfill demand despite uncertainty
        - Cost efficiency (30%): control holding + stockout costs
        - Constraint adherence (30%): no capacity or budget violations

        This genuinely challenges frontier models because:
        - Demand is noisy + seasonal (spikes mid-episode)
        - Supplier often delivers less than ordered
        - Budget runs out if agent over-orders
        """
        if tracker.total_demand == 0:
            return self._result(1.0, 1.0, 1.0, 1.0)

        # service level
        service_level = round(
            min(tracker.total_fulfilled / tracker.total_demand, 1.0), 4
        )

        # cost efficiency — stricter than Task 2
        total_cost = tracker.total_holding_cost + tracker.total_stockout_cost
        benchmark_cost = 0.08 * tracker.total_revenue   # tighter benchmark
        cost_ratio = total_cost / max(benchmark_cost, 1.0)
        cost_efficiency = round(max(0.0, 1.0 - (cost_ratio - 1.0) * 0.20), 4)

        # constraint adherence — both capacity AND budget violations
        capacity_rate = tracker.capacity_violations / max(tracker.steps, 1)
        budget_rate   = tracker.budget_violations   / max(tracker.steps, 1)
        constraint = round(
            max(0.0, 1.0 - capacity_rate * 2.0 - budget_rate * 3.0), 4
        )

        # seasonal handling bonus
        # reward agents that didn't stockout during the spike window
        # approximated by checking if fulfillment stayed high mid-episode
        mid_start = tracker.steps // 3
        mid_end   = 2 * tracker.steps // 3
        mid_rewards = tracker.reward_history[mid_start:mid_end]
        if mid_rewards:
            mid_avg = sum(mid_rewards) / len(mid_rewards)
            overall_avg = sum(tracker.reward_history) / len(tracker.reward_history)
            seasonal_bonus = 0.05 if mid_avg >= overall_avg * 0.8 else 0.0
        else:
            seasonal_bonus = 0.0

        # final score
        final = round(
            0.40 * service_level +
            0.30 * cost_efficiency +
            0.30 * constraint +
            seasonal_bonus,
            4
        )
        final = max(0.0, min(1.0, final))

        return self._result(final, service_level, cost_efficiency, constraint)

    def _result(self, score, sl, ce, ca):
        return {
            "task_id":              self.task_id,
            "score":                max(0.0, min(1.0, score)),
            "service_level":        sl,
            "cost_efficiency":      ce,
            "constraint_adherence": ca,
            "details": {
                "weight_service_level":        0.40,
                "weight_cost_efficiency":      0.30,
                "weight_constraint_adherence": 0.30,
                "seasonal_bonus":              "up to +0.05",
            }
        }


# ─────────────────────────────────────────
# TASK 4 GRADER — extreme
# Strictest scoring — most agents score < 0.3
# ─────────────────────────────────────────
class Task4Grader(BaseGrader):
    def __init__(self):
        super().__init__("task4_extreme")

    def score(self, tracker: EpisodeTracker) -> dict:
        """
        Task 4 extreme rubric:
        - Service level (35%): fulfill demand during chaos
        - Cost efficiency (35%): survive on tight budget
        - Constraint adherence (30%): capacity + budget violations

        Designed so that:
        - Random agent scores ~0.05–0.10
        - Heuristic agent scores ~0.10–0.20
        - Smart LLM agent scores ~0.25–0.45
        - Near-optimal agent scores ~0.50–0.70
        """
        if tracker.total_demand == 0:
            return self._result(1.0, 1.0, 1.0, 1.0)

        # service level — strict
        service_level = round(
            min(tracker.total_fulfilled / tracker.total_demand, 1.0), 4
        )

        # cost efficiency — very strict benchmark (5% of revenue)
        total_cost = tracker.total_holding_cost + tracker.total_stockout_cost
        benchmark  = 0.05 * tracker.total_revenue
        cost_ratio = total_cost / max(benchmark, 1.0)
        cost_efficiency = round(max(0.0, 1.0 - (cost_ratio - 1.0) * 0.25), 4)

        # constraint adherence — both capacity and budget
        capacity_rate = tracker.capacity_violations / max(tracker.steps, 1)
        budget_rate   = tracker.budget_violations   / max(tracker.steps, 1)
        constraint    = round(
            max(0.0, 1.0 - capacity_rate * 3.0 - budget_rate * 4.0), 4
        )

        # chaos survival bonus
        # reward agents that maintained positive reward mid-episode
        if tracker.reward_history:
            positive_steps = sum(
                1 for r in tracker.reward_history if r > 0
            )
            survival_rate  = positive_steps / len(tracker.reward_history)
            chaos_bonus    = round(0.05 * survival_rate, 4)
        else:
            chaos_bonus = 0.0

        # final score
        final = round(
            0.35 * service_level +
            0.35 * cost_efficiency +
            0.30 * constraint +
            chaos_bonus,
            4
        )
        final = max(0.0, min(1.0, final))

        return self._result(final, service_level, cost_efficiency, constraint)

    def _result(self, score, sl, ce, ca):
        return {
            "task_id":              self.task_id,
            "score":                max(0.0, min(1.0, score)),
            "service_level":        sl,
            "cost_efficiency":      ce,
            "constraint_adherence": ca,
            "details": {
                "weight_service_level":        0.35,
                "weight_cost_efficiency":      0.35,
                "weight_constraint_adherence": 0.30,
                "chaos_bonus":                 "up to +0.05",
                "note": "Designed to challenge frontier LLMs",
            }
        }


# ─────────────────────────────────────────
# GRADER REGISTRY
# ─────────────────────────────────────────
GRADER_REGISTRY = {
    "task1_easy":    Task1Grader,
    "task2_medium":  Task2Grader,
    "task3_hard":    Task3Grader,
    "task4_extreme": Task4Grader,
}

def get_grader(task_id: str) -> BaseGrader:
    if task_id not in GRADER_REGISTRY:
        raise ValueError(f"Unknown task '{task_id}'. Available: {list(GRADER_REGISTRY.keys())}")
    return GRADER_REGISTRY[task_id]()
