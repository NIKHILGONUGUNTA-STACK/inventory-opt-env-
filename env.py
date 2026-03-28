# ─────────────────────────────────────────
# env.py
# Core InventoryEnv class
# Implements full OpenEnv interface:
#   reset() → Observation
#   step()  → StepResult
#   state() → dict
# ─────────────────────────────────────────
from models import Observation, Action, RewardInfo, StepResult, TaskInfo, EpisodeScore
from tasks import get_task, list_tasks, TaskConfig
from graders import get_grader, EpisodeTracker
from reward import make_shaper_for_task


class InventoryEnv:
    """
    Smart Inventory Optimization Environment.

    Simulates a warehouse where an AI agent decides
    how much inventory to order each day to maximize
    service level while minimizing costs.

    Supports 3 tasks of increasing difficulty:
      task1_easy   — deterministic demand, reliable supplier
      task2_medium — stochastic demand, variable lead times
      task3_hard   — seasonal spikes, unreliable supplier, budget
    """

    def __init__(self, task_id: str = "task1_easy"):
        self.task_id = task_id
        self._load_task(task_id)

    # ─────────────────────────────────────
    # INTERNAL SETUP
    # ─────────────────────────────────────
    def _load_task(self, task_id: str):
        """Load task config, grader, reward shaper."""
        self.task: TaskConfig      = get_task(task_id)
        self.grader: object        = get_grader(task_id)
        self.shaper: object        = make_shaper_for_task(task_id)
        self.tracker: EpisodeTracker = EpisodeTracker()
        self.sim                   = self.task.simulator
        self._last_obs: dict       = {}
        self._done: bool           = False
        self._step_count: int      = 0
        self._total_reward: float  = 0.0
        self._episode_score: float = 0.0

    # ─────────────────────────────────────
    # RESET
    # ─────────────────────────────────────
    def reset(self, task_id: str = None) -> Observation:
        """
        Reset environment to start of a new episode.
        Optionally switch to a different task.
        Returns the initial Observation.
        """
        if task_id and task_id != self.task_id:
            self.task_id = task_id
            self._load_task(task_id)

        # reset simulator + tracker
        self.sim.reset()
        self.tracker.reset()
        self._done        = False
        self._step_count  = 0
        self._total_reward = 0.0
        self._episode_score = 0.0

        # get initial observation by doing a zero-order step
        initial_forecast = self.sim.demand_model.get_forecast(0, 1.0)

        obs_dict = {
            "stock":                self.sim.stock,
            "demand_forecast":      initial_forecast,
            "lead_time":            self.sim.supplier.lead_time_min,
            "in_transit":           0,
            "supplier_reliability": self.sim.supplier.reliability,
            "holding_cost":         self.sim.holding_cost,
            "stockout_penalty":     self.sim.stockout_penalty,
            "warehouse_capacity":   self.sim.warehouse_capacity,
            "days_remaining":       self.sim.episode_length,
            "budget_remaining":     self.sim.budget,
            "current_day":          0,
            "season_factor":        1.0,
        }

        self._last_obs = obs_dict
        return Observation(**obs_dict)

    # ─────────────────────────────────────
    # STEP
    # ─────────────────────────────────────
    def step(self, action: Action) -> StepResult:
        """
        Execute one step in the environment.

        1. Validate action
        2. Run simulator
        3. Compute reward breakdown
        4. Record in tracker
        5. Check if episode is done
        6. Return StepResult
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() before stepping again."
            )

        # clamp order to max allowed
        order_qty = max(0, min(action.order_qty, self.task.max_order_qty))

        # run one simulator step
        step_result = self.sim.step(order_qty)

        # compute reward breakdown
        reward_breakdown = self.shaper.compute(step_result)

        # shaped bonus (guides agent toward good inventory state)
        bonus = self.shaper.compute_shaped_bonus(
            stock           = step_result["stock"],
            demand_forecast = step_result["demand_forecast"],
            lead_time       = step_result["lead_time"],
            in_transit      = step_result["in_transit"],
            days_remaining  = step_result["days_remaining"],
        )
        reward_breakdown["total_reward"] = round(
            reward_breakdown["total_reward"] + bonus, 4
        )

        # record in tracker
        self.tracker.record(step_result, reward_breakdown)

        # accumulate totals
        self._step_count  += 1
        self._total_reward += reward_breakdown["total_reward"]
        self._done         = step_result["done"]

        # if episode ended, compute final score
        if self._done:
            result = self.grader.score(self.tracker)
            self._episode_score = result["score"]

        # build observation
        obs = Observation(
            stock                = step_result["stock"],
            demand_forecast      = step_result["demand_forecast"],
            lead_time            = step_result["lead_time"],
            in_transit           = step_result["in_transit"],
            supplier_reliability = step_result["supplier_reliability"],
            holding_cost         = step_result["holding_cost"],
            stockout_penalty     = step_result["stockout_penalty"],
            warehouse_capacity   = step_result["warehouse_capacity"],
            days_remaining       = step_result["days_remaining"],
            budget_remaining     = step_result["budget_remaining"],
            current_day          = step_result["current_day"],
            season_factor        = step_result["season_factor"],
        )
        self._last_obs = obs.model_dump()

        # build reward info
        reward_info = RewardInfo(
            total_reward              = reward_breakdown["total_reward"],
            revenue                   = reward_breakdown["revenue"],
            holding_cost_incurred     = reward_breakdown["holding_cost_incurred"],
            stockout_cost_incurred    = reward_breakdown["stockout_cost_incurred"],
            capacity_violation_penalty= reward_breakdown["capacity_violation_penalty"],
            units_fulfilled           = reward_breakdown["units_fulfilled"],
            units_unmet               = reward_breakdown["units_unmet"],
            step_penalty              = reward_breakdown["step_penalty"],
        )

        return StepResult(
            observation = obs,
            reward      = reward_breakdown["total_reward"],
            done        = self._done,
            info        = reward_info,
        )

    # ─────────────────────────────────────
    # STATE
    # ─────────────────────────────────────
    def state(self) -> dict:
        """
        Returns full current environment state.
        Used by OpenEnv spec for inspection.
        """
        return {
            "task_id":        self.task_id,
            "current_day":    self._step_count,
            "done":           self._done,
            "total_reward":   round(self._total_reward, 4),
            "episode_score":  self._episode_score,
            "observation":    self._last_obs,
            "tracker": {
                "total_demand":        self.tracker.total_demand,
                "total_fulfilled":     self.tracker.total_fulfilled,
                "total_unmet":         self.tracker.total_unmet,
                "steps":               self.tracker.steps,
                "capacity_violations": self.tracker.capacity_violations,
                "budget_violations":   self.tracker.budget_violations,
            }
        }

    # ─────────────────────────────────────
    # TASK INFO
    # ─────────────────────────────────────
    def get_task_info(self) -> TaskInfo:
        """Returns metadata about the current task."""
        return TaskInfo(
            task_id       = self.task.task_id,
            name          = self.task.name,
            difficulty    = self.task.difficulty,
            description   = self.task.description,
            episode_length= self.task.episode_length,
            max_order_qty = self.task.max_order_qty,
            num_products  = self.task.num_products,
        )

    def list_tasks(self) -> list:
        """Returns info for all available tasks."""
        return [
            TaskInfo(
                task_id        = t.task_id,
                name           = t.name,
                difficulty     = t.difficulty,
                description    = t.description,
                episode_length = t.episode_length,
                max_order_qty  = t.max_order_qty,
                num_products   = t.num_products,
            )
            for t in list_tasks()
        ]

    # ─────────────────────────────────────
    # SCORE (end of episode)
    # ─────────────────────────────────────
    def score(self) -> EpisodeScore:
        """
        Returns final graded score for the completed episode.
        Call only after done=True.
        """
        if not self._done:
            raise RuntimeError("Episode not finished yet. Keep stepping until done=True.")

        result = self.grader.score(self.tracker)
        return EpisodeScore(
            task_id               = result["task_id"],
            score                 = result["score"],
            service_level         = result["service_level"],
            cost_efficiency       = result["cost_efficiency"],
            constraint_adherence  = result["constraint_adherence"],
            details               = result.get("details"),
        )
