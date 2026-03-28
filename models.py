from pydantic import BaseModel, Field
from typing import Optional, List


# ─────────────────────────────────────────
# OBSERVATION — what the agent sees
# ─────────────────────────────────────────
class Observation(BaseModel):
    stock: int = Field(..., description="Current inventory level in units")
    demand_forecast: float = Field(..., description="Predicted demand for next period (may be noisy in hard tasks)")
    lead_time: int = Field(..., description="Days until next ordered batch arrives")
    in_transit: int = Field(..., description="Units already ordered and on the way")
    supplier_reliability: float = Field(..., description="Probability supplier fulfills full order (0.0–1.0)")
    holding_cost: float = Field(..., description="Cost per unit per day for storing inventory")
    stockout_penalty: float = Field(..., description="Cost per unit of unmet customer demand")
    warehouse_capacity: int = Field(..., description="Maximum units that can be stored")
    days_remaining: int = Field(..., description="Steps left in this episode")
    budget_remaining: float = Field(..., description="Remaining budget (used in Task 3)")
    current_day: int = Field(..., description="Current day in the episode")
    season_factor: float = Field(1.0, description="Demand multiplier for seasonal spikes (Task 3)")


# ─────────────────────────────────────────
# ACTION — what the agent decides
# ─────────────────────────────────────────
class Action(BaseModel):
    order_qty: int = Field(..., ge=0, description="Number of units to order this step (0 = do nothing)")


# ─────────────────────────────────────────
# REWARD — breakdown of what happened
# ─────────────────────────────────────────
class RewardInfo(BaseModel):
    total_reward: float = Field(..., description="Net reward this step")
    revenue: float = Field(..., description="Revenue from fulfilled demand")
    holding_cost_incurred: float = Field(..., description="Cost of holding current stock")
    stockout_cost_incurred: float = Field(..., description="Penalty for unmet demand")
    capacity_violation_penalty: float = Field(..., description="Penalty for exceeding warehouse capacity")
    units_fulfilled: int = Field(..., description="Units actually sold this step")
    units_unmet: int = Field(..., description="Units of demand not fulfilled")
    step_penalty: float = Field(-0.01, description="Small fixed penalty per step")


# ─────────────────────────────────────────
# STEP RESULT — full response from step()
# ─────────────────────────────────────────
class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: RewardInfo


# ─────────────────────────────────────────
# TASK INFO — describes a task to the agent
# ─────────────────────────────────────────
class TaskInfo(BaseModel):
    task_id: str = Field(..., description="Unique task identifier e.g. 'task1_easy'")
    name: str = Field(..., description="Human readable task name")
    difficulty: str = Field(..., description="easy / medium / hard")
    description: str = Field(..., description="What the agent needs to do")
    episode_length: int = Field(..., description="Number of steps per episode")
    max_order_qty: int = Field(..., description="Maximum units agent can order per step")
    num_products: int = Field(..., description="Number of products in this task")


# ─────────────────────────────────────────
# EPISODE SCORE — returned by grader
# ─────────────────────────────────────────
class EpisodeScore(BaseModel):
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Final normalized score 0.0–1.0")
    service_level: float = Field(..., description="Fraction of demand fulfilled")
    cost_efficiency: float = Field(..., description="Normalized cost score")
    constraint_adherence: float = Field(..., description="How well agent respected constraints")
    details: Optional[dict] = Field(None, description="Extra breakdown info")