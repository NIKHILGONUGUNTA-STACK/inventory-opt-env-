import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────
# SUPPLIER — models one supplier
# ─────────────────────────────────────────
@dataclass
class Supplier:
    name: str
    reliability: float        # probability of full fulfillment
    lead_time_min: int        # minimum days to deliver
    lead_time_max: int        # maximum days to deliver

    def get_lead_time(self) -> int:
        """Returns actual lead time for this order (random within range)."""
        return random.randint(self.lead_time_min, self.lead_time_max)

    def fulfill_order(self, order_qty: int) -> int:
        """
        Simulates supplier fulfillment.
        - Full order if reliability check passes
        - Partial (50–90%) if it fails
        """
        if random.random() <= self.reliability:
            return order_qty  # full delivery
        else:
            # partial delivery between 50% and 90%
            partial_rate = random.uniform(0.5, 0.9)
            return int(order_qty * partial_rate)


# ─────────────────────────────────────────
# DEMAND MODEL — generates customer demand
# ─────────────────────────────────────────
@dataclass
class DemandModel:
    base_demand: float          # average daily demand
    noise_level: float          # 0.0 = deterministic, 1.0 = high noise
    seasonal: bool = False      # whether to apply seasonal spikes

    def get_demand(self, day: int, season_factor: float = 1.0) -> int:
        """
        Returns actual customer demand for this day.
        - Task 1: noise_level=0.0 → always base_demand
        - Task 2: noise_level=0.2 → ±20% variance
        - Task 3: noise_level=0.4 + seasonal spikes
        """
        demand = self.base_demand * season_factor

        if self.noise_level > 0:
            noise = random.gauss(0, self.noise_level * self.base_demand)
            demand = max(0, demand + noise)

        return int(round(demand))

    def get_forecast(self, day: int, season_factor: float = 1.0) -> float:
        """
        Returns a demand forecast (what the agent sees).
        In Task 3 this is noisy — not equal to actual demand.
        """
        true_demand = self.base_demand * season_factor

        if self.noise_level > 0:
            # forecast error: agent sees a slightly wrong number
            forecast_noise = random.gauss(0, 0.15 * self.base_demand)
            return max(0, true_demand + forecast_noise)

        return true_demand

    def get_season_factor(self, day: int, episode_length: int) -> float:
        """
        Returns seasonal multiplier.
        Spike in the middle third of the episode (simulates a peak season).
        """
        if not self.seasonal:
            return 1.0

        # spike during middle third of episode
        third = episode_length // 3
        if third <= day < 2 * third:
            # smooth sine-based spike peaking at 2x demand
            progress = (day - third) / third
            spike = 1.0 + math.sin(progress * math.pi)
            return round(spike, 3)

        return 1.0


# ─────────────────────────────────────────
# IN-TRANSIT PIPELINE — tracks pending orders
# ─────────────────────────────────────────
@dataclass
class InTransitPipeline:
    """
    Tracks orders that are on the way.
    Each entry is (arrival_day, quantity).
    """
    orders: List[Tuple[int, int]] = field(default_factory=list)

    def add_order(self, arrival_day: int, quantity: int):
        """Place a new order that will arrive on arrival_day."""
        if quantity > 0:
            self.orders.append((arrival_day, quantity))

    def receive_deliveries(self, current_day: int) -> int:
        """
        Returns total units arriving today.
        Removes those orders from the pipeline.
        """
        arriving = [qty for day, qty in self.orders if day <= current_day]
        self.orders = [(day, qty) for day, qty in self.orders if day > current_day]
        return sum(arriving)

    def total_in_transit(self) -> int:
        """Total units currently on the way."""
        return sum(qty for _, qty in self.orders)

    def reset(self):
        """Clear all pending orders."""
        self.orders = []


# ─────────────────────────────────────────
# SIMULATOR — ties everything together
# ─────────────────────────────────────────
class InventorySimulator:
    def __init__(
        self,
        demand_model: DemandModel,
        supplier: Supplier,
        warehouse_capacity: int,
        holding_cost: float,
        stockout_penalty: float,
        revenue_per_unit: float,
        episode_length: int,
        initial_stock: int,
        budget: float = float("inf"),
        order_cost_per_unit: float = 1.0,
    ):
        self.demand_model = demand_model
        self.supplier = supplier
        self.warehouse_capacity = warehouse_capacity
        self.holding_cost = holding_cost
        self.stockout_penalty = stockout_penalty
        self.revenue_per_unit = revenue_per_unit
        self.episode_length = episode_length
        self.initial_stock = initial_stock
        self.budget = budget
        self.order_cost_per_unit = order_cost_per_unit

        # runtime state
        self.pipeline = InTransitPipeline()
        self.stock = initial_stock
        self.current_day = 0
        self.budget_remaining = budget
        self.season_factor = 1.0

    def reset(self):
        """Reset to start of episode."""
        self.pipeline.reset()
        self.stock = self.initial_stock
        self.current_day = 0
        self.budget_remaining = self.budget
        self.season_factor = 1.0

    def step(self, order_qty: int) -> dict:
        """
        Simulate one day:
        1. Receive deliveries arriving today
        2. Observe actual demand
        3. Fulfill demand from stock
        4. Place new order (supplier processes it)
        5. Compute reward
        6. Advance day
        """
        # 1. receive deliveries
        arrived = self.pipeline.receive_deliveries(self.current_day)
        self.stock = min(self.stock + arrived, self.warehouse_capacity)

        # 2. seasonal factor + actual demand
        self.season_factor = self.demand_model.get_season_factor(
            self.current_day, self.episode_length
        )
        actual_demand = self.demand_model.get_demand(
            self.current_day, self.season_factor
        )

        # 3. fulfill demand
        units_fulfilled = min(self.stock, actual_demand)
        units_unmet = actual_demand - units_fulfilled
        self.stock -= units_fulfilled

        # 4. place new order (clamp to budget)
        order_qty = self._apply_budget(order_qty)
        if order_qty > 0:
            actual_delivery = self.supplier.fulfill_order(order_qty)
            lead_time = self.supplier.get_lead_time()
            arrival_day = self.current_day + lead_time
            self.pipeline.add_order(arrival_day, actual_delivery)
            order_cost = order_qty * self.order_cost_per_unit
            self.budget_remaining = max(0, self.budget_remaining - order_cost)

        # 5. capacity violation check
        capacity_violation = max(0, self.stock - self.warehouse_capacity)
        self.stock = min(self.stock, self.warehouse_capacity)

        # 6. reward
        revenue = self.revenue_per_unit * units_fulfilled
        holding = self.holding_cost * self.stock
        stockout = self.stockout_penalty * units_unmet
        cap_penalty = 10.0 * capacity_violation
        step_penalty = 0.01
        total_reward = revenue - holding - stockout - cap_penalty - step_penalty

        # 7. forecast for next step
        next_forecast = self.demand_model.get_forecast(
            self.current_day + 1, self.season_factor
        )

        self.current_day += 1
        done = self.current_day >= self.episode_length

        return {
            "stock": self.stock,
            "demand_forecast": next_forecast,
            "lead_time": self.supplier.lead_time_min,
            "in_transit": self.pipeline.total_in_transit(),
            "supplier_reliability": self.supplier.reliability,
            "holding_cost": self.holding_cost,
            "stockout_penalty": self.stockout_penalty,
            "warehouse_capacity": self.warehouse_capacity,
            "days_remaining": self.episode_length - self.current_day,
            "budget_remaining": self.budget_remaining,
            "current_day": self.current_day,
            "season_factor": self.season_factor,
            "reward": total_reward,
            "done": done,
            "units_fulfilled": units_fulfilled,
            "units_unmet": units_unmet,
            "revenue": revenue,
            "holding_cost_incurred": holding,
            "stockout_cost_incurred": stockout,
            "capacity_violation_penalty": cap_penalty,
            "step_penalty": -step_penalty,
            "actual_demand": actual_demand,
        }

    def _apply_budget(self, order_qty: int) -> int:
        """Clamp order quantity to what the remaining budget allows."""
        if self.budget >= 999999:  # no budget limit
            return order_qty
        max_affordable = int(self.budget_remaining / self.order_cost_per_unit)
        return min(order_qty, max_affordable)
