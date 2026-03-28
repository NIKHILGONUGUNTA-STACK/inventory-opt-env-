from dataclasses import dataclass
from simulator import InventorySimulator, DemandModel, Supplier


# ─────────────────────────────────────────
# TASK CONFIG — defines one task's settings
# ─────────────────────────────────────────
@dataclass
class TaskConfig:
    task_id: str
    name: str
    difficulty: str
    description: str
    episode_length: int
    max_order_qty: int
    num_products: int
    simulator: InventorySimulator


# ─────────────────────────────────────────
# TASK 1 — EASY (deterministic)
# ─────────────────────────────────────────
def make_task1() -> TaskConfig:
    demand = DemandModel(
        base_demand=50,
        noise_level=0.0,       # perfectly predictable
        seasonal=False,
    )
    supplier = Supplier(
        name="ReliableSupplier",
        reliability=1.0,       # always delivers full order
        lead_time_min=2,
        lead_time_max=2,       # fixed 2-day lead time
    )
    sim = InventorySimulator(
        demand_model=demand,
        supplier=supplier,
        warehouse_capacity=500,
        holding_cost=0.5,
        stockout_penalty=2.0,
        revenue_per_unit=5.0,
        episode_length=30,
        initial_stock=100,
        budget=999999.0,   # no budget limit
        order_cost_per_unit=1.0,
    )
    return TaskConfig(
        task_id="task1_easy",
        name="Deterministic Inventory Control",
        difficulty="easy",
        description=(
            "Single product, single reliable supplier. "
            "Demand is fixed at 50 units/day, lead time is always 2 days. "
            "Learn the basic reorder point: keep stock above safety level "
            "without overloading the warehouse."
        ),
        episode_length=30,
        max_order_qty=200,
        num_products=1,
        simulator=sim,
    )


# ─────────────────────────────────────────
# TASK 2 — MEDIUM (stochastic)
# ─────────────────────────────────────────
def make_task2() -> TaskConfig:
    demand = DemandModel(
        base_demand=70,
        noise_level=0.2,       # ±20% demand variance
        seasonal=False,
    )
    supplier = Supplier(
        name="VariableSupplier",
        reliability=0.85,      # 15% chance of partial delivery
        lead_time_min=1,
        lead_time_max=5,       # lead time varies 1–5 days
    )
    sim = InventorySimulator(
        demand_model=demand,
        supplier=supplier,
        warehouse_capacity=400,    # tighter capacity constraint
        holding_cost=0.8,
        stockout_penalty=3.0,
        revenue_per_unit=6.0,
        episode_length=45,
        initial_stock=120,
        budget=999999.0,
        order_cost_per_unit=1.5,
    )
    return TaskConfig(
        task_id="task2_medium",
        name="Stochastic Demand with Variable Lead Times",
        difficulty="medium",
        description=(
            "Demand fluctuates ±20% around 70 units/day. "
            "Supplier reliability is 85% with lead times of 1–5 days. "
            "Warehouse capacity is limited to 400 units. "
            "Agent must buffer against uncertainty without overstocking."
        ),
        episode_length=45,
        max_order_qty=300,
        num_products=1,
        simulator=sim,
    )


# ─────────────────────────────────────────
# TASK 3 — HARD (fully uncertain)
# ─────────────────────────────────────────
def make_task3() -> TaskConfig:
    demand = DemandModel(
        base_demand=80,
        noise_level=0.4,       # high demand uncertainty
        seasonal=True,         # mid-episode demand spike (up to 2x)
    )
    supplier = Supplier(
        name="UnreliableSupplier",
        reliability=0.65,      # frequent partial failures
        lead_time_min=2,
        lead_time_max=7,       # wide lead time range
    )
    sim = InventorySimulator(
        demand_model=demand,
        supplier=supplier,
        warehouse_capacity=350,    # tight warehouse
        holding_cost=1.2,
        stockout_penalty=5.0,      # high penalty for stockouts
        revenue_per_unit=8.0,
        episode_length=60,
        initial_stock=150,
        budget=5000.0,             # strict budget constraint
        order_cost_per_unit=2.0,
    )
    return TaskConfig(
        task_id="task3_hard",
        name="Seasonal Demand with Unreliable Suppliers",
        difficulty="hard",
        description=(
            "High demand uncertainty (±40%) with mid-episode seasonal spike. "
            "Supplier reliability is only 65% with lead times of 2–7 days. "
            "Tight warehouse (350 units) and strict budget ($5000 total). "
            "Agent must forecast risk, plan ahead, and manage under pressure."
        ),
        episode_length=60,
        max_order_qty=300,
        num_products=1,
        simulator=sim,
    )


# ─────────────────────────────────────────
# REGISTRY — all tasks in one place
# ─────────────────────────────────────────
TASK_REGISTRY = {
    "task1_easy":   make_task1,
    "task2_medium": make_task2,
    "task3_hard":   make_task3,
}

def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]()

def list_tasks() -> list:
    return [get_task(tid) for tid in TASK_REGISTRY]
