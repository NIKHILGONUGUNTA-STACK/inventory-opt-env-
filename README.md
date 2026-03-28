---
title: Smart Inventory Optimization Environment
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - inventory-management
  - supply-chain
---

# Smart Inventory Optimization Environment

A production-grade OpenEnv environment simulating real-world inventory
management. An AI agent observes warehouse state and decides how much
inventory to order each day.

## Environment Description

The agent operates a warehouse where it must balance:
- **Service level** — fulfilling customer demand without stockouts
- **Holding costs** — avoiding excess inventory
- **Constraints** — warehouse capacity and budget limits

At each timestep the agent observes the current state and decides
`order_qty` — how many units to order from the supplier.

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `order_qty` | integer ≥ 0 | Units to order this step |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `stock` | int | Current inventory level |
| `demand_forecast` | float | Predicted next-period demand |
| `lead_time` | int | Days until delivery arrives |
| `in_transit` | int | Units already ordered, en route |
| `supplier_reliability` | float | Fulfillment probability 0–1 |
| `holding_cost` | float | Cost per unit per day |
| `stockout_penalty` | float | Cost per unmet demand unit |
| `warehouse_capacity` | int | Maximum storable units |
| `days_remaining` | int | Steps left in episode |
| `budget_remaining` | float | Remaining budget (Task 3) |
| `current_day` | int | Current episode day |
| `season_factor` | float | Seasonal demand multiplier |

## Tasks

### Task 1 — Easy (Deterministic)
- Fixed demand: 50 units/day
- Supplier reliability: 100%
- Lead time: exactly 2 days
- Episode length: 30 days
- No capacity or budget constraints
- **Learn**: basic reorder point logic

### Task 2 — Medium (Stochastic)
- Demand: 70 units/day ± 20% noise
- Supplier reliability: 85%
- Lead time: 1–5 days (random)
- Warehouse capacity: 400 units
- Episode length: 45 days
- **Learn**: buffering under uncertainty

### Task 3 — Hard (Fully Uncertain)
- Demand: 80 units/day ± 40% noise + seasonal spike
- Supplier reliability: 65%
- Lead time: 2–7 days (random)
- Warehouse capacity: 350 units
- Budget: $5000 total
- Episode length: 60 days
- **Learn**: risk planning, seasonal forecasting

## Reward Function
```
reward = revenue
       - holding_cost
       - stockout_penalty
       - capacity_violation_penalty
       - step_penalty (0.01)
       + shaping_bonus
```

Provides **continuous partial feedback** every step.

## Grading (0.0 – 1.0)

| Task | Service Level | Cost Efficiency | Constraint Adherence |
|------|--------------|-----------------|----------------------|
| Task 1 | 80% | 10% | 10% |
| Task 2 | 50% | 35% | 15% |
| Task 3 | 40% | 30% | 30% |

## Baseline Scores (heuristic agent)

| Task | Score | Service Level | Cost Efficiency |
|------|-------|--------------|-----------------|
| task1_easy | 1.000 | 1.000 | 1.000 |
| task2_medium | 0.807 | 0.924 | 0.756 |
| task3_hard | 0.223 | 0.412 | 0.000 |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Start new episode |
| POST | `/step` | Execute one action |
| GET | `/state` | Current env state |
| GET | `/score` | Final episode score |

## Setup
```bash
# install
pip install -r requirements.txt

# run server
python server.py

# run baseline agent
python inference.py
```

## Docker
```bash
docker build -t inventory-opt-env .
docker run -p 7860:7860 inventory-opt-env
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face / API key |
- [OpenEnv Spec](https://huggingface.co/spaces/openrlbench/openenv)
- [Pydantic Validation](https://docs.pydantic.dev/)
