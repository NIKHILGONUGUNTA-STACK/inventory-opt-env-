# ─────────────────────────────────────────
# inference.py
# Baseline agent for Smart Inventory Optimization
# Uses OpenAI client to query an LLM agent
# Reads credentials from environment variables:
#   API_BASE_URL — LLM API endpoint
#   MODEL_NAME   — model identifier
#   HF_TOKEN     — Hugging Face / API key
# ─────────────────────────────────────────
import os
import json
import time
import requests
from openai import OpenAI

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

client = OpenAI(
    api_key  = HF_TOKEN or "dummy",
    base_url = API_BASE_URL,
)

TASKS = ["task1_easy", "task2_medium", "task3_hard"]


# ─────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert inventory management agent.
Your job is to decide how many units to order each day to:
1. Avoid stockouts (running out of stock = lost revenue + penalty)
2. Avoid overstocking (too much stock = high holding costs)
3. Respect warehouse capacity and budget constraints

You will receive the current warehouse state as JSON and must respond
with ONLY a JSON object in this exact format:
{"order_qty": <integer>}

Key rules:
- order_qty must be a non-negative integer
- Consider lead time: ordered stock arrives after lead_time days
- Consider in_transit: stock already ordered but not yet arrived
- Consider days_remaining: don't over-order near episode end
- Consider budget_remaining: don't exceed budget in Task 3
- A simple heuristic: order enough to cover (lead_time + 1) days of forecast demand
  minus current stock minus in_transit, but never exceed warehouse capacity"""


# ─────────────────────────────────────────
# AGENT DECISION
# ─────────────────────────────────────────
def get_agent_action(observation: dict, task_info: dict) -> int:
    """
    Ask the LLM agent what to order given current observation.
    Falls back to heuristic if LLM call fails.
    """
    user_prompt = f"""Current inventory state:
{json.dumps(observation, indent=2)}

Task: {task_info.get('name', '')}
Difficulty: {task_info.get('difficulty', '')}

Decide how many units to order. Respond ONLY with:
{{"order_qty": <integer>}}"""

    try:
        response = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens  = 50,
            temperature = 0.0,
        )
        content = response.choices[0].message.content.strip()
        # parse JSON response
        content = content.replace("```json", "").replace("```", "").strip()
        parsed  = json.loads(content)
        order_qty = int(parsed.get("order_qty", 0))
        return max(0, order_qty)

    except Exception as e:
        print(f"  [LLM fallback] {e}")
        return _heuristic_action(observation)


def _heuristic_action(obs: dict) -> int:
    """
    Fallback heuristic when LLM is unavailable.
    Order enough to cover (lead_time + 2) days of forecast demand.
    """
    stock        = obs.get("stock", 0)
    forecast     = obs.get("demand_forecast", 50)
    lead_time    = obs.get("lead_time", 2)
    in_transit   = obs.get("in_transit", 0)
    capacity     = obs.get("warehouse_capacity", 500)
    days_left    = obs.get("days_remaining", 1)
    budget       = obs.get("budget_remaining")
    if budget is None:
        budget = float("inf")

    # target: cover lead_time + 2 days of demand
    target_stock = forecast * (lead_time + 2)
    available    = stock + in_transit
    order_qty    = max(0, int(target_stock - available))

    # respect capacity
    order_qty = min(order_qty, capacity - stock)

    # respect budget (order_cost assumed 1.0–2.0 per unit)
    if budget < float("inf"):
        order_qty = min(order_qty, int(budget / 2.0))

    # don't over-order near end of episode
    if days_left <= lead_time:
        order_qty = 0

    return max(0, order_qty)


# ─────────────────────────────────────────
# ENV API HELPERS
# ─────────────────────────────────────────
def env_reset(task_id: str) -> dict:
    r = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(order_qty: int) -> dict:
    r = requests.post(
        f"{ENV_URL}/step",
        json={"order_qty": order_qty},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_score() -> dict:
    r = requests.get(f"{ENV_URL}/score", timeout=30)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────
# RUN ONE EPISODE
# ─────────────────────────────────────────
def run_episode(task_id: str) -> dict:
    print(f"\n{'='*50}")
    print(f"  Task: {task_id}")
    print(f"{'='*50}")

    # reset
    reset_data = env_reset(task_id)
    obs        = reset_data["observation"]
    task_info  = reset_data["task_info"]

    print(f"  Episode length : {task_info['episode_length']} days")
    print(f"  Max order qty  : {task_info['max_order_qty']}")
    print(f"  Initial stock  : {obs['stock']}")
    print(f"  Forecast       : {obs['demand_forecast']}")

    total_reward   = 0.0
    total_fulfilled = 0
    total_unmet    = 0
    step           = 0

    # run episode
    while True:
        order_qty = get_agent_action(obs, task_info)

        result = env_step(order_qty)
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]
        info   = result["info"]

        total_reward    += reward
        total_fulfilled += info["units_fulfilled"]
        total_unmet     += info["units_unmet"]
        step            += 1

        if step % 10 == 0 or done:
            print(
                f"  Day {step:>3} | stock={obs['stock']:>4} | "
                f"order={order_qty:>4} | reward={reward:>8.2f} | "
                f"fulfilled={info['units_fulfilled']:>3} | unmet={info['units_unmet']:>3}"
            )

        if done:
            break

        time.sleep(0.05)   # small delay to avoid hammering server

    # final score
    score_data = env_score()
    print(f"\n  --- Final Score ---")
    print(f"  Score              : {score_data['score']}")
    print(f"  Service level      : {score_data['service_level']}")
    print(f"  Cost efficiency    : {score_data['cost_efficiency']}")
    print(f"  Constraint adhere  : {score_data['constraint_adherence']}")
    print(f"  Total reward       : {round(total_reward, 2)}")
    print(f"  Units fulfilled    : {total_fulfilled}")
    print(f"  Units unmet        : {total_unmet}")

    return {
        "task_id":            task_id,
        "score":              score_data["score"],
        "service_level":      score_data["service_level"],
        "cost_efficiency":    score_data["cost_efficiency"],
        "constraint_adherence": score_data["constraint_adherence"],
        "total_reward":       round(total_reward, 2),
    }


# ─────────────────────────────────────────
# MAIN — run all 3 tasks
# ─────────────────────────────────────────
def main():
    print("\nSmart Inventory Optimization — Baseline Inference")
    print(f"Model   : {MODEL_NAME}")
    print(f"Env URL : {ENV_URL}")
    print(f"Tasks   : {TASKS}")

    results = []
    for task_id in TASKS:
        result = run_episode(task_id)
        results.append(result)

    # summary
    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    print(f"  {'Task':<22} {'Score':>7} {'Service':>9} {'Cost Eff':>10}")
    print(f"  {'-'*50}")
    for r in results:
        print(
            f"  {r['task_id']:<22} "
            f"{r['score']:>7.4f} "
            f"{r['service_level']:>9.4f} "
            f"{r['cost_efficiency']:>10.4f}"
        )

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average score across all tasks: {avg_score:.4f}")
    print(f"\n  Baseline inference complete.")

    # save results to file
    with open("baseline_scores.json", "w") as f:
        json.dump({
            "model":   MODEL_NAME,
            "results": results,
            "average_score": avg_score,
        }, f, indent=2)
    print("  Results saved to baseline_scores.json")


if __name__ == "__main__":
    main()
