import os
import json
import time
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

client = OpenAI(
    api_key  = HF_TOKEN or "dummy",
    base_url = API_BASE_URL,
)

TASKS = ["task1_easy", "task2_medium", "task3_hard", "task4_extreme"]

SYSTEM_PROMPT = """You are an expert inventory management agent.
Decide how many units to order each day to avoid stockouts and overstocking.
Respond ONLY with JSON: {"order_qty": <integer>}
Rules:
- order_qty must be a non-negative integer
- Cover (lead_time + 2) days of forecast demand
- Never exceed warehouse_capacity
- Respect budget_remaining in Task 3 and 4"""


def get_agent_action(observation: dict, task_info: dict) -> tuple:
    """Returns (order_qty, error_str)"""
    user_prompt = f"""Inventory state:
{json.dumps(observation, indent=2)}
Task: {task_info.get('name','')} ({task_info.get('difficulty','')})
Respond ONLY with: {{"order_qty": <integer>}}"""

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
        content = content.replace("```json","").replace("```","").strip()
        parsed  = json.loads(content)
        return max(0, int(parsed.get("order_qty", 0))), None

    except Exception as e:
        return _heuristic_action(observation), str(e)


def _heuristic_action(obs: dict) -> int:
    stock     = obs.get("stock", 0)
    forecast  = obs.get("demand_forecast", 50)
    lead_time = obs.get("lead_time", 2)
    in_transit= obs.get("in_transit", 0)
    capacity  = obs.get("warehouse_capacity", 500)
    days_left = obs.get("days_remaining", 1)
    budget    = obs.get("budget_remaining", float("inf"))

    target    = forecast * (lead_time + 2)
    available = stock + in_transit
    order_qty = max(0, int(target - available))
    order_qty = min(order_qty, capacity - stock)

    if budget < float("inf"):
        order_qty = min(order_qty, int(budget / 2.0))
    if days_left <= lead_time:
        order_qty = 0

    return max(0, order_qty)


def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(order_qty: int) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"order_qty": order_qty}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_score() -> dict:
    r = requests.get(f"{ENV_URL}/score", timeout=30)
    r.raise_for_status()
    return r.json()


def run_episode(task_id: str) -> dict:
    # reset
    reset_data = env_reset(task_id)
    obs        = reset_data["observation"]
    task_info  = reset_data["task_info"]

    # ── MANDATORY [START] LOG ──
    print(f"[START] task={task_id} env=inventory-opt model={MODEL_NAME}", flush=True)

    rewards      = []
    total_reward = 0.0
    step         = 0
    last_error   = None

    while True:
        order_qty, error = get_agent_action(obs, task_info)
        last_error = error

        result = env_step(order_qty)
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]
        step  += 1
        total_reward += reward
        rewards.append(round(reward, 2))

        error_str = error if error else "null"
        done_str  = "true" if done else "false"

        # ── MANDATORY [STEP] LOG ──
        print(
            f"[STEP] step={step} action={order_qty} "
            f"reward={reward:.2f} done={done_str} error={error_str}",
            flush=True
        )

        if done:
            break
        time.sleep(0.02)

    # get final score
    score_data = env_score()
    score      = score_data.get("score", 0.0)
    success    = score > 0.0
    rewards_str = ",".join(str(r) for r in rewards)

    # ── MANDATORY [END] LOG ──
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step} score={score:.2f} rewards={rewards_str}",
        flush=True
    )

    return {
        "task_id":      task_id,
        "score":        score,
        "service_level":      score_data.get("service_level", 0),
        "cost_efficiency":    score_data.get("cost_efficiency", 0),
        "constraint_adherence": score_data.get("constraint_adherence", 0),
        "total_reward": round(total_reward, 2),
    }


def main():
    results = []
    for task_id in TASKS:
        result = run_episode(task_id)
        results.append(result)

    # save scores
    avg = sum(r["score"] for r in results) / len(results)
    with open("baseline_scores.json", "w") as f:
        json.dump({"model": MODEL_NAME, "results": results, "average_score": avg}, f, indent=2)


if __name__ == "__main__":
    main()
