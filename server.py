# ─────────────────────────────────────────
# server.py
# FastAPI server exposing OpenEnv interface
# Endpoints:
#   GET  /health
#   GET  /tasks
#   POST /reset
#   POST /step
#   GET  /state
#   GET  /score
# ─────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env import InventoryEnv
from models import Action, StepResult, Observation, TaskInfo, EpisodeScore


# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────
app = FastAPI(
    title="Smart Inventory Optimization Environment",
    description=(
        "A production-grade OpenEnv environment simulating "
        "real-world inventory management for training AI agents."
    ),
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Smart Inventory Optimization Environment</title>
  <style>
    body { font-family: sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; background: #0f0f0f; color: #e0e0e0; }
    h1   { color: #4ade80; font-size: 2rem; margin-bottom: 4px; }
    h2   { color: #94a3b8; font-size: 1rem; font-weight: 400; margin-top: 0; }
    .badge { display:inline-block; background:#1e293b; border:1px solid #334155; border-radius:6px; padding:4px 10px; font-size:13px; margin:4px; color:#94a3b8; }
    .badge span { color:#4ade80; font-weight:600; }
    table { width:100%; border-collapse:collapse; margin:20px 0; }
    th    { background:#1e293b; padding:10px; text-align:left; color:#94a3b8; font-size:13px; border:1px solid #334155; }
    td    { padding:10px; border:1px solid #1e293b; font-size:14px; }
    tr:hover td { background:#1e293b; }
    .easy   { color:#4ade80; font-weight:600; }
    .medium { color:#fbbf24; font-weight:600; }
    .hard   { color:#f87171; font-weight:600; }
    .extreme{ color:#c084fc; font-weight:600; }
    .endpoint { background:#1e293b; border-radius:8px; padding:16px; margin:8px 0; display:flex; align-items:center; gap:12px; }
    .method-get  { background:#1d4ed8; color:#fff; border-radius:4px; padding:3px 8px; font-size:12px; font-weight:700; min-width:44px; text-align:center; }
    .method-post { background:#15803d; color:#fff; border-radius:4px; padding:3px 8px; font-size:12px; font-weight:700; min-width:44px; text-align:center; }
    .path { font-family:monospace; color:#4ade80; font-size:15px; }
    .desc { color:#94a3b8; font-size:13px; margin-left:auto; }
    a { color:#4ade80; }
    .score { color:#4ade80; font-weight:700; }
    .hero { background:#1e293b; border-radius:12px; padding:24px; margin:20px 0; border:1px solid #334155; }
  </style>
</head>
<body>

<h1>📦 Smart Inventory Optimization</h1>
<h2>A production-grade OpenEnv environment for training AI inventory agents</h2>

<div>
  <span class="badge">version <span>1.0.0</span></span>
  <span class="badge">framework <span>FastAPI</span></span>
  <span class="badge">spec <span>OpenEnv</span></span>
  <span class="badge">tasks <span>4</span></span>
  <span class="badge"><a href="/docs">Swagger UI →</a></span>
</div>

<div class="hero">
  <b>What this environment simulates</b><br><br>
  An AI agent operates a warehouse and decides how many units to order each day.
  It must balance <b style="color:#4ade80">service level</b> (fulfilling customer demand),
  <b style="color:#fbbf24">holding costs</b> (avoiding excess stock), and
  <b style="color:#f87171">stockout penalties</b> (avoiding shortages) — under increasing uncertainty.
</div>

<h3>Tasks</h3>
<table>
  <tr>
    <th>Task ID</th><th>Difficulty</th><th>Demand</th><th>Supplier</th><th>Baseline Score</th>
  </tr>
  <tr>
    <td><code>task1_easy</code></td>
    <td><span class="easy">Easy</span></td>
    <td>Fixed 50/day</td>
    <td>100% reliable, 2-day lead</td>
    <td><span class="score">1.000</span></td>
  </tr>
  <tr>
    <td><code>task2_medium</code></td>
    <td><span class="medium">Medium</span></td>
    <td>±20% noise</td>
    <td>85% reliable, 1–5 day lead</td>
    <td><span class="score">0.836</span></td>
  </tr>
  <tr>
    <td><code>task3_hard</code></td>
    <td><span class="hard">Hard</span></td>
    <td>±40% + seasonal spike</td>
    <td>65% reliable, 2–7 day lead</td>
    <td><span class="score">0.231</span></td>
  </tr>
  <tr>
    <td><code>task4_extreme</code></td>
    <td><span class="extreme">Extreme</span></td>
    <td>±60% + crashes + spikes</td>
    <td>40% reliable, 3–10 day lead</td>
    <td><span class="score">TBD</span></td>
  </tr>
</table>

<h3>Reward Function</h3>
<div class="hero" style="font-family:monospace; font-size:14px; line-height:2">
  reward = <span style="color:#4ade80">revenue × units_fulfilled</span><br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;− <span style="color:#fbbf24">holding_cost × stock</span><br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;− <span style="color:#f87171">stockout_penalty × units_unmet</span><br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;− <span style="color:#c084fc">capacity_violation_penalty</span><br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;− <span style="color:#94a3b8">step_penalty (0.01)</span><br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ <span style="color:#38bdf8">shaping_bonus</span>
</div>

<h3>API Endpoints</h3>
<div class="endpoint"><span class="method-get">GET</span><span class="path">/health</span><span class="desc">Health check</span></div>
<div class="endpoint"><span class="method-get">GET</span><span class="path">/tasks</span><span class="desc">List all tasks</span></div>
<div class="endpoint"><span class="method-post">POST</span><span class="path">/reset</span><span class="desc">Start new episode</span></div>
<div class="endpoint"><span class="method-post">POST</span><span class="path">/step</span><span class="desc">Execute one action</span></div>
<div class="endpoint"><span class="method-get">GET</span><span class="path">/state</span><span class="desc">Current env state</span></div>
<div class="endpoint"><span class="method-get">GET</span><span class="path">/score</span><span class="desc">Final episode score</span></div>

<br>
<p style="color:#475569; font-size:13px">
  Built for Meta Scaler Hackathon · OpenEnv framework · 
  <a href="/docs">Interactive docs →</a>
</p>

</body>
</html>
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# global environment instance
env = InventoryEnv(task_id="task1_easy")


# ─────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: Optional[str] = "task1_easy"

class StepRequest(BaseModel):
    order_qty: int


# ─────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    current_task: str
    current_day: int
    done: bool

class ResetResponse(BaseModel):
    observation: Observation
    task_info: TaskInfo

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

class StateResponse(BaseModel):
    state: dict

class TaskListResponse(BaseModel):
    tasks: list


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint.
    Automated ping checks this returns 200.
    """
    s = env.state()
    return HealthResponse(
        status       = "ok",
        version      = "1.0.0",
        current_task = env.task_id,
        current_day  = s["current_day"],
        done         = s["done"],
    )


@app.get("/tasks", response_model=TaskListResponse)
def list_tasks():
    """
    List all available tasks with metadata.
    """
    tasks = env.list_tasks()
    return TaskListResponse(
        tasks=[t.model_dump() for t in tasks]
    )


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    """
    Reset the environment to start a new episode.
    Optionally switch to a different task.

    Example:
      POST /reset
      {"task_id": "task2_medium"}
    """
    valid_tasks = ["task1_easy", "task2_medium", "task3_hard"]
    if request.task_id not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{request.task_id}'. Must be one of {valid_tasks}"
        )

    obs      = env.reset(task_id=request.task_id)
    task_info = env.get_task_info()

    return ResetResponse(
        observation = obs,
        task_info   = task_info,
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Execute one step in the environment.
    Agent provides order_qty (units to order).

    Example:
      POST /step
      {"order_qty": 60}
    """
    if env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call POST /reset to start a new episode."
        )

    if request.order_qty < 0:
        raise HTTPException(
            status_code=400,
            detail="order_qty must be >= 0"
        )

    action = Action(order_qty=request.order_qty)
    result = env.step(action)

    return StepResponse(
        observation = result.observation,
        reward      = result.reward,
        done        = result.done,
        info        = result.info.model_dump(),
    )


@app.get("/state", response_model=StateResponse)
def state():
    """
    Returns the full current environment state.
    Useful for inspection and debugging.
    """
    return StateResponse(state=env.state())


@app.get("/score", response_model=EpisodeScore)
def score():
    """
    Returns the final graded score for the completed episode.
    Only valid after done=True.
    """
    if not env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode not finished yet. Keep stepping until done=True."
        )
    return env.score()


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
