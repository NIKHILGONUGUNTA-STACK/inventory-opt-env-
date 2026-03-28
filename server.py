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
