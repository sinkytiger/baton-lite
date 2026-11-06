"""Handoffs: replay a recorded prefix, then let another model continue, plus the policies that pick the turn.

A capable-only trajectory is recorded once. A handoff after turn k replays the first k actions in a fresh
workspace (checking that each turn leaves the same file tree as recorded), then the receiver continues with
either the whole prefix in its context or only a log file, as in the paper. Because the receiver decodes
greedily, the continuation from turn k does not depend on which policy chose k, so the continuation from every
k is computed once and every policy becomes a lookup. Cost of a handoff = the source's API cost for turns 1..k;
the local receiver costs nothing.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from . import sandbox
from .agent import HANDOFF_LOG, MAX_TURNS, Trajectory, Turn, history, run, task_message, write_handoff_log
from .data import Task, materialize

DETECTOR_SYSTEM = """You monitor an AI agent that is solving a data-analysis task with an expensive model. After \
each turn you decide whether the rest of the task can now be handed to a much smaller, cheaper model that will \
continue from the same workspace.

The task is ready for handoff only when all three hold:
1. A substantive artifact exists: a working script, computed intermediate results, or a draft result.json.
2. Actual execution has given task-relevant feedback: code has run on the data and the agent has seen the \
results. A failed run counts if it narrowed down the problem.
3. The remaining work is focused: local fixes, checks, or writing the final output, not open exploration or a \
change of approach.

Answer HANDOFF_READY or HANDOFF_NOT_READY."""

DETECTOR_SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string", "maxLength": 400},
    "judgement": {"type": "string", "enum": ["HANDOFF_READY", "HANDOFF_NOT_READY"]}},
    "required": ["reason", "judgement"]}

FIXED = (0.25, 0.5, 0.75)


def fresh(task: Task, root: Path) -> tuple[Path, Path]:
    if root.exists():
        shutil.rmtree(root)
    ws, scratch = root / "ws", root / "scratch"
    materialize(task, ws)
    return ws, scratch


def replay(turns: list[Turn], ws: Path, scratch: Path) -> int | None:
    """Re-executes the recorded actions; returns the first turn whose file tree differs, or None."""
    for t in turns:
        if t.kind == "code":
            sandbox.run(t.code, ws, scratch)
        if sandbox.tree_hash(ws) != t.tree:
            return t.n
    return None


def handoff_points(source: Trajectory) -> list[int]:
    """Turns after which a handoff can happen: at least one turn done, at least one turn left to the receiver."""
    n = len(source.turns)
    last = n - 1 if source.end == "submit" else n
    return list(range(1, min(last, MAX_TURNS - 1) + 1))


def continue_from(task: Task, source: Trajectory, k: int, receiver, transfer: str, root: Path,
                  max_turns: int = MAX_TURNS) -> Trajectory:
    ws, scratch = fresh(task, root)
    prefix = copy.deepcopy(source.turns[:k])
    traj = Trajectory(task.tid, [source.models[0], receiver.model], turns=prefix, handoff_at=k, transfer=transfer)
    bad = replay(prefix, ws, scratch)
    if bad is not None:
        traj.end = f"replay_mismatch@{bad}"
        traj.score = {"correct": False, "status": "replay_mismatch", "trap": None}
        return traj
    if transfer == "incontext":
        start = [task_message(task, max_turns)] + history(prefix)
    elif transfer == "logfile":
        write_handoff_log(prefix, ws)
        m = task_message(task, max_turns)
        start = [{"role": "user", "content": m["content"] + "\n\n" + HANDOFF_LOG.format(k=k)}]
    else:
        raise ValueError(transfer)
    return run(task, receiver, ws, scratch, traj, start, max_turns)


# ------------------------------------------------------------------ policies

def fixed_point(source: Trajectory, p: float) -> int | None:
    pts = handoff_points(source)
    if not pts:
        return None
    return min(max(round(p * len(source.turns)), pts[0]), pts[-1])


def rule_point(source: Trajectory) -> int | None:
    """The paper's three readiness signs, read from the workspace record: an agent-made file exists (artifact),
    the turn's code ran without error after the agent had already looked at the data (feedback), and the turn
    opened no data file it had not opened before (the remaining work is focused)."""
    seen: set[str] = set()
    for t in source.turns:
        data_reads = {r for r in t.reads if r.startswith("data/") or r == "README.md"}
        new = data_reads - seen
        looked_before = bool(seen)
        seen |= data_reads
        if t.n not in handoff_points(source):
            continue
        if t.agent_files and t.kind == "code" and t.returncode == 0 and looked_before and not new:
            return t.n
    return None


def render(task: Task, turns: list[Turn], budget: int = 24000) -> str:
    parts = []
    for t in turns:
        obs = t.observation if len(t.observation) < 1500 else t.observation[:700] + "\n...\n" + t.observation[-700:]
        parts.append(f"## Turn {t.n}\nAgent:\n{t.response.strip()}\n\nResult:\n{obs}")
    body = ""
    for p in reversed(parts):     # keep the most recent turns that fit
        if len(body) + len(p) > budget:
            body = "[earlier turns omitted]\n\n" + body
            break
        body = p + "\n\n" + body
    return f"# Task\n{task.prompt}\n\n# Trajectory so far\n{body}\nIs the task ready for handoff now?"


def detect(task: Task, source: Trajectory, chat) -> list[dict]:
    """Asks a judge model after every possible handoff turn; returns one record per turn."""
    out = []
    for k in handoff_points(source):
        r = chat.reply(DETECTOR_SYSTEM, [{"role": "user", "content": render(task, source.turns[:k])}],
                       schema=DETECTOR_SCHEMA)
        try:
            obj = json.loads(r.text)
            j = obj.get("judgement")
        except ValueError:
            obj, j = {}, None
        out.append({"turn": k, "judgement": j, "reason": obj.get("reason", ""), "usage": r.usage})
    return out


def prompt_point(judgements: list[dict]) -> int | None:
    return next((j["turn"] for j in judgements if j["judgement"] == "HANDOFF_READY"), None)
