"""A minimal agent loop in the style of mini-SWE-agent: one action per turn, a linear history, nothing else.

The paper's scaffold gives the model a shell. Here the only action is a Python snippet run in the sandbox
(see sandbox.py), because the run executes model-written code on a personal laptop. Everything else follows the
same shape: the model writes a short THOUGHT and one code block, sees the return code and output, and says
SUBMIT when result.json is done. A trajectory records every turn with its cost and the workspace state after it,
so any prefix can be replayed exactly and handed to another model.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import sandbox
from .data import Task, score

MAX_TURNS = 20

SYSTEM = """You are a financial data analyst working in a workspace directory. You solve the task by running \
Python code, one snippet per turn.

Every reply has two parts. First a line starting with THOUGHT: that says in one or two sentences what you will \
do next and why. Then exactly one code block:

```python
# your code
```

The code runs in a fresh Python 3.12 process with the workspace as the current directory. Use print() to see \
results; only files persist between turns. You can use the standard library (csv, json, re, math, statistics, \
decimal, pathlib) and numpy. There is no network and no shell.

Start by reading README.md. When result.json is written and you have checked it, reply with a THOUGHT line and \
the single word SUBMIT instead of a code block."""

TASK = """Task:
{prompt}

You have at most {max_turns} turns."""

HANDOFF_LOG = """Another agent has already worked on this task in this workspace for {k} turns, and you now take \
over. Its log is in `.handoff/log.jsonl`: one JSON record per turn with the fields turn, response, action, \
returncode, stdout and stderr. Inspect the final records first, then continue from the current state of the \
workspace."""

FORMAT_ERROR = ("Format error: {why} Reply with a line starting with THOUGHT: and exactly one ```python code "
                "block, or with a THOUGHT line and the single word SUBMIT when result.json is done.")

_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def parse(text: str) -> tuple[str, str]:
    """('code', source) | ('submit', '') | ('error', reason)."""
    blocks = _BLOCK.findall(text)
    if len(blocks) == 1:
        return "code", blocks[0]
    if len(blocks) > 1:
        return "error", f"found {len(blocks)} code blocks; use exactly one."
    if re.search(r"(?m)^\s*\**SUBMIT\**\s*$", text) or text.strip().endswith("SUBMIT"):
        return "submit", ""
    return "error", "no code block and no SUBMIT."


@dataclass
class Turn:
    n: int
    model: str
    response: str
    kind: str                       # code | submit | error | refusal
    code: str = ""
    observation: str = ""
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    reads: list = field(default_factory=list)
    writes: list = field(default_factory=list)
    tree: str = ""                  # workspace hash after the turn
    agent_files: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    cost: float = 0.0
    raw: list | None = None         # Claude content blocks, for the same model to continue


@dataclass
class Trajectory:
    task: str
    models: list
    turns: list = field(default_factory=list)
    end: str = ""                   # submit | max_turns | refusal | model_error
    handoff_at: int | None = None   # turns 1..handoff_at came from the source model
    transfer: str | None = None
    score: dict = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return sum(t.cost for t in self.turns)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def load(path: str | Path) -> "Trajectory":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["turns"] = [Turn(**t) for t in d["turns"]]
        return Trajectory(**d)


def history(turns: list[Turn]) -> list[dict]:
    """Model-agnostic messages for a list of turns (assistant reply, then the observation)."""
    msgs = []
    for t in turns:
        msgs.append({"role": "assistant", "content": t.response, "raw": t.raw, "model": t.model})
        if t.kind in ("code", "error"):
            msgs.append({"role": "user", "content": t.observation})
    return msgs


def act(turn: Turn, workspace: Path, scratch: Path) -> None:
    """Executes the turn's action (if any) and fills in what happened."""
    if turn.kind == "code":
        r = sandbox.run(turn.code, workspace, scratch)
        turn.returncode, turn.stdout, turn.stderr = r.returncode, r.stdout, r.stderr
        turn.reads, turn.writes = r.reads, r.writes
        turn.observation = r.observation()
    elif turn.kind == "error":
        turn.observation = FORMAT_ERROR.format(why=turn.code)
    turn.tree = sandbox.tree_hash(workspace)
    turn.agent_files = sandbox.agent_files(workspace)


def run(task: Task, chat, workspace: Path, scratch: Path, traj: Trajectory, start: list[dict],
        max_turns: int = MAX_TURNS, on_turn=None) -> Trajectory:
    """Continues traj with chat until SUBMIT, the turn limit or a refusal. start is the message list the model
    sees before its first turn (the task, plus any transferred history)."""
    msgs = list(start)
    while len(traj.turns) < max_turns:
        n = len(traj.turns) + 1
        reply = chat.reply(SYSTEM, msgs)
        if reply.stop == "refusal":
            traj.turns.append(Turn(n, chat.model, reply.text, "refusal", usage=reply.usage, cost=reply.cost, raw=reply.raw))
            traj.end = "refusal"
            break
        kind, payload = parse(reply.text)
        turn = Turn(n, chat.model, reply.text, kind, code=payload, usage=reply.usage, cost=reply.cost, raw=reply.raw)
        act(turn, workspace, scratch)
        traj.turns.append(turn)
        msgs += history([turn])
        if on_turn:
            on_turn(traj)
        if kind == "submit":
            traj.end = "submit"
            break
    else:
        traj.end = "max_turns"
    traj.score = score(task, workspace)
    return traj


def task_message(task: Task, max_turns: int = MAX_TURNS) -> dict:
    return {"role": "user", "content": TASK.format(prompt=task.prompt, max_turns=max_turns)}


def write_handoff_log(turns: list[Turn], workspace: Path) -> None:
    d = workspace / ".handoff"
    d.mkdir(exist_ok=True)
    with open(d / "log.jsonl", "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps({"turn": t.n, "response": t.response, "action": t.code if t.kind == "code" else t.kind,
                                "returncode": t.returncode, "stdout": t.stdout, "stderr": t.stderr},
                               ensure_ascii=False) + "\n")
