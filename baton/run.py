"""Command line: record trajectories, compute handoff continuations, judge readiness, report.

    python -m baton solo --model claude-sonnet-5-5 --split dev        # capable-only (paid)
    python -m baton solo --model qwen3:4b --split test                 # efficient-only (local)
    python -m baton sweep --source claude-sonnet-5-5 --receiver qwen3:4b --transfer incontext
    python -m baton detect --source claude-sonnet-5-5 --judge qwen3:4b
    python -m baton logfile --source claude-sonnet-5-5 --receiver qwen3:4b
    python -m baton reverse --source qwen3:4b --receiver claude-sonnet-5-5   # efficient-to-capable (paid)
    python -m baton spent
    python -m baton report --out reports/results.md

Every command resumes from what is on disk. A paid run stops before any request once the API spend recorded
under runs/ reaches --budget dollars (default 10).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .agent import Trajectory, history, run, task_message
from .data import build_tasks
from .handoff import (FIXED, continue_from, detect, fixed_point, fresh, handoff_points, prompt_point, replay,
                      rule_point)
from .models import ModelError, make_chat

RUNS = Path("runs")
WORK = Path("C:/Users/sinky/dev/baton-work") if sys.platform == "win32" else Path("work")


def tag(model: str) -> str:
    return model.replace(":", "_")


def solo_path(model: str, tid: str) -> Path:
    return RUNS / "solo" / tag(model) / f"{tid}.json"


def spent() -> float:
    """API dollars recorded in every saved trajectory (finished or partial), counting each turn once."""
    total = 0.0
    for f in RUNS.rglob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or "turns" not in d:
            continue
        k = d.get("handoff_at") or 0
        # a handoff's prefix turns are copies of its source's turns, which are already counted
        own = d["turns"][k:] if d.get("transfer") else d["turns"]
        total += sum(t.get("cost", 0.0) for t in own)
    return total


class BudgetExceeded(Exception):
    pass


class Guard:
    """Wraps a paid chat: refuses to send a request once the recorded spend reaches the budget."""

    def __init__(self, chat, budget: float):
        self.chat, self.budget, self.model = chat, budget, chat.model

    def reply(self, system, messages, **kw):
        s = spent()
        if s >= self.budget:
            raise BudgetExceeded(f"recorded API spend ${s:.2f} reached the ${self.budget:.2f} budget")
        return self.chat.reply(system, messages, **kw)


def chat_for(model: str, budget: float, **kw):
    c = make_chat(model, **kw)
    return Guard(c, budget) if model.startswith("claude-") else c


def pick(split: str, only: list[str] | None):
    return [t for t in build_tasks() if (split == "all" or t.split == split) and (not only or t.tid in only)]


def _resume_or_start(task, model: str, path: Path, chat, root: Path) -> Trajectory:
    """Runs one solo trajectory, picking up a partial one saved after an interruption."""
    partial = path.with_suffix(".partial.json")
    ws, scratch = fresh(task, root)
    traj, start = Trajectory(task.tid, [model]), [task_message(task)]
    if partial.exists():
        old = Trajectory.load(partial)
        if replay(old.turns, ws, scratch) is None:
            traj, start = old, [task_message(task)] + history(old.turns)
            print(f"  resuming {task.tid} after turn {len(old.turns)}", flush=True)
        else:
            ws, scratch = fresh(task, root)
    out = run(task, chat, ws, scratch, traj, start, on_turn=lambda tr: tr.save(partial))
    out.save(path)
    partial.unlink(missing_ok=True)
    return out


def cmd_solo(a) -> int:
    chat = chat_for(a.model, a.budget)
    for task in pick(a.split, a.tasks):
        path = solo_path(a.model, task.tid)
        if path.exists():
            continue
        t0 = time.time()
        tr = _resume_or_start(task, a.model, path, chat, WORK / "solo" / tag(a.model) / task.tid)
        print(f"[solo {a.model}] {task.tid}: {tr.score['status']} trap={tr.score.get('trap')} turns={len(tr.turns)} "
              f"end={tr.end} ${tr.cost:.4f} {time.time() - t0:.0f}s | total spent ${spent():.3f}", flush=True)
    return 0


def _sources(a):
    for task in pick(a.split, a.tasks):
        p = solo_path(a.source, task.tid)
        if not p.exists():
            print(f"  no {a.source} trajectory for {task.tid}; run solo first")
            continue
        yield task, Trajectory.load(p)


def sweep_path(source: str, receiver: str, transfer: str, tid: str, k: int) -> Path:
    return RUNS / "handoff" / f"{tag(source)}__{tag(receiver)}" / transfer / tid / f"k{k:02d}.json"


def _continue(task, source, k, receiver_chat, receiver, transfer, a):
    p = sweep_path(a.source, receiver, transfer, task.tid, k)
    if p.exists():
        return Trajectory.load(p)
    t0 = time.time()
    tr = continue_from(task, source, k, receiver_chat, transfer, WORK / "handoff" / tag(a.source) / tag(receiver) / task.tid)
    tr.save(p)
    print(f"[{transfer} {a.source}->{receiver}] {task.tid} k={k}/{len(source.turns)}: {tr.score['status']} "
          f"trap={tr.score.get('trap')} turns={len(tr.turns)} end={tr.end} {time.time() - t0:.0f}s", flush=True)
    return tr


def cmd_sweep(a) -> int:
    rc = chat_for(a.receiver, a.budget)
    for task, src in _sources(a):
        for k in handoff_points(src):
            _continue(task, src, k, rc, a.receiver, a.transfer, a)
    return 0


def detector_path(source: str, judge: str, tid: str) -> Path:
    return RUNS / "detector" / f"{tag(source)}__{tag(judge)}" / f"{tid}.json"


def cmd_detect(a) -> int:
    judge = chat_for(a.judge, a.budget)
    for task, src in _sources(a):
        p = detector_path(a.source, a.judge, task.tid)
        if p.exists():
            continue
        js = detect(task, src, judge)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
        print(f"[detect {a.judge}] {task.tid}: " + "".join("R" if j["judgement"] == "HANDOFF_READY" else "." for j in js),
              flush=True)
    return 0


def policy_points(task, src, source: str, judge: str) -> dict[str, int | None]:
    pts = {f"fixed{int(p * 100)}": fixed_point(src, p) for p in FIXED}
    pts["rule"] = rule_point(src)
    dp = detector_path(source, judge, task.tid)
    if dp.exists():
        pts["prompt"] = prompt_point(json.loads(dp.read_text(encoding="utf-8")))
    return pts


def cmd_logfile(a) -> int:
    rc = chat_for(a.receiver, a.budget)
    for task, src in _sources(a):
        ks = {k for k in policy_points(task, src, a.source, a.judge).values() if k}
        for k in sorted(ks):
            _continue(task, src, k, rc, a.receiver, "logfile", a)
    return 0


def cmd_reverse(a) -> int:
    """Efficient-to-capable: the local model's own trajectory, handed to the capable model at 50% progress."""
    rc = chat_for(a.receiver, a.budget)
    for task, src in _sources(a):
        k = fixed_point(src, 0.5)
        if k is None:
            continue
        _continue(task, src, k, rc, a.receiver, "incontext", a)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="baton", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--split", default="test", choices=["dev", "test", "all"])
        sp.add_argument("--tasks", nargs="*")
        sp.add_argument("--budget", type=float, default=10.0)
    sp = sub.add_parser("solo"); common(sp); sp.add_argument("--model", required=True)
    sp = sub.add_parser("sweep"); common(sp); sp.add_argument("--source", required=True)
    sp.add_argument("--receiver", default="qwen3:4b"); sp.add_argument("--transfer", default="incontext",
                                                                        choices=["incontext", "logfile"])
    sp = sub.add_parser("detect"); common(sp); sp.add_argument("--source", required=True)
    sp.add_argument("--judge", default="qwen3:4b")
    sp = sub.add_parser("logfile"); common(sp); sp.add_argument("--source", required=True)
    sp.add_argument("--receiver", default="qwen3:4b"); sp.add_argument("--judge", default="qwen3:4b")
    sp = sub.add_parser("reverse"); common(sp); sp.add_argument("--source", default="qwen3:4b")
    sp.add_argument("--receiver", required=True)
    sub.add_parser("spent")
    sp = sub.add_parser("report"); sp.add_argument("--split", default="test", choices=["dev", "test", "all"])
    sp.add_argument("--out"); sp.add_argument("--csv", help="also write one row per recorded trajectory here")
    a = p.parse_args(argv)
    try:
        if a.cmd == "solo":
            return cmd_solo(a)
        if a.cmd == "sweep":
            return cmd_sweep(a)
        if a.cmd == "detect":
            return cmd_detect(a)
        if a.cmd == "logfile":
            return cmd_logfile(a)
        if a.cmd == "reverse":
            return cmd_reverse(a)
    except BudgetExceeded as exc:
        print(f"stopped: {exc}")
        return 3
    except ModelError as exc:
        print(f"model unavailable: {exc}; rerun to resume")
        return 2
    if a.cmd == "spent":
        print(f"${spent():.4f}")
        return 0
    from .report import CSV_FIELDS, outcome_rows, report
    text = report(RUNS, a.split)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    if a.csv:
        import csv
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(outcome_rows(RUNS, a.split))
    print(text)
    return 0
