"""Accuracy-cost tables, paired comparisons and the pre-registered hypothesis checks."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .agent import Trajectory
from .data import build_tasks
from .handoff import FIXED, fixed_point, handoff_points, prompt_point, rule_point

B = 10_000


def _load(p: Path) -> Trajectory | None:
    return Trajectory.load(p) if p.exists() else None


def boot_diff(a: list[float], b: list[float], seed: int = 0) -> tuple[float, float, float]:
    """Mean of a - b over tasks with a 95% paired bootstrap interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def mcnemar(a: list[bool], b: list[bool]) -> float:
    """Exact two-sided McNemar p-value on paired outcomes."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z, p = 1.96, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


class Pair:
    """Everything recorded for one capable source handing off to one local receiver."""

    def __init__(self, runs: Path, source: str, receiver: str, judge: str, tasks):
        from .run import detector_path, solo_path, sweep_path
        self.source, self.receiver, self.tasks = source, receiver, tasks
        self.src = {t.tid: _load(solo_path(source, t.tid)) for t in tasks}
        self.local = {t.tid: _load(solo_path(receiver, t.tid)) for t in tasks}
        self.sweep = {}
        for t in tasks:
            s = self.src[t.tid]
            for transfer in ("incontext", "logfile"):
                for k in (handoff_points(s) if s else []):
                    tr = _load(sweep_path(source, receiver, transfer, t.tid, k))
                    if tr:
                        self.sweep[(transfer, t.tid, k)] = tr
        self.judge = {}
        for t in tasks:
            p = detector_path(source, judge, t.tid)
            if p.exists():
                self.judge[t.tid] = json.loads(p.read_text(encoding="utf-8"))

    def complete(self) -> bool:
        return all(self.src.values())

    def points(self, policy: str, tid: str) -> int | None:
        s = self.src[tid]
        if policy.startswith("fixed"):
            return fixed_point(s, int(policy[5:]) / 100)
        if policy == "rule":
            return rule_point(s)
        if policy == "prompt":
            return prompt_point(self.judge[tid]) if tid in self.judge else None
        raise ValueError(policy)

    def outcome(self, policy: str, transfer: str, tid: str):
        """(correct, cost, handoff turn or None, total turns) of a policy on one task, or None if not run yet."""
        s = self.src[tid]
        if policy == "capable":
            return s.score["correct"], s.cost, None, len(s.turns)
        if policy == "local":
            lt = self.local[tid]
            return (lt.score["correct"], 0.0, None, len(lt.turns)) if lt else None
        if policy == "oracle":   # analysis only: the earliest handoff that ends correct, if any
            for k in handoff_points(s):
                tr = self.sweep.get((transfer, tid, k))
                if tr is None:
                    return None
                if tr.score["correct"]:
                    return True, sum(t.cost for t in s.turns[:k]), k, len(s.turns)
            return s.score["correct"], s.cost, None, len(s.turns)
        if policy == "prompt" and tid not in self.judge:
            return None
        k = self.points(policy, tid)
        if k is None:   # the policy never fires: the capable model finishes alone
            return s.score["correct"], s.cost, None, len(s.turns)
        tr = self.sweep.get((transfer, tid, k))
        if tr is None:
            return None
        return tr.score["correct"], sum(t.cost for t in s.turns[:k]), k, len(s.turns)

    def column(self, policy, transfer):
        rows = [self.outcome(policy, transfer, t.tid) for t in self.tasks]
        return None if any(r is None for r in rows) else rows


def _pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}"


def pair_section(pair: Pair, reverse: dict | None) -> list[str]:
    n = len(pair.tasks)
    cap = pair.column("capable", "incontext")
    base_cost = sum(r[1] for r in cap) / n
    lines = [f"## {pair.source} → {pair.receiver}", "",
             "| Condition | Transfer | Correct | Accuracy % (95% CI) | API $ per task | Cost vs capable-only | "
             "Handoff turn / length | vs capable-only, pts (95% CI) | McNemar p |",
             "|---|---|---|---|---|---|---|---|---|"]
    conds = [("capable", "-"), ("local", "-")] + [(f"fixed{int(p * 100)}", "incontext") for p in FIXED] + \
            [("rule", "incontext"), ("prompt", "incontext"), ("fixed50", "logfile"), ("rule", "logfile"),
             ("prompt", "logfile"), ("oracle", "incontext")]
    for policy, transfer in conds:
        col = pair.column(policy, "incontext" if transfer == "-" else transfer)
        if col is None:
            lines.append(f"| {policy} | {transfer} | not run | | | | | | |")
            continue
        k = sum(r[0] for r in col)
        lo, hi = wilson(k, n)
        cost = sum(r[1] for r in col) / n
        ks = [(r[2], r[3]) for r in col if r[2]]
        when = f"{np.mean([a / b for a, b in ks]) * 100:.0f}% ({len(ks)}/{n} handed off)" if ks else "-"
        if policy == "capable":
            diff, p = "-", "-"
        else:
            m, dlo, dhi = boot_diff([r[0] for r in col], [r[0] for r in cap])
            diff, p = f"{100 * m:+.1f} ({100 * dlo:+.1f} to {100 * dhi:+.1f})", f"{mcnemar([r[0] for r in col], [r[0] for r in cap]):.3f}"
        rel = f"{100 * (cost / base_cost - 1):+.0f}%" if base_cost else "-"
        lines.append(f"| {policy} | {transfer} | {k}/{n} | {_pct(k / n)} ({_pct(lo)}–{_pct(hi)}) | {cost:.4f} | {rel} | "
                     f"{when} | {diff} | {p} |")
    if reverse is not None:
        rows = [reverse.get(t.tid) for t in pair.tasks]
        if all(r is not None for r in rows):
            k = sum(r[0] for r in rows)
            lo, hi = wilson(k, n)
            cost = sum(r[1] for r in rows) / n
            m, dlo, dhi = boot_diff([r[0] for r in rows], [r[0] for r in cap])
            lines.append(f"| reverse ({pair.receiver} → {pair.source}) at 50% | incontext | {k}/{n} | {_pct(k / n)} "
                         f"({_pct(lo)}–{_pct(hi)}) | {cost:.4f} | {100 * (cost / base_cost - 1):+.0f}% | - | "
                         f"{100 * m:+.1f} ({100 * dlo:+.1f} to {100 * dhi:+.1f}) | "
                         f"{mcnemar([r[0] for r in rows], [r[0] for r in cap]):.3f} |")
    lines += ["", "Handoff curve (in-context): share of continuations that end correct, by when the handoff happens",
              "", "| Handoff after | continuations | correct % |", "|---|---|---|"]
    bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    for lo_, hi_ in bins:
        xs = [tr.score["correct"] for (tr_, tid, k), tr in pair.sweep.items()
              if tr_ == "incontext" and lo_ <= k / len(pair.src[tid].turns) < hi_]
        lines.append(f"| {int(lo_ * 100)}–{min(100, int(hi_ * 100))}% of the capable run | {len(xs)} | "
                     f"{_pct(np.mean(xs)) if xs else 'n/a'} |")
    if pair.judge:
        js = [j["judgement"] for v in pair.judge.values() for j in v]
        c = Counter(js)
        first = [prompt_point(v) for v in pair.judge.values()]
        lines += ["", f"Prompt detector ({len(js)} judgements on {len(pair.judge)} tasks): READY {c['HANDOFF_READY']}, "
                  f"NOT_READY {c['HANDOFF_NOT_READY']}, unparsable {len(js) - c['HANDOFF_READY'] - c['HANDOFF_NOT_READY']}; "
                  f"fires at turn 1 on {sum(1 for f in first if f == 1)} tasks, never on {sum(1 for f in first if f is None)}."]
    mism = sum(1 for tr in pair.sweep.values() if tr.score.get("status") == "replay_mismatch")
    lines += ["", f"Replay mismatches (prefix did not reproduce the recorded workspace): {mism} of {len(pair.sweep)}."]
    traps = Counter()
    for (transfer, tid, k), tr in pair.sweep.items():
        if transfer == "incontext" and not tr.score["correct"]:
            traps[tr.score.get("trap") or tr.score.get("status")] += 1
    lines += ["", "Wrong in-context continuations by cause: " + ", ".join(f"{k} {v}" for k, v in traps.most_common())]
    return lines


CSV_FIELDS = ["run", "source", "receiver", "transfer", "task", "template", "handoff_after", "source_turns", "turns",
              "end", "correct", "status", "trap", "source_cost_usd", "receiver_cost_usd", "aborted_turns"]


def outcome_rows(runs: Path, split: str = "test") -> list[dict]:
    """One row per recorded trajectory (alone or handoff), enough to recount every table without the raw runs."""
    tids = {t.tid for t in build_tasks() if split == "all" or t.split == split}
    rows, solo_len = [], {}
    for f in sorted((runs / "solo").glob("*/*.json")):
        tr = Trajectory.load(f)
        if tr.task not in tids:
            continue
        solo_len[(tr.models[0], tr.task)] = len(tr.turns)
        rows.append({"run": "alone", "source": tr.models[0], "receiver": "", "transfer": "", "task": tr.task,
                     "handoff_after": "", "turns": len(tr.turns), "source_cost_usd": round(tr.cost, 6),
                     "receiver_cost_usd": 0.0, "tr": tr})
    for f in sorted((runs / "handoff").glob("*/*/*/k*.json")):
        tr = Trajectory.load(f)
        if tr.task not in tids:
            continue
        k = tr.handoff_at
        rows.append({"run": "handoff", "source": tr.models[0], "receiver": tr.models[1], "transfer": tr.transfer,
                     "task": tr.task, "handoff_after": k, "turns": len(tr.turns),
                     "source_cost_usd": round(sum(t.cost for t in tr.turns[:k]), 6),
                     "receiver_cost_usd": round(sum(t.cost for t in tr.turns[k:]), 6), "tr": tr})
    for r in rows:
        tr = r.pop("tr")
        r.update(template=r["task"].rsplit("-", 1)[0], source_turns=solo_len.get((r["source"], r["task"]), ""),
                 end=tr.end, correct=int(bool(tr.score.get("correct"))), status=tr.score.get("status"),
                 trap=tr.score.get("trap") or "",
                 aborted_turns=sum(1 for t in tr.turns if t.usage.get("aborted")))
    return rows


def report(runs: Path, split: str = "test") -> str:
    from .run import solo_path, sweep_path
    tasks = [t for t in build_tasks() if split == "all" or t.split == split]
    out = [f"# Results ({split} split, {len(tasks)} tasks)", ""]
    for source in ("claude-sonnet-5-5", "claude-haiku-5-5"):
        pair = Pair(runs, source, "qwen3:4b", "qwen3:4b", tasks)
        if not pair.complete():
            out += [f"## {source} → qwen3:4b", "", "not complete", ""]
            continue
        reverse = None
        if source == "claude-sonnet-5-5":
            reverse = {}
            for t in tasks:
                lt = _load(solo_path("qwen3:4b", t.tid))
                if lt is None:
                    continue
                k = fixed_point(lt, 0.5)
                if k is None:
                    reverse[t.tid] = (lt.score["correct"], 0.0)
                    continue
                tr = _load(sweep_path("qwen3:4b", source, "incontext", t.tid, k))
                if tr:
                    reverse[t.tid] = (tr.score["correct"], sum(x.cost for x in tr.turns[k:]))
        out += pair_section(pair, reverse) + [""]
    return "\n".join(out)
