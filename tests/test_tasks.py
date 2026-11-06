"""The gold answers must follow from the workspace files alone. A reference solver that reads only the files
(never the generator's internal values) recomputes every answer, and every planted trap must change it."""

import csv
import io
import json
import re

import pytest

from baton.data import TEMPLATES, build_tasks, digest, matches, materialize, score

SHA = "df0d0bac5d17c67a"  # frozen at pre-registration


@pytest.fixture(scope="module")
def tasks():
    return build_tasks()


def _load(files):
    companies = {r["ticker"]: r for r in csv.DictReader(io.StringIO(files["data/companies.csv"]))}
    unit = {"thousands": 1e3, "millions": 1e6}
    vals = {}
    for r in csv.DictReader(io.StringIO(files["data/financials.csv"])):
        m = 1e6 if r["item"] == "shares_outstanding" else unit[companies[r["ticker"]]["units"]]
        vals[(r["ticker"], int(r["fiscal_year"]), r["basis"], r["item"])] = float(r["value"]) * m
    for r in csv.DictReader(io.StringIO(files["data/amendments.csv"])):
        vals[(r["ticker"], int(r["fiscal_year"]), r["basis"], r["item"])] = \
            float(r["restated_value"]) * unit[companies[r["ticker"]]["units"]]
    splits = {r["ticker"]: float(r["ratio"]) for r in csv.DictReader(io.StringIO(files["data/corporate_actions.csv"]))}
    return companies, vals, splits


def _one_offs(note):
    total = 0.0
    for line in note.splitlines():
        if line.startswith("Net income for fiscal 2025 includes"):
            amt = float(re.search(r"of \$([\d.]+) million", line).group(1)) * 1e6
            total += -amt if "gain" in line else amt
    return total


def solve(task):
    f = task.files
    companies, v, splits = _load(f)
    c = lambda t, y, i: v[(t, y, "consolidated", i)]
    tickers = sorted(companies)
    if task.template == "margin_streak":
        m = lambda t, y: c(t, y, "operating_income") / c(t, y, "revenue")
        return sorted(t for t in tickers if m(t, 2024) > m(t, 2023) and m(t, 2025) > m(t, 2024))
    if task.template == "growth_top3":
        return sorted(tickers, key=lambda t: -(c(t, 2025, "revenue") / c(t, 2024, "revenue")))[:3]
    if task.template == "size_screen":
        cut = float(re.search(r"\$([\d,]+) million", task.question).group(1).replace(",", "")) * 1e6
        return sorted(t for t in tickers if c(t, 2025, "revenue") >= cut)
    if task.template == "adjusted_roe":
        sector = re.search(r"in the (\w+) sector", task.question).group(1)
        out = {}
        for t in tickers:
            if companies[t]["sector"] != sector:
                continue
            ni = c(t, 2025, "net_income") + _one_offs(f[f"data/notes/{t}_FY2025.md"])
            eq = (c(t, 2024, "total_equity") + c(t, 2025, "total_equity")) / 2
            out[t] = round(100 * ni / eq, 1)
        return out
    if task.template == "payout":
        return {t: round(100 * c(t, 2025, "dividends_paid") / c(t, 2025, "net_income"), 1) for t in tickers
                if c(t, 2025, "dividends_paid") > 0 and c(t, 2025, "net_income") > 0}
    if task.template == "split_dps":
        t = re.search(r"\((\w+)\)", task.question).group(1)
        a = c(t, 2024, "dividends_paid") / (c(t, 2024, "shares_outstanding") * splits[t])
        b = c(t, 2025, "dividends_paid") / c(t, 2025, "shares_outstanding")
        return {"dps_2024": round(a, 2), "dps_2025": round(b, 2), "change_pct": round(100 * (b / a - 1), 1)}
    raise ValueError(task.template)


def test_counts_and_determinism(tasks):
    assert len(tasks) == 36
    assert sum(t.split == "dev" for t in tasks) == 6 and sum(t.split == "test" for t in tasks) == 30
    assert {t.template for t in tasks} == set(TEMPLATES)
    assert digest(tasks) == digest(build_tasks())
    if SHA:
        assert digest(tasks) == SHA


def test_reference_solver_reproduces_gold(tasks):
    for t in tasks:
        assert matches(t.kind, solve(t), t.gold, t.tol), (t.tid, solve(t), t.gold)


def test_every_trap_changes_the_answer(tasks):
    for t in tasks:
        assert t.traps
        for name, wrong in t.traps.items():
            assert not matches(t.kind, wrong, t.gold, t.tol), (t.tid, name)


def test_score_reads_result_file(tasks, tmp_path):
    t = next(x for x in tasks if x.template == "payout")
    ws = materialize(t, tmp_path / "ws")
    assert score(t, ws)["status"] == "no_result"
    (ws / "result.json").write_text(json.dumps({"payout": t.gold}), encoding="utf-8")
    assert score(t, ws)["correct"]
    trap = next(iter(t.traps))
    (ws / "result.json").write_text(json.dumps({"payout": t.traps[trap]}), encoding="utf-8")
    s = score(t, ws)
    assert not s["correct"] and s["trap"] is not None
