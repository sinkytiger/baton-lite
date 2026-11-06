"""Synthetic financial-analysis tasks with exact answers.

Each task is a small workspace of fictional Canadian issuers: a data dictionary, consolidated and parent-only
statements, filed restatements, year-end notes and corporate actions. The agent must explore the files, write
code, and save result.json. Every task plants traps (a restatement that changes the answer, mixed reporting
units, parent-only rows, prior-year items in the notes, a share split), and the generator keeps a task only if
each planted trap changes the answer, so a wrong answer can be traced to the trap it fell into.

    from baton.data import build_tasks, materialize
    tasks = build_tasks()            # 6 dev + 30 test, deterministic
    materialize(tasks[0], "work/t")  # writes the workspace files
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

YEARS = (2023, 2024, 2025)
UNITS = {"thousands": 1_000, "millions": 1_000_000}
SECTORS = ["Industrials", "Consumer", "Technology", "Energy", "Materials", "Utilities"]
# Invented names; any resemblance to a real issuer is unintended.
NAMES = {
    "Industrials": ["Aldergrove Rail", "Coldstream Freight", "Quillbrook Machining", "Saltmarsh Aerostructures",
                    "Thornbury Valves", "Wexcombe Fabrication"],
    "Consumer": ["Bracken Foods", "Larchmont Grocers", "Pinecrest Apparel", "Eastgate Outfitters",
                 "Willowbrook Home", "Marrowfield Bakeries"],
    "Technology": ["Cedarline Software", "Foxglove Cloud", "Moraine Data", "Quintessa Analytics",
                   "Brightwater Payments", "Halvorsen Robotics"],
    "Energy": ["Driftwood Energy", "Riverbend Pipelines", "Yellowpoint Solar", "Corvane Gas",
               "Ashcombe Drilling", "Stonefell Refining"],
    "Materials": ["Elkhorn Materials", "Tamarack Chemicals", "Quarry Point Cement", "Inlet Copper",
                  "Deer Lake Paper", "Birchwood Packaging"],
    "Utilities": ["Fernhill Utilities", "Upland Water", "Hollyburn Hydro", "Northcliff Power",
                  "Ravelstone Gas Distribution", "Kettle Bay Transmission"],
}
TEMPLATES = ["margin_streak", "adjusted_roe", "growth_top3", "size_screen", "payout", "split_dps"]
DEV_SEEDS = {t: [101] for t in TEMPLATES}
TEST_SEEDS = {t: [201, 202, 203, 204, 205] for t in TEMPLATES}
N_COMPANIES = 8

README = """# Workspace

All companies are fictional. Amounts are in Canadian dollars.

## Files

- `data/companies.csv`: one row per company: `ticker`, `name`, `sector`, `units`. `units` is the unit of every
  amount the company reports in `financials.csv` and `amendments.csv`: `thousands` or `millions` of dollars.
- `data/financials.csv`: originally filed annual figures, one row per value: `ticker`, `fiscal_year`, `basis`,
  `item`, `value`. `basis` is `consolidated` (the group) or `parent` (the parent company alone). Items:
  `revenue`, `operating_income`, `net_income`, `total_equity` (at year end), `dividends_paid`, and
  `shares_outstanding` (at year end, in millions of shares for every company, whatever its `units`).
- `data/amendments.csv`: restatements filed after the original statements: `ticker`, `fiscal_year`, `basis`,
  `item`, `restated_value`, `filed`. A restated value replaces the original value for that company, year,
  basis and item.
- `data/notes/<TICKER>_FY2025.md`: management notes to the fiscal 2025 statements. Amounts in the notes are
  written out in words and figures (for example "$12.5 million") and are not in the company's reporting unit.
- `data/corporate_actions.csv`: share splits: `ticker`, `effective`, `action`, `ratio` (new shares per old
  share).
"""

FORMAT = """
Write your answer to `result.json` in the workspace root, exactly in this format:
{fmt}
Use consolidated figures unless the task says otherwise, and apply every restatement in `data/amendments.csv`."""


@dataclass
class Company:
    ticker: str
    name: str
    sector: str
    units: str
    # true consolidated values in dollars (after restatement), by item then year
    cons: dict[str, dict[int, float]]
    parent: dict[str, dict[int, float]]
    one_offs: list[dict] = field(default_factory=list)   # FY2025 after-tax items: {"kind": gain|charge, "amount"}
    prior_one_off: dict | None = None                    # an FY2024 item mentioned in the FY2025 note
    split: float | None = None                           # ratio, effective in FY2025


@dataclass
class Task:
    tid: str
    template: str
    seed: int
    split: str
    question: str
    answer_format: str
    gold: object
    kind: str                                            # set | list | dict | fields
    tol: dict = field(default_factory=dict)              # tolerance per value
    traps: dict = field(default_factory=dict)            # trap name -> answer that trap produces
    files: dict = field(default_factory=dict)            # relative path -> text

    @property
    def prompt(self) -> str:
        return self.question + FORMAT.format(fmt=self.answer_format)


# ------------------------------------------------------------------ company generation

def _ticker(name: str, used: set[str]) -> str:
    letters = "".join(w[0] for w in name.split()).upper()
    base = (letters + name.replace(" ", "").upper()[1:])[:4]
    t = base
    i = 0
    while t in used:
        i += 1
        t = base[:3] + "ABCDEFGHJKLMNPQRSTUVWXYZ"[i % 24]
    used.add(t)
    return t


def _company(rng: random.Random, name: str, sector: str, used: set[str]) -> Company:
    units = rng.choice(["thousands", "millions"])
    rev = {2023: rng.uniform(300e6, 4.0e9)}
    for y in (2024, 2025):
        rev[y] = rev[y - 1] * (1 + rng.uniform(-0.08, 0.22))
    base_m = rng.uniform(0.06, 0.24)
    opm = {y: max(0.01, base_m + rng.uniform(-0.04, 0.04)) for y in YEARS}
    oi = {y: rev[y] * opm[y] for y in YEARS}
    ni = {y: oi[y] * rng.uniform(0.55, 0.8) for y in YEARS}
    if rng.random() < 0.15:
        ni[2025] = -rev[2025] * rng.uniform(0.01, 0.05)
    eq = {2023: rev[2023] * rng.uniform(0.5, 1.4)}
    payout = rng.choice([0.0, rng.uniform(0.2, 0.7)])
    div = {y: max(0.0, ni[y]) * payout if payout else 0.0 for y in YEARS}
    for y in (2024, 2025):
        eq[y] = eq[y - 1] + ni[y] - div[y]
    shares = {2023: rng.uniform(20e6, 150e6)}
    split = None
    if rng.random() < 0.3:
        split = rng.choice([2.0, 3.0])
    for y in (2024, 2025):
        shares[y] = shares[y - 1] * (1 + rng.uniform(-0.02, 0.03))
    if split:
        shares[2025] *= split
    cons = {"revenue": rev, "operating_income": oi, "net_income": ni, "total_equity": eq, "dividends_paid": div,
            "shares_outstanding": shares}
    parent = {k: {y: v * rng.uniform(0.3, 0.85) for y, v in d.items()} for k, d in cons.items()
              if k != "shares_outstanding"}
    parent["shares_outstanding"] = dict(shares)
    one_offs = []
    for _ in range(rng.choice([0, 1, 1, 2])):
        one_offs.append({"kind": rng.choice(["gain", "charge"]), "amount": round(rng.uniform(4, 60), 1) * 1e6})
    prior = {"kind": rng.choice(["gain", "charge"]), "amount": round(rng.uniform(4, 40), 1) * 1e6} \
        if rng.random() < 0.5 else None
    return Company(_ticker(name, used), name, sector, units, cons, parent, one_offs, prior, split)


def _companies(rng: random.Random, n: int = N_COMPANIES, sector_min: str | None = None) -> list[Company]:
    used: set[str] = set()
    pool = [(s, nm) for s in SECTORS for nm in NAMES[s]]
    rng.shuffle(pool)
    picked = pool[:n]
    if sector_min:   # make sure one sector has at least three companies
        same = [p for p in pool if p[0] == sector_min][:3]
        picked = same + [p for p in pool if p not in same][: n - 3]
        rng.shuffle(picked)
    return [_company(rng, nm, s, used) for s, nm in picked]


# ------------------------------------------------------------------ file rendering

def _fmt_value(v: float, units: str) -> str:
    x = v / UNITS[units]
    return f"{x:.1f}" if units == "millions" else f"{x:.0f}"


def _parse(s: str, units: str) -> float:
    return float(s) * UNITS[units]


def _words(amount: float) -> str:
    return f"${amount / 1e6:.1f} million"


def _note(rng: random.Random, c: Company) -> str:
    lines = [f"# {c.name}: notes to the fiscal 2025 financial statements", ""]
    lines.append(rng.choice([
        "Demand in our core markets was steady through the year, and we continued to invest in capacity.",
        "The year was shaped by input-cost pressure in the first half and a recovery in volumes in the second.",
        "We completed the integration of last year's acquisitions and simplified our reporting segments.",
    ]))
    lines.append("")
    lines.append("## Items affecting comparability")
    lines.append("")
    if not c.one_offs:
        lines.append("There were no unusual items in fiscal 2025.")
    for o in c.one_offs:
        pre_tax = o["amount"] / rng.uniform(0.7, 0.8)
        if o["kind"] == "gain":
            lines.append(f"Net income for fiscal 2025 includes a one-time after-tax gain of {_words(o['amount'])} "
                         f"(pre-tax: {_words(pre_tax)}) on the sale of a non-core property.")
        else:
            lines.append(f"Net income for fiscal 2025 includes an after-tax restructuring charge of {_words(o['amount'])} "
                         f"(pre-tax: {_words(pre_tax)}) related to the closure of two facilities.")
    if c.prior_one_off:
        p = c.prior_one_off
        what = "a one-time after-tax gain" if p["kind"] == "gain" else "an after-tax impairment charge"
        lines.append(f"For comparison, fiscal 2024 net income included {what} of {_words(p['amount'])}, "
                     "which did not recur in fiscal 2025.")
    lines.append("")
    lines.append("## Outlook")
    lines.append("")
    lines.append(rng.choice([
        "We expect capital spending of roughly the same size as this year.",
        "We plan to keep the dividend policy unchanged.",
        "Management will review the capital structure during the coming year.",
    ]))
    return "\n".join(lines) + "\n"


def _csv(rows: list[list]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()


@dataclass
class Book:
    """What the workspace says: originally filed values (in dollars) and the restatements."""
    companies: list[Company]
    filed: dict[tuple[str, int, str, str], float]          # (ticker, year, basis, item) -> dollars, as filed
    amended: dict[tuple[str, int, str, str], float]        # restated values, dollars

    def value(self, ticker, year, item, basis="consolidated", amend=True) -> float:
        k = (ticker, year, basis, item)
        return self.amended[k] if amend and k in self.amended else self.filed[k]


def _book(rng: random.Random, cs: list[Company], amend_keys: list[tuple[str, int, str]]) -> Book:
    """Rounds every value to what the files show, then files an original value that differs from the true one
    for each key in amend_keys (the restatement carries the true value)."""
    filed, amended = {}, {}
    for c in cs:
        for basis, d in (("consolidated", c.cons), ("parent", c.parent)):
            for item, by_year in d.items():
                for y, v in by_year.items():
                    unit = "millions" if item == "shares_outstanding" else c.units
                    s = f"{v / 1e6:.2f}" if item == "shares_outstanding" else _fmt_value(v, unit)
                    filed[(c.ticker, y, basis, item)] = float(s) * (1e6 if item == "shares_outstanding" else UNITS[unit])
    for t, y, item in amend_keys:
        c = next(c for c in cs if c.ticker == t)
        k = (t, y, "consolidated", item)
        true = filed[k]
        wrong = true * (1 + rng.choice([-1, 1]) * rng.uniform(0.04, 0.15))
        filed[k] = float(_fmt_value(wrong, c.units)) * UNITS[c.units]
        amended[k] = true
    return Book(cs, filed, amended)


def _render(rng: random.Random, book: Book) -> dict[str, str]:
    cs = book.companies
    files = {"README.md": README}
    files["data/companies.csv"] = _csv([["ticker", "name", "sector", "units"]] +
                                       [[c.ticker, c.name, c.sector, c.units] for c in cs])
    rows = [["ticker", "fiscal_year", "basis", "item", "value"]]
    for c in cs:
        for basis in ("consolidated", "parent"):
            for item in ("revenue", "operating_income", "net_income", "total_equity", "dividends_paid",
                         "shares_outstanding"):
                for y in YEARS:
                    v = book.filed[(c.ticker, y, basis, item)]
                    s = f"{v / 1e6:.2f}" if item == "shares_outstanding" else _fmt_value(v, c.units)
                    rows.append([c.ticker, y, basis, item, s])
    body = rows[1:]
    rng.shuffle(body)
    files["data/financials.csv"] = _csv([rows[0]] + body)
    arows = [["ticker", "fiscal_year", "basis", "item", "restated_value", "filed"]]
    for (t, y, basis, item), v in sorted(book.amended.items()):
        c = next(c for c in cs if c.ticker == t)
        arows.append([t, y, basis, item, _fmt_value(v, c.units), f"2026-0{rng.randint(2, 6)}-{rng.randint(10, 28)}"])
    files["data/amendments.csv"] = _csv(arows)
    for c in cs:
        files[f"data/notes/{c.ticker}_FY2025.md"] = _note(rng, c)
    crows = [["ticker", "effective", "action", "ratio"]]
    for c in cs:
        if c.split:
            crows.append([c.ticker, f"2025-0{rng.randint(3, 9)}-15", "split", f"{c.split:g}"])
    files["data/corporate_actions.csv"] = _csv(crows)
    return files


# ------------------------------------------------------------------ templates

def _opm(book, t, y, amend=True, basis="consolidated"):
    return book.value(t, y, "operating_income", basis, amend) / book.value(t, y, "revenue", basis, amend)


def _margin_streak(rng, seed):
    cs = _companies(rng)
    keys = []
    for c in rng.sample(cs, 3):
        keys.append((c.ticker, rng.choice([2024, 2025]), rng.choice(["operating_income", "revenue"])))
    book = _book(rng, cs, keys)

    def streak(amend=True, basis="consolidated"):
        return sorted(c.ticker for c in cs if _opm(book, c.ticker, 2024, amend, basis) > _opm(book, c.ticker, 2023, amend, basis)
                      and _opm(book, c.ticker, 2025, amend, basis) > _opm(book, c.ticker, 2024, amend, basis))
    gold = streak()
    traps = {"no_restatement": streak(amend=False), "parent_basis": streak(basis="parent")}
    if not 1 <= len(gold) <= 5:
        return None
    q = ("Which companies improved their consolidated operating margin (operating income divided by revenue) in "
         "both fiscal 2024 and fiscal 2025, each compared with the year before?")
    return book, q, '{"tickers": ["ABCD", "EFGH"]}  (tickers sorted alphabetically; [] if none)', gold, "set", {}, traps


def _adjusted_roe(rng, seed):
    sector = rng.choice(SECTORS)
    cs = _companies(rng, sector_min=sector)
    members = [c for c in cs if c.sector == sector]
    for c in members:   # every member has a FY2025 item, and at least one mentions a prior-year item
        if not c.one_offs:
            c.one_offs = [{"kind": rng.choice(["gain", "charge"]), "amount": round(rng.uniform(5, 60), 1) * 1e6}]
        if c.cons["net_income"][2025] <= 0:
            c.cons["net_income"][2025] = c.cons["revenue"][2025] * 0.06
    if not any(c.prior_one_off for c in members):
        members[0].prior_one_off = {"kind": "gain", "amount": round(rng.uniform(5, 40), 1) * 1e6}
    if not any(c.units == "thousands" for c in members):
        members[0].units = "thousands"
    keys = [(rng.choice(members).ticker, 2025, rng.choice(["net_income", "total_equity"]))]
    keys.append((rng.choice([c for c in cs if c not in members]).ticker, 2025, "net_income"))
    book = _book(rng, cs, keys)

    def roe(c, amend=True, adjust=True, prior=False, unit_slip=False):
        ni = book.value(c.ticker, 2025, "net_income", amend=amend)
        adj = 0.0
        if adjust:
            for o in c.one_offs + ([c.prior_one_off] if prior and c.prior_one_off else []):
                amt = o["amount"]
                if unit_slip:   # note amounts read as if they were in the company's reporting unit
                    amt = amt / 1e6 * UNITS[c.units]
                adj += -amt if o["kind"] == "gain" else amt
        eq = (book.value(c.ticker, 2024, "total_equity", amend=amend) + book.value(c.ticker, 2025, "total_equity", amend=amend)) / 2
        return round(100 * (ni + adj) / eq, 1)
    gold = {c.ticker: roe(c) for c in members}
    traps = {"no_restatement": {c.ticker: roe(c, amend=False) for c in members},
             "unadjusted": {c.ticker: roe(c, adjust=False) for c in members},
             "prior_year_item": {c.ticker: roe(c, prior=True) for c in members},
             "unit_slip": {c.ticker: roe(c, unit_slip=True) for c in members}}
    q = (f"For every company in the {sector} sector, compute fiscal 2025 adjusted return on equity: net income "
         "excluding the after-tax items affecting comparability that the fiscal 2025 notes report for fiscal 2025 "
         "(take out one-time gains, add back charges), divided by the average of fiscal 2024 and fiscal 2025 "
         "year-end total equity. Report it in percent, rounded to one decimal.")
    return (book, q, '{"roe": {"ABCD": 12.3, "EFGH": 8.0}}  (one entry per company in the sector)', gold, "dict",
            {"*": 0.15}, traps)


def _growth_top3(rng, seed):
    cs = _companies(rng)

    def growth(book, t, amend=True, basis="consolidated"):
        return book.value(t, 2025, "revenue", basis, amend) / book.value(t, 2024, "revenue", basis, amend) - 1
    keys = [(c.ticker, rng.choice([2024, 2025]), "revenue") for c in rng.sample(cs, 3)]
    book = _book(rng, cs, keys)

    def top(amend=True, basis="consolidated"):
        return [c.ticker for c in sorted(cs, key=lambda c: -growth(book, c.ticker, amend, basis))[:3]]
    gold = top()
    gs = sorted((growth(book, c.ticker) for c in cs), reverse=True)
    if gs[2] - gs[3] < 0.002:
        return None
    traps = {"no_restatement": top(amend=False), "parent_basis": top(basis="parent")}
    q = ("Which three companies had the fastest consolidated revenue growth in fiscal 2025 compared with "
         "fiscal 2024? Rank them from fastest to slowest.")
    return book, q, '{"top3": ["ABCD", "EFGH", "IJKL"]}  (fastest first)', gold, "list", {}, traps


def _size_screen(rng, seed):
    cs = _companies(rng)
    if len({c.units for c in cs}) < 2:
        return None
    true = sorted(c.cons["revenue"][2025] / 1e6 for c in cs)
    cut = math.floor((true[3] + true[4]) / 2 / 10) * 10
    # restate the two companies nearest the cut, so a missed restatement can move one across it
    near = sorted(cs, key=lambda c: abs(c.cons["revenue"][2025] / 1e6 - cut))[:2]
    keys = [(c.ticker, 2025, "revenue") for c in near]
    book = _book(rng, cs, keys)
    revs = sorted(book.value(c.ticker, 2025, "revenue") / 1e6 for c in cs)
    if not revs[3] < cut < revs[4]:
        return None

    def screen(amend=True, unit_slip=None):
        out = []
        for c in cs:
            v = book.value(c.ticker, 2025, "revenue", amend=amend)
            if unit_slip:   # every value read as if it were in the same unit
                v = v / UNITS[c.units] * UNITS[unit_slip]
            if v / 1e6 >= cut:
                out.append(c.ticker)
        return sorted(out)
    gold = screen()
    traps = {"no_restatement": screen(amend=False), "all_thousands": screen(unit_slip="thousands"),
             "all_millions": screen(unit_slip="millions")}
    q = f"Which companies reported consolidated revenue of at least ${cut:,} million in fiscal 2025?"
    return book, q, '{"tickers": ["ABCD", "EFGH"]}  (tickers sorted alphabetically)', gold, "set", {}, traps


def _payout(rng, seed):
    cs = _companies(rng)
    payers = [c for c in cs if c.cons["dividends_paid"][2025] > 0]
    if len(payers) < 3:
        return None
    if not any(c.cons["net_income"][2025] <= 0 for c in payers):   # a dividend payer with a loss, so the rule matters
        c = rng.choice(payers)
        c.cons["net_income"][2025] = -c.cons["revenue"][2025] * rng.uniform(0.01, 0.04)
    profitable = [c for c in payers if c.cons["net_income"][2025] > 0]
    if not profitable:
        return None
    keys = [(rng.choice(profitable).ticker, 2025, rng.choice(["dividends_paid", "net_income"]))]
    book = _book(rng, cs, keys)

    def table(amend=True, keep_losses=False):
        out = {}
        for c in cs:
            d = book.value(c.ticker, 2025, "dividends_paid", amend=amend)
            n = book.value(c.ticker, 2025, "net_income", amend=amend)
            if d > 0 and (n > 0 or keep_losses):
                out[c.ticker] = round(100 * d / n, 1)
        return out
    gold = table()
    if len(gold) < 2:
        return None
    traps = {"no_restatement": table(amend=False), "kept_loss_makers": table(keep_losses=True)}
    q = ("Compute the fiscal 2025 dividend payout ratio (consolidated dividends paid divided by consolidated net "
         "income), in percent rounded to one decimal, for every company that paid dividends in fiscal 2025. "
         "Leave out companies with a net loss in fiscal 2025.")
    return book, q, '{"payout": {"ABCD": 45.2, "EFGH": 30.0}}', gold, "dict", {"*": 0.15}, traps


def _split_dps(rng, seed):
    cs = _companies(rng)
    payers = [c for c in cs if c.cons["dividends_paid"][2024] > 0 and c.cons["dividends_paid"][2025] > 0]
    if not payers:
        return None
    target = rng.choice(payers)
    target.split = target.split or rng.choice([2.0, 3.0])
    if target.cons["shares_outstanding"][2025] < target.cons["shares_outstanding"][2024] * 1.5:
        target.cons["shares_outstanding"][2025] *= target.split
        target.parent["shares_outstanding"][2025] = target.cons["shares_outstanding"][2025]
    keys = [(target.ticker, rng.choice([2024, 2025]), "dividends_paid")]
    book = _book(rng, cs, keys)
    t = target.ticker

    def dps(amend=True, adjust=True):
        d24 = book.value(t, 2024, "dividends_paid", amend=amend)
        d25 = book.value(t, 2025, "dividends_paid", amend=amend)
        s24 = book.value(t, 2024, "shares_outstanding") * (target.split if adjust else 1)
        s25 = book.value(t, 2025, "shares_outstanding")
        a, b = d24 / s24, d25 / s25
        return {"dps_2024": round(a, 2), "dps_2025": round(b, 2), "change_pct": round(100 * (b / a - 1), 1)}
    gold = dps()
    traps = {"no_restatement": dps(amend=False), "split_ignored": dps(adjust=False)}
    q = (f"For {target.name} ({t}), compute dividends per share for fiscal 2024 and fiscal 2025 (consolidated "
         "dividends paid divided by year-end shares outstanding), with fiscal 2024 restated on the fiscal 2025 "
         "share basis so that a share split during fiscal 2025 does not distort the comparison. Also give the "
         "percent change from fiscal 2024 to fiscal 2025, computed from the unrounded per-share values.")
    fmt = '{"dps_2024": 1.23, "dps_2025": 1.31, "change_pct": 6.5}  (dollars per share to two decimals, change to one)'
    return book, q, fmt, gold, "fields", {"dps_2024": 0.006, "dps_2025": 0.006, "change_pct": 0.15}, traps


BUILDERS = {"margin_streak": _margin_streak, "adjusted_roe": _adjusted_roe, "growth_top3": _growth_top3,
            "size_screen": _size_screen, "payout": _payout, "split_dps": _split_dps}


def _differs(kind, a, b, tol) -> bool:
    return not matches(kind, a, b, tol)


def build_task(template: str, seed: int, split: str) -> Task:
    """Rejection-samples until every planted trap changes the answer."""
    for attempt in range(500):
        rng = random.Random(f"{template}:{seed}:{attempt}")
        out = BUILDERS[template](rng, seed)
        if out is None:
            continue
        book, q, fmt, gold, kind, tol, traps = out
        if not all(_differs(kind, gold, v, tol) for v in traps.values()):
            continue
        files = _render(rng, book)
        tid = f"{template}-{seed}"
        return Task(tid, template, seed, split, q, fmt, gold, kind, tol, traps, files)
    raise RuntimeError(f"could not build {template}:{seed}")


def build_tasks() -> list[Task]:
    tasks = []
    for split, seeds in (("dev", DEV_SEEDS), ("test", TEST_SEEDS)):
        for t in TEMPLATES:
            for s in seeds[t]:
                tasks.append(build_task(t, s, split))
    return tasks


def digest(tasks: list[Task]) -> str:
    h = hashlib.sha256()
    for t in tasks:
        h.update(json.dumps([t.tid, t.prompt, t.gold, sorted(t.files.items())], sort_keys=True).encode())
    return h.hexdigest()[:16]


def materialize(task: Task, root: str | Path) -> Path:
    root = Path(root)
    for rel, text in task.files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    return root


# ------------------------------------------------------------------ scoring

def _num_ok(a, b, tol) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def matches(kind: str, answer, gold, tol: dict) -> bool:
    if kind == "set":
        return isinstance(answer, list) and sorted(map(str, answer)) == sorted(gold)
    if kind == "list":
        return isinstance(answer, list) and [str(x) for x in answer] == list(gold)
    if kind == "dict":
        return (isinstance(answer, dict) and set(answer) == set(gold)
                and all(_num_ok(answer[k], gold[k], tol.get(k, tol.get("*", 0))) for k in gold))
    if kind == "fields":
        return isinstance(answer, dict) and all(_num_ok(answer.get(k), v, tol.get(k, 0)) for k, v in gold.items())
    raise ValueError(kind)


RESULT_KEY = {"margin_streak": "tickers", "size_screen": "tickers", "growth_top3": "top3", "adjusted_roe": "roe",
              "payout": "payout", "split_dps": None}


def read_result(task: Task, workspace: str | Path):
    """The answer the agent left in result.json, or None if missing or unreadable."""
    p = Path(workspace) / "result.json"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    key = RESULT_KEY[task.template]
    return obj if key is None else (obj.get(key) if isinstance(obj, dict) else None)


def score(task: Task, workspace: str | Path) -> dict:
    ans = read_result(task, workspace)
    if ans is None:
        return {"correct": False, "status": "no_result", "trap": None}
    if matches(task.kind, ans, task.gold, task.tol):
        return {"correct": True, "status": "correct", "trap": None}
    trap = next((name for name, v in task.traps.items() if matches(task.kind, ans, v, task.tol)), None)
    return {"correct": False, "status": "wrong", "trap": trap}
