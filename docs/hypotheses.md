# Pre-registered hypotheses

Written and committed before any model ran on the test split. The task generator, the scoring rules, the
agent prompt and this file are frozen at that commit; later changes are listed at the end with a reason.

## Background

AgentBaton (Li et al., arXiv:2610.35959) studies one long agentic task shared by two models: a capable model
works first and hands the rest to an efficient model once, at a suitable point. In the paper,
capable-to-efficient handoffs "at suitable stages of task progress largely preserve or even improve accuracy",
efficient-to-capable handoffs "generally underperform capable-only execution", and a learned 2B detector that
watches for three readiness signs cuts API cost by 65.9% on average over 11 model pairs. The paper's efficient
model is Qwen3.8-27B; its benchmarks are Terminal-Bench 2.1 and DeepSWE.

This study asks whether the capable-to-efficient result holds when the efficient model is far weaker (qwen3:4b
on a 6 GB laptop GPU) and the work is multi-step financial data analysis.

## Setup

* Tasks: 6 templates (margin streak, adjusted ROE, revenue-growth ranking, size screen, payout ratio,
  split-adjusted dividends per share) over workspaces of fictional Canadian issuers with planted traps
  (restatements, mixed reporting units, parent-only rows, prior-year items in notes, share splits). Dev split:
  6 tasks (one per template). Test split: 30 tasks (five per template). Every gold answer is recomputed from
  the workspace files by an independent reference solver in the tests, and every planted trap changes the
  answer.
* Agent: one Python action per turn in a sandboxed interpreter (standard library and numpy), linear history,
  at most 20 turns in total across both models, answer in `result.json`, then SUBMIT.
* Capable sources: Claude Sonnet 5.5 (main pair) and Claude Haiku 5.5 (secondary pair), adaptive thinking at
  effort medium, prompt caching on, no sampling parameters (the API does not accept them), one trajectory per
  task. Efficient receiver and prompt-based judge: qwen3:4b through Ollama, temperature 0, thinking off,
  8,192-token context. Local models cost $0, as in the paper.
* A handoff after turn k replays the source's first k actions in a fresh workspace and checks the file tree
  after every turn against the record; a mismatch is reported and the handoff counts as wrong. The receiver
  then continues with the whole prefix in its context (in-context transfer) or only a log file (log-file
  transfer). The continuation from every possible k is computed, so a policy is a lookup.
* Policies: fixed progress at 25%, 50% and 75% of the source's own trajectory length (uses future information,
  as the paper's fixed-progress baseline does), a rule detector (the paper's three readiness signs read from
  the workspace record: an agent-made file exists, the turn ran without error after the agent had looked at
  the data, the turn opened no new data file), and a prompt detector (qwen3:4b asked after every turn with the
  paper's three signs). A policy that never fires leaves the capable model to finish alone.
* Cost: the source's API cost for turns 1..k; capable-only is the full trajectory's cost.
* Scoring: `result.json` as it stands when the episode ends (after SUBMIT or at the turn limit), compared with
  the gold answer by exact set or order match, or within a stated tolerance for numbers.

## Settings fixed on the dev split

* qwen3:4b runs with an 8,192-token context. At 16,384 Ollama put a fifth of the model on the CPU and one turn
  took minutes; at 8,192 the model sits entirely on the 6 GB GPU. Conversations that do not fit drop their
  oldest turns, keeping the system prompt and the task.
* qwen3:4b agent turns are constrained to a JSON object {thought, action, code} and rendered back into the
  shared THOUGHT-plus-code-block format. Unconstrained, it wrote 1,536 tokens of prose and no code block on two
  consecutive turns.
* The agent's interpreter has no pandas: the Windows application-control policy on this laptop blocks its
  DLLs. The standard library and numpy are available to every model.
* The turn limit was lowered from 30 to 20 after the dev runs: Claude finished every dev task in 5 to 8 turns,
  and qwen3:4b runs that did not finish used all 30.

## What the dev split showed (6 tasks, disclosed before the test run)

Sonnet 5.5 alone 6/6 at $0.025 per task; Haiku 5.5 alone 6/6 at $0.002; qwen3:4b alone 0/6 (it never opened
README.md). Sonnet → qwen3:4b, in-context: fixed 25% 1/6, 50% 2/6, 75% 4/6, rule detector 4/6 at a 15% cost
cut (it fired at 76% of the run on average), prompt detector 3/6. The prompt detector judged NOT_READY for the
first three turns on every dev task and then READY, so it was not degenerate there. Log-file transfer with the
rule detector 5/6. Efficient-to-capable 6/6 at 73% more cost than Sonnet alone. No replay mismatch in 55
replays. The hypotheses below come from the paper; where the dev split already points the other way (the cost
cut in H1, the detector in H4), they are kept as the paper's claims and tested on the test split.

## Hypotheses

**H1 (replication).** For Sonnet 5.5 → qwen3:4b, the rule detector with in-context transfer cuts the mean API
cost per task by at least 40% against capable-only, and is not much less accurate: the lower bound of the 95%
paired bootstrap interval of (handoff − capable-only) accuracy is above −15 points.

**H2 (direction).** Efficient-to-capable (qwen3:4b's own trajectory handed to Sonnet 5.5 at 50% of its length,
in-context) is less accurate than capable-only.

**H3 (timing).** Later handoffs end correct more often: fixed 75% is more accurate than fixed 25% (Sonnet pair,
in-context).

**H4 (prompt detector).** The prompt-based detector is degenerate, as in the paper: on at least 80% of tasks it
fires at the first possible turn or never.

Exploratory, with no prediction: log-file versus in-context transfer, the Haiku 5.5 pair, the handoff curve over
all k, and which traps the receiver falls into.

## Decision rules

* Paired comparisons over the 30 test tasks, 95% bootstrap interval (10,000 resamples of tasks) and an exact
  McNemar test. A difference counts only when the interval excludes zero.
* H1 holds when both parts hold, fails when the accuracy interval lies entirely below zero or the cost cut is
  under 40%, and is inconclusive otherwise.
* Every hypothesis is reported whichever way it comes out.
* The dev split may be used to fix prompts, parsing and settings. Each test configuration runs once; reruns
  only after a crash or a model-server failure, never because of a result.
* API budget: $10 for the whole study, enforced in code before every paid request.

## Changes after registration

(none yet)
