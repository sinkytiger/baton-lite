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

* 2026-11-06, no effect on answers: a Claude refusal now records its `stop_details` category. The one refusal
  in the test run (Sonnet 5.5 taking over adjusted_roe-204 from qwen3:4b, turn 13) happened before this change,
  so its category is unknown; as registered, it ends the episode and counts as wrong.
* 2026-11-06, no effect on answers: the laptop slept from 17:01 to 19:16 during the Sonnet in-context sweep
  (Windows event 42, sleep reason "Button or Lid"); the continuation running then resumed where it was, so only
  its wall-clock time is off. `scripts/stay_awake.py` now holds the system-required execution state against idle
  sleep until the run log shows the end; it does not stop a closed lid or a shutdown, and no power setting was
  changed.
* 2026-11-06, model-server failure: one qwen3:4b request in the Sonnet log-file runs ended with Ollama's "token
  repeat limit reached" error (HTTP 500). As registered, the step was rerun and resumed from the saved turns.
* 2026-11-06 21:08 to 2026-11-07 17:29, interruption: Windows shut down (fast-startup shutdown: Kernel-Power 42
  with a user logoff; the log does not say who started it) during the Sonnet log-file runs, which ended every
  process. The run resumed from the files on disk.
* 2026-11-07, server build change: starting the Ollama app after the shutdown triggered its auto-updater, which
  left Ollama 0.40.1 running. The earlier runs most likely used 0.40.14, the build the app reported on the
  morning of 2026-11-07. Asking the new server again for three recorded qwen3:4b turns (same messages, same
  options) gave three different replies; whether the old server would have repeated itself was not tested. Every
  registered hypothesis (H1 to H4) uses runs that finished before the change. The Sonnet log-file runs for
  margin_streak-201 to 205 and adjusted_roe-201 to 204 came from the old server and the other Sonnet log-file runs
  from 0.40.1, so log-file versus in-context (exploratory) straddles the change.
* 2026-11-07 22:32 to 2026-11-08 19:14, interruption: a second Windows fast-startup shutdown of the same kind,
  during the Haiku 5.5 in-context sweep. The run resumed from the files on disk on the same Ollama build (0.40.1).
* 2026-11-08 21:53 to 2026-11-09 08:13, interruption and second build change: a third shutdown of the same kind,
  near the end of the Haiku 5.5 in-context sweep. At the next login the Ollama app installed 0.40.15 (07:18). The
  Haiku detector and in-context runs up to split_dps-204 came from 0.40.1; the remaining in-context runs and all
  Haiku log-file runs come from 0.40.15, started with `ollama serve` so that the app's updater cannot change it
  again. The Haiku pair (exploratory) therefore also straddles a build change.
* 2026-11-09, repeatability within one install: asking the Ollama install of 2026-11-09 (see the correction
  below) again for ten qwen3:4b receiver turns it had recorded earlier the same day (the first two receiver turns
  of five Haiku log-file continuations, same messages, same options) gave 3 identical replies and 7 different ones
  (scripts/repeat_check.py, output in docs/repeatability_2026-11-09.txt). Temperature 0 with a fixed seed does not
  make this setup reliably repeat itself, so the three differences seen after the 2026-11-07 build change cannot be
  put down to the build: they are consistent with run-to-run variation. Each continuation in this study is one draw.
* 2026-11-09, model-server rule, Haiku log-file runs only: in the Haiku log-file continuation from payout-202 at
  k = 4, Ollama aborted a qwen3:4b reply with "token repeat limit reached" on every one of 30 tries (5 tries per
  request, 6 runs of the step), and the step stopped. Rerunning, the registered response to a server failure,
  could not get past it. Such an abort is the model's own reply cut off for repeating itself, so from then on it
  counts as an empty reply, reported to the agent as a format error, after which the episode goes on and the turn
  counts toward the 20-turn limit. A reply cut off at the 1,536-token limit was already a format error, though it
  keeps its partial text. The rule touches
  only the Haiku log-file continuations that ran after it (payout-202 from k = 4, payout-203 to 205 and
  split_dps-201 to 205); no earlier run is changed. The one earlier abort (Sonnet log-file runs, 2026-11-06)
  succeeded when the step was rerun. In the end the rule never applied: rerun with it in place, the payout-202
  continuation from k = 4 finished without an abort, and no later reply was aborted (0 aborted turns recorded).
* 2026-11-09, corrections to the run's history above, found by an independent audit and checked against the Windows
  and Ollama logs. The laptop also slept a second time on 2026-11-06, for under a minute around 19:16, with the same
  logged reason. On 2026-11-07 the updater passed through 0.40.15 and 0.40.7 within five minutes before stopping at
  0.40.1; neither received a model (chat) request. Of the 113 Sonnet log-file runs, 82 (on 21 tasks) came from
  0.40.1. The 2026-11-09 07:18 install of 0.40.15 did not complete: upgrade.log records "DeleteFile failed; code 5",
  "User canceled the installation process" and "Rolling back changes". ollama.exe and the app report 0.40.15, but
  lib/ollama (llama-server.exe, libllama.dll, ggml-base.dll, cuda_v12/ggml-cuda.dll) holds files dated 2026-10-08,
  and the two llama-server crashes in the Windows log that day carry that build's time stamp, so these are most
  likely the files 0.40.1 installed. The runs after 08:13 on 2026-11-09, including the repeatability check, used
  this mixed install. The Windows log also shows four llama-server crashes during the run (2026-11-06 16:29 and
  20:30, 2026-11-09 10:02 and 10:22). The client retried failed requests up to five times without recording them,
  so the run log shows only failures that stopped a step.
* 2026-11-09, corrections to how this file describes the paper, found by an independent audit after the test run
  and checked against the paper's text. The tests, thresholds and verdicts do not change; the text above stays as
  registered.
  * Background: the paper's efficient model is not only Qwen3.8-27B. Section 3 uses it alone, but the 65.9% comes
    from section 4.3, where GPT-5.6 Luna, a cloud model whose cost is counted, also takes over. The detector is
    initialized from Qwen 3.5 2B and trained on handoff points annotated with Codex, using criteria that follow
    the three readiness signs.
  * H4: "degenerate, as in the paper" is wrong. The paper prompts the capable model itself, not a separate judge,
    and reports that Kimi K3 never handed off, GLM-5.3 did on 6 of 113 tasks and GPT-5.6 Sol did on every task at
    $3.75 against $5.32 alone. It reports no first-turn handoffs and does not use the word "degenerate". H4 here is
    this study's own test of a qwen3:4b judge.
  * H3 is a reading of the paper's progress tables, not a claim the paper states. The paper says readiness depends
    on more than elapsed turns and that on Terminal-Bench "later handoffs do not monotonically improve accuracy".
  * H1 is called a replication, but it tests a hand-written stand-in for the three signs, not the learned detector.
    The stand-in is narrower than the paper's signs: the paper also accepts a runnable prototype as the artifact and
    counts a failed check that exposes a localized problem as feedback.
