# baton-lite

**Can an expensive model start a financial-analysis task and hand the rest to a 4B model on a laptop?** A small
check of AgentBaton ([Li et al., arXiv:2610.35959](https://arxiv.org/abs/2610.35959)) with Claude Sonnet 5.5 and
Haiku 5.5 as the capable models and qwen3:4b, running locally on a 6 GB GPU, as the model that takes over.

> Status: finished. Test run on 30 tasks, 2026-11-06 to 2026-11-09: $1.72 of API calls, $2.14 with the 6 dev tasks.
> All companies in the data are fictional. Nothing here is financial advice.

## Findings

Main pair, Claude Sonnet 5.5 → qwen3:4b, 30 test tasks, one run per task:

1. **The handoff missed the pre-registered bar.** A file-based rule standing in for the paper's three readiness
   signs kept 26 of 30 tasks correct (Sonnet alone: 30 of 30) and cut API cost by only 16%. The pre-registered
   hypothesis (at least a 40% cut, with little accuracy lost) fails on both counts.
2. **The rule only fired once the answer was written.** The only file Sonnet made was `result.json`, on every
   task, so the rule's "an agent-made file exists" meant "the answer exists". On all 30 tasks the workspace at
   the handoff was identical to Sonnet's final one. The four losses are qwen3:4b redoing the work and overwriting
   a correct answer.
3. **Before the answer existed, the 4B model usually got it wrong.** Over every possible point before the answer
   was written, 30 of 110 in-context continuations ended correct. Log-file runs covered only the points a policy
   picked: 1 of 80 ended correct there, against 26 of the same 80 in-context.
4. **Later handoffs ended correct more often** (pre-registered, supported): 8 of 30 at 25% of the run, 18 of 30
   at 75%.
5. **Handoffs that ended correct existed, but the detectors did not pick them reliably.** In hindsight, taking the
   earliest handoff that ended correct gives 30 of 30 at 50% less cost: a handoff on 29 tasks, and Sonnet alone on
   split_dps-202, where none ended correct. It is picked from the outcomes of single draws, so it is optimistic.
   The rule detector got 26 of 30 but cut only 16%; the prompt detector cut 47% but got 16 of 30.
6. **The reverse direction cost more.** qwen3:4b starting and Sonnet taking over at 50% ended correct on 29 of 30
   tasks at 52% more cost than Sonnet alone. The one miss was a refusal by Sonnet.
7. **A cheaper API model on its own cost far less and lost little.** Haiku 5.5 alone solved 29 of 30 at $0.0018
   per task, 8% of Sonnet's cost. That is cheaper than every Sonnet → qwen3:4b handoff rule and at least as accurate
   (+10 pts against the rule detector, interval -3.3 to +23.3, so not a settled difference). This comparison was
   not pre-registered.

Secondary pair, Haiku 5.5 → qwen3:4b (exploratory): the same pattern. The rule fired with only SUBMIT left on all
30 tasks, ended correct on 27 of 30 and cut cost by 12%.

## The question

AgentBaton lets a capable model work on a long agentic task and hand it, once, to a cheaper model at a suitable
point. The paper finds that capable-to-efficient handoffs at suitable stages "largely preserve or even improve
accuracy", while efficient-to-capable handoffs "generally underperform capable-only execution" with limited or
no cost reduction. Its learned 2B detector cuts API cost by 65.9% on average across 11 model pairs and two
benchmarks. The model taking over in those pairs is Qwen3.8-27B (counted at $0 as locally deployable) or GPT-5.6
Luna (a cloud model). This repository asks what happens when the model taking over is much smaller, a 4B model
that costs nothing to run, and the work is multi-step financial data analysis.

## What is compared

Each task is a workspace of fictional Canadian issuers: a data dictionary, consolidated and parent-only
statements, filed restatements, year-end notes and share splits. The agent explores the files, writes Python and
saves `result.json`. Six templates (margin streaks, adjusted ROE, revenue-growth ranking, size screens, payout
ratios, split-adjusted dividends per share) each plant traps, and every trap changes the answer.

| Condition | What happens |
|---|---|
| capable only | Sonnet 5.5 (or Haiku 5.5) solves the task alone |
| efficient only | qwen3:4b solves it alone |
| fixed 25 / 50 / 75% | the capable model hands off after that share of its own run |
| rule detector | hands off at the first turn where a file-based stand-in for the paper's three readiness signs holds |
| prompt detector | qwen3:4b, given the three signs, reads the run after every turn and says HANDOFF_READY or not |
| hindsight best | the earliest handoff point that ended correct, or the capable model alone if none did (analysis only, uses the outcome) |
| reverse | qwen3:4b starts, Sonnet 5.5 takes over at 50% |

The receiver gets the prefix either in its context or as a log file, as in the paper. A handoff replays the
recorded prefix in a fresh workspace and checks the file tree after every turn, so the continuation starts from
exactly the state the capable model left. In-context, the continuation from every possible handoff point was run,
so each policy is a lookup; log-file runs cover the points the policies picked. Cost is the capable model's API
cost up to the handoff; the local model costs $0, as in the paper. In the reverse condition the cost is Sonnet's
after it takes over.

The hypotheses and decision rules were committed before the test run: [docs/hypotheses.md](docs/hypotheses.md).

## Results

### Sonnet 5.5 → qwen3:4b

| Condition | Transfer | Correct | API $ per task | Cost vs Sonnet alone | Handoff at (share of turns) | vs Sonnet alone, pts (95% CI) |
|---|---|---|---|---|---|---|
| Sonnet 5.5 alone | - | 30/30 | 0.0220 | - | - | - |
| qwen3:4b alone | - | 0/30 | 0 | -100% | - | -100.0 |
| fixed 25% | in-context | 8/30 | 0.0032 | -85% | 28% | -73.3 (-86.7 to -56.7) |
| fixed 50% | in-context | 12/30 | 0.0092 | -58% | 49% | -60.0 (-76.7 to -43.3) |
| fixed 75% | in-context | 18/30 | 0.0161 | -27% | 72% | -40.0 (-56.7 to -23.3) |
| rule detector | in-context | 26/30 | 0.0185 | -16% | 78% | -13.3 (-26.7 to -3.3) |
| prompt detector | in-context | 16/30 | 0.0117 | -47% | 58% | -46.7 (-63.3 to -30.0) |
| fixed 50% | log file | 2/30 | 0.0092 | -58% | 49% | -93.3 (-100.0 to -83.3) |
| rule detector | log file | 25/30 | 0.0185 | -16% | 78% | -16.7 (-30.0 to -3.3) |
| prompt detector | log file | 9/30 | 0.0117 | -47% | 58% | -70.0 (-86.7 to -53.3) |
| hindsight best | in-context | 30/30 | 0.0110 | -50% | 55% (29 tasks) | +0.0 |
| reverse at 50% | in-context | 29/30 | 0.0335 | +52% | - | -3.3 (-10.0 to +0.0) |

Intervals are 95% paired bootstrap intervals over tasks (10,000 resamples). Full tables with Wilson intervals and
McNemar p-values: [reports/results.md](reports/results.md). One row per recorded test-split run, including the
log-file runs at fixed 25% and 75% that the tables leave out (Sonnet: 0 and 15 of 30):
[reports/outcomes.csv](reports/outcomes.csv).

### Pre-registered hypotheses

| | Hypothesis | Result | Verdict |
|---|---|---|---|
| H1 | Rule detector, in-context: at least 40% cheaper, accuracy loss interval above -15 pts | 16% cheaper; -13.3 pts (-26.7 to -3.3) | fails |
| H2 | Reverse direction less accurate than Sonnet alone | -3.3 pts (-10.0 to +0.0); interval includes 0 | not supported |
| H3 | Fixed 75% more accurate than fixed 25% | +33.3 pts (+13.3 to +53.3), McNemar p = 0.006 | supported |
| H4 | Prompt detector degenerate (turn 1 or never on at least 80% of tasks) | 0 of 30 tasks | fails |

The pre-registration calls H4 "as in the paper". That was my misreading: the paper prompts the capable model
itself to hand off, not a separate judge, and reports that Kimi K3 never did, GLM-5.3 did on 6 of 113 tasks and
GPT-5.6 Sol did on every task. The test and its threshold stand as registered; the correction is logged in
[docs/hypotheses.md](docs/hypotheses.md#changes-after-registration).

### Why the rule detector looks accurate but saves little

The rule detector is a hand-written stand-in for the paper's three signs, read from the workspace record: an
agent-made file exists, a turn ran without error after the agent had looked at the data, and the turn opened no
new data file. It is narrower than the paper, which also accepts a runnable prototype as the artifact and counts
a failed check that exposes a localized problem as feedback. Here the agent's code ran inline and was not saved,
and the only file it made was the answer. So the rule fired after the answer was final on all 30 tasks, with only
SUBMIT left on 22 of them. What was left for qwen3:4b was to submit. It rewrote `result.json` on 11 tasks, and on 4
of them the new answer was wrong.

The looser reading, as judged by qwen3:4b, did not rescue the handoff here. The prompt detector was given the
broader signs (a working script or computed intermediate results count as the artifact, a failed run can count as
feedback) and handed over earlier, at 58% of the turns, but ended correct on only 16 of 30.

The cost cut is smaller than the share of turns handed over because the turns left are the cheap ones: after the
rule fires Sonnet only checks and submits, about $0.0025 a turn against $0.0039 before. The rule detector handed
over after 78% of Sonnet's turns, which had used 84% of its cost.

### When the handoff happens

Every continuation, by how far into Sonnet's run it started (in-context):

| Handoff after | Continuations | Ended correct |
|---|---|---|
| 0–25% of the run | 29 | 13.8% |
| 25–50% | 40 | 27.5% |
| 50–75% | 51 | 47.1% |
| 75–100% | 32 | 90.6% |

Most of the rise comes from the answer already being written. Continuations that started after Sonnet's final
`result.json` stayed correct 38 of 42 times; those that started before it ended correct 30 of 110 times.

### Log file versus in-context

With only a log file to read, qwen3:4b did worse: fixed 50% fell from 12 to 2 correct (-33.3 pts, -50.0 to
-16.7), the prompt detector from 16 to 9 (-23.3 pts, -40.0 to -6.7). Before the answer was written, 1 of 80 log-file
runs ended correct against 26 of the same 80 in-context, and 69 of the 80 ran out of turns. The rule detector, which
hands over a finished answer, stayed level in total (26 to 25), though 7 tasks flipped (3 up, 4 down). Most of the
Sonnet log-file runs (82 of 113, on 21 of 30 tasks) ran on a different Ollama build from all the Sonnet in-context
runs (see below). The comparison was registered as exploratory, and this adds a further caveat.

### Where qwen3:4b went wrong

Alone, qwen3:4b solved none of the 30 tasks: it wrote a wrong answer on 24 and no answer on 6, and it opened the
data dictionary (`README.md` in the workspace) on 1 task. Of the 84 in-context continuations that did not end
correct, 27 left no usable answer (23 never wrote one; 4 wrote a `result.json` the scorer could not use, 3 without
the required key and 1 left empty), 10 gave exactly
the answer one planted trap produces (the no-restatement answer 7 times, a unit mix-up answer 3 times; a match
shows the answer, not the cause) and 47 gave some other wrong answer.

### Haiku 5.5 → qwen3:4b (exploratory)

| Condition | Transfer | Correct | API $ per task | Cost vs Haiku alone | Handoff at (share of turns) | vs Haiku alone, pts (95% CI) |
|---|---|---|---|---|---|---|
| Haiku 5.5 alone | - | 29/30 | 0.0018 | - | - | - |
| fixed 25% | in-context | 7/30 | 0.0002 | -88% | 26% | -73.3 (-86.7 to -56.7) |
| fixed 50% | in-context | 7/30 | 0.0006 | -69% | 46% | -73.3 (-86.7 to -56.7) |
| fixed 75% | in-context | 19/30 | 0.0013 | -29% | 74% | -33.3 (-50.0 to -16.7) |
| rule detector | in-context | 27/30 | 0.0016 | -12% | 82% | -6.7 (-16.7 to +0.0) |
| prompt detector | in-context | 11/30 | 0.0009 | -51% | 60% (29 tasks) | -60.0 (-76.7 to -43.3) |
| fixed 50% | log file | 0/30 | 0.0006 | -69% | 46% | -96.7 (-100.0 to -90.0) |
| rule detector | log file | 25/30 | 0.0016 | -12% | 82% | -13.3 (-26.7 to -3.3) |
| prompt detector | log file | 5/30 | 0.0009 | -51% | 60% (29 tasks) | -80.0 (-93.3 to -66.7) |
| hindsight best | in-context | 29/30 | 0.0011 | -38% | 59% (27 tasks) | +0.0 |

Haiku alone missed one task, adjusted_roe-204. Haiku needed 5 to 7 turns, and the rule fired with only SUBMIT left
on all 30 tasks. Of its three in-context misses, one is Haiku's own wrong answer and two are qwen3:4b overwriting a
correct one. Continuations that started before Haiku's answer was written ended correct 27 of 110 times in-context
and 4 of 90 times with a log file (23 of the same 90 in-context). Fixed 75% again beat fixed 25% (19 against 7 of
30, +40.0 pts, +16.7 to +60.0). The prompt detector fired at turn 1 on no task and never on one. Most of this
pair's in-context runs came from Ollama 0.40.1 and all of its log-file runs from the partly installed 0.40.15 of
2026-11-09 (see below).

## How this differs from the paper

The 65.9% in the paper and the 16% here are not comparable: the receivers, tasks, detectors and averaging all
differ.

* The receiver is a 4B model on a laptop, not Qwen3.8-27B or GPT-5.6 Luna.
* The tasks are short: Sonnet 5.5 needed 4 to 9 turns (6.1 on average), so there are only 3 to 8 handoff points
  per task.
* The detectors are a hand-written rule and a qwen3:4b judge. The paper trains a 2B detector, initialized from
  Qwen 3.5 2B, on handoff points annotated with Codex; its own prompting test asks the capable model itself. That
  detector handed off at 26.6–41.1% of the source run on average on DeepSWE and 55.6–68.9% on Terminal-Bench; the
  rule here did so at 78%.
* In its analysis experiments the paper runs the capable models at max reasoning effort and Qwen3.8-27B at xhigh.
  Here Sonnet and Haiku run at medium effort and qwen3:4b with thinking off.
* The agent acts through Python snippets, not a shell.
* 30 tasks, one run each. qwen3:4b's replies are not reliably repeatable even at temperature 0 (see below), so
  each continuation is one draw.

## What went wrong during the run

The run's history is logged with dates in [docs/hypotheses.md](docs/hypotheses.md#changes-after-registration).

* **Ollama builds.** The Ollama app's auto-updater changed the install twice as far as the runs are concerned.
  The early runs most likely used 0.40.14. From 2026-11-07 the runs used 0.40.1 (that day the updater passed
  through 0.40.15 and 0.40.7 within five minutes, before any model request). On 2026-11-09 an install of 0.40.15
  was cancelled and rolled back partway: `ollama.exe` reports 0.40.15, but the libraries that run the model
  (llama-server, ggml, the CUDA backend) are still files dated 2026-10-08, most likely 0.40.1's. The runs after
  08:13 that day used this mixed install. Every run behind H1 to H4 finished before the first change. The Sonnet
  log-file runs straddle the first change and the Haiku pair the second.
* **Repeatability.** Asked again for ten recorded receiver turns on the mixed install, qwen3:4b gave the same
  reply 3 times and a different one 7 times ([script](scripts/repeat_check.py),
  [output](docs/repeatability_2026-11-09.txt)), so a build change cannot be told apart from ordinary run-to-run
  variation here.
* **Shutdowns and sleep.** Windows shut down three times during the run, and the laptop slept twice on 2026-11-06
  (17:01 to 19:16, then under a minute; logged reason: button or lid). After each shutdown the run resumed from the
  files on disk; after the sleeps the running process simply went on.
* **Aborted replies and crashes.** Ollama sometimes stops a reply that keeps repeating itself. The run log only
  shows failures that stopped a step: the client retried a failed request up to five times without recording it,
  and the Windows log shows four llama-server crashes during the run (2026-11-06 16:29 and 20:30, 2026-11-09 10:02
  and 10:22). The first abort that stopped a step was got past by rerunning the step. The second (Haiku log-file
  runs) was not, so from then on such a reply counts as an empty reply and so a format error; a reply cut off at the token limit was already a format error, though it keeps
  its partial text. In the end the rule never applied: rerun with it in place, the same continuation finished
  without an abort, and no later reply was aborted.
* **One refusal.** Sonnet refused once when taking over adjusted_roe-204 from qwen3:4b. As registered, that ends
  the episode and counts as wrong.
* **No pandas.** The laptop's application-control policy blocks pandas, so the agent had the standard library and
  numpy.

## Run it

Needs Python 3.12, [Ollama](https://ollama.com) with `qwen3:4b`, and an Anthropic API key in
`ANTHROPIC_API_KEY` (or in `~/.harnessforge/anthropic_api_key.txt`, outside the repository).

```bash
pip install -r requirements.txt
```

```bash
pytest -q
```

```bash
bash scripts/run_all.sh
```

Every step resumes from what is on disk, and paid steps stop before any request once the recorded API spend
reaches `BUDGET` dollars (default 10). Almost all of the run time is qwen3:4b.

## Safety of the agent's code

The agent's only action is a Python snippet. It runs in a fresh interpreter whose working directory is the
task workspace, behind an audit hook that refuses file access outside the workspace (apart from reading the Python
installation), process creation, network access, ctypes and console input, with an environment stripped of credentials. It keeps a cooperative model from
touching the rest of the machine by accident; it is not a security boundary against hostile code.

## Credits

The handoff design, the three readiness signs and the cost accounting follow the AgentBaton paper. The paper and
its arXiv page link no code release (checked 2026-11-09); everything here was written from its description.

## License

MIT
