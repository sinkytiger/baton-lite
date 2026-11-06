# baton-lite

**Can an expensive model start a financial-analysis task and hand the rest to a 4B model on a laptop?** A small
check of AgentBaton ([Li et al., arXiv:2610.35959](https://arxiv.org/abs/2610.35959)) with Claude Sonnet 5.5 and
Haiku 5.5 as the capable models and qwen3:4b, running locally on a 6 GB GPU, as the model that takes over.

> Status: experiment in progress. Results will be added after the test-split run.
> All companies in the data are fictional. Nothing here is financial advice.

## The question

AgentBaton lets a capable model work on a long agentic task and hand it, once, to a cheaper model at a suitable
point. The paper reports that capable-to-efficient handoffs at the right time largely keep accuracy while cutting
API cost by 65.9% on average, and that the reverse direction does not pay. Its efficient model is Qwen3.8-27B.
This repository asks whether that holds when the model taking over is much weaker, a 4B model that costs nothing
to run, and the work is multi-step financial data analysis.

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
| rule detector | hands off at the first turn showing the paper's three readiness signs in the workspace record |
| prompt detector | qwen3:4b reads the run after every turn and says HANDOFF_READY or not |
| reverse | qwen3:4b starts, Sonnet 5.5 takes over at 50% |

The receiver gets the prefix either in its context or as a log file, as in the paper. A handoff replays the
recorded prefix in a fresh workspace and checks the file tree after every turn, so the continuation starts from
exactly the state the capable model left.

The hypotheses and decision rules were committed before the test run: [docs/hypotheses.md](docs/hypotheses.md).

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
reaches `BUDGET` dollars (default 10).

## Safety of the agent's code

The agent's only action is a Python snippet. It runs in a fresh interpreter whose working directory is the
task workspace, behind an audit hook that refuses file access outside the workspace, process creation, network
access, ctypes and console input, with an environment stripped of credentials. It keeps a cooperative model from
touching the rest of the machine by accident; it is not a security boundary against hostile code.

## Credits

The handoff design, the three readiness signs and the cost accounting follow the AgentBaton paper. The paper has
no released code; everything here was written from its description.

## License

MIT
