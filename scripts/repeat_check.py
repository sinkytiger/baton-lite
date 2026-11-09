"""Does the running Ollama server give qwen3:4b the same reply it recorded earlier?

Rebuilds the exact message list a recorded receiver turn saw (no code is executed: the recorded observations are
reused), asks the running server again, and compares the JSON action byte for byte. It needs Ollama and qwen3:4b,
and it should run on an idle server.

    python scripts/repeat_check.py runs/handoff/claude-haiku-5-5__qwen3_4b logfile 1794179580 5

Arguments: a handoff folder, the transfer (incontext or logfile), a Unix time (only continuations written after it
are used, so that they come from the server being tested) and how many continuations to replay (two receiver
turns each). The output from 2026-11-09 is in docs/repeatability_2026-11-09.txt.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baton.agent import HANDOFF_LOG, SYSTEM, Trajectory, history, task_message  # noqa: E402
from baton.data import build_tasks  # noqa: E402
from baton.models import OllamaChat  # noqa: E402

folder, transfer, after, n = Path(sys.argv[1]) / sys.argv[2], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
tasks = {t.tid: t for t in build_tasks()}
files = [f for f in sorted(folder.glob("*-2*/k*.json")) if f.stat().st_mtime > after]
if not files:
    sys.exit("no recorded continuations found: stop")
random.Random(7).shuffle(files)
chat = OllamaChat("qwen3:4b")
same = diff = 0
for f in files[:n]:
    traj = Trajectory.load(f)
    task, k = tasks[traj.task], traj.handoff_at
    prefix = traj.turns[:k]
    if transfer == "incontext":
        start = [task_message(task)] + history(prefix)
    else:
        m = task_message(task)
        start = [{"role": "user", "content": m["content"] + "\n\n" + HANDOFF_LOG.format(k=k)}]
    for j in range(2):                      # the receiver's first two turns
        if k + j >= len(traj.turns):
            break
        rec = traj.turns[k + j]
        new = chat.reply(SYSTEM, start + history(traj.turns[k:k + j]))
        old_raw = rec.usage.get("raw_json")
        ok = new.usage.get("raw_json") == old_raw
        same += ok
        diff += not ok
        print(f"{transfer:9} {traj.task} k={k} turn {k + j + 1}: {'same' if ok else 'DIFFERENT'}", flush=True)
        if not ok:
            print("   old:", (old_raw or "")[:160].replace("\n", " "))
            print("   new:", (new.usage.get("raw_json") or "")[:160].replace("\n", " "))
print(f"identical {same}, different {diff}")
