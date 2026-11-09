import json

from baton.agent import Trajectory, parse, run, task_message
from baton.data import build_tasks
from baton.handoff import continue_from, fixed_point, fresh, handoff_points, replay, rule_point
from baton.models import Reply, cost_of


class Scripted:
    """A fake model that plays back fixed replies."""

    def __init__(self, model, replies):
        self.model, self.replies, self.seen = model, list(replies), []

    def reply(self, system, messages, **kw):
        self.seen.append(list(messages))
        return Reply(self.replies.pop(0), None, {"output_tokens": 10}, 0.01)


def _task():
    return next(t for t in build_tasks() if t.tid == "payout-101")


def _solution(task):
    return ("THOUGHT: write the answer.\n```python\nimport json\n"
            f"json.dump({{'payout': {json.dumps(task.gold)}}}, open('result.json', 'w'))\nprint('ok')\n```")


def test_parse():
    assert parse("THOUGHT: x\n```python\nprint(1)\n```") == ("code", "print(1)\n")
    assert parse("THOUGHT: done\nSUBMIT") == ("submit", "")
    assert parse("THOUGHT: done\n\n**SUBMIT**")[0] == "submit"
    assert parse("```python\na\n```\n```python\nb\n```")[0] == "error"
    assert parse("just talk")[0] == "error"


def test_loop_scores_and_records(tmp_path):
    task = _task()
    look = "THOUGHT: look.\n```python\nprint(open('README.md').read()[:50])\n```"
    chat = Scripted("fake-capable", [look, "no action here", _solution(task), "THOUGHT: done\nSUBMIT"])
    ws, scratch = fresh(task, tmp_path / "a")
    tr = run(task, chat, ws, scratch, Trajectory(task.tid, ["fake-capable"]), [task_message(task)])
    assert tr.end == "submit" and tr.score["correct"]
    assert [t.kind for t in tr.turns] == ["code", "error", "code", "submit"]
    assert tr.turns[0].reads == ["README.md"] and tr.turns[2].agent_files == ["result.json"]
    assert abs(tr.cost - 0.04) < 1e-9


def test_replay_and_handoff(tmp_path):
    task = _task()
    look = "THOUGHT: look.\n```python\nprint(open('data/companies.csv').read())\n```"
    src = run(task, Scripted("fake-capable", [look, _solution(task), "THOUGHT: done\nSUBMIT"]),
              *fresh(task, tmp_path / "src"), Trajectory(task.tid, ["fake-capable"]), [task_message(task)])
    assert handoff_points(src) == [1, 2]
    ws, scratch = fresh(task, tmp_path / "re")
    assert replay(src.turns, ws, scratch) is None
    # in-context: the receiver sees the prefix as history and only has to submit
    recv = Scripted("fake-local", ["THOUGHT: done\nSUBMIT"])
    tr = continue_from(task, src, 2, recv, "incontext", tmp_path / "h1")
    assert tr.score["correct"] and tr.handoff_at == 2 and len(tr.turns) == 3
    assert recv.seen[0][1]["content"] == src.turns[0].response
    # log file: the receiver starts a new session pointed at .handoff/log.jsonl
    recv = Scripted("fake-local", ["THOUGHT: read.\n```python\nprint(open('.handoff/log.jsonl').read()[:80])\n```",
                                   "THOUGHT: done\nSUBMIT"])
    tr = continue_from(task, src, 1, recv, "logfile", tmp_path / "h2")
    assert len(recv.seen[0]) == 1 and ".handoff/log.jsonl" in recv.seen[0][0]["content"]
    assert not tr.score["correct"]   # the receiver never wrote result.json
    assert tr.turns[1].returncode == 0


def test_policies(tmp_path):
    task = _task()
    look = "THOUGHT: look.\n```python\nprint(open('data/companies.csv').read())\n```"
    check = "THOUGHT: check.\n```python\nprint(open('result.json').read())\n```"
    src = run(task, Scripted("fake-capable", [look, _solution(task), check, "THOUGHT: done\nSUBMIT"]),
              *fresh(task, tmp_path / "src"), Trajectory(task.tid, ["fake-capable"]), [task_message(task)])
    # turn 2 writes result.json; turn 3 runs without error and opens no new data file
    assert rule_point(src) == 2 or rule_point(src) == 3
    assert fixed_point(src, 0.5) == 2 and fixed_point(src, 0.25) == 1 and fixed_point(src, 0.75) == 3


def test_cost():
    u = {"input_tokens": 1_000_000, "cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
         "output_tokens": 1_000_000}
    assert abs(cost_of("claude-sonnet-5-5", u) - (2 + 2.5 + 0.2 + 10)) < 1e-9
    assert cost_of("qwen3:4b", u) == 0


class _Failing:
    def __init__(self, message):
        self.message, self.calls = message, 0

    def chat(self, **kw):
        self.calls += 1
        raise RuntimeError(self.message)


def _ollama(client):
    from baton.models import OllamaChat
    chat = OllamaChat.__new__(OllamaChat)
    chat.client, chat.model, chat.num_ctx, chat.num_predict, chat.think = client, "qwen3:4b", 8192, 1536, False
    chat.char_budget = 10_000
    return chat


def test_repeat_abort_is_a_format_error_not_an_outage(monkeypatch):
    import pytest
    from baton import models
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    client = _Failing("prediction aborted, token repeat limit reached (status code: 500)")
    r = _ollama(client).reply("sys", [{"role": "user", "content": "task"}])
    assert client.calls == 1 and r.text == "" and r.stop == "aborted"
    assert r.usage["aborted"] == models.REPEAT_ABORT and parse(r.text)[0] == "error"
    outage = _Failing("connection refused")
    with pytest.raises(models.ModelError):
        _ollama(outage).reply("sys", [{"role": "user", "content": "task"}])
    assert outage.calls == 5
    judged = _Failing("prediction aborted, token repeat limit reached (status code: 500)")
    with pytest.raises(models.ModelError):      # a judge call is not an agent turn: still treated as a failure
        _ollama(judged).reply("sys", [{"role": "user", "content": "task"}], schema={"type": "object"})
