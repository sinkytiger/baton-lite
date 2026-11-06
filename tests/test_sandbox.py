import os

import pytest

from baton import sandbox


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    (w / "data").mkdir(parents=True)
    (w / "data" / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    return w


@pytest.fixture
def scratch(tmp_path):
    return tmp_path / "scratch"


def test_reads_and_writes_inside_workspace(ws, scratch):
    r = sandbox.run("import csv, json\nrows = list(csv.reader(open('data/a.csv')))\n"
                    "json.dump({'n': len(rows)}, open('result.json', 'w'))\nprint(rows)", ws, scratch)
    assert r.returncode == 0, r.stderr
    assert "['x', 'y']" in r.stdout
    assert (ws / "result.json").exists()
    assert r.reads == ["data/a.csv"] and r.writes == ["result.json"]


def test_numpy_works(ws, scratch):
    r = sandbox.run("import numpy as np\nprint(np.mean([1, 2, 3]))", ws, scratch)
    assert r.returncode == 0, r.stderr
    assert "2.0" in r.stdout


@pytest.mark.parametrize("code", [
    "open(r'C:\\Windows\\win.ini').read()",
    "open('../outside.txt', 'w').write('x')",
    "import subprocess; subprocess.run(['cmd', '/c', 'echo hi'])",
    "import os; os.system('echo hi')",
    "import socket; socket.create_connection(('example.com', 80))",
    "import os; os.remove(r'C:\\Windows\\win.ini')",
    "import os; print(os.listdir('C:\\\\Users'))",
    "import ctypes; ctypes.CDLL('kernel32')",
    "input()",
])
def test_blocked(ws, scratch, code):
    r = sandbox.run(code, ws, scratch)
    assert r.returncode != 0
    assert "sandbox:" in r.stderr or "PermissionError" in r.stderr, r.stderr
    assert not (ws.parent / "outside.txt").exists()


def test_environment_has_no_keys(ws, scratch, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    r = sandbox.run("import os; print(sorted(k for k in os.environ if 'KEY' in k or 'TOKEN' in k))", ws, scratch)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


def test_timeout(ws, scratch, monkeypatch):
    monkeypatch.setattr(sandbox, "TIMEOUT", 2)
    r = sandbox.run("while True: pass", ws, scratch)
    assert r.timed_out and r.returncode == -1


def test_tree_hash_tracks_content(ws):
    h1 = sandbox.tree_hash(ws)
    (ws / "x.py").write_text("print(1)", encoding="utf-8")
    h2 = sandbox.tree_hash(ws)
    (ws / ".handoff").mkdir()
    (ws / ".handoff" / "log.jsonl").write_text("{}", encoding="utf-8")
    assert h1 != h2 and sandbox.tree_hash(ws) == h2
    assert sandbox.agent_files(ws) == ["x.py"]
