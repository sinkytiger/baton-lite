"""Runs one agent action (a Python snippet) inside a task workspace.

The agent's only action is Python code. Each action runs in a fresh interpreter whose working directory is the
workspace, with an audit hook installed before the code starts: file access outside the workspace (except
reading the Python installation itself), starting processes, network access, ctypes and console input are
refused. The environment is stripped, so API keys never reach the agent's code. This keeps a cooperative model
from touching the rest of the laptop by accident; it is not a security boundary against hostile code.

The hook also records which workspace files the action opened, which the rule-based handoff detector uses.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TIMEOUT = 60
OUT_LIMIT = 2000       # characters of stdout and of stderr shown to the model
SKIP_DIRS = {"__pycache__", ".handoff"}

WRAPPER = r'''
import os, sys, json
ws_real = os.path.realpath(sys.argv[1])
ws = os.path.normcase(ws_real)
log_path = os.path.realpath(sys.argv[2])
code = open(sys.argv[3], encoding="utf-8").read()
roots = {os.path.normcase(os.path.realpath(p)) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)}
reads, writes = set(), set()

def _inside(p, root):
    return p == root or p.startswith(root.rstrip("\\/") + os.sep)

def _norm(p):
    return os.path.normcase(os.path.realpath(os.fsdecode(p)))

BLOCKED = ("subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.startfile", "os.fork",
           "os.forkpty", "os.kill", "socket.", "urllib.Request", "ctypes.", "winreg.", "_winapi.CreateProcess",
           "_winapi.OpenProcess", "_winapi.TerminateProcess", "_winapi.CreateJunction", "builtins.input",
           "webbrowser.open", "msvcrt.")
PATH_EVENTS = {"os.remove": 1, "os.rmdir": 1, "os.mkdir": 1, "os.rename": 2, "os.replace": 2, "os.link": 2,
               "os.symlink": 2, "os.chmod": 1, "os.truncate": 1, "os.utime": 1, "shutil.rmtree": 1,
               "shutil.move": 2, "shutil.copyfile": 2, "os.chdir": 1, "os.listdir": 1, "os.scandir": 1,
               "glob.glob": 1, "os.walk": 1}

def hook(event, args):
    if event.startswith(BLOCKED):
        raise PermissionError(f"sandbox: {event} is not allowed")
    if event == "open":
        path, mode, flags = args
        if path is None or isinstance(path, int):
            return
        p = _norm(path)
        writing = bool(mode and any(ch in str(mode) for ch in "wax+")) or bool(
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
        if _inside(p, ws):   # record the path with its original case (normcase lowercases on Windows)
            rel = os.path.relpath(os.path.realpath(os.fsdecode(path)), ws_real).replace(os.sep, "/")
            (writes if writing else reads).add(rel)
            return
        if p == os.path.normcase(log_path):
            return
        if not writing and any(_inside(p, r) for r in roots):
            return
        raise PermissionError(f"sandbox: access outside the workspace is not allowed ({os.fsdecode(path)})")
    n = PATH_EVENTS.get(event)
    if n:
        for a in args[:n]:
            if a is None or isinstance(a, int):
                continue
            p = _norm(a if a not in ("", b"") else ".")
            if not _inside(p, ws) and not (event in ("os.listdir", "os.scandir") and any(_inside(p, r) for r in roots)):
                raise PermissionError(f"sandbox: {event} outside the workspace is not allowed")

def _save():
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"reads": sorted(reads), "writes": sorted(writes)}, f)

import atexit, linecache, traceback
try:
    import numpy   # loads its DLLs through ctypes, so it is imported before the hook blocks ctypes
except ImportError:
    pass
linecache.cache["<action>"] = (len(code), None, code.splitlines(True), "<action>")
# show the agent's own source lines in tracebacks and leave this wrapper's frame out
sys.excepthook = lambda t, v, tb: traceback.print_exception(t, v, tb.tb_next if tb and tb.tb_next else tb)
atexit.register(_save)
sys.addaudithook(hook)
del atexit
g = {"__name__": "__main__", "__builtins__": __builtins__}
exec(compile(code, "<action>", "exec"), g)
'''


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str
    reads: list[str]
    writes: list[str]
    timed_out: bool = False

    def observation(self) -> str:
        return (f"<returncode>{self.returncode}</returncode>\n<stdout>\n{_clip(self.stdout)}\n</stdout>\n"
                f"<stderr>\n{_clip(self.stderr)}\n</stderr>")


def _clip(s: str, n: int = OUT_LIMIT) -> str:
    s = s.rstrip("\n")
    if len(s) <= n:
        return s
    half = n // 2
    return s[:half] + f"\n... [{len(s) - n} characters omitted] ...\n" + s[-half:]


def _env() -> dict[str, str]:
    keep = ("SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})
    return env


def run(code: str, workspace: str | Path, scratch: str | Path, python: str | None = None) -> Result:
    """Runs code with the workspace as current directory. scratch is a directory outside the workspace for the
    wrapper, the code file and the access log."""
    ws = Path(workspace).resolve()
    scratch = Path(scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    wrapper, code_file, log = scratch / "wrapper.py", scratch / "action.py", scratch / "access.json"
    wrapper.write_text(WRAPPER, encoding="utf-8")
    code_file.write_text(code, encoding="utf-8")
    log.unlink(missing_ok=True)
    try:
        p = subprocess.run([python or sys.executable, "-I", "-B", "-X", "utf8", str(wrapper), str(ws), str(log), str(code_file)],
                           cwd=ws, env=_env(), capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=TIMEOUT)
        rc, out, err, timed_out = p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        rc, err, timed_out = -1, f"Timed out after {TIMEOUT} seconds.", True
    acc = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {"reads": [], "writes": []}
    err = err.replace(str(wrapper), "wrapper")
    return Result(rc, out, err, acc["reads"], acc["writes"], timed_out)


def tree_hash(workspace: str | Path) -> str:
    """Content hash of every file in the workspace (excluding caches and the handoff log)."""
    ws = Path(workspace)
    h = hashlib.sha256()
    for p in sorted(ws.rglob("*")):
        rel = p.relative_to(ws)
        if p.is_file() and not (set(rel.parts) & SKIP_DIRS):
            h.update(str(rel.as_posix()).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def agent_files(workspace: str | Path) -> list[str]:
    """Files the agent created (anything outside README.md and data/)."""
    ws = Path(workspace)
    out = []
    for p in sorted(ws.rglob("*")):
        rel = p.relative_to(ws)
        if p.is_file() and rel.parts[0] not in ("data", "README.md") and not (set(rel.parts) & SKIP_DIRS):
            out.append(rel.as_posix())
    return out
