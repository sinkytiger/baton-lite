"""Chat backends with one interface: Claude (paid, capable) and Ollama (local, free).

Messages are model-agnostic dicts {"role": "user" | "assistant", "content": str, "raw": ..., "model": ...}.
A Claude assistant turn keeps its raw content blocks (thinking included) so the same model can continue its own
conversation; any other model only ever sees the text, as in the paper's in-context transfer.

Credentials: the ANTHROPIC_API_KEY environment variable or, when it is not set, a key file outside the
repository (default ~/.harnessforge/anthropic_api_key.txt, override with BATON_KEY_FILE). The key is read at
run time and never logged or written anywhere.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# $ per million tokens (cached Claude API price table, 2026-10-06). Cache writes cost 1.25x the input price
# (5-minute cache), cache reads 0.1x. Local models cost nothing, as in the paper.
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}


class ModelError(Exception):
    pass


@dataclass
class Reply:
    text: str
    raw: list | None
    usage: dict
    cost: float
    stop: str = "end_turn"


def cost_of(model: str, usage: dict) -> float:
    if model not in PRICES:
        return 0.0
    pin, pout = PRICES[model]
    return (usage.get("input_tokens", 0) * pin + usage.get("cache_creation_input_tokens", 0) * pin * 1.25
            + usage.get("cache_read_input_tokens", 0) * pin * 0.1 + usage.get("output_tokens", 0) * pout) / 1e6


def key_file() -> Path:
    return Path(os.environ.get("BATON_KEY_FILE") or Path.home() / ".harnessforge" / "anthropic_api_key.txt")


def resolve_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    f = key_file()
    if f.exists():
        return f.read_text(encoding="utf-8-sig").strip() or None
    return None


class ClaudeChat:
    """Claude with adaptive thinking at a fixed effort, automatic prompt caching of the growing conversation,
    and no sampling parameters (current models reject them). No refusal fallback: a fallback model would answer
    in the capable model's place and blur the comparison, so a refusal ends the episode and is counted."""

    def __init__(self, model: str, effort: str = "medium", max_tokens: int = 16000):
        import anthropic
        key = resolve_api_key()
        if not key:
            raise ModelError(f"no Anthropic API key: set ANTHROPIC_API_KEY or create {key_file()}")
        self.client = anthropic.Anthropic(api_key=key, max_retries=5, timeout=600)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def _messages(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "assistant" and m.get("raw") and m.get("model") == self.model:
                out.append({"role": "assistant", "content": m["raw"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def reply(self, system: str, messages: list[dict]) -> Reply:
        import anthropic
        try:
            r = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system, messages=self._messages(messages),
                thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"})
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) as exc:
            raise ModelError(f"{self.model}: {exc}") from exc
        u = r.usage
        usage = {"input_tokens": u.input_tokens or 0, "output_tokens": u.output_tokens or 0,
                 "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
                 "cache_read_input_tokens": u.cache_read_input_tokens or 0, "request_id": r._request_id}
        if r.stop_reason == "refusal" and getattr(r, "stop_details", None):
            usage["refusal_category"] = r.stop_details.category
        text = "".join(b.text for b in r.content if b.type == "text")
        raw = r.to_dict()["content"]
        return Reply(text, raw, usage, cost_of(self.model, usage), r.stop_reason or "end_turn")


ACTION_SCHEMA = {"type": "object", "properties": {
    "thought": {"type": "string", "maxLength": 500},
    "action": {"type": "string", "enum": ["code", "submit"]},
    "code": {"type": "string", "maxLength": 6000}},
    "required": ["thought", "action", "code"]}


def action_text(obj: dict) -> str:
    """The structured reply rendered in the same text format every other model writes."""
    thought = " ".join(str(obj.get("thought", "")).split())
    if obj.get("action") == "submit":
        return f"THOUGHT: {thought}\nSUBMIT"
    code = str(obj.get("code", "")).strip("\n")
    return f"THOUGHT: {thought}\n```python\n{code}\n```"


REPEAT_ABORT = "token repeat limit reached"


class OllamaChat:
    """A local model through Ollama: greedy decoding, thinking off, and a character budget that drops the oldest
    turns (keeping the system prompt and the task) when the conversation would not fit in the context window.
    8,192 tokens is the largest window that keeps qwen3:4b entirely on a 6 GB GPU; at 16,384 Ollama moved a
    fifth of the model to the CPU and a single turn took minutes.

    Agent turns are constrained to a JSON object {thought, action, code} and rendered back into the shared text
    format: on the dev task, unconstrained qwen3:4b wrote 1,536 tokens of prose and no code block, twice."""

    def __init__(self, model: str, num_ctx: int = 8192, num_predict: int = 1536, think: bool = False):
        import ollama
        self.client = ollama.Client(host=HOST, timeout=900)
        self.model, self.num_ctx, self.num_predict, self.think = model, num_ctx, num_predict, think
        self.char_budget = int((num_ctx - num_predict - 512) * 3.0)

    def _fit(self, system: str, messages: list[dict]) -> tuple[list[dict], int]:
        msgs = [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in messages]
        size = lambda ms: sum(len(m["content"]) for m in ms)
        dropped = 0
        while size(msgs) > self.char_budget and len(msgs) > 4:
            del msgs[2:4]       # oldest assistant/user pair after the task message
            dropped += 1
        if dropped:
            msgs.insert(2, {"role": "user", "content": f"[{2 * dropped} earlier messages were omitted to fit the context window.]"})
            msgs.insert(3, {"role": "assistant", "content": "Understood. I will continue from the current state."})
        return msgs, dropped

    def reply(self, system: str, messages: list[dict], schema: dict | None = None) -> Reply:
        msgs, dropped = self._fit(system, messages)
        kw = {"think": self.think} if self.model.startswith("qwen3") else {}
        agent_turn = schema is None
        kw["format"] = ACTION_SCHEMA if agent_turn else schema
        last = None
        for attempt in range(5):
            try:
                t = time.time()
                r = self.client.chat(model=self.model, messages=msgs, options={
                    "temperature": 0, "seed": 7, "num_ctx": self.num_ctx, "num_predict": self.num_predict}, **kw)
                usage = {"prompt_tokens": r.prompt_eval_count or 0, "output_tokens": r.eval_count or 0,
                         "seconds": round(time.time() - t, 2), "dropped_pairs": dropped}
                text = r.message.content or ""
                if agent_turn:
                    usage["raw_json"] = text
                    try:
                        text = action_text(json.loads(text))
                    except ValueError:
                        pass   # a cut-off object stays as it is and the agent loop reports a format error
                return Reply(text, None, usage, 0.0, r.done_reason or "stop")
            except Exception as exc:  # noqa: BLE001 - connection resets, server restarts
                if agent_turn and REPEAT_ABORT in str(exc):
                    # Ollama stopped a reply that kept repeating itself. That is the model's own reply, cut off,
                    # not an outage: like a reply cut off at num_predict, it becomes a format error and the
                    # episode goes on (added 2026-11-09, after one continuation aborted 30 times in a row).
                    usage = {"prompt_tokens": 0, "output_tokens": 0, "seconds": 0.0, "dropped_pairs": dropped,
                             "raw_json": "", "aborted": REPEAT_ABORT}
                    return Reply("", None, usage, 0.0, "aborted")
                last = exc
                time.sleep(10 * 2 ** attempt)
        raise ModelError(f"{self.model}: {last}")


def make_chat(model: str, **kw):
    return ClaudeChat(model, **kw) if model.startswith("claude-") else OllamaChat(model, **kw)
