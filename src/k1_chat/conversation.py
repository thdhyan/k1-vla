"""Voice conversation: transcript in, short spoken reply out. With a tool Registry the model can also call tools
(robot_tools.py: every motion goes through the arbiter); without one this is speech only."""
import json
import re
import sys
import threading
from pathlib import Path
from typing import Callable, Iterable

_STOP = re.compile(r"\b(stop|halt|freeze)\b", re.I)


def is_stop(text: str) -> bool:
    """Spoken stop words: the brain e-stops on these without asking the LLM."""
    return bool(_STOP.search(text))


def load_system(base: Path) -> str:
    """system_prompt.md (persona) followed by every knowledge/*.md in name order."""
    parts = [(base / "system_prompt.md").read_text().strip()]
    parts += [f.read_text().strip() for f in sorted((base / "knowledge").glob("*.md"))]
    return "\n\n".join(parts)


SYSTEM = load_system(Path(__file__).parent)  # edit system_prompt.md / knowledge/*.md; k1voice.sh syncs them
FALLBACK = "Sorry, I could not think of an answer just now."


class Conversation:
    """Thread-safe. A reply is dropped (None) when a newer utterance arrived while the LLM was thinking; the newer call
    then carries both user turns merged into one message."""

    def __init__(self, llm, system: str = SYSTEM, max_turns: int = 20, history_file: Path | None = None,
                 tools=None, log=None, max_tool_calls: int = 3, max_rounds: int = 4, one_call_per_round: bool = False,
                 context: Callable[[], str] | None = None):
        self._llm, self._system, self._max, self._file = llm, system, max_turns, history_file
        self._tools, self._log, self._max_calls, self._max_rounds = tools, log, max_tool_calls, max_rounds
        self._context = context  # live robot state, re-read before every LLM round, never stored in history
        self._one_per_round = one_call_per_round  # robot: each step must see the previous result (look, then turn, ...)
        self.history: list[dict] = self._load()
        self._lock, self._seq = threading.Lock(), 0
        self._tool_lock = threading.Lock()  # async voice chat: never two motions at once

    def _event(self, kind: str, **data) -> None:
        if self._log:
            self._log.add(kind, **data)

    def _answer(self, msgs: list[dict]) -> str:
        """LLM -> tool calls -> tool results -> LLM ... until it answers in text. Tool messages stay in this turn only."""
        if self._tools is None:
            return self._llm.chat(msgs, max_tokens=2000) or ""  # reasoning models spend part of this thinking
        msgs, used = list(msgs), 0
        for _ in range(self._max_rounds):
            if self._context:
                try:
                    state = self._context()
                except Exception as e:  # noqa: BLE001
                    state = f"Robot state unknown ({type(e).__name__})"
                msgs[0] = {"role": "system", "content": f"{self._system}\n\n{state}"}
                self._event("state", text=state)
            self._event("llm_request", text=msgs[-1].get("content") if isinstance(msgs[-1].get("content"), str) else "(multimodal)")
            m = self._llm.complete(msgs, tools=self._tools.schemas(), max_tokens=2000)
            calls = m.get("tool_calls") or []
            self._event("llm_reply", text=m.get("content") or "", tool_calls=[c["function"]["name"] for c in calls],
                        **({"thinking": m["reasoning"][-1500:]} if m.get("reasoning") else {}))  # console shows why it chose
            if not calls:
                return m.get("content") or ""
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
            for i, c in enumerate(calls):
                if self._one_per_round and i > 0:
                    result = "not run: call one tool at a time and choose the next one from the previous result"
                elif used >= self._max_calls:
                    result = "error: tool call limit for this turn reached, answer the user now"
                else:
                    used += 1
                    with self._tool_lock:
                        result = self._tools.run(c["function"]["name"], c["function"].get("arguments") or "")
                msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})
        return ""

    def _load(self) -> list[dict]:
        try:
            h = json.loads(self._file.read_text()) if self._file else []
            return h[-(2 * self._max - 1):] if isinstance(h, list) else []
        except (OSError, ValueError):
            return []

    def _save(self) -> None:
        if self._file:
            self._file.write_text(json.dumps(self.history))

    def reply(self, text: str) -> str | None:
        with self._lock:
            self._seq += 1
            mine = self._seq
            if self.history and self.history[-1]["role"] == "user":
                self.history[-1]["content"] += " " + text
            else:
                self.history.append({"role": "user", "content": text})
            self.history = self.history[-(2 * self._max - 1):]
            msgs = [{"role": "system", "content": self._system}, *map(dict, self.history)]
        ok = True
        try:
            answer = self._answer(msgs).strip()
        except Exception as e:
            print(f"[llm] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            answer = ""
        if not answer:
            answer, ok = FALLBACK, False
        with self._lock:
            if mine != self._seq:
                return None
            if ok:
                self.history.append({"role": "assistant", "content": answer})
            else:
                self.history.pop()
            self._save()
            return answer


def run_voice_chat(transcripts: Iterable[str], conv: Conversation, say: Callable[[str], None]) -> None:
    for text in transcripts:
        say(conv.reply(text))


def run_async_voice_chat(transcripts: Iterable[str], conv: Conversation, say: Callable[[str], None]) -> None:
    """Keep listening while the LLM thinks and the robot speaks: each transcript starts its own LLM call in a thread,
    and a call superseded by a newer utterance is never spoken."""
    speak_lock = threading.Lock()
    threads: list[threading.Thread] = []

    def handle(text: str) -> None:
        answer = conv.reply(text)
        if answer is not None:
            with speak_lock:
                say(answer)

    for text in transcripts:
        t = threading.Thread(target=handle, args=(text,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


WAKE_WORDS = "hey robot,hi robot,okay robot,ok robot,hey booster"


# left over around the name: "He, Apollo" (ASR for "hey"), "okay so Apollo" -> nothing left = bare wake word
FILLER = r"(?i)^(?:(?:hey|he|hi|hello|ok|okay|oh|yo|so|um|uh|a)\b[\s,.!?]*)+$"


class WakeGate:
    """Only pass utterances that contain a wake phrase. check() returns the text to answer (wake phrase removed,
    "" = just the wake phrase) or None to ignore. A bare wake phrase arms the gate for the next utterance only
    (within arm_s), so "Apollo." -> "Yes?" -> "turn left" works but room talk afterwards is ignored.
    window_s > 0 also passes follow-ups for that long after touch() (the robot spoke); 0 = name every time.
    List longer phrases first ("hey apollo" before "apollo") so the whole phrase is stripped.
    Stop words are handled before this and never need the wake phrase."""

    def __init__(self, words: str = WAKE_WORDS, window_s: float = 0.0, clock=None, arm_s: float = 10.0):
        import time
        self._words = [self._norm(w) for w in words.split(",") if w.strip()]
        self._window, self._arm, self._clock = window_s, arm_s, clock or time.monotonic
        self._until = self._armed = -1.0

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())

    def touch(self) -> None:
        self._until = self._clock() + self._window

    def arm(self) -> None:
        """Pass the next utterance (within arm_s) without the wake phrase: the robot asked or chose keep_listening."""
        self._armed = self._clock() + self._arm

    def check(self, text: str) -> str | None:
        n = f" {self._norm(text)} "
        now = self._clock()
        for w in self._words:
            if f" {w} " in n:
                self.touch()
                rest = re.sub(r"(?i)\b" + r"\W+".join(map(re.escape, w.split())) + r"\b", " ", text)
                rest = re.sub(FILLER, "", " ".join(rest.split()).strip(" ,.!?")).strip(" ,.!?")
                self._armed = now + self._arm if not rest else -1.0
                return rest
        if now < self._armed:
            self._armed = -1.0
            return text
        if now < self._until:
            self.touch()
            return text
        return None
