"""Brain: one robot = backend + arbiter + safety + tools + conversation + web console. tests/test_brain.py.

Laptop, virtual robot, typed input (no audio, no hardware):
  PYTHONPATH=src uv run python -m k1_chat.brain --name A2 --backend sim            # console: http://127.0.0.1:8091/
On the robot the voice loop uses this through `robot_audio agent` (see PLAN_TOOLS.md).
"""
import argparse
import sys
import threading

from k1_chat.arbiter import VelocityArbiter
from k1_chat.backend import Loco, Mode, SimRobot
from k1_chat.console import make_server
from k1_chat.conversation import SYSTEM, Conversation, is_stop
from k1_chat.events import EventLog
from k1_chat.modes import ModeFSM
from k1_chat.robot_tools import build_registry
from k1_chat.safety import Safety
from k1_chat.script_tools import add_script_tools
from k1_chat.tools import Tool

TOOLS_PROMPT = ("\n\nYou have a body. Use the tools to walk, turn, move your head, look, wave, shake hands, dab, cheer or "
                "change mode when the person asks. The robot state line above is live: trust it over your memory. "
                "Walking and gestures only work in walking mode; if the state says prepare, call set_mode walking first. "
                "Use the amounts the person asks for (a little turn is about 20 degrees, turn around is 180, "
                "a step is about 0.3 m). To face something, use the angle from detect or look, not a fixed 90. "
                "Forward walking stops by itself before obstacles; if a tool says it stopped or did not move, say so. "
                "Call one tool at a time and pick the next step from the last result. "
                "Seeing: detect is fast (objects, angles, distances); look is slower but describes details. When asked "
                "what is around you or to find something, check several directions on your own: detect or look, turn "
                "your head (set_head) to about 45 left and right, and turn your body if needed, then answer once. "
                "Use watch for things that move. move_and_watch scans while turning. "
                "To follow someone use follow_person. Put the head back to centre (yaw 0) when done. "
                "For a fixed sequence you may use macro; for a loop you may use run_script. "
                "Never claim you moved or saw something unless a tool result in this turn says so. "
                "Do not ask permission for simple safe moves the person asked for; just do them. "
                "After using tools, reply in one short sentence. "
                "People must say your name to talk to you. If you ask them a question or the chat clearly goes on, "
                "call keep_listening so their next sentence reaches you without your name; otherwise do not call it.")
MAX_TOOL_CALLS, MAX_ROUNDS = 10, 12  # enough for a look/head/turn search then a gesture


class Brain:
    def __init__(self, name: str, backend, llm, system: str = SYSTEM, history_file=None, extra_tools=None, battery=None,
                 perception=None):
        self.name, self.backend = name, backend
        self.log = EventLog()
        self.arbiter = VelocityArbiter()
        self.loco = Loco(backend)
        self.fsm = ModeFSM(self.loco)
        self.safety = Safety(self.loco, self.arbiter, self.log, fsm=self.fsm)
        self.battery, self.perception = battery, perception
        self.tools = build_registry(self.loco, self.arbiter, self.log, fsm=self.fsm, battery=battery, perception=perception)
        if extra_tools:
            extra_tools(self.tools, self.log)  # e.g. vision.add_look_tool
        add_script_tools(self.tools, stopped=lambda: self.arbiter.estopped)  # last: scripts can call every tool above
        self.listen_next = False  # set by keep_listening; the agent then opens the wake gate for one more utterance
        self.tools.add(Tool("keep_listening", "Hear the person's next sentence without the wake word (use after asking "
                            "a question or when the conversation continues). Not calling it ends the conversation.",
                            {}, self._keep_listening))
        self.conv = Conversation(llm, system=system + TOOLS_PROMPT, history_file=history_file,
                                 tools=self.tools, log=self.log, max_tool_calls=MAX_TOOL_CALLS,
                                 max_rounds=MAX_ROUNDS, one_call_per_round=True, context=self.context)
        self._turn = threading.Lock()  # one turn at a time: a new utterance merged into a running turn re-ran its tools
        self._server = None

    def context(self) -> str:
        """Live state line for every LLM round: the model must not guess its mode or what is in front of it."""
        try:
            mode = self.loco.get_mode()
        except Exception:
            mode = None
        parts = [f"mode {mode.name.lower() if isinstance(mode, Mode) else 'unknown'}"
                 + ("" if mode == Mode.WALKING else " (cannot walk or gesture until set_mode walking)")]
        if self.arbiter.estopped:
            parts.append("e-stop latched: chat motion is off")
        b = self.battery() if self.battery else None
        if b:
            parts.append(f"battery {b['soc']:.0f}%" + (" LOW: say so and avoid walking" if b["soc"] < 20 else ""))
        if self.perception is not None:
            from k1_chat.perception import describe
            parts.append(describe(self.perception.get()))
        return "Robot state right now: " + "; ".join(parts) + "."

    def state(self) -> dict:
        mode = getattr(self.backend, "mode", None)
        if mode is None:
            try:
                mode = self.loco.get_mode()
            except Exception:
                mode = None
        out = self.arbiter.output()
        return {"robot": self.name, "mode": mode.name if isinstance(mode, Mode) else str(mode),
                "twist": [out.vx, out.vy, out.vyaw]}

    def _keep_listening(self) -> str:
        self.listen_next = True
        return "ok: the next sentence reaches you without your name"

    def handle(self, text: str) -> str | None:
        self.listen_next = False
        self.log.add("heard", text=text)
        if is_stop(text):
            self.safety.estop("voice")
            answer = "Stopping."
        else:
            with self._turn:  # "stop" above never waits for this
                answer = self.conv.reply(text)
        if answer is not None:
            self.log.add("said", text=answer)
        return answer

    def serve_console(self, host: str = "0.0.0.0", port: int = 8091, live=None, debug=None, calibrate=None, wake="") -> str:
        self._server = make_server(host, port, self.name, self.log, self.safety, state=self.state, live=live, debug=debug,
                                   calibrate=calibrate, wake=wake)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        h, p = self._server.server_address[:2]
        return f"http://{h}:{p}/"

    def close(self) -> None:
        if self._server:
            self._server.shutdown()


def make_backend(kind: str):
    if kind == "sim":
        return SimRobot(mode=Mode.WALKING)
    from k1_chat.booster_robot import RosLocoBackend
    return RosLocoBackend()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="A2")
    ap.add_argument("--backend", choices=["sim", "ros"], default="sim")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv()
    from k1_chat.llm import from_env
    b = Brain(args.name, make_backend(args.backend), from_env())
    print(f"console: {b.serve_console(args.host, args.port)}  (type to talk, Ctrl-D to quit)", flush=True)
    for line in sys.stdin:
        if line.strip():
            print("K1:", b.handle(line.strip()), flush=True)
    b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
