"""Fleet registry loaded from fleet.json (single source of truth for robot IPs)."""
import json
from pathlib import Path

_FILE = Path(__file__).resolve().parents[2] / "fleet.json"


def load(path: Path = _FILE) -> dict:
    return json.loads(path.read_text())


def robot_ip(name: str, fleet: dict | None = None) -> str:
    fleet = fleet or load()
    ip = fleet["robots"][name.upper()]["ip"]
    if not ip:
        raise KeyError(f"{name} has no IP ({fleet['robots'][name.upper()]['status']})")
    return ip


def active(fleet: dict | None = None) -> dict[str, str]:
    fleet = fleet or load()
    return {n: r["ip"] for n, r in fleet["robots"].items() if r["ip"]}
