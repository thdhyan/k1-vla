"""LLM panel on the laptop: logging proxy robots -> self-hosted vLLM on a GPU box, plus load/unload buttons so the GPU
is only held while we use it. tests/test_llm_panel.py.

  laptop:  PYTHONPATH=src uv run python -m k1_chat.llm_panel --ssh-host dl --gpu 3      # page + proxy on http://192.168.50.250:8000/
  robots:  K1_CHAT_BASE_URL=http://192.168.50.250:8000/v1 K1_CHAT_MODEL=nemotron-bolt  (vision: K1_VISION_*=..., cosmos-reason2)

  GET  /                     page: load/unload buttons, GPU + container status, every request in/out, last image sent
  GET|POST /v1/...           OpenAI API, routed by "model" to the ssh tunnel of that container (logged)
  GET  /api/events?since=N   {"events": [...], "status": {...}}
  GET  /api/frame.jpg        last image a vision request carried
  POST /api/action/<name>    run scripts/serve_llm.sh for one of ACTIONS (one at a time), output streamed into events
No auth: bind it to the robots' LAN (enp2s0 192.168.50.250), never 0.0.0.0 on eduroam.
"""
import argparse
import base64
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, urlparse

import httpx

from k1_chat.events import EventLog

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "serve_llm.sh"
# model name served by vLLM -> (container role, local tunnel port)
MODELS = {"nemotron-bolt": ("chat", 18000), "cosmos-reason2": ("vision", 18001)}
ACTIONS = {"load_chat": ("up", "chat"), "unload_chat": ("down", "chat"),
           "load_vision": ("up", "vision"), "unload_vision": ("down", "vision"), "unload_all": ("down", "all")}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>K1 LLM panel</title><style>
body{font:14px system-ui;margin:0;background:#111;color:#eee}header{position:sticky;top:0;background:#1b1b1b;padding:10px 16px;
display:flex;gap:8px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333;z-index:1}h1{font-size:18px;margin:0 8px 0 0}
button{background:#2a4;color:#fff;border:0;border-radius:6px;padding:8px 12px;cursor:pointer}button.u{background:#933}
#st{color:#aaa;white-space:pre-wrap;padding:8px 16px;border-bottom:1px solid #333;font:12px ui-monospace,monospace}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,2fr);gap:16px;padding:16px}@media(max-width:800px){main{grid-template-columns:1fr}}
img{max-width:100%;border-radius:6px;background:#222;min-height:60px}#ev div{border-bottom:1px solid #2a2a2a;padding:6px 0;white-space:pre-wrap;word-break:break-word}
.k{display:inline-block;min-width:92px;font-weight:600}.llm_in .k{color:#c9f}.llm_out .k{color:#fc6}.action .k,.action_out .k,.action_done .k{color:#6cf}
.error .k{color:#f55}.t{color:#777;font-size:12px;margin-right:8px}</style>
<header><h1>K1 LLM panel</h1>
<button data-a="load_chat">load chat (Bolt)</button><button data-a="load_vision">load vision (Cosmos-Reason2)</button>
<button class=u data-a="unload_chat">unload chat</button><button class=u data-a="unload_vision">unload vision</button>
<button class=u data-a="unload_all">unload all</button></header><div id=st>connecting...</div>
<main><section><h3>Last image sent</h3><img id=fr alt="no image yet"></section>
<section><h3>Requests and actions (newest first)</h3><div id=ev></div></section></main><script>
let since=0;const ev=document.getElementById('ev'),esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function txt(e){const{seq,t,kind,text,...r}=e;return (Object.keys(r).length?JSON.stringify(r)+'\\n':'')+(text??'')}
document.querySelectorAll('button[data-a]').forEach(b=>b.onclick=async()=>{const r=await fetch('/api/action/'+b.dataset.a,{method:'POST'});
if(!r.ok)alert(await r.text())});
async function poll(){try{const d=await(await fetch('/api/events?since='+since)).json();
for(const e of d.events){since=e.seq;const div=document.createElement('div');div.className=e.kind;
div.innerHTML=`<span class=t>${new Date(e.t*1000).toLocaleTimeString()}</span><span class=k>${esc(e.kind)}</span>${esc(txt(e))}`;
ev.prepend(div);if(e.kind=='llm_in'&&e.image)document.getElementById('fr').src='/api/frame.jpg?'+e.seq}
while(ev.children.length>400)ev.lastChild.remove();document.getElementById('st').textContent=JSON.stringify(d.status,null,1)}
catch(x){document.getElementById('st').textContent='panel unreachable'}}poll();setInterval(poll,1000)</script>"""


def summarize(messages: list[dict]) -> str:
    """Readable transcript of a request: text kept, image bytes replaced by their size."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif p.get("type") == "image_url":
                    url = p.get("image_url", {}).get("url", "")
                    parts.append(f"[image {len(url.partition(',')[2])} b64 chars]")
            c = " ".join(parts)
        calls = [f"{t['function']['name']}({t['function'].get('arguments', '')})" for t in m.get("tool_calls") or []]
        out.append(f"{m.get('role', '?')}: {c or ''}{' -> ' + ', '.join(calls) if calls else ''}")
    return "\n".join(out)


def last_image(messages: list[dict]) -> bytes | None:
    for m in reversed(messages):
        for p in m.get("content") if isinstance(m.get("content"), list) else []:
            url = p.get("image_url", {}).get("url", "") if p.get("type") == "image_url" else ""
            if url.startswith("data:image"):
                try:
                    return base64.b64decode(url.partition(",")[2])
                except ValueError:
                    return None
    return None


def make_panel(host: str, port: int, routes: dict[str, str], log: EventLog,
               runner: Callable[[str], Iterable[str]], status: Callable[[], dict] = dict) -> ThreadingHTTPServer:
    """routes: served model name -> upstream base URL (without /v1). runner(action) yields output lines."""
    page = PAGE.encode()
    busy = threading.Lock()

    def run_action(name: str) -> None:
        try:
            log.add("action", text=name)
            for line in runner(name):
                log.add("action_out", text=line)
            log.add("action_done", text=name)
        except Exception as e:
            log.add("error", text=f"{name}: {e}")
        finally:
            busy.release()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=str).encode())

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif u.path == "/api/events":
                since = int((parse_qs(u.query).get("since") or ["0"])[0])
                self._json({"events": log.since(since), "status": status()})
            elif u.path == "/api/frame.jpg":
                f = log.frame()
                self._send(200, f[0], "image/jpeg") if f else self._json({"error": "no image"}, 404)
            elif u.path == "/v1/models":
                self._json({"object": "list", "data": [{"id": m, "object": "model"} for m in routes]})
            elif u.path.startswith("/api/action/"):
                self._json({"error": "use POST"}, 405)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path.startswith("/api/action/"):
                name = self.path.rsplit("/", 1)[1]
                if name not in ACTIONS:
                    return self._json({"error": f"unknown action {name!r}"}, 404)
                if not busy.acquire(blocking=False):
                    return self._json({"error": "another action is running"}, 409)
                threading.Thread(target=run_action, args=(name,), daemon=True).start()
                return self._json({"started": name}, 202)
            if not self.path.startswith("/v1/"):
                return self._json({"error": "not found"}, 404)
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                body = json.loads(raw)
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            model, msgs = body.get("model"), body.get("messages") or []
            up = routes.get(model)
            if not up:
                log.add("error", model=model, text="model not loaded / unknown")
                return self._json({"error": f"model {model!r} not loaded; have {sorted(routes)}"}, 404)
            img = last_image(msgs)
            if img:
                log.set_frame(img, model=model)
            log.add("llm_in", model=model, image=bool(img), tools=[t.get("function", {}).get("name") for t in body.get("tools") or []],
                    text=summarize(msgs))
            t = time.monotonic()
            try:
                r = httpx.post(up + self.path, content=raw, timeout=300,
                               headers={"Content-Type": "application/json", "Authorization": self.headers.get("Authorization", "")})
            except httpx.HTTPError as e:
                log.add("error", model=model, text=f"upstream: {e}")
                return self._json({"error": f"upstream unreachable: {e}"}, 502)
            ms = round((time.monotonic() - t) * 1000)
            try:
                msg = r.json()["choices"][0]["message"]
                think = msg.get("reasoning") or msg.get("reasoning_content")
                text = summarize([msg]) + (f"\n[reasoning] {think[:600]}" if think else "")
            except (ValueError, KeyError, IndexError, TypeError):
                text = r.text[:1000]
            log.add("llm_out" if r.status_code < 400 else "error", model=model, ms=ms, code=r.status_code, text=text)
            self._send(r.status_code, r.content, r.headers.get("Content-Type", "application/json"))

    return ThreadingHTTPServer((host, port), H)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh-host", default="dl")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--host", default="192.168.50.250", help="robots' LAN address of this laptop")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    log, state = EventLog(), {}

    def runner(name: str):
        verb, role = ACTIONS[name]
        args = [verb, a.ssh_host, a.gpu, role] if verb == "up" else [verb, a.ssh_host, role]
        p = subprocess.Popen([str(SCRIPT), *args],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        yield from (line.rstrip() for line in p.stdout)
        yield f"exit {p.wait()}"

    def tunnel_and_status():
        fwd = [x for _, port in MODELS.values() for x in ("-L", f"{port}:localhost:{port - 10000}")]
        tun = None
        while True:
            if tun is None or tun.poll() is not None:
                tun = subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", *fwd, a.ssh_host])
            st = {"tunnel": "up" if tun.poll() is None else "down"}
            try:
                st["remote"] = subprocess.run([str(SCRIPT), "status", a.ssh_host, a.gpu], capture_output=True, text=True,
                                              timeout=20).stdout.strip().splitlines()
            except subprocess.TimeoutExpired:
                st["remote"] = ["status timed out"]
            for m, (_, port) in MODELS.items():
                try:
                    st[m] = "ready" if httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2).status_code == 200 else "not ready"
                except httpx.HTTPError:
                    st[m] = "not loaded"
            state.clear(), state.update(st)
            time.sleep(10)

    threading.Thread(target=tunnel_and_status, daemon=True).start()
    routes = {m: f"http://127.0.0.1:{port}" for m, (_, port) in MODELS.items()}
    s = make_panel(a.host, a.port, routes, log, runner, status=lambda: dict(state))
    print(f"LLM panel: http://{a.host}:{a.port}/  (gpu {a.gpu} on {a.ssh_host})", flush=True)
    s.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
