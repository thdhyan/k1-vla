"""Per-robot web console, served by the brain on the robot (default :8091). tests/test_console.py.

  GET  /                  page: STOP button, state, last frame sent to the VLM, live event timeline (VLM in/out, tool calls)
  GET  /api/events?since=N  {"events": [...], "state": {...}}
  GET  /api/frame.jpg     latest camera frame sent to the vision model
  GET  /api/live.jpg      current head camera frame (404 when the brain has no camera); the page polls it ~4 Hz
  GET  /api/calibrate     mic calibration status; POST starts it (3 s silence, 5 s speech -> VAD threshold, saved)
  GET  /api/debug         k1_chat.debug.collect(): memory, GPU, OOM kills, nemo/kokoro/LLM health, processes (404 if none)
  POST /api/estop         Safety.estop: latch chat motion, zero velocity, robot -> PREPARE
  POST /api/release       unlatch chat motion (mode unchanged); /api/chat_on is the same
  POST /api/chat_off      latch chat motion, zero velocity, mode unchanged (operator takes over)
  POST /api/mode/prepare  ModeFSM.request: stand up from damping / leave walking. /api/mode/walking: prepare -> walking.
                          Never damping; refused while a motion or another mode change runs.
No auth: anyone on the robots' LAN can stop or release. The physical remote stays the primary e-stop.
"""
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

from k1_chat.backend import Mode
from k1_chat.events import EventLog
from k1_chat.safety import Safety

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>K1 __NAME__ console</title><style>
body{font:14px system-ui;margin:0;background:#111;color:#eee}header{position:sticky;top:0;background:#1b1b1b;padding:10px 16px;
display:flex;gap:12px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333;z-index:1}h1{font-size:18px;margin:0}
#stop{background:#d22;color:#fff;font:bold 22px system-ui;border:0;border-radius:8px;padding:12px 28px;cursor:pointer}
.b{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:8px 14px;cursor:pointer}
#st{color:#aaa}.es{color:#f55;font-weight:bold}#cm{font-weight:bold;padding:4px 10px;border-radius:12px}.on{background:#174;color:#cfc}.off{background:#722;color:#fcc}main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.4fr);gap:16px;padding:16px}
@media(max-width:800px){main{grid-template-columns:1fr}}img{max-width:100%;border-radius:6px;background:#222;min-height:60px}
#ev div{border-bottom:1px solid #2a2a2a;padding:6px 0;white-space:pre-wrap;word-break:break-word}.k{display:inline-block;min-width:92px;font-weight:600}
.heard .k{color:#6cf}.said .k{color:#8e8}.llm_request .k{color:#c9f}.llm_reply .k{color:#fc6}.tool_call .k,.tool_result .k{color:#f9a}
.estop .k,.error .k{color:#f55}.t{color:#777;font-size:12px;margin-right:8px}
#sys table{border-collapse:collapse;width:100%;font-size:13px}#sys td{border-bottom:1px solid #2a2a2a;padding:3px 6px;vertical-align:top;word-break:break-word}
.ok{color:#8e8}.bad{color:#f55;font-weight:bold}.warn{color:#fc6}</style>
<header><h1>K1 __NAME__</h1><button id=stop title="robot goes to PREPARE (stands still)">STOP</button>
<span id=cm></span><button class=b id=on>chat motion ON</button><button class=b id=off>chat motion OFF</button>
<button class=b id=prep title="stand up / leave walking">stand up (prepare)</button><button class=b id=walk>walking mode</button>
<button class=b id=cal title="3 s silence, then 5 s speech: sets the mic threshold">calibrate mic</button><span id=calst class=t></span>
<span id=st>connecting...</span><span id=hb class=t>health: loading...</span>
<div id=wk style="flex-basis:100%;color:#fc6">__WAKE__</div></header>
<main><section><h3>Live camera</h3><img id=lv alt="no live camera"><h3>Last frame sent to VLM</h3><img id=fr alt="no frame yet"><div id=fp class=t></div>
<h3>System <span id=sysage class=t></span></h3><div id=sys>loading...</div></section>
<section><h3>Events (newest first)</h3><div id=ev></div></section></main><script>
let since=0;const ev=document.getElementById('ev'),esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function txt(e){const{seq,t,kind,...r}=e;return Object.keys(r).length==1&&'text'in r?r.text:JSON.stringify(r)}
async function post(p){const r=await fetch(p,{method:'POST'});document.getElementById('st').textContent=p+': '+await r.text()}
const B=(id,p)=>document.getElementById(id).onclick=()=>post(p);B('stop','/api/estop');B('on','/api/chat_on');B('off','/api/chat_off');
B('prep','/api/mode/prepare');B('walk','/api/mode/walking');const cm=document.getElementById('cm');
async function poll(){try{const d=await(await fetch('/api/events?since='+since)).json();
for(const e of d.events){since=e.seq;const div=document.createElement('div');div.className=e.kind;
div.innerHTML=`<span class=t>${new Date(e.t*1000).toLocaleTimeString()}</span><span class=k>${esc(e.kind)}</span>${esc(txt(e))}`;
ev.prepend(div);if(e.kind=='frame'){document.getElementById('fr').src='/api/frame.jpg?'+e.seq;document.getElementById('fp').textContent=e.prompt||''}}
while(ev.children.length>400)ev.lastChild.remove();const s=d.state;
cm.textContent=s.estopped?'CHAT MOTION OFF':'CHAT MOTION ON';cm.className=s.estopped?'off':'on';
document.getElementById('st').textContent='mode '+s.mode}
catch(x){document.getElementById('st').textContent='brain unreachable: use the remote'}}poll();setInterval(poll,500)
const lv=document.getElementById('lv');lv.onload=()=>setTimeout(live,250);lv.onerror=()=>setTimeout(live,2000);
function live(){lv.src='/api/live.jpg?'+Date.now()}live()
const calst=document.getElementById('calst');
function calshow(c){if(c.error&&!c.phase){calst.textContent=c.error;return}const r=c.result;
calst.innerHTML=c.phase=='quiet'||c.phase=='speak'?`<b class=warn>${esc(c.prompt)}</b> ${c.seconds_left}s`:
r?(r.ok?`<span class=ok>threshold ${r.previous} -> ${r.threshold}</span> (noise ${r.noise_p95}, speech ${r.speech_median}, peak ${r.speech_peak})`
:`<span class=bad>speech barely above noise (noise ${r.noise_p95}, peak ${r.speech_peak}): kept ${r.previous}. Speak louder/closer.</span>`):`threshold ${c.threshold}`;
if(c.phase=='quiet'||c.phase=='speak')setTimeout(calpoll,300)}
async function calpoll(){try{calshow(await(await fetch('/api/calibrate')).json())}catch(x){calst.textContent='calibrate: '+x}}
document.getElementById('cal').onclick=async()=>{calshow(await(await fetch('/api/calibrate',{method:'POST'})).json())};calpoll()
const C=(ok,s)=>`<span class=${ok===true?'ok':ok===false?'bad':'warn'}>${esc(s)}</span>`;
function llm(x){if(!x||x.configured===false)return C(null,'not configured');if(x.error)return C(false,x.model+': '+x.error);
return C(x.ok,x.model+' @ '+x.base)+(x.rtt_s!=null?` rtt ${x.rtt_s}s, free tier ${x.free_tier}, usage ${x.usage}, rate ${esc(JSON.stringify(x.rate_limit))}`:'')+(x.served?' serves '+esc(x.served.join(',')):'')}
async function sys(){try{const r=await fetch('/api/debug');if(r.status==404){document.getElementById('sys').textContent='no debug data';return}
const d=await r.json(),m=d.memory,n=d.nemo_asr,a=d.robot_audio_service,o=d.oom_kills_this_boot,rows=[];
rows.push(['memory',C(m.available_mb>500,m.available_mb+' MB free')+` of ${m.total_mb} MB, swap ${m.swap_used_mb}/${m.swap_total_mb} MB`]);
rows.push(['GPU load',d.gpu_load_pct==null?'n/a':C(d.gpu_load_pct<90?true:null,d.gpu_load_pct+' %')]);
rows.push(['OOM kills (boot)',o.error?C(false,o.error):C(o.count==0,o.count)+(o.last.length?'<br>'+o.last.map(esc).join('<br>'):'')]);
rows.push(['ASR server',C(n.ok,n.ok?'healthy':'DOWN')+` ${esc(n.model||'')} @ ${esc(n.url)}, local nemo-asrd systemd ${esc(n.systemd.ActiveState||'?')}, restarts ${esc(n.systemd.NRestarts||'?')}`+(n.error?' '+esc(n.error):'')+` (agent uses ${esc(d.asr_backend)})`]);
rows.push(['robot audio svc',a.error?C(false,a.error):C(a.ok,a.stage)+' '+esc(a.note)]);
rows.push(['chat LLM',llm(d.chat_llm)]);rows.push(['vision LLM',llm(d.vision_llm)]);
for(const[k,v]of Object.entries(d.timings))rows.push(['last '+k,C(v.ok,v.secs+' s')+` ${v.ago_s}s ago ${esc(v.detail)}`]);
for(const w of d.processes.watched||[])rows.push(['proc '+w.name,w.count?`${w.count}x ${w.rss_mb} MB pids ${w.pids.join(',')}`:'<span class=t>not running</span>']);
for(const p of d.processes.top||[])rows.push(['top '+p.pid,`${p.rss_mb} MB, ${p.age_s}s: ${esc(p.cmd)}`]);
const tts=Object.entries(d.timings).find(([k])=>k=='tts'),nr=+(n.systemd.NRestarts||0);
document.getElementById('hb').innerHTML=`ASR ${n.ok?C(true,'OK'):C(false,'DOWN')}${nr?C(null,' ('+nr+' restarts)'):''} | Kokoro ${tts?C(tts[1].ok,tts[1].secs+'s '+tts[1].ago_s+'s ago'):C(null,'loaded, not used yet')} | RAM ${C(m.available_mb>500?true:m.available_mb>250?null:false,m.available_mb+' MB')} | OOM ${C(o.count==0,o.count||0)}`;
document.getElementById('sys').innerHTML='<table>'+rows.map(([k,v])=>`<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join('')+'</table>';
document.getElementById('sysage').textContent=new Date(d.t*1000).toLocaleTimeString()}
catch(x){document.getElementById('sys').textContent='debug error: '+x;document.getElementById('hb').innerHTML=C(false,'agent unreachable (Kokoro/brain process dead?)')}}sys();setInterval(sys,2000)</script>"""


def make_server(host: str, port: int, name: str, log: EventLog, safety: Safety,
                state: Callable[[], dict] = dict,
               live: Callable[[], bytes | None] | None = None,
               debug: Callable[[], dict] | None = None, calibrate=None, wake: str = "") -> ThreadingHTTPServer:
    page = PAGE.replace("__NAME__", name).replace("__WAKE__", html.escape(wake)).encode()

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
                self._json({"events": log.since(since), "state": {**state(), "estopped": safety.arbiter.estopped}})
            elif u.path == "/api/frame.jpg":
                f = log.frame()
                self._send(200, f[0], "image/jpeg") if f else self._json({"error": "no frame"}, 404)
            elif u.path == "/api/live.jpg":
                jpeg = live() if live else None
                self._send(200, jpeg, "image/jpeg") if jpeg else self._json({"error": "no live camera"}, 404 if not live else 503)
            elif u.path == "/api/calibrate":
                self._json(calibrate.status()) if calibrate else self._json({"error": "no mic calibration"}, 404)
            elif u.path == "/api/debug":
                self._json(debug()) if debug else self._json({"error": "no debug"}, 404)
            elif u.path.startswith("/api/"):
                self._json({"error": "use POST"}, 405)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path == "/api/estop":
                self._json(safety.estop("console"))
            elif self.path in ("/api/release", "/api/chat_on"):
                safety.release("console")
                self._json({"released": True})
            elif self.path == "/api/calibrate":
                if not calibrate:
                    self._json({"error": "no mic calibration"}, 404)
                else:
                    log.add("calibrate", text="mic calibration started from console")
                    self._json(calibrate.start())
            elif self.path == "/api/chat_off":
                self._json(safety.chat_off("console"))
            elif self.path in ("/api/mode/prepare", "/api/mode/walking"):
                mode = self.path.rsplit("/", 1)[1]
                err = safety.fsm.request(Mode.PREPARE if mode == "prepare" else Mode.WALKING)
                log.add("console_mode", mode=mode, result=err or "ok")
                self._json({"result": err or f"now in {mode} mode"})
            else:
                self._json({"error": "not found"}, 404)

    return ThreadingHTTPServer((host, port), H)
