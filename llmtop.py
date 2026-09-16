#!/usr/bin/env python3
"""llmtop - htop-artige Uebersicht ueber lokale LLM-Backends.

Zeigt fuer llama.cpp, Ollama und Lemonade Server an, ob sie laufen, welches
Modell geladen ist, wie viel Speicher es belegt und was gerade durchgeht.
Dazu iGPU- und NPU-Auslastung.

Nur Standardbibliothek, Python >= 3.11.

Wichtig: socket-aktivierte llama.cpp-Backends werden NIE ueber ihren
Socket-Port abgefragt - das wuerde den Modell-Ladevorgang ausloesen. Der
Zustand kommt aus systemd, Messwerte nur vom internen Backend-Port und nur
wenn der Dienst ohnehin schon laeuft.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

VERSION = "0.1.0"
HTTP_TIMEOUT = 1.5
CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


# --------------------------------------------------------------------------
# kleine Helfer
# --------------------------------------------------------------------------

def read_text(path: str | Path) -> str | None:
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


def read_int(path: str | Path) -> int | None:
    raw = read_text(path)
    if raw is None:
        return None
    try:
        return int(raw.split()[0])
    except (ValueError, IndexError):
        return None


def http_json(url: str, timeout: float = HTTP_TIMEOUT):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def http_text(url: str, timeout: float = HTTP_TIMEOUT) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def human_bytes(n: float | None, *, digits: int = 1) -> str:
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if abs(n) < 1024.0 or unit == "T":
            if unit == "B":
                return f"{int(n)} B"
            return f"{n:.{digits}f} {unit}iB"
        n /= 1024.0
    return f"{n:.{digits}f} TiB"


def human_ctx(n: int | None) -> str:
    """Kontextfenster kompakt. 131072 -> 128k, 34630 -> 33.8k."""
    if not n:
        return "-"
    if n < 1024:
        return str(n)
    k = n / 1024
    return f"{k:.0f}k" if abs(k - round(k)) < 0.05 else f"{k:.1f}k"


def human_count(n: int | None) -> str:
    if n is None:
        return "-"
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n/1000:.1f}k"
    return f"{n/1_000_000:.2f}M"


def human_delta(seconds: float | None) -> str:
    """Zeitspanne kompakt: 45s, 12m, 3h04, 2t05h."""
    if seconds is None:
        return "-"
    seconds = int(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    if seconds < 60:
        return f"{sign}{seconds}s"
    if seconds < 3600:
        return f"{sign}{seconds//60}m{seconds%60:02d}"
    if seconds < 86400:
        return f"{sign}{seconds//3600}h{(seconds%3600)//60:02d}"
    return f"{sign}{seconds//86400}t{(seconds%86400)//3600:02d}h"


def parse_iso(ts: str | None) -> float | None:
    """ISO-8601 aus Go/Python zu Unix-Zeit. Nanosekunden werden gekuerzt."""
    if not ts:
        return None
    cleaned = re.sub(r"(\.\d{6})\d+", r"\1", ts.replace("Z", "+00:00"))
    try:
        import datetime

        return datetime.datetime.fromisoformat(cleaned).timestamp()
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Prozesse
# --------------------------------------------------------------------------

@dataclass
class ProcInfo:
    pid: int
    ppid: int = 0
    comm: str = ""
    argv: list[str] = field(default_factory=list)
    rss: int = 0
    cpu_ticks: int = 0

    @property
    def cmdline(self) -> str:
        return " ".join(self.argv)


def proc_read(pid: int) -> ProcInfo | None:
    base = f"/proc/{pid}"
    raw = read_text(f"{base}/stat")
    if raw is None:
        return None
    # comm kann Leerzeichen und Klammern enthalten, darum ab der letzten ")".
    try:
        rest = raw[raw.rindex(")") + 2:].split()
        comm = raw[raw.index("(") + 1:raw.rindex(")")]
    except ValueError:
        return None
    if len(rest) < 22:
        return None
    try:
        ppid = int(rest[1])
        utime, stime = int(rest[11]), int(rest[12])
        rss_pages = int(rest[21])
    except ValueError:
        return None
    argv_raw = read_text(f"{base}/cmdline") or ""
    argv = [a for a in argv_raw.split("\0") if a]
    return ProcInfo(pid, ppid, comm, argv, rss_pages * PAGE_SIZE, utime + stime)


def proc_all() -> list[ProcInfo]:
    procs = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        info = proc_read(int(entry))
        if info is not None:
            procs.append(info)
    return procs


def proc_scan(pattern: re.Pattern[str],
              procs: list[ProcInfo] | None = None) -> list[ProcInfo]:
    """Prozesse nach Programmnamen suchen.

    Bewusst nur comm und der Basename von argv[0] - nicht die ganze
    Kommandozeile. earlyoom laeuft mit "--prefer ^(python3?|llama-server|
    ollama)$" und wuerde sonst als Backend durchgehen.
    """
    found = []
    for info in (procs if procs is not None else proc_all()):
        names = [info.comm]
        if info.argv:
            names.append(os.path.basename(info.argv[0]))
        if any(pattern.search(n) for n in names):
            found.append(info)
    return found


DRM_SIZE_RE = re.compile(r"^(\d+)\s*(KiB|MiB|GiB|B)?$")


def drm_proc_stats(pid: int) -> dict | None:
    """GPU-Speicher und GPU-Zeit eines Prozesses aus /proc/<pid>/fdinfo.

    Auf Systemen mit gemeinsamem Speicher (Strix Halo und Verwandte) liegt das
    Modell im GTT und taucht in RSS gar nicht auf - erst drm-resident-gtt zeigt
    die wahre Belegung. Lesbar nur fuer eigene Prozesse; fremde Dienste wie ein
    als root laufendes Ollama liefern nichts.
    """
    fd_dir = f"/proc/{pid}/fdinfo"
    try:
        entries = os.listdir(fd_dir)
    except OSError:
        return None
    clients: dict[str, dict[str, int]] = {}
    for name in entries:
        try:
            with open(f"{fd_dir}/{name}", "r", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if "drm-driver" not in text:
            continue
        fields: dict[str, str] = {}
        for line in text.splitlines():
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        client = fields.get("drm-client-id") or name

        def size(key: str) -> int:
            m = DRM_SIZE_RE.match(fields.get(key, ""))
            if not m:
                return 0
            scale = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20,
                     "GiB": 1 << 30}.get(m.group(2) or "B", 1)
            return int(m.group(1)) * scale

        engine = 0
        for key, value in fields.items():
            if key.startswith("drm-engine-"):
                try:
                    engine += int(value.split()[0])
                except (ValueError, IndexError):
                    pass
        entry = {"mem": size("drm-resident-gtt") + size("drm-resident-vram"),
                 "engine_ns": engine}
        # Mehrere Deskriptoren koennen denselben DRM-Client meinen - sonst
        # zaehlt der Speicher mehrfach.
        old = clients.get(client)
        if old is None or entry["mem"] > old["mem"]:
            clients[client] = entry
    if not clients:
        return None
    return {"mem": sum(c["mem"] for c in clients.values()),
            "engine_ns": sum(c["engine_ns"] for c in clients.values())}


class GpuTimeTracker:
    """GPU-Auslastung je Prozess aus der kumulierten Engine-Zeit."""

    def __init__(self) -> None:
        self._prev: dict[int, tuple[float, int]] = {}

    def percent(self, pid: int, engine_ns: int) -> float | None:
        now = time.monotonic()
        prev = self._prev.get(pid)
        self._prev[pid] = (now, engine_ns)
        if prev is None:
            return None
        elapsed = now - prev[0]
        if elapsed <= 0:
            return None
        delta = engine_ns - prev[1]
        if delta < 0:
            return None
        return min(100.0, delta / 1e9 / elapsed * 100.0)


class CpuTracker:
    """CPU-Prozent je PID aus /proc-Deltas, ohne externe Tools."""

    def __init__(self) -> None:
        self._prev: dict[int, tuple[float, int]] = {}

    def percent(self, pid: int, ticks: int) -> float | None:
        now = time.monotonic()
        prev = self._prev.get(pid)
        self._prev[pid] = (now, ticks)
        if prev is None:
            return None
        elapsed = now - prev[0]
        if elapsed <= 0:
            return None
        return max(0.0, (ticks - prev[1]) / CLK_TCK / elapsed * 100.0)

    def forget_except(self, pids: set[int]) -> None:
        for pid in list(self._prev):
            if pid not in pids:
                del self._prev[pid]


class RateTracker:
    """tok/s aus monoton wachsenden Zaehlern."""

    def __init__(self) -> None:
        self._prev: dict[str, tuple[float, float]] = {}

    def rate(self, key: str, total: float | None) -> float | None:
        if total is None:
            return None
        now = time.monotonic()
        prev = self._prev.get(key)
        self._prev[key] = (now, total)
        if prev is None:
            return None
        elapsed = now - prev[0]
        delta = total - prev[1]
        if elapsed <= 0 or delta < 0:  # Zaehler zurueckgesetzt
            return None
        return delta / elapsed


# --------------------------------------------------------------------------
# llama-server Kommandozeile auswerten
# --------------------------------------------------------------------------

LLAMA_FLAGS = {
    "port": ("--port",),
    "host": ("--host",),
    "model": ("-m", "--model"),
    "alias": ("-a", "--alias", "--model-alias"),
    "ctx": ("-c", "--ctx-size"),
    "ngl": ("-ngl", "--gpu-layers", "--n-gpu-layers"),
    "parallel": ("-np", "--parallel"),
}


def parse_llama_argv(argv: list[str]) -> dict:
    """Die fuer die Anzeige relevanten llama-server-Optionen herausziehen."""
    out: dict = {}
    lookup = {flag: name for name, flags in LLAMA_FLAGS.items() for flag in flags}
    for i, tok in enumerate(argv):
        key = None
        value = None
        if "=" in tok and tok.startswith("-"):
            flag, _, value = tok.partition("=")
            key = lookup.get(flag)
        elif tok in lookup and i + 1 < len(argv):
            key, value = lookup[tok], argv[i + 1]
        if key and value is not None:
            out[key] = value
    for numeric in ("port", "ctx", "ngl", "parallel"):
        if numeric in out:
            try:
                out[numeric] = int(out[numeric])
            except ValueError:
                out.pop(numeric)
    if "--flash-attn" in argv or "-fa" in argv:
        out["flash_attn"] = True
    if "--metrics" in argv:
        out["metrics"] = True
    if "--no-slots" in argv:
        out["slots"] = False
    spec = next((argv[i + 1] for i, t in enumerate(argv)
                 if t == "--spec-type" and i + 1 < len(argv)), None)
    if spec:
        out["draft"] = spec
    return out


def model_label(path: str | None, alias: str | None = None) -> str:
    """Sprechender Name: Alias, sonst Verzeichnis/Datei statt sha256-Blob."""
    if alias:
        return alias
    if not path:
        return "-"
    p = Path(path)
    if p.name.startswith("sha256-") or p.name.startswith("sha256:"):
        return f"blob {p.name[7:19]}"
    stem = p.stem
    # "Modell-00001-of-00003" auf den Grundnamen kuerzen
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
    parent = p.parent.name
    if parent and parent.lower() not in {"models", "gguf", ".", "/"} and len(stem) < 12:
        return f"{parent}/{stem}"
    return stem


# --------------------------------------------------------------------------
# systemd
# --------------------------------------------------------------------------

class Systemd:
    """Duenne Huelle um systemctl. Fehlt systemd, liefert alles leere Werte."""

    def __init__(self) -> None:
        self.available = shutil.which("systemctl") is not None

    def _run(self, args: list[str]) -> str:
        if not self.available:
            return ""
        try:
            res = subprocess.run(
                ["systemctl", *args], capture_output=True, text=True,
                timeout=5, stdin=subprocess.DEVNULL,
            )
            return res.stdout
        except (subprocess.SubprocessError, OSError):
            return ""

    def show(self, unit: str, props: list[str], user: bool) -> dict[str, str]:
        scope = ["--user"] if user else []
        out = self._run([*scope, "show", unit, "--no-pager",
                         "--property=" + ",".join(props)])
        result = {}
        for line in out.splitlines():
            key, _, value = line.partition("=")
            if key:
                result[key] = value
        return result

    def units(self, pattern: str, user: bool) -> list[str]:
        scope = ["--user"] if user else []
        out = self._run([*scope, "list-units", "--all", "--no-pager",
                         "--plain", "--no-legend", pattern])
        names = []
        for line in out.splitlines():
            parts = line.split()
            if parts:
                names.append(parts[0].lstrip("● ").strip())
        return [n for n in names if n]

    def cat(self, unit: str, user: bool) -> str:
        scope = ["--user"] if user else []
        return self._run([*scope, "cat", unit, "--no-pager"])


ARGV_RE = re.compile(r"argv\[\]=(.*?)\s+;")


def execstart_argv(show_value: str) -> list[str]:
    """argv der letzten ExecStart-Zeile aus `systemctl show --property=ExecStart`.

    Diese Form wird bevorzugt, weil systemd die Specifier (%h, %t, ...) dort
    bereits expandiert hat - in `systemctl cat` stehen sie noch roh drin.
    """
    matches = ARGV_RE.findall(show_value or "")
    if not matches:
        return []
    return matches[-1].split()


def resolve_llama_config(argv: list[str], depth: int = 0) -> dict:
    """llama-server-Optionen finden, auch wenn ExecStart auf ein Skript zeigt."""
    if not argv:
        return {}
    cfg = parse_llama_argv(argv)
    if cfg.get("port") or cfg.get("model"):
        return cfg
    if depth > 1:
        return cfg
    # ExecStart zeigt auf ein Wrapper-Skript: dessen Inhalt nach llama-server
    # durchsuchen, ohne es auszufuehren.
    script = Path(os.path.expanduser(argv[0]))
    if not script.is_file():
        return cfg
    try:
        body = script.read_text(errors="replace")
    except OSError:
        return cfg
    body = body.replace("\\\n", " ")
    vars_: dict[str, str] = {}
    for line in body.splitlines():
        line = line.strip()
        m = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.+)$", line)
        if m:
            vars_[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    home = os.path.expanduser("~")

    def expand(text: str) -> str:
        def sub(m: re.Match[str]) -> str:
            return vars_.get(m.group(1) or m.group(2), m.group(0))
        for _ in range(4):
            new = text.replace("${HOME}", home).replace("$HOME", home)
            new = re.sub(r"\$\{(\w+)\}|\$(\w+)", sub, new)
            if new == text:
                break
            text = new
        return text
    for line in body.splitlines():
        if "llama-server" not in line:
            continue
        tokens = [t.strip('"').strip("'") for t in expand(line).split()]
        cfg.update(parse_llama_argv(tokens))
    if (cfg.get("model") or cfg.get("port")) or "llama-server" not in body:
        return cfg
    # Die Aufrufzeile trug nichts Brauchbares - typisch fuer Skripte, die die
    # Optionen erst in einem Bash-Array sammeln und mit "${ARGS[@]}" uebergeben.
    # Dann das ganze Skript als Argumentvorrat lesen.
    flat = expand(body).replace("(", " ").replace(")", " ")
    tokens = [t.strip('"').strip("'") for t in flat.split()]
    cfg.update(parse_llama_argv(tokens))
    return cfg


# --------------------------------------------------------------------------
# Datenmodell
# --------------------------------------------------------------------------

RUNNING, SLEEPING, STOPPED, ABSENT = "running", "sleeping", "stopped", "absent"


@dataclass
class Backend:
    kind: str
    name: str
    state: str = ABSENT
    detail: str = ""
    model: str = "-"
    ctx: int | None = None
    mem: int | None = None
    mem_kind: str = ""
    cpu: float | None = None
    gpu_mem: int | None = None
    gpu_util: float | None = None
    tps: float | None = None
    busy: bool = False
    slots_busy: int = 0
    slots_total: int = 0
    idle_in: float | None = None
    since: float | None = None
    last_use: float | None = None
    port: int | None = None
    pid: int | None = None
    extras: list[str] = field(default_factory=list)
    children: list["Backend"] = field(default_factory=list)


@dataclass
class Snapshot:
    backends: list[Backend] = field(default_factory=list)
    gpu: dict = field(default_factory=dict)
    npu: dict = field(default_factory=dict)
    system: dict = field(default_factory=dict)
    taken: float = 0.0


# --------------------------------------------------------------------------
# Sammler
# --------------------------------------------------------------------------

class Collector:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.sd = Systemd()
        self.cpu = CpuTracker()
        self.gputime = GpuTimeTracker()
        self.rates = RateTracker()
        self.guarded_ports: set[int] = set()
        self._npu_name: str | None = None
        self._npu_probed = False
        self._guards_primed = False
        self._prev_cpu_total: tuple[int, int] | None = None

    # -- llama.cpp ---------------------------------------------------------

    def _llama_live(self, be: Backend, host: str, port: int, use_metrics: bool) -> None:
        """Messwerte vom laufenden Backend holen. Nur interne Ports, nie Sockets."""
        if port in self.guarded_ports:
            be.extras.append("Messung uebersprungen (Socket-Port)")
            return
        base = f"http://{host}:{port}"
        slots = http_json(f"{base}/slots")
        if isinstance(slots, list):
            be.slots_total = len(slots)
            be.slots_busy = sum(1 for s in slots if s.get("is_processing"))
            be.busy = be.slots_busy > 0
            if be.ctx is None and slots:
                be.ctx = slots[0].get("n_ctx")
            decoded = 0
            for s in slots:
                nxt = s.get("next_token")
                if isinstance(nxt, list):
                    nxt = nxt[0] if nxt else {}
                if isinstance(nxt, dict):
                    decoded += int(nxt.get("n_decoded") or 0)
            be.tps = self.rates.rate(f"llama:{host}:{port}", float(decoded))
        elif isinstance(slots, dict) and slots.get("error"):
            be.extras.append("/slots deaktiviert")
        if use_metrics and be.tps is None:
            # Nur als Rueckfall: llamacpp:tokens_predicted_total wird erst beim
            # Auftragsende fortgeschrieben und steht waehrend der Generierung
            # still - /slots zaehlt dagegen live mit.
            text = http_text(f"{base}/metrics")
            if text:
                be.tps = self._prom_tokens_per_second(text, f"llama-m:{host}:{port}")
        if be.ctx is None or be.model == "-":
            props = http_json(f"{base}/props")
            if isinstance(props, dict):
                if be.model == "-":
                    be.model = model_label(props.get("model_path"))
                gen = props.get("default_generation_settings") or {}
                be.ctx = be.ctx or gen.get("n_ctx")
                be.slots_total = be.slots_total or (props.get("total_slots") or 0)

    def _attach_gpu(self, be: Backend) -> None:
        """GPU-Speicher und GPU-Zeit nachtragen, soweit /proc es hergibt."""
        if not be.pid:
            return
        stats = drm_proc_stats(be.pid)
        if not stats:
            return
        if stats["mem"]:
            be.gpu_mem = stats["mem"]
        be.gpu_util = self.gputime.percent(be.pid, stats["engine_ns"])

    def _prom_tokens_per_second(self, text: str, key: str) -> float | None:
        total = None
        for line in text.splitlines():
            if line.startswith("llamacpp:tokens_predicted_total"):
                try:
                    total = float(line.split()[-1])
                except (ValueError, IndexError):
                    return None
                break
        return self.rates.rate(key, total)

    def collect_llama(self, owned_pids: set[int]) -> list[Backend]:
        backends: list[Backend] = []
        claimed_pids: set[int] = set()

        handled: set[str] = set()
        for user_scope in (True, False):
            for sock in self.sd.units(self.cfg["llama"]["unit_glob_socket"], user_scope):
                info = self.sd.show(sock, ["Listen", "Triggers", "ActiveState"], user_scope)
                listen = info.get("Listen", "")
                for m in re.finditer(r"(?::|\b)(\d{2,5})\s*\(Stream\)", listen):
                    self.guarded_ports.add(int(m.group(1)))
                # Der gleichnamige Dienst zuerst: haengt vor dem Backend ein
                # Proxy (systemd-socket-proxyd), steht der zwar in Triggers,
                # kennt aber weder Modell noch Kontext.
                candidates = [sock.replace(".socket", ".service"),
                              *(info.get("Triggers") or "").split()]
                handled.update(candidates)
                chosen = fallback = None
                for cand in candidates:
                    be = self._llama_from_unit(cand, user_scope, sock, listen)
                    if be is None:
                        continue
                    fallback = fallback or be
                    if be.model != "-":
                        chosen = be
                        break
                chosen = chosen or fallback
                if chosen:
                    backends.append(chosen)
                    if chosen.pid:
                        claimed_pids.add(chosen.pid)

        for user_scope in (True, False):
            for svc in self.sd.units(self.cfg["llama"]["unit_glob_service"], user_scope):
                if svc in handled:
                    continue
                handled.add(svc)
                be = self._llama_from_unit(svc, user_scope, None, "")
                # Ohne erkennbares Modell und ohne laufenden Prozess ist es
                # kein llama-Backend, sondern Beiwerk wie ein Socket-Proxy.
                if be is None or (be.model == "-" and not be.pid):
                    continue
                backends.append(be)
                if be.pid:
                    claimed_pids.add(be.pid)

        # freistehende llama-server, die zu keiner Unit gehoeren
        for proc in proc_scan(re.compile(r"^llama-server$")):
            if proc.pid in claimed_pids or proc.pid in owned_pids:
                continue
            if proc.ppid in owned_pids:
                continue
            cfg = parse_llama_argv(proc.argv)
            be = Backend(kind="llama", name=cfg.get("alias") or f"llama-server:{proc.pid}",
                         state=RUNNING, detail="freistehend", pid=proc.pid,
                         port=cfg.get("port"), ctx=cfg.get("ctx"),
                         model=model_label(cfg.get("model"), cfg.get("alias")),
                         mem=proc.rss, mem_kind="RSS")
            be.cpu = self.cpu.percent(proc.pid, proc.cpu_ticks)
            self._attach_gpu(be)
            self._llama_extras(be, cfg)
            if cfg.get("port"):
                self._llama_live(be, cfg.get("host") or "127.0.0.1", cfg["port"],
                                 bool(cfg.get("metrics")))
            backends.append(be)
        return backends

    def _llama_extras(self, be: Backend, cfg: dict) -> None:
        if cfg.get("draft"):
            be.extras.append(f"draft {cfg['draft']}")
        if cfg.get("flash_attn"):
            be.extras.append("fa")
        if cfg.get("ngl"):
            be.extras.append(f"ngl {cfg['ngl']}")

    def _llama_from_unit(self, service: str, user_scope: bool,
                         socket_unit: str | None, listen: str) -> Backend | None:
        props = self.sd.show(service, [
            "ActiveState", "SubState", "MainPID", "Description",
            "ActiveEnterTimestampMonotonic", "InactiveEnterTimestampMonotonic",
            "LoadState", "ExecStart",
        ], user_scope)
        if not props or props.get("LoadState") not in ("loaded",):
            return None
        name = service.replace(".service", "")
        active = props.get("ActiveState", "")
        be = Backend(kind="llama", name=name)
        cfg = resolve_llama_config(execstart_argv(props.get("ExecStart", "")))
        be.ctx = cfg.get("ctx")
        be.model = model_label(cfg.get("model"), cfg.get("alias"))
        self._llama_extras(be, cfg)
        path = cfg.get("model")
        if path and not path.startswith("$") and not os.path.exists(path):
            be.extras.append("Modelldatei fehlt")

        sock_port = None
        m = re.search(r"(\d{2,5})\s*\(Stream\)", listen or "")
        if m:
            sock_port = int(m.group(1))

        now_mono = time.monotonic()
        if active == "active":
            be.state = RUNNING
            pid = int(props.get("MainPID") or 0)
            enter = props.get("ActiveEnterTimestampMonotonic")
            if enter and enter.isdigit():
                be.since = now_mono - int(enter) / 1e6
            be.port = sock_port or cfg.get("port")
            if pid:
                info = proc_read(pid)
                if info is None:  # MainPID zeigt auf den Wrapper, Kind suchen
                    info = next((p for p in proc_scan(re.compile(r"^llama-server$"))
                                 if p.ppid == pid), None)
                if info:
                    be.pid = info.pid
                    be.mem, be.mem_kind = info.rss, "RSS"
                    be.cpu = self.cpu.percent(info.pid, info.cpu_ticks)
                    self._attach_gpu(be)
                    live = parse_llama_argv(info.argv)
                    cfg = {**cfg, **live}
            if cfg.get("port"):
                self._llama_live(be, cfg.get("host") or "127.0.0.1", cfg["port"],
                                 bool(cfg.get("metrics")))
        elif socket_unit:
            be.state = SLEEPING
            be.port = sock_port
            be.detail = f"Socket {sock_port}" if sock_port else "Socket aktiv"
            left = props.get("InactiveEnterTimestampMonotonic")
            if left and left.isdigit() and int(left) > 0:
                be.since = now_mono - int(left) / 1e6
        else:
            be.state = STOPPED
            be.detail = props.get("SubState", "")
        return be

    # -- Ollama ------------------------------------------------------------

    def _ollama_blob_map(self) -> dict[str, str]:
        """sha256-Blob des Modell-Layers -> Modellname:Tag, aus den Manifesten."""
        roots = [Path(p) for p in self.cfg["ollama"]["model_dirs"] if Path(p).is_dir()]
        mapping: dict[str, str] = {}
        for root in roots:
            manifests = root / "manifests"
            if not manifests.is_dir():
                continue
            for path in manifests.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    data = json.loads(path.read_text(errors="replace"))
                except (OSError, ValueError):
                    continue
                for layer in data.get("layers", []):
                    if layer.get("mediaType", "").endswith("image.model"):
                        digest = str(layer.get("digest", "")).replace("sha256:", "")
                        if digest:
                            mapping[digest] = self._manifest_name(
                                path.relative_to(manifests).parts)
        return mapping

    @staticmethod
    def _manifest_name(parts: tuple[str, ...]) -> str:
        """Manifestpfad -> Name wie in /api/ps.

        registry.ollama.ai/library/gemma4/12b          -> gemma4:12b
        registry.ollama.ai/ns/nexus-medical/latest     -> ns/nexus-medical:latest
        hf.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Q4_K_M
            -> hf.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q4_K_M
        """
        if len(parts) < 2:
            return ":".join(parts)
        path, tag = list(parts[:-1]), parts[-1]
        # Die Standardregistry taucht im Namen nie auf, "library" auch nicht.
        if path and path[0] in ("registry.ollama.ai", "ollama.com"):
            path = path[1:]
            if path[:1] == ["library"]:
                path = path[1:]
        return f"{'/'.join(path)}:{tag}"

    def collect_ollama(self) -> tuple[Backend, set[int]]:
        host = self.cfg["ollama"]["url"].rstrip("/")
        be = Backend(kind="ollama", name="ollama")
        procs = proc_all()
        server = next((p for p in proc_scan(re.compile(r"^ollama$"), procs)
                       if "serve" in p.cmdline), None)
        pids: set[int] = set()

        props = self.sd.show("ollama.service", ["ActiveState", "SubState", "MainPID",
                                                "ActiveEnterTimestampMonotonic",
                                                "LoadState"], False)
        if props.get("LoadState") == "loaded":
            if props.get("ActiveState") == "active":
                be.state = RUNNING
                enter = props.get("ActiveEnterTimestampMonotonic")
                if enter and enter.isdigit():
                    be.since = time.monotonic() - int(enter) / 1e6
            else:
                be.state = STOPPED
                be.detail = props.get("SubState", "")
        if server is not None:
            be.state = RUNNING
            be.pid = server.pid
            pids.add(server.pid)
        elif be.state == ABSENT:
            return be, pids

        if be.state != RUNNING:
            return be, pids

        m = re.search(r":(\d+)", host)
        be.port = int(m.group(1)) if m else None

        ver = http_json(f"{host}/api/version")
        if isinstance(ver, dict) and ver.get("version"):
            be.extras.append(f"v{ver['version']}")

        # Jedes Kind von "ollama serve" ist ein Runner - egal ob es als
        # llama-server oder als "ollama runner" auftritt.
        runners = [p for p in procs if p.ppid in pids and server and p.pid != server.pid]
        for r in runners:
            pids.add(r.pid)

        ps = http_json(f"{host}/api/ps")
        if not isinstance(ps, dict):
            be.detail = "API nicht erreichbar"
            return be, pids
        models = ps.get("models") or []
        if not models:
            be.detail = "kein Modell geladen"
            be.model = "-"
            return be, pids

        blobs = self._ollama_blob_map() if runners else {}
        now = time.time()
        for entry in models:
            full = entry.get("name", "?")
            child = Backend(kind="ollama-model", name=full.rsplit("/", 1)[-1],
                            state=RUNNING, model=full,
                            ctx=entry.get("context_length"))
            vram = entry.get("size_vram") or 0
            total = entry.get("size") or 0
            child.mem = total or None
            child.mem_kind = "GPU" if vram >= total > 0 else ("GPU-Anteil" if vram else "RAM")
            if 0 < vram < total:
                child.extras.append(f"{vram/total*100:.0f}% GPU")
            details = entry.get("details") or {}
            if details.get("parameter_size"):
                child.extras.append(details["parameter_size"])
            if details.get("quantization_level"):
                child.extras.append(details["quantization_level"])
            expires = parse_iso(entry.get("expires_at"))
            if expires:
                child.idle_in = expires - now

            runner = self._match_runner(entry, runners, blobs)
            if runner is not None:
                child.pid = runner.pid
                child.cpu = self.cpu.percent(runner.pid, runner.cpu_ticks)
                self._attach_gpu(child)
                rcfg = parse_llama_argv(runner.argv)
                child.port = rcfg.get("port")
                if rcfg.get("port"):
                    self._llama_live(child, rcfg.get("host") or "127.0.0.1",
                                     rcfg["port"], bool(rcfg.get("metrics")))
            be.children.append(child)

        if len(be.children) == 1:
            first = be.children[0]
            be.model, be.ctx, be.mem, be.mem_kind = first.model, first.ctx, first.mem, first.mem_kind
            be.tps, be.busy, be.idle_in = first.tps, first.busy, first.idle_in
            be.slots_busy, be.slots_total = first.slots_busy, first.slots_total
            be.cpu, be.gpu_mem, be.gpu_util = first.cpu, first.gpu_mem, first.gpu_util
            be.extras.extend(first.extras)
            be.children = []
        else:
            be.model = f"{len(be.children)} Modelle"
            be.busy = any(c.busy for c in be.children)
            be.mem = sum(c.mem or 0 for c in be.children) or None
            be.mem_kind = "GPU"
            be.gpu_mem = sum(c.gpu_mem or 0 for c in be.children) or None
        return be, pids

    @staticmethod
    def _match_runner(entry: dict, runners: list[ProcInfo],
                      blobs: dict[str, str]) -> ProcInfo | None:
        if not runners:
            return None
        name = entry.get("name")
        for r in runners:
            cfg = parse_llama_argv(r.argv)
            blob = Path(cfg.get("model", "")).name.replace("sha256-", "")
            if blob and blobs.get(blob) == name:
                return r
        return runners[0] if len(runners) == 1 else None

    # -- Lemonade ----------------------------------------------------------

    def collect_lemonade(self) -> tuple[Backend, set[int]]:
        be = Backend(kind="lemonade", name="lemonade")
        pids: set[int] = set()
        daemon = next(iter(proc_scan(re.compile(r"^lemond$|^lemonade-server$"))), None)
        url = self.cfg["lemonade"]["url"].rstrip("/")
        if daemon is not None:
            pids.add(daemon.pid)
            be.pid = daemon.pid
            dcfg = {}
            for i, tok in enumerate(daemon.argv):
                if tok == "--port" and i + 1 < len(daemon.argv):
                    dcfg["port"] = daemon.argv[i + 1]
                if tok == "--host" and i + 1 < len(daemon.argv):
                    dcfg["host"] = daemon.argv[i + 1]
            if dcfg.get("port"):
                url = f"http://{dcfg.get('host', '127.0.0.1')}:{dcfg['port']}"
                be.port = int(dcfg["port"])
            be.cpu = self.cpu.percent(daemon.pid, daemon.cpu_ticks)

        props = self.sd.show(self.cfg["lemonade"]["unit"],
                             ["ActiveState", "SubState", "LoadState",
                              "ActiveEnterTimestampMonotonic"], False)
        if props.get("LoadState") == "loaded":
            if props.get("ActiveState") == "active":
                be.state = RUNNING
                enter = props.get("ActiveEnterTimestampMonotonic")
                if enter and enter.isdigit():
                    be.since = time.monotonic() - int(enter) / 1e6
            else:
                be.state = STOPPED
                be.detail = props.get("SubState", "")
        elif daemon is not None:
            be.state = RUNNING
        if be.state != RUNNING:
            return be, pids

        health = http_json(f"{url}/api/v1/health")
        if not isinstance(health, dict):
            be.detail = "API nicht erreichbar"
            return be, pids
        if health.get("version"):
            be.extras.append(f"v{health['version']}")

        uptime = read_text("/proc/uptime")
        uptime_s = float(uptime.split()[0]) if uptime else None
        loaded = health.get("all_models_loaded") or []
        if not loaded:
            be.model = "-"
            be.detail = "kein Modell geladen"
            return be, pids

        stats = http_json(f"{url}/api/v1/stats")
        for entry in loaded:
            child = Backend(kind="lemonade-model", state=RUNNING,
                            name=entry.get("model_name", "?"),
                            model=entry.get("model_name", "?"),
                            ctx=(entry.get("recipe_options") or {}).get("ctx_size"))
            child.busy = bool(entry.get("is_busy") or entry.get("is_streaming"))
            child.detail = str(entry.get("backend_health") or entry.get("status") or "")
            recipe = entry.get("recipe")
            device = entry.get("device")
            if recipe:
                child.extras.append(f"{recipe}/{device}" if device else str(recipe))
            if entry.get("type") and entry["type"] != "llm":
                child.extras.append(str(entry["type"]))
            if entry.get("pinned"):
                child.extras.append("pinned")
            if not entry.get("backend_alive", True):
                child.state = STOPPED
                child.detail = "Backend tot"
            last_use = entry.get("last_use")
            if uptime_s and isinstance(last_use, (int, float)) and last_use > 0:
                idle = uptime_s - last_use / 1000.0
                if -60 < idle < uptime_s + 60:
                    child.last_use = max(0.0, idle)
            pid = entry.get("pid")
            if isinstance(pid, int) and pid > 0:
                pids.add(pid)
                info = proc_read(pid)
                if info:
                    child.pid = pid
                    child.mem, child.mem_kind = info.rss, "RSS"
                    child.cpu = self.cpu.percent(pid, info.cpu_ticks)
                    self._attach_gpu(child)
                    for kid in proc_scan(re.compile(r"^llama-server$|^sd-server$")):
                        if kid.ppid == pid:
                            pids.add(kid.pid)
            backend_url = str(entry.get("backend_url") or "")
            m = re.search(r"https?://([^:/]+):(\d+)", backend_url)
            if m and recipe == "llamacpp":
                child.port = int(m.group(2))
                self._llama_live(child, m.group(1), int(m.group(2)), True)
            elif m:
                child.port = int(m.group(2))
            if stats and child.busy and child.tps is None:
                tps = stats.get("tokens_per_second")
                if isinstance(tps, (int, float)):
                    child.tps = float(tps)
            be.children.append(child)

        active = health.get("model_loaded")
        be.busy = any(c.busy for c in be.children)
        be.mem = sum(c.mem or 0 for c in be.children) or None
        be.mem_kind = "RSS"
        be.gpu_mem = sum(c.gpu_mem or 0 for c in be.children) or None
        if len(be.children) == 1:
            first = be.children[0]
            be.model, be.ctx, be.tps = first.model, first.ctx, first.tps
            be.gpu_mem, be.gpu_util = first.gpu_mem, first.gpu_util
            be.extras.extend(first.extras)
            be.detail = first.detail
            be.children = []
        else:
            be.model = f"{len(be.children)} Modelle"
            if active:
                be.detail = f"vorn: {active}"
        if stats and isinstance(stats.get("output_tokens_total"), int):
            be.extras.append(f"{human_count(stats['output_tokens_total'])} tok gesamt")
        return be, pids

    # -- GPU ---------------------------------------------------------------

    def collect_gpu(self) -> dict:
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            dev = card / "device"
            if not (dev / "mem_info_gtt_used").exists():
                continue
            gpu: dict = {"name": read_text(dev / "product_name") or "AMD GPU",
                         "busy": read_int(dev / "gpu_busy_percent"),
                         "vram_used": read_int(dev / "mem_info_vram_used"),
                         "vram_total": read_int(dev / "mem_info_vram_total"),
                         "gtt_used": read_int(dev / "mem_info_gtt_used"),
                         "gtt_total": read_int(dev / "mem_info_gtt_total"),
                         "card": card.name}
            sclk = read_text(dev / "pp_dpm_sclk") or ""
            current = [ln for ln in sclk.splitlines() if ln.endswith("*")]
            if current:
                gpu["sclk"] = current[0].split(":")[-1].replace("*", "").strip()
            for hwmon in (dev / "hwmon").glob("hwmon*"):
                power = read_int(hwmon / "power1_average") or read_int(hwmon / "power1_input")
                if power:
                    gpu["watt"] = power / 1e6
                temp = read_int(hwmon / "temp1_input")
                if temp:
                    gpu["temp"] = temp / 1000.0
            return gpu
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=4,
                    stdin=subprocess.DEVNULL).stdout.strip()
            except (subprocess.SubprocessError, OSError):
                return {}
            if out:
                f = [x.strip() for x in out.splitlines()[0].split(",")]
                def num(idx, scale=1.0):
                    try:
                        return float(f[idx]) * scale
                    except (ValueError, IndexError):
                        return None
                return {"name": f[0], "busy": num(1), "vram_used": num(2, 1 << 20),
                        "vram_total": num(3, 1 << 20), "watt": num(4), "temp": num(5)}
        return {}

    # -- NPU ---------------------------------------------------------------

    def _npu_product(self) -> str | None:
        """Geraetename einmalig ueber xrt-smi holen; der Aufruf ist traege."""
        if self._npu_probed:
            return self._npu_name
        self._npu_probed = True
        binary = next((p for p in self.cfg["npu"]["xrt_smi"]
                       if Path(os.path.expanduser(p)).is_file()), None) or shutil.which("xrt-smi")
        if not binary:
            return None
        try:
            out = subprocess.run([os.path.expanduser(binary), "examine"],
                                 capture_output=True, text=True, timeout=10,
                                 stdin=subprocess.DEVNULL).stdout
        except (subprocess.SubprocessError, OSError):
            return None
        m = re.search(r"Processor\s*:\s*(.+)", out)
        if m:
            self._npu_name = m.group(1).strip()
        m = re.search(r"NPU Firmware Version\s*:\s*(\S+)", out)
        if m:
            self._npu_name = f"{self._npu_name or 'NPU'}"
        return self._npu_name

    def collect_npu(self) -> dict:
        devices = sorted(Path("/sys/class/accel").glob("accel*")) if \
            Path("/sys/class/accel").is_dir() else []
        if not devices:
            return {}
        dev = devices[0] / "device"
        npu: dict = {"device": devices[0].name,
                     "fw": read_text(dev / "fw_version"),
                     "power_state": read_text(dev / "power_state"),
                     "driver": None,
                     "users": []}
        driver = dev / "driver"
        if driver.is_symlink():
            npu["driver"] = os.path.basename(os.readlink(driver))
        npu["name"] = self._npu_product()
        # Auslastung liefert amdxdna nur ueber debugfs (root). Ersatzweise:
        # wer haelt das Geraet offen?
        busy = read_int(dev / "npu_busy_percent")
        if busy is not None:
            npu["busy"] = busy
        npu["users"] = self._accel_users()
        return npu

    @staticmethod
    def _accel_users() -> list[tuple[int, str]]:
        """Prozesse mit offenem /dev/accel/*. Ohne root nur die eigenen."""
        users: list[tuple[int, str]] = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            fd_dir = f"/proc/{entry}/fd"
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue
            for fd in fds:
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if "/dev/accel/" in target:
                    users.append((int(entry), read_text(f"/proc/{entry}/comm") or "?"))
                    break
        return users

    # -- System ------------------------------------------------------------

    def collect_system(self) -> dict:
        info: dict = {}
        mem = read_text("/proc/meminfo") or ""
        fields = {}
        for line in mem.splitlines():
            key, _, value = line.partition(":")
            try:
                fields[key] = int(value.split()[0]) * 1024
            except (ValueError, IndexError):
                continue
        info["mem_total"] = fields.get("MemTotal")
        info["mem_available"] = fields.get("MemAvailable")
        if info["mem_total"] and info["mem_available"] is not None:
            info["mem_used"] = info["mem_total"] - info["mem_available"]
        load = read_text("/proc/loadavg") or ""
        info["load"] = load.split()[:3] if load else []
        uptime = read_text("/proc/uptime")
        if uptime:
            info["uptime"] = float(uptime.split()[0])

        stat = read_text("/proc/stat") or ""
        first = stat.splitlines()[0] if stat else ""
        parts = [int(x) for x in first.split()[1:] if x.isdigit()]
        if len(parts) >= 4:
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
            total = sum(parts)
            if self._prev_cpu_total is not None:
                d_total = total - self._prev_cpu_total[0]
                d_idle = idle - self._prev_cpu_total[1]
                if d_total > 0:
                    info["cpu"] = max(0.0, (1 - d_idle / d_total) * 100.0)
            self._prev_cpu_total = (total, idle)
        return info

    # -- alles zusammen ----------------------------------------------------

    def _prime_guards(self) -> None:
        """Socket-Ports sperren, bevor irgendein Sammler HTTP spricht."""
        for user_scope in (True, False):
            for sock in self.sd.units(self.cfg["llama"]["unit_glob_socket"], user_scope):
                listen = self.sd.show(sock, ["Listen"], user_scope).get("Listen", "")
                for m in re.finditer(r"(?::|\b)(\d{2,5})\s*\(Stream\)", listen):
                    self.guarded_ports.add(int(m.group(1)))
        self._guards_primed = True

    def snapshot(self) -> Snapshot:
        if not self._guards_primed:
            self._prime_guards()
        snap = Snapshot(taken=time.time())
        with ThreadPoolExecutor(max_workers=5) as pool:
            f_ollama = pool.submit(self.collect_ollama)
            f_lemon = pool.submit(self.collect_lemonade)
            f_gpu = pool.submit(self.collect_gpu)
            f_npu = pool.submit(self.collect_npu)
            f_sys = pool.submit(self.collect_system)
            ollama, o_pids = f_ollama.result()
            lemon, l_pids = f_lemon.result()
            snap.gpu = f_gpu.result()
            snap.npu = f_npu.result()
            snap.system = f_sys.result()
        llama = self.collect_llama(o_pids | l_pids)
        snap.backends = [*llama, ollama, lemon]
        live = {b.pid for b in snap.backends if b.pid}
        live |= {c.pid for b in snap.backends for c in b.children if c.pid}
        self.cpu.forget_except(live)
        return snap


# --------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------

DEFAULT_CFG: dict = {
    "ui": {"interval": 2.0, "ascii": False},
    "llama": {
        "unit_glob_socket": "llama-*.socket",
        "unit_glob_service": "llama-*.service",
    },
    "ollama": {
        "url": "http://127.0.0.1:11434",
        "model_dirs": [
            "/var/lib/ollama/.ollama/models",
            "/usr/share/ollama/.ollama/models",
            os.path.expanduser("~/.ollama/models"),
        ],
    },
    "lemonade": {"url": "http://127.0.0.1:8000", "unit": "lemond.service"},
    "npu": {"xrt_smi": ["/opt/xilinx/xrt/bin/xrt-smi", "/opt/xilinx/xrt/bin/unwrapped/xrt-smi"]},
}


def load_config(path: str | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CFG))  # tiefe Kopie
    candidates = [path] if path else [
        os.environ.get("LLMTOP_CONFIG"),
        os.path.join(os.environ.get("XDG_CONFIG_HOME",
                                    os.path.expanduser("~/.config")), "llmtop", "config.toml"),
    ]
    for cand in candidates:
        if not cand or not os.path.isfile(cand) or tomllib is None:
            continue
        try:
            with open(cand, "rb") as fh:
                user = tomllib.load(fh)
        except (OSError, ValueError) as exc:
            print(f"llmtop: Konfiguration {cand} unlesbar: {exc}", file=sys.stderr)
            continue
        for section, values in user.items():
            if isinstance(values, dict):
                cfg.setdefault(section, {}).update(values)
            else:
                cfg[section] = values
        break

    host = os.environ.get("OLLAMA_HOST")
    if host:
        if not host.startswith("http"):
            host = f"http://{host}" if ":" in host else f"http://{host}:11434"
        cfg["ollama"]["url"] = host
    models = os.environ.get("OLLAMA_MODELS")
    if models:
        cfg["ollama"]["model_dirs"].insert(0, models)
    lemo = os.environ.get("LEMONADE_URL")
    if lemo:
        cfg["lemonade"]["url"] = lemo
    return cfg


# --------------------------------------------------------------------------
# Darstellung
# --------------------------------------------------------------------------

Seg = tuple[str, str]  # (Text, Stilname)

STATE_STYLE = {RUNNING: "ok", SLEEPING: "idle", STOPPED: "warn", ABSENT: "dim"}
STATE_WORD = {RUNNING: "laeuft", SLEEPING: "schlaeft", STOPPED: "gestoppt", ABSENT: "fehlt"}
KIND_TITLE = {"llama": "llama.cpp", "ollama": "Ollama", "lemonade": "Lemonade"}


def bar(pct: float | None, width: int, ascii_mode: bool) -> str:
    if pct is None:
        return " " * width
    filled = int(round(max(0.0, min(100.0, pct)) / 100.0 * width))
    if ascii_mode:
        return "#" * filled + "." * (width - filled)
    return "█" * filled + "░" * (width - filled)


def busy_style(pct: float | None) -> str:
    if pct is None:
        return "dim"
    if pct >= 85:
        return "bad"
    if pct >= 50:
        return "warn"
    return "ok"


class Renderer:
    def __init__(self, ascii_mode: bool) -> None:
        self.ascii = ascii_mode
        self.dot_on = "*" if ascii_mode else "●"
        self.dot_off = "o" if ascii_mode else "○"
        self.sep = "-" if ascii_mode else "─"
        self.mid = "|" if ascii_mode else "·"

    def rows(self, snap: Snapshot, width: int, interval: float) -> list[list[Seg]]:
        out: list[list[Seg]] = []
        out.extend(self._header(snap, width))
        out.append([])
        by_kind: dict[str, list[Backend]] = {}
        for be in snap.backends:
            by_kind.setdefault(be.kind, []).append(be)
        for kind in ("llama", "ollama", "lemonade"):
            group = by_kind.get(kind) or []
            if not group:
                continue
            out.append(self._rule(KIND_TITLE[kind], width))
            for be in group:
                out.extend(self._backend(be, width, indent=0))
                for child in be.children:
                    out.extend(self._backend(child, width, indent=2))
            out.append([])
        if not any(by_kind.values()):
            out.append([("  Kein Backend gefunden.", "dim")])
        out.append([(f"  Aktualisierung alle {interval:g}s", "dim"),
                    ("  |  q beenden  +/- Intervall  r sofort", "dim")])
        return out

    def _rule(self, title: str, width: int) -> list[Seg]:
        pad = max(0, width - len(title) - 5)
        return [(f"{self.sep}{self.sep} ", "dim"), (title, "title"),
                (" " + self.sep * pad, "dim")]

    def _header(self, snap: Snapshot, width: int) -> list[list[Seg]]:
        sysinfo = snap.system
        rows: list[list[Seg]] = []
        host = os.uname().nodename
        head = [("llmtop", "title"), (f" {VERSION}", "dim"), ("  ", ""), (host, "hi")]
        if sysinfo.get("uptime"):
            head.append((f"  up {human_delta(sysinfo['uptime'])}", "dim"))
        if sysinfo.get("load"):
            head.append(("  load " + " ".join(sysinfo["load"]), "dim"))
        rows.append(head)

        bw = 12 if width < 100 else 18
        cpu = sysinfo.get("cpu")
        line: list[Seg] = [("CPU ", "label"),
                           (bar(cpu, bw, self.ascii), busy_style(cpu)),
                           (f" {cpu:4.0f}%" if cpu is not None else "    -", "")]
        if sysinfo.get("mem_total"):
            used, total = sysinfo.get("mem_used"), sysinfo["mem_total"]
            pct = used / total * 100 if used else None
            line += [("   RAM ", "label"),
                     (f"{human_bytes(used)}/{human_bytes(total)}", ""),
                     (f" ({pct:.0f}%)" if pct else "", "dim")]
        rows.append(line)

        gpu = snap.gpu
        if gpu:
            busy = gpu.get("busy")
            line = [("GPU ", "label"), (bar(busy, bw, self.ascii), busy_style(busy)),
                    (f" {busy:4.0f}%" if busy is not None else "    -", "")]
            if gpu.get("gtt_total"):
                line += [("   GTT ", "label"),
                         (f"{human_bytes(gpu.get('gtt_used'))}/{human_bytes(gpu['gtt_total'])}", "")]
            if gpu.get("vram_total") and (gpu.get("vram_total") or 0) > (2 << 30):
                line += [("  VRAM ", "label"),
                         (f"{human_bytes(gpu.get('vram_used'))}/{human_bytes(gpu['vram_total'])}", "")]
            if gpu.get("temp"):
                line.append((f"  {gpu['temp']:.0f}°C", "dim"))
            if gpu.get("watt"):
                line.append((f" {gpu['watt']:.0f}W", "dim"))
            if gpu.get("name") and width > 110:
                line.append((f"  {gpu['name']}", "dim"))
            rows.append(line)

        npu = snap.npu
        if npu:
            users = npu.get("users") or []
            if npu.get("busy") is not None:
                state_seg = (bar(npu["busy"], bw, self.ascii), busy_style(npu["busy"]))
                tail = [(f" {npu['busy']:4.0f}%", "")]
            elif users:
                state_seg = (f"belegt: {', '.join(c for _, c in users[:3])}", "ok")
                tail = []
            else:
                state_seg = ("frei", "idle")
                tail = [("  (Auslastung nur via debugfs/root)", "dim")]
            line = [("NPU ", "label"), state_seg, *tail]
            meta = []
            if npu.get("driver"):
                meta.append(npu["driver"])
            if npu.get("fw"):
                meta.append(f"FW {npu['fw']}")
            if npu.get("power_state"):
                meta.append(npu["power_state"])
            if meta:
                line.append(("   " + f" {self.mid} ".join(meta), "dim"))
            rows.append(line)
        return rows

    def _backend(self, be: Backend, width: int, indent: int) -> list[list[Seg]]:
        pad = " " * (2 + indent)
        dot = self.dot_on if be.state == RUNNING else self.dot_off
        style = STATE_STYLE.get(be.state, "dim")
        name_w = 18 if indent == 0 else 16
        head: list[Seg] = [
            (pad, ""), (dot + " ", style),
            (be.name[:name_w].ljust(name_w) + " ", "hi" if be.state == RUNNING else ""),
            (STATE_WORD.get(be.state, be.state).ljust(9), style),
        ]
        if be.busy:
            head.append(("aktiv ", "bad"))
        if be.detail and not (be.busy and be.detail.lower() in
                              ("busy", "aktiv", "processing", "streaming")):
            head.append((be.detail + "  ", "dim"))
        if be.port:
            head.append((f":{be.port} ", "dim"))
        if be.pid:
            head.append((f"pid {be.pid} ", "dim"))
        if be.since is not None and be.state in (RUNNING, SLEEPING):
            word = "seit" if be.state == RUNNING else "ruht"
            head.append((f"{word} {human_delta(be.since)}", "dim"))
        rows = [head]

        if be.state in (ABSENT,):
            return rows

        info = " " * (4 + indent)
        second: list[Seg] = [(info, "")]
        second.append((be.model, "model"))
        bits: list[Seg] = []
        if be.ctx:
            bits.append((f"ctx {human_ctx(be.ctx)}", ""))
        if be.gpu_mem:
            bits.append((f"{human_bytes(be.gpu_mem)} GPU", ""))
            if be.mem:
                bits.append((f"{human_bytes(be.mem)} RSS", "dim"))
        elif be.mem:
            bits.append((f"{human_bytes(be.mem)} {be.mem_kind}".strip(), ""))
        for extra in be.extras:
            bits.append((extra, "dim"))
        for seg in bits:
            second.append((f" {self.mid} ", "dim"))
            second.append(seg)
        rows.append(second)

        third: list[Seg] = [(info, "")]
        has = False
        if be.cpu is not None:
            third += [("CPU ", "label"), (f"{be.cpu:5.1f}%", busy_style(be.cpu))]
            has = True
        if be.gpu_util is not None:
            third += [("   GPU ", "label"),
                      (f"{be.gpu_util:5.1f}%", busy_style(be.gpu_util))]
            has = True
        if be.slots_total:
            third += [("   Slots ", "label"),
                      (f"{be.slots_busy}/{be.slots_total}",
                       "bad" if be.slots_busy else "")]
            has = True
        if be.tps is not None:
            third += [("   ", ""), (f"{be.tps:.1f} tok/s",
                                    "bad" if be.tps > 0.05 else "dim")]
            has = True
        if be.last_use is not None and not be.busy:
            third += [("   ", ""), (f"zuletzt vor {human_delta(be.last_use)}", "dim")]
            has = True
        if be.idle_in is not None:
            word = "entlaedt in" if be.idle_in > 0 else "entladen seit"
            third += [("   ", ""), (f"{word} {human_delta(abs(be.idle_in))}",
                                    "warn" if 0 < be.idle_in < 120 else "dim")]
            has = True
        if has:
            rows.append(third)
        return rows


# --------------------------------------------------------------------------
# Ausgabe: einmalig, JSON, TUI
# --------------------------------------------------------------------------

ANSI = {
    "": "\033[0m", "dim": "\033[2m", "ok": "\033[32m", "idle": "\033[36m",
    "warn": "\033[33m", "bad": "\033[31m", "hi": "\033[1m", "title": "\033[1;34m",
    "label": "\033[2m", "model": "\033[35m",
}


def print_rows(rows: list[list[Seg]], color: bool) -> None:
    for row in rows:
        if not color:
            print("".join(text for text, _ in row).rstrip())
            continue
        parts = []
        for text, style in row:
            parts.append(f"{ANSI.get(style, '')}{text}\033[0m" if style else text)
        print("".join(parts).rstrip())


def backend_to_dict(be: Backend) -> dict:
    data = {
        "kind": be.kind, "name": be.name, "state": be.state, "detail": be.detail,
        "model": be.model, "ctx": be.ctx, "memory_bytes": be.mem,
        "memory_kind": be.mem_kind, "cpu_percent": be.cpu,
        "gpu_memory_bytes": be.gpu_mem, "gpu_percent": be.gpu_util,
        "tokens_per_second": be.tps, "busy": be.busy,
        "slots_busy": be.slots_busy, "slots_total": be.slots_total,
        "unload_in_seconds": be.idle_in, "since_seconds": be.since,
        "last_use_seconds_ago": be.last_use,
        "port": be.port, "pid": be.pid, "extras": be.extras,
    }
    if be.children:
        data["models"] = [backend_to_dict(c) for c in be.children]
    return data


def snapshot_to_dict(snap: Snapshot) -> dict:
    return {
        "taken": snap.taken,
        "backends": [backend_to_dict(b) for b in snap.backends],
        "gpu": snap.gpu, "npu": snap.npu, "system": snap.system,
    }


def run_tui(collector: Collector, interval: float, ascii_mode: bool) -> int:
    import curses
    import select
    import threading

    state: dict = {"snap": None, "err": None, "interval": interval, "stop": False,
                   "force": threading.Event()}

    def worker() -> None:
        while not state["stop"]:
            try:
                state["snap"] = collector.snapshot()
                state["err"] = None
            except Exception as exc:  # Sammeln darf die Anzeige nie killen
                state["err"] = f"{type(exc).__name__}: {exc}"
            state["force"].wait(state["interval"])
            state["force"].clear()

    def draw(stdscr) -> int:
        curses.curs_set(0)
        stdscr.nodelay(True)
        pairs = {}
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            spec = {"ok": curses.COLOR_GREEN, "idle": curses.COLOR_CYAN,
                    "warn": curses.COLOR_YELLOW, "bad": curses.COLOR_RED,
                    "title": curses.COLOR_BLUE, "model": curses.COLOR_MAGENTA}
            for i, (name, color) in enumerate(spec.items(), start=1):
                curses.init_pair(i, color, -1)
                pairs[name] = curses.color_pair(i)
            pairs["hi"] = curses.A_BOLD
            pairs["dim"] = curses.A_DIM
            pairs["label"] = curses.A_DIM
            pairs["title"] = pairs["title"] | curses.A_BOLD

        renderer = Renderer(ascii_mode)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            while True:  # alle anliegenden Tasten abarbeiten
                try:
                    key = stdscr.getch()
                except curses.error:
                    key = -1
                if key == -1:
                    break
                if key in (ord("q"), ord("Q"), 27):
                    state["stop"] = True
                    state["force"].set()
                    return 0
                if key in (ord("+"), ord("=")):
                    state["interval"] = min(60.0, state["interval"] + 0.5)
                elif key == ord("-"):
                    state["interval"] = max(0.5, state["interval"] - 0.5)
                elif key in (ord("r"), ord("R"), ord(" ")):
                    state["force"].set()

            height, width = stdscr.getmaxyx()
            stdscr.erase()
            snap = state["snap"]
            if snap is None:
                stdscr.addnstr(0, 0, "llmtop sammelt Daten ...", width - 1)
            else:
                rows = renderer.rows(snap, width - 1, state["interval"])
                for y, row in enumerate(rows):
                    if y >= height - 1:
                        break
                    x = 0
                    for text, style in row:
                        if x >= width - 1:
                            break
                        chunk = text[: max(0, width - 1 - x)]
                        try:
                            stdscr.addnstr(y, x, chunk, width - 1 - x,
                                           pairs.get(style, 0))
                        except curses.error:
                            pass
                        x += len(chunk)
            if state["err"]:
                try:
                    stdscr.addnstr(height - 1, 0, f"Fehler: {state['err']}"[:width - 1],
                                   width - 1, pairs.get("bad", 0))
                except curses.error:
                    pass
            stdscr.refresh()
            # Bewusst select statt curses.napms(): napms gibt die GIL nicht
            # frei und wuerde den Sammel-Thread aushungern - die Anzeige blieb
            # dann sekundenlang auf "sammelt Daten" stehen.
            try:
                select.select([sys.stdin], [], [], 0.12)
            except (OSError, ValueError):
                time.sleep(0.12)

    return curses.wrapper(draw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmtop",
        description="Zustand lokaler LLM-Backends: llama.cpp, Ollama, Lemonade, "
                    "dazu GPU und NPU.")
    parser.add_argument("-1", "--once", action="store_true",
                        help="einmal ausgeben statt TUI")
    parser.add_argument("--json", action="store_true", help="JSON ausgeben und beenden")
    parser.add_argument("-n", "--interval", type=float, default=None,
                        help="Aktualisierungsintervall in Sekunden (Standard 2)")
    parser.add_argument("--ascii", action="store_true",
                        help="nur ASCII, keine Blockzeichen")
    parser.add_argument("--no-color", action="store_true", help="ohne Farbe (mit --once)")
    parser.add_argument("--config", help="Pfad zur Konfigurationsdatei")
    parser.add_argument("--version", action="version", version=f"llmtop {VERSION}")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    interval = args.interval if args.interval is not None else float(cfg["ui"]["interval"])
    ascii_mode = args.ascii or bool(cfg["ui"].get("ascii"))
    collector = Collector(cfg)

    def sampled() -> Snapshot:
        """Zweimal messen: CPU-Prozent und tok/s entstehen erst aus dem Delta."""
        collector.snapshot()
        time.sleep(min(1.0, max(0.3, interval / 2)))
        return collector.snapshot()

    if args.json:
        snap = sampled()
        print(json.dumps(snapshot_to_dict(snap), indent=2, ensure_ascii=False))
        return 0

    if args.once:
        snap = sampled()
        color = sys.stdout.isatty() and not args.no_color
        print_rows(Renderer(ascii_mode).rows(snap, shutil.get_terminal_size().columns - 1,
                                             interval), color)
        return 0

    if not sys.stdout.isatty():
        print("llmtop: kein Terminal, nutze --once oder --json", file=sys.stderr)
        return 2
    try:
        return run_tui(collector, interval, ascii_mode)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
