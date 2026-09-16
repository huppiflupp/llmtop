#!/usr/bin/env python3
"""llmtop - an htop-style overview of local LLM backends.

Shows, for llama.cpp, Ollama and Lemonade Server, whether they are up, which
model is loaded, how much memory it holds and what is going through it right
now, plus integrated GPU and NPU state.

Standard library only, Python >= 3.11.

Important: socket-activated llama.cpp backends are never probed on their
socket port - that would trigger a model load just to answer "is it running?".
State comes from systemd, and measurements only from the internal backend port
and only while the service is already up.
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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

VERSION = "0.6.0"
HTTP_TIMEOUT = 1.5
CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
HISTORY = 1024  # samples kept per graph, independent of the drawn width


# --------------------------------------------------------------------------
# small helpers
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
    """Context window, compact. 131072 -> 128k, 34630 -> 33.8k."""
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
    """Compact duration: 45s, 12m, 3h04, 2d05h."""
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
    return f"{sign}{seconds//86400}d{(seconds%86400)//3600:02d}h"


def parse_iso(ts: str | None) -> float | None:
    """ISO-8601 from Go/Python to unix time. Nanoseconds get truncated."""
    if not ts:
        return None
    cleaned = re.sub(r"(\.\d{6})\d+", r"\1", ts.replace("Z", "+00:00"))
    try:
        import datetime

        return datetime.datetime.fromisoformat(cleaned).timestamp()
    except ValueError:
        return None


# --------------------------------------------------------------------------
# braille drawing
# --------------------------------------------------------------------------
#
# A braille cell is a 2x4 dot matrix, so one character holds two samples
# horizontally and four steps vertically - the same trick btop uses for its
# graphs. Dot bit values within U+2800:
#
#     left column, top to bottom : 0x01 0x02 0x04 0x40
#     right column, top to bottom: 0x08 0x10 0x20 0x80

BRAILLE_BASE = 0x2800
DOT_BITS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))
CELL_FULL = chr(BRAILLE_BASE | 0xFF)
CELL_LEFT = chr(BRAILLE_BASE | 0x47)
CELL_TRACK = chr(BRAILLE_BASE | 0xC0)   # bottom row only: an empty rail
CELL_EMPTY = chr(BRAILLE_BASE)

# Green through yellow to red, as xterm-256 indices. Terminals limited to
# eight colours fall back to plain green/yellow/red (see Palette).
GRADIENT = (46, 82, 118, 154, 190, 226, 220, 214, 208, 202, 196)
GRAD_N = len(GRADIENT)


def grad_style(fraction: float) -> str:
    """Pick a gradient step for a 0..1 value."""
    idx = int(max(0.0, min(1.0, fraction)) * (GRAD_N - 1) + 0.5)
    return f"grad{idx}"


Seg = tuple[str, str]  # (text, style name)


class Graph:
    """A scrolling braille graph that keeps more history than it draws.

    The buffer is deliberately much wider than any terminal, so resizing the
    window only changes how much of the past is visible - no samples are lost
    and the graph reflows instantly.
    """

    def __init__(self) -> None:
        self.samples: deque[float] = deque(maxlen=HISTORY)
        self.seen = True

    def push(self, value: float | None) -> None:
        self.samples.append(float(value) if value is not None else 0.0)

    def scale(self, floor: float = 1.0) -> float:
        return max(floor, max(self.samples, default=0.0))

    def render(self, width: int, height: int, maxval: float,
               ascii_mode: bool) -> list[list[Seg]]:
        """Draw the graph as `height` rows of `width` characters."""
        width = max(1, width)
        height = max(1, height)
        maxval = max(1e-9, maxval)
        need = width * 2
        data = list(self.samples)[-need:]
        data = [0.0] * (need - len(data)) + data

        if ascii_mode:
            ramp = " .:-=+*#%@"
            rows = []
            for row in range(height):
                line = []
                for col in range(width):
                    value = max(data[2 * col], data[2 * col + 1]) / maxval
                    band = (height - 1 - row) / height
                    local = max(0.0, min(1.0, (value - band) * height))
                    line.append((ramp[int(local * (len(ramp) - 1))],
                                 grad_style(value)))
                rows.append(line)
            return rows

        total_dots = height * 4
        rows: list[list[Seg]] = []
        for row in range(height):
            line: list[Seg] = []
            for col in range(width):
                bits = 0
                for side in (0, 1):
                    value = data[2 * col + side] / maxval
                    # At least one dot, so an idle graph still shows a
                    # baseline instead of vanishing - same as btop does.
                    filled = max(1, int(round(max(0.0, min(1.0, value)) * total_dots)))
                    for k in range(4):
                        if (row * 4 + k) >= total_dots - filled:
                            bits |= DOT_BITS[side][k]
                if height > 1:
                    # Tall graphs get btop's vertical gradient: the higher a
                    # dot sits, the hotter its colour.
                    band = (total_dots - (row * 4 + 2)) / total_dots
                    style = grad_style(band)
                else:
                    style = grad_style(max(data[2 * col], data[2 * col + 1]) / maxval)
                line.append((chr(BRAILLE_BASE | bits) if bits else CELL_EMPTY,
                             style if bits else "track"))
            rows.append(line)
        return rows


def mem_pct(used: float | None, total: float | None) -> float | None:
    return (used or 0) / total * 100 if total else None


def gpu_mem(gpu: dict) -> tuple[str, float | None, float | None]:
    """The GPU's memory pool: GTT on unified memory, VRAM otherwise."""
    if gpu.get("gtt_total"):
        return "GTT", gpu.get("gtt_used"), gpu["gtt_total"]
    return "VRAM", gpu.get("vram_used"), gpu.get("vram_total")


def meter(pct: float | None, width: int, ascii_mode: bool) -> list[Seg]:
    """A horizontal level bar with a left-to-right gradient, like btop's."""
    width = max(1, width)
    if ascii_mode:
        filled = int(round(max(0.0, min(100.0, pct or 0.0)) / 100.0 * width))
        return [("#" * filled, grad_style(0.5)), ("." * (width - filled), "track")]
    if pct is None:
        return [(CELL_TRACK * width, "track")]
    dots = int(round(max(0.0, min(100.0, pct)) / 100.0 * width * 2))
    out: list[Seg] = []
    for col in range(width):
        position = col / max(1, width - 1)
        if dots >= (col + 1) * 2:
            out.append((CELL_FULL, grad_style(position)))
        elif dots == col * 2 + 1:
            out.append((CELL_LEFT, grad_style(position)))
        else:
            out.append((CELL_TRACK, "track"))
    return out


# --------------------------------------------------------------------------
# processes
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
    # comm may contain spaces and parentheses, so start after the last ")".
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
    """Find processes by program name.

    Deliberately only comm and the basename of argv[0], never the full command
    line: earlyoom runs with "--prefer ^(python3?|llama-server|ollama)$" and
    would otherwise be counted as a backend.
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
    """Per-process GPU memory and GPU time from /proc/<pid>/fdinfo.

    On unified-memory systems (Strix Halo and relatives) the model lives in
    GTT and never shows up in RSS - only drm-resident-gtt reveals what is
    really held. Readable for own processes only; a service running as its own
    user, such as Ollama, yields nothing.
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
        # Several descriptors can refer to the same DRM client; without this
        # the memory would be counted more than once.
        old = clients.get(client)
        if old is None or entry["mem"] > old["mem"]:
            clients[client] = entry
    if not clients:
        return None
    return {"mem": sum(c["mem"] for c in clients.values()),
            "engine_ns": sum(c["engine_ns"] for c in clients.values())}


LEAD_MIN = 0.10  # below this share of the device nothing counts as leading


def leading_device(cpu_frac: float | None, gpu_frac: float | None) -> str | None:
    """Which device carries a backend: "cpu", "gpu" or None when idle.

    CPU is measured against all logical CPUs, GPU against the one device, so
    both are a share of what that side has. A GPU model keeps one or two
    cores polling (a few percent of the machine) while the GPU sits near
    100%; a CPU model is the other way round.
    """
    cpu, gpu = cpu_frac or 0.0, gpu_frac or 0.0
    if max(cpu, gpu) < LEAD_MIN:
        return None
    return "cpu" if cpu > gpu else "gpu"


def attribute_gpu(snap: Snapshot) -> None:
    """Fill in the GPU load of a backend whose own figure is unreadable.

    Services running as their own user (Ollama, Lemonade) hide their fdinfo,
    so their GPU load is unknown even while they are the only thing on the
    GPU. When exactly one such backend is working, it gets whatever part of
    the device load the readable processes do not account for, marked as an
    estimate. With two or more candidates nothing is guessed.
    """
    busy = snap.gpu.get("busy")
    if busy is None:
        return
    leaves = [c for be in snap.backends for c in (be.children or [be])]
    known = sum(n.gpu_util for n in leaves if n.gpu_util is not None)
    unknown = [n for n in leaves if n.gpu_util is None
               and (n.busy or (n.tps or 0) > 0.05)]
    if len(unknown) == 1:
        unknown[0].gpu_util = max(0.0, busy - known)
        unknown[0].gpu_estimated = True


class GpuTimeTracker:
    """Per-process GPU load from the cumulative engine time."""

    def __init__(self) -> None:
        self._prev: dict[int, tuple[float, int]] = {}
        self._touched: set[int] = set()

    def sweep(self) -> None:
        """Forget processes that were not measured in this round."""
        for pid in [p for p in self._prev if p not in self._touched]:
            del self._prev[pid]
        self._touched.clear()

    def percent(self, pid: int, engine_ns: int) -> float | None:
        self._touched.add(pid)
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
    """Per-PID CPU percentage from /proc deltas, without external tools."""

    def __init__(self) -> None:
        self._prev: dict[int, tuple[float, int]] = {}
        self._touched: set[int] = set()

    def percent(self, pid: int, ticks: int) -> float | None:
        self._touched.add(pid)
        now = time.monotonic()
        prev = self._prev.get(pid)
        self._prev[pid] = (now, ticks)
        if prev is None:
            return None
        elapsed = now - prev[0]
        if elapsed <= 0:
            return None
        return max(0.0, (ticks - prev[1]) / CLK_TCK / elapsed * 100.0)

    def sweep(self) -> None:
        """Forget processes that were not measured in this round.

        Tracking what was touched rather than which PIDs end up displayed
        matters: a single loaded model is folded into its parent row, and
        its runner PID would otherwise be dropped - and its CPU reading with
        it - on every round.
        """
        for pid in [p for p in self._prev if p not in self._touched]:
            del self._prev[pid]
        self._touched.clear()


class RateTracker:
    """tokens/s from monotonically growing counters."""

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
        if elapsed <= 0 or delta < 0:  # counter was reset
            return None
        return delta / elapsed


# --------------------------------------------------------------------------
# reading llama-server command lines
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
    """Pull the llama-server options that matter for the display."""
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
    """A readable name: the alias, else directory/file rather than a blob."""
    if alias:
        return alias
    if not path:
        return "-"
    p = Path(path)
    if p.name.startswith("sha256-") or p.name.startswith("sha256:"):
        return f"blob {p.name[7:19]}"
    stem = p.stem
    # Shorten "Model-00001-of-00003" back to the base name.
    stem = re.sub(r"-\d{5}-of-\d{5}$", "", stem)
    parent = p.parent.name
    if parent and parent.lower() not in {"models", "gguf", ".", "/"} and len(stem) < 12:
        return f"{parent}/{stem}"
    return stem


# --------------------------------------------------------------------------
# systemd
# --------------------------------------------------------------------------

class Systemd:
    """A thin shell around systemctl. Without systemd everything comes back empty."""

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


ARGV_RE = re.compile(r"argv\[\]=(.*?)\s+;")


def execstart_argv(show_value: str) -> list[str]:
    """argv of the last ExecStart from `systemctl show --property=ExecStart`.

    That form is preferred because systemd has already expanded the specifiers
    (%h, %t, ...) there - `systemctl cat` still shows them raw.
    """
    matches = ARGV_RE.findall(show_value or "")
    if not matches:
        return []
    return matches[-1].split()


def resolve_llama_config(argv: list[str], depth: int = 0) -> dict:
    """Find llama-server options even when ExecStart points at a script."""
    if not argv:
        return {}
    cfg = parse_llama_argv(argv)
    if cfg.get("port") or cfg.get("model"):
        return cfg
    if depth > 1:
        return cfg
    # ExecStart points at a wrapper script: read it for llama-server options,
    # without executing anything.
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
    # The invocation line carried nothing useful - typical for scripts that
    # collect options in a bash array and pass them as "${ARGS[@]}". Then read
    # the whole script as a pool of arguments.
    flat = expand(body).replace("(", " ").replace(")", " ")
    tokens = [t.strip('"').strip("'") for t in flat.split()]
    cfg.update(parse_llama_argv(tokens))
    return cfg


# --------------------------------------------------------------------------
# data model
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
    gpu_estimated: bool = False  # gpu_util inferred from the device total
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

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass
class Snapshot:
    backends: list[Backend] = field(default_factory=list)
    gpu: dict = field(default_factory=dict)
    npu: dict = field(default_factory=dict)
    system: dict = field(default_factory=dict)
    taken: float = 0.0


# --------------------------------------------------------------------------
# collectors
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
        self._machine: dict | None = None

    # -- llama.cpp ---------------------------------------------------------

    def _attach_gpu(self, be: Backend) -> None:
        """Add GPU memory and GPU time, as far as /proc allows."""
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

    def _llama_live(self, be: Backend, host: str, port: int, use_metrics: bool) -> None:
        """Measure a running backend. Internal ports only, never sockets."""
        if port in self.guarded_ports:
            be.extras.append("skipped measurement (socket port)")
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
            be.extras.append("/slots disabled")
        if use_metrics and be.tps is None:
            # Fallback only: llamacpp:tokens_predicted_total is written when a
            # task finishes and stands still during generation - /slots counts
            # along live.
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

    def _llama_extras(self, be: Backend, cfg: dict) -> None:
        if cfg.get("draft"):
            be.extras.append(f"draft {cfg['draft']}")
        if cfg.get("flash_attn"):
            be.extras.append("fa")
        if cfg.get("ngl"):
            be.extras.append(f"ngl {cfg['ngl']}")

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
                # The like-named service first: if a proxy (systemd-socket-proxyd)
                # sits in front of the backend it is what Triggers names, but it
                # knows neither model nor context.
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
                # With no recognisable model and no running process this is not
                # a llama backend but scaffolding, such as a socket proxy.
                if be is None or (be.model == "-" and not be.pid):
                    continue
                backends.append(be)
                if be.pid:
                    claimed_pids.add(be.pid)

        # free-standing llama-server processes belonging to no unit
        for proc in proc_scan(re.compile(r"^llama-server$")):
            if proc.pid in claimed_pids or proc.pid in owned_pids:
                continue
            if proc.ppid in owned_pids:
                continue
            cfg = parse_llama_argv(proc.argv)
            be = Backend(kind="llama", name=cfg.get("alias") or f"llama-server:{proc.pid}",
                         state=RUNNING, detail="standalone", pid=proc.pid,
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
            be.extras.append("model file missing")

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
                if info is None:  # MainPID is the wrapper, look for the child
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
            be.detail = "socket-activated"
            left = props.get("InactiveEnterTimestampMonotonic")
            if left and left.isdigit() and int(left) > 0:
                be.since = now_mono - int(left) / 1e6
        else:
            be.state = STOPPED
            be.detail = props.get("SubState", "")
        return be

    # -- Ollama ------------------------------------------------------------

    def _ollama_blob_map(self) -> dict[str, str]:
        """Model-layer blob sha256 -> model name:tag, taken from the manifests."""
        roots = [Path(os.path.expanduser(p)) for p in self.cfg["ollama"]["model_dirs"]]
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
        """Manifest path -> the name as /api/ps reports it.

        registry.ollama.ai/library/gemma4/12b          -> gemma4:12b
        registry.ollama.ai/ns/nexus-medical/latest     -> ns/nexus-medical:latest
        hf.co/user/Some-Model-GGUF/Q4_K_M              -> hf.co/user/Some-Model-GGUF:Q4_K_M
        """
        if len(parts) < 2:
            return ":".join(parts)
        path, tag = list(parts[:-1]), parts[-1]
        # The default registry never appears in the name, nor does "library".
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

        # Every child of "ollama serve" is a runner, whether it shows up as
        # llama-server or as "ollama runner".
        runners = [p for p in procs if p.ppid in pids and server and p.pid != server.pid]
        for r in runners:
            pids.add(r.pid)

        ps = http_json(f"{host}/api/ps")
        if not isinstance(ps, dict):
            be.detail = "API unreachable"
            return be, pids
        models = ps.get("models") or []
        if not models:
            be.detail = "no model loaded"
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
            child.mem_kind = "GPU" if vram >= total > 0 else ("part GPU" if vram else "RAM")
            if 0 < vram < total:
                child.extras.append(f"{vram/total*100:.0f}% GPU")
            details = entry.get("details") or {}
            if details.get("parameter_size"):
                child.extras.append(details["parameter_size"])
            if details.get("quantization_level"):
                child.extras.append(details["quantization_level"])
            expires = parse_iso(entry.get("expires_at"))
            # While a request runs Ollama reports a zero time (year 1); that
            # is "not scheduled", not "already unloaded".
            if expires and expires > 946684800:
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
            be.model = f"{len(be.children)} models"
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
            be.detail = "API unreachable"
            return be, pids
        if health.get("version"):
            be.extras.append(f"v{health['version']}")

        uptime = read_text("/proc/uptime")
        uptime_s = float(uptime.split()[0]) if uptime else None
        loaded = health.get("all_models_loaded") or []
        if not loaded:
            be.model = "-"
            be.detail = "no model loaded"
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
                child.detail = "backend down"
            last_use = entry.get("last_use")
            # last_use is monotonic milliseconds, so it only means something
            # measured against /proc/uptime.
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
            be.last_use = first.last_use
            be.extras.extend(first.extras)
            be.detail = first.detail
            be.children = []
        else:
            be.model = f"{len(be.children)} models"
            if active:
                be.detail = f"front: {active}"
        if stats and isinstance(stats.get("output_tokens_total"), int):
            be.extras.append(f"{human_count(stats['output_tokens_total'])} tok total")
        return be, pids

    # -- GPU ---------------------------------------------------------------

    def collect_gpu(self) -> dict:
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            dev = card / "device"
            if not (dev / "mem_info_gtt_used").exists():
                continue
            name = read_text(dev / "product_name")
            if not name:
                # Integrated GPUs rarely name themselves in sysfs, but the CPU
                # model string does: "... w/ Radeon 8060S".
                m = re.search(r"\bw/\s*(.+)$", self.machine().get("cpu_name") or "")
                name = m.group(1).strip() if m else "AMD GPU"
            gpu: dict = {"name": name,
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
        """Fetch the device name via xrt-smi once; the call is slow."""
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
        # amdxdna only exposes utilisation through debugfs, which needs root.
        # Without it, fall back to: who has the device open?
        busy = read_int(dev / "npu_busy_percent")
        if busy is not None:
            npu["busy"] = busy
        npu["users"] = self._accel_users()
        return npu

    @staticmethod
    def _accel_users() -> list[tuple[int, str]]:
        """Processes holding /dev/accel/* open. Without root, only our own."""
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

    # -- system ------------------------------------------------------------

    def machine(self) -> dict:
        """Host, machine model and CPU name - read once, they do not change."""
        if self._machine is not None:
            return self._machine
        junk = {"", "to be filled by o.e.m.", "default string", "system product name",
                "system manufacturer", "not specified", "none", "o.e.m."}

        def dmi(name: str) -> str | None:
            value = (read_text(f"/sys/class/dmi/id/{name}") or "").strip()
            return None if value.lower() in junk else value

        cpu_name = None
        for line in (read_text("/proc/cpuinfo") or "").splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
        vendor, product = dmi("sys_vendor"), dmi("product_name")
        machine = (product or "").replace("_", " ")
        if vendor and vendor.lower() not in machine.lower():
            machine = f"{vendor} {machine}".strip()
        self._machine = {"host": os.uname().nodename, "machine": machine or None,
                         "cpu_name": cpu_name, "threads": os.cpu_count()}
        return self._machine

    def collect_system(self) -> dict:
        info: dict = dict(self.machine())
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

    # -- everything together ------------------------------------------------

    def _prime_guards(self) -> None:
        """Block socket ports before any collector speaks HTTP."""
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
        attribute_gpu(snap)
        self.cpu.sweep()
        self.gputime.sweep()
        return snap


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

DEFAULT_CFG: dict = {
    "ui": {"interval": 2.0, "ascii": False, "graph_height": "auto",
           "background": 234},
    "llama": {
        "unit_glob_socket": "llama-*.socket",
        "unit_glob_service": "llama-*.service",
    },
    "ollama": {
        "url": "http://127.0.0.1:11434",
        "model_dirs": [
            "/var/lib/ollama/.ollama/models",
            "/usr/share/ollama/.ollama/models",
            "~/.ollama/models",
        ],
    },
    "lemonade": {"url": "http://127.0.0.1:8000", "unit": "lemond.service"},
    "npu": {"xrt_smi": ["/opt/xilinx/xrt/bin/xrt-smi",
                        "/opt/xilinx/xrt/bin/unwrapped/xrt-smi"]},
}


def load_config(path: str | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CFG))  # deep copy
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
            print(f"llmtop: cannot read config {cand}: {exc}", file=sys.stderr)
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
# rendering
# --------------------------------------------------------------------------

STATE_STYLE = {RUNNING: "ok", SLEEPING: "idle", STOPPED: "warn", ABSENT: "dim"}
STATE_WORD = {RUNNING: "running", SLEEPING: "asleep", STOPPED: "stopped",
              ABSENT: "absent"}
KIND_TITLE = {"llama": "llama.cpp", "ollama": "Ollama", "lemonade": "Lemonade"}


ELLIPSIS = "…"


def fit(text: str, width: int) -> str:
    """Exactly `width` characters: padded, or cut with an ellipsis."""
    if width <= 0:
        return ""
    if len(text) <= width:
        return text.ljust(width)
    return text[: width - 1] + ELLIPSIS


class Line:
    """One output line of fixed width.

    Fields go in left to right. Once something fails to fit, the line is
    closed: a shorter field further along must never slip into an earlier
    column, because that slipping is exactly what makes a display wander.
    """

    def __init__(self, width: int) -> None:
        self.width = max(1, width)
        self.segs: list[Seg] = []
        self.used = 0
        self.full = False

    @property
    def left(self) -> int:
        return self.width - self.used

    def add(self, text: str, style: str = "") -> bool:
        if not text:
            return True
        if self.full or len(text) > self.left:
            self.full = True
            return False
        self.segs.append((text, style))
        self.used += len(text)
        return True

    def add_segs(self, segs: list[Seg]) -> bool:
        total = sum(len(t) for t, _ in segs)
        if self.full or total > self.left:
            self.full = True
            return False
        self.segs.extend(segs)
        self.used += total
        return True

    def add_clipped(self, text: str, style: str = "", minimum: int = 4) -> bool:
        """Append the trailing field, shortened with an ellipsis if needed."""
        if not text:
            return True
        if self.full or self.left < minimum:
            self.full = True
            return False
        if len(text) > self.left:
            text = text[: self.left - 1] + ELLIPSIS
        return self.add(text, style)

    def padded(self) -> list[Seg]:
        if self.used < self.width:
            return [*self.segs, (" " * (self.width - self.used), "")]
        return list(self.segs)


class Layout:
    """Sizes derived from the current terminal, recomputed on every frame.

    Every panel spans the full width. From SPLIT_MIN columns on, each panel is
    divided down the middle: system and identity on the left, GPU and live
    measurements on the right. Below that the halves are stacked. Either way
    every field keeps a fixed column.
    """

    SPLIT_MIN = 90

    def __init__(self, width: int, height: int, graph_rows: int) -> None:
        self.width = max(40, width)
        self.height = max(8, height)
        self.inner = self.width - 4  # inside "│ " ... " │"
        self.split = self.width >= self.SPLIT_MIN
        if self.split:
            self.left_w = (self.inner - 3) // 2
            self.right_w = self.inner - 3 - self.left_w
            self.divider_col = 2 + self.left_w + 1
        else:
            self.left_w = self.right_w = self.inner
            self.divider_col = None
        self.name_w = 18 if self.left_w >= 52 else (14 if self.left_w >= 42 else 11)
        self.show_pid = self.left_w >= 62
        # Load and memory graphs share their rows like btop's boxes: CPU
        # with RAM, GPU with GTT. An odd row goes to the load graph.
        graph_rows = max(2, graph_rows)
        self.graph_h = (graph_rows + 1) // 2
        self.mem_h = graph_rows // 2


MEM_KIND = {"GPU": "GPU", "part GPU": "GPU", "RAM": "RAM", "RSS": "RSS"}
LABEL_W = 5  # "CPU  "
BOX_STYLE = {"system": "b_system", "llama": "b_llama",
             "ollama": "b_ollama", "lemonade": "b_lemonade"}


class Renderer:
    def __init__(self, ascii_mode: bool, graph_height: str | int = "auto") -> None:
        self.ascii = ascii_mode
        self.graph_height = graph_height
        if ascii_mode:
            self.dot_on, self.dot_off, self.mid = "*", "o", "|"
            self.h, self.v = "-", "|"
            self.corners = ("+", "+", "+", "+")
            self.tees = ("+", "+")
            self.tab_top = self.tab_bottom = ("[", "]")
        else:
            self.dot_on, self.dot_off, self.mid = "●", "○", "·"
            self.h, self.v = "─", "│"
            self.corners = ("╭", "╮", "╰", "╯")
            self.tees = ("┬", "┴")
            self.tab_top = ("┐", "┌")      # ┐title┌ like btop
            self.tab_bottom = ("┘", "└")   # ┘title└
        self.graphs: dict[str, Graph] = {}
        self._last_fed = -1.0

    # -- history ----------------------------------------------------------

    def graph(self, key: str) -> Graph:
        g = self.graphs.get(key)
        if g is None:
            g = self.graphs[key] = Graph()
        g.seen = True
        return g

    def feed(self, snap: Snapshot) -> None:
        """Take one sample per snapshot - never per drawn frame."""
        if snap.taken == self._last_fed:
            return
        self._last_fed = snap.taken
        for g in self.graphs.values():
            g.seen = False
        self.graph("cpu").push(snap.system.get("cpu"))
        self.graph("gpu").push(snap.gpu.get("busy"))
        self.graph("ram").push(mem_pct(snap.system.get("mem_used"),
                                       snap.system.get("mem_total")))
        self.graph("gtt").push(mem_pct(*gpu_mem(snap.gpu)[1:]))
        for be in snap.backends:
            for node in (be, *be.children):
                if node.tps is not None:
                    self.graph(f"tps:{node.key}").push(node.tps)
        # Drop graphs of backends that are gone, so they cannot pile up.
        for key in [k for k, g in self.graphs.items()
                    if not g.seen and k.startswith("tps:")]:
            del self.graphs[key]

    # -- panels -------------------------------------------------------------

    def _border(self, lay: Layout, top: bool, style: str,
                left: list[Seg] | None = None, center: list[Seg] | None = None,
                right: list[Seg] | None = None) -> list[Seg]:
        """A panel edge with optional title tabs set into the line."""
        width = lay.width
        chars = [self.h] * width
        styles = [style] * width
        tl, tr, bl, br = self.corners
        chars[0], chars[-1] = (tl, tr) if top else (bl, br)
        if lay.divider_col is not None:
            chars[lay.divider_col] = self.tees[0] if top else self.tees[1]
        opening, closing = self.tab_top if top else self.tab_bottom
        taken: list[tuple[int, int]] = []

        def place(segs: list[Seg] | None, where: str) -> None:
            if not segs:
                return
            cells = [(opening, style)]
            for text, sty in segs:
                cells.extend((ch, sty) for ch in text)
            cells.append((closing, style))
            n = len(cells)
            pos = {"left": 2, "right": width - 2 - n, "center": (width - n) // 2}[where]
            if where == "center":
                # Centre in the space the side tabs leave, keeping one line
                # character on either side.
                lo = max([b + 1 for a, b in taken if a < width // 2] + [2])
                hi = min([a - 1 for a, b in taken if a >= width // 2] + [width - 2])
                if hi - lo < n:
                    return
                pos = min(max(pos, lo), hi - n)
            if pos < 1 or pos + n > width - 1:
                return
            if any(pos < b and a < pos + n for a, b in taken):
                return  # would collide with a tab placed earlier
            for i, (ch, sty) in enumerate(cells):
                chars[pos + i], styles[pos + i] = ch, sty
            taken.append((pos, pos + n))

        # Priority when space is short: left title, then right, then center.
        place(left, "left")
        place(right, "right")
        place(center, "center")

        segs: list[Seg] = []
        for ch, sty in zip(chars, styles):
            if segs and segs[-1][1] == sty:
                segs[-1] = (segs[-1][0] + ch, sty)
            else:
                segs.append((ch, sty))
        return segs

    def _panel(self, lay: Layout, style: str, left: list[list[Seg]],
               right: list[list[Seg]], top: dict, bottom: dict) -> list[list[Seg]]:
        rows = [self._border(lay, True, style, **top)]
        edge = (self.v, style)
        if lay.split:
            blank_l = [(" " * lay.left_w, "")]
            blank_r = [(" " * lay.right_w, "")]
            for i in range(max(len(left), len(right))):
                rows.append([edge, (" ", ""),
                             *(left[i] if i < len(left) else blank_l),
                             (f" {self.v} ", style),
                             *(right[i] if i < len(right) else blank_r),
                             (" ", ""), edge])
        else:
            for row in (*left, *right):
                rows.append([edge, (" ", ""), *row, (" ", ""), edge])
        rows.append(self._border(lay, False, style, **bottom))
        return rows

    # -- system panel ------------------------------------------------------

    def _meter_segs(self, pct: float | None, width: int) -> list[Seg]:
        return meter(pct, width, self.ascii)

    @staticmethod
    def _pct_seg(value: float | None) -> Seg:
        """Always five characters wide."""
        if value is None:
            return ("    -", "dim")
        return (f"{value:4.0f}%", grad_style(value / 100.0))

    def _graph_rows(self, key: str, label: str, value: float | None, width: int,
                    height: int, live: bool,
                    tail: list[Seg] | None = None) -> list[list[Seg]]:
        """Label, history graph (or meter when not live), then `tail` -
        the percentage unless given."""
        if tail is None:
            tail = [(" ", ""), self._pct_seg(value)]
        graph_w = max(4, width - LABEL_W - sum(len(t) for t, _ in tail))
        if live:
            body = self.graph(key).render(graph_w, height, 100.0, self.ascii)
        else:
            body = [self._meter_segs(value, graph_w)]
        rows = []
        for i, cells in enumerate(body):
            line = Line(width)
            line.add(f"{label:<4} " if i == 0 else " " * LABEL_W, "label")
            line.add_segs(cells)
            if i == 0:
                line.add_segs(tail)
            rows.append(line.padded())
        return rows

    def _mem_rows(self, key: str, label: str, used: float | None, total: float | None,
                  width: int, height: int, live: bool, extra: str = "") -> list[list[Seg]]:
        text = f"{human_bytes(used)}/{human_bytes(total)}".rjust(19)
        pct = mem_pct(used, total)
        if not live:
            return self._graph_rows(key, label, pct, width, height, live,
                                    [(" " + text, ""), (extra, "dim")])
        # Live: the figures get a line of their own above the graph, so the
        # graph lines up with the load graph above it.
        head = Line(width)
        head.add(f"{label:<4} ", "label")
        head.add(text.strip())
        head.add(extra.rjust(head.left), "dim")
        return [head.padded(), *self._graph_rows(key, "", pct, width, height, live)]

    def _text_row(self, width: int, label: str, text: str, style: str = "dim") -> list[Seg]:
        line = Line(width)
        line.add(f"{label:<4} ", "label")
        line.add_clipped(text, style)
        return line.padded()

    def _npu_row(self, npu: dict, width: int) -> list[Seg]:
        line = Line(width)
        line.add("NPU  ", "label")
        users = npu.get("users") or []
        if npu.get("busy") is not None:
            line.add_segs(self._meter_segs(npu["busy"], 8))
            line.add(" ")
            line.add(*self._pct_seg(npu["busy"]))
        elif users:
            line.add(fit("in use: " + ", ".join(c for _, c in users[:3]), 14), "ok")
        else:
            line.add(fit("idle", 14), "idle")
        meta = [m for m in (npu.get("driver"),
                            f"fw {npu['fw']}" if npu.get("fw") else None,
                            npu.get("power_state")) if m]
        line.add_clipped(f" {self.mid} ".join(meta), "dim")
        return line.padded()

    def _system_panel(self, snap: Snapshot, lay: Layout, live: bool) -> list[list[Seg]]:
        sysinfo, gpu, npu = snap.system, snap.gpu, snap.npu

        cpu_name = sysinfo.get("cpu_name") or "CPU"
        if sysinfo.get("threads"):
            cpu_name += f" {self.mid} {sysinfo['threads']} threads"
        left = [self._text_row(lay.left_w, "", cpu_name, "label")]
        left += self._graph_rows("cpu", "CPU", sysinfo.get("cpu"), lay.left_w,
                                 lay.graph_h, live)
        if sysinfo.get("mem_total"):
            left += self._mem_rows("ram", "RAM", sysinfo.get("mem_used"),
                                   sysinfo["mem_total"], lay.left_w, lay.mem_h, live)
        if sysinfo.get("load"):
            left.append(self._text_row(lay.left_w, "load", "  ".join(sysinfo["load"]), ""))

        right: list[list[Seg]] = []
        if gpu:
            name = gpu.get("name") or "GPU"
            if gpu.get("sclk"):
                name += f" {self.mid} {gpu['sclk']}"
            right.append(self._text_row(lay.right_w, "", name, "label"))
            right += self._graph_rows("gpu", "GPU", gpu.get("busy"), lay.right_w,
                                      lay.graph_h, live)
            tail = ""
            if gpu.get("temp") is not None or gpu.get("watt") is not None:
                temp = f"{gpu['temp']:4.0f}°C" if gpu.get("temp") is not None else " " * 6
                watt = f"{gpu['watt']:5.0f}W" if gpu.get("watt") is not None else " " * 6
                tail = f" {temp}{watt}"
            label, used, total = gpu_mem(gpu)
            if total:
                right += self._mem_rows("gtt", label, used, total, lay.right_w,
                                        lay.mem_h, live, tail)
        if npu:
            right.append(self._npu_row(npu, lay.right_w))

        host: list[Seg] = [(sysinfo.get("host") or os.uname().nodename, "hi")]
        uptime = f"up {human_delta(sysinfo['uptime'])}" if sysinfo.get("uptime") else ""
        if sysinfo.get("machine"):
            full = sum(len(t) for t, _ in host) + 3 + len(sysinfo["machine"])
            # host tab + clock tab + uptime tab, each with tab chars and gaps
            if full + 2 + 10 + len(uptime) + 2 + 10 <= lay.width:
                host += [(f" {self.mid} ", "dim"), (sysinfo["machine"], "label")]
        top = {"left": host,
               "center": [(time.strftime("%H:%M:%S"), "clock")],
               "right": [(uptime, "label")] if uptime else None}
        return self._panel(lay, BOX_STYLE["system"], left, right, top, {})

    # -- backend panels ----------------------------------------------------

    @staticmethod
    def _timer(be: Backend) -> Seg:
        if be.state == SLEEPING and be.since is not None:
            return (f"idle {human_delta(be.since)}", "dim")
        if be.idle_in is not None and not be.busy:
            if be.idle_in > 0:
                return (f"unloads in {human_delta(be.idle_in)}",
                        "warn" if be.idle_in < 120 else "dim")
            return ("unloaded", "dim")
        if be.last_use is not None and not be.busy:
            return (f"used {human_delta(be.last_use)} ago", "dim")
        if be.since is not None and be.state == RUNNING:
            return (f"up {human_delta(be.since)}", "dim")
        return ("", "")

    def _backend_left(self, be: Backend, lay: Layout, indent: int) -> list[list[Seg]]:
        width = lay.left_w
        running = be.state == RUNNING
        style = STATE_STYLE.get(be.state, "dim")

        head = Line(width)
        head.add(" " * indent)
        head.add((self.dot_on if running else self.dot_off) + " ", style)
        # Children lose their indent from the name column, so the state column
        # lines up for parents and children alike.
        head.add(fit(be.name, max(8, lay.name_w - indent)) + " ", "hi" if running else "")
        head.add(fit(STATE_WORD.get(be.state, be.state), 8) + " ", style)
        head.add("busy " if be.busy else "     ", "bad")
        head.add(fit(f":{be.port}" if be.port else "", 7), "dim")
        if lay.show_pid:
            head.add(fit(f"pid {be.pid}" if be.pid else "", 12), "dim")
        detail = be.detail
        if be.busy and detail.lower() in ("busy", "processing", "streaming"):
            detail = ""
        head.add_clipped(detail, "dim")

        model = Line(width)
        model.add(" " * (2 + indent))
        model.add_clipped(be.model, "model", minimum=6)
        for extra in be.extras:
            if not model.add_segs([(f" {self.mid} ", "dim"), (extra, "dim")]):
                break
        return [head.padded(), model.padded()]

    def _backend_right(self, be: Backend, lay: Layout,
                       live: bool) -> list[tuple[list[Seg], bool]]:
        width = lay.right_w
        indent = "" if lay.split else "  "

        # line 1: cpu | gpu | tok/s | sparkline
        load = Line(width)
        has_load = be.cpu is not None or be.gpu_util is not None or be.tps is not None
        if has_load:
            load.add(indent)
            # CPU in cores: a percentage of one core next to the whole-machine
            # graph above reads as a contradiction (191% there, 6% here).
            cpu_frac = be.cpu / 100.0 / (os.cpu_count() or 1) if be.cpu is not None else None
            gpu_frac = be.gpu_util / 100.0 if be.gpu_util is not None else None
            lead = leading_device(cpu_frac, gpu_frac)
            load.add("CPU " if lead == "cpu" else "cpu ", "lead" if lead == "cpu" else "label")
            if be.cpu is None:
                load.add("         -", "dim")
            else:
                load.add(f"{be.cpu / 100.0:4.1f} cores",
                         "lead" if lead == "cpu" else grad_style(cpu_frac))
            load.add("  ")
            load.add("GPU " if lead == "gpu" else "gpu ", "lead" if lead == "gpu" else "label")
            if be.gpu_util is None:
                load.add("     -", "dim")
            else:
                text = (f"~{be.gpu_util:.0f}%".rjust(6) if be.gpu_estimated
                        else f"{be.gpu_util:5.1f}%")
                load.add(text, "lead" if lead == "gpu" else grad_style(gpu_frac))
            load.add("  ")
            if be.tps is None:
                load.add(f"{'-':>6} tok/s", "dim")
            else:
                load.add(f"{be.tps:6.1f} tok/s", "bad" if be.tps > 0.05 else "dim")
                spark_w = load.left - 1
                if live and spark_w >= 4:
                    g = self.graph(f"tps:{be.key}")
                    load.add(" ")
                    load.add_segs(g.render(spark_w, 1, g.scale(10.0), self.ascii)[0])

        # line 2: memory | ctx | slots | timer
        size = Line(width)
        mem, kind = ((be.gpu_mem, "GPU") if be.gpu_mem
                     else (be.mem, MEM_KIND.get(be.mem_kind, "")))
        timer = self._timer(be)
        has_size = bool(mem or be.ctx or be.slots_total or timer[0])
        if has_size:
            size.add(indent)
            size.add(f"{human_bytes(mem) if mem else '-':>9} {kind:<3}",
                     "" if mem else "dim")
            size.add("  ctx ", "label")
            size.add(f"{human_ctx(be.ctx) if be.ctx else '-':>5}", "" if be.ctx else "dim")
            size.add("  slots ", "label")
            slots = f"{be.slots_busy}/{be.slots_total}" if be.slots_total else "-"
            size.add(f"{slots:>5}", "bad" if be.slots_busy else ("" if be.slots_total else "dim"))
            size.add("  ")
            size.add_clipped(*timer)
        return [(load.padded(), has_load), (size.padded(), has_size)]

    @staticmethod
    def _summary(group: list[Backend]) -> str:
        counts: dict[str, int] = {}
        for be in group:
            counts[be.state] = counts.get(be.state, 0) + 1
            for child in be.children:
                if child.busy:
                    counts["busy"] = counts.get("busy", 0) + 1
            if be.busy and not be.children:
                counts["busy"] = counts.get("busy", 0) + 1
        order = ("busy", RUNNING, SLEEPING, STOPPED, ABSENT)
        words = {"busy": "busy", **STATE_WORD}
        return " · ".join(f"{counts[k]} {words[k]}" for k in order if counts.get(k))

    # -- everything --------------------------------------------------------

    def rows(self, snap: Snapshot, width: int, height: int, interval: float,
             live: bool) -> list[list[Seg]]:
        if self.graph_height == "auto":
            # Graphs take up whatever rows the panels leave free, so the
            # screen fills at any window height. Each extra graph row adds
            # the same number of lines (one split, two stacked), so one
            # trial build is enough to size them.
            out = self._build(snap, Layout(width, height, 2), interval, live)
            spare = height - len(out)
            if spare <= 0:
                return out
            step = len(self._build(snap, Layout(width, height, 3), interval, live)) - len(out)
            if step <= 0:
                return out
            return self._build(snap, Layout(width, height, 2 + spare // step), interval, live)
        try:
            gh = max(1, min(4, int(self.graph_height)))
        except (TypeError, ValueError):
            gh = 1
        return self._build(snap, Layout(width, height, 2 * gh), interval, live)

    def _build(self, snap: Snapshot, lay: Layout, interval: float,
               live: bool) -> list[list[Seg]]:
        out = self._system_panel(snap, lay, live)
        by_kind: dict[str, list[Backend]] = {}
        for be in snap.backends:
            by_kind.setdefault(be.kind, []).append(be)
        kinds = [k for k in ("llama", "ollama", "lemonade") if by_kind.get(k)]

        keys: list[Seg] = []
        if live:
            for key, word in (("q", " quit  "), ("+/-", " interval  "), ("r", " refresh")):
                keys += [(key, "hi"), (word, "label")]
        footer = {"left": keys or None,
                  "right": [(f"llmtop {VERSION} {self.mid} every {interval:g}s", "label")]}

        if not kinds:
            out += self._panel(lay, BOX_STYLE["system"],
                               [Line(lay.left_w).padded()], [],
                               {"left": [("backends", "title")]}, footer)
            return out
        for n, kind in enumerate(kinds):
            group = by_kind[kind]
            left: list[list[Seg]] = []
            right: list[list[Seg]] = []
            for be in group:
                for node, indent in ((be, 0), *((c, 2) for c in be.children)):
                    lrows = self._backend_left(node, lay, indent)
                    rrows = self._backend_right(node, lay, live)
                    if lay.split:
                        left.extend(lrows)
                        right.extend(r for r, _ in rrows)
                    else:
                        # Stacked: keep each backend's lines together.
                        left.extend(lrows)
                        left.extend(r for r, has in rrows if has)
            top = {"left": [(KIND_TITLE[kind], "title")],
                   "right": [(self._summary(group), "label")]}
            out += self._panel(lay, BOX_STYLE[kind], left, right, top,
                               footer if n == len(kinds) - 1 else {})
        return out


# --------------------------------------------------------------------------
# colours
# --------------------------------------------------------------------------

DEFAULT_BACKGROUND = 234  # xterm-256 dark grey
BASE_FG = 252

# style -> (xterm-256 foreground, 8-colour fallback, attribute)
PALETTE: dict[str, tuple[int, str | None, str]] = {
    "":           (BASE_FG, None, ""),
    "dim":        (243, None, "dim"),
    "label":      (246, None, ""),
    "hi":         (255, None, "bold"),
    "title":      (255, None, "bold"),
    "clock":      (230, None, "bold"),
    "ok":         (114, "green", ""),
    "idle":       (74, "cyan", ""),
    "warn":       (179, "yellow", ""),
    "bad":        (203, "red", ""),
    "lead":       (203, "red", "bold"),  # the device a backend runs on
    "model":      (182, "magenta", ""),
    "track":      (238, None, "dim"),
    # panel frames, one colour per panel like btop's boxes
    "b_system":   (65, "green", ""),
    "b_llama":    (137, "yellow", ""),
    "b_ollama":   (61, "blue", ""),
    "b_lemonade": (131, "red", ""),
}


def style_spec(style: str) -> tuple[int, str | None, str]:
    if style.startswith("grad"):
        try:
            idx = int(style[4:])
        except ValueError:
            return PALETTE[""]
        frac = idx / (GRAD_N - 1)
        return (GRADIENT[idx], "green" if frac < 0.45 else
                "yellow" if frac < 0.8 else "red", "")
    return PALETTE.get(style, PALETTE[""])


def parse_background(value) -> int | None:
    """Config/CLI value -> xterm-256 index, or None for the terminal's own."""
    if value is None:
        return DEFAULT_BACKGROUND
    if isinstance(value, int):
        return value if 0 <= value <= 255 else DEFAULT_BACKGROUND
    text = str(value).strip().lower()
    if text in ("", "none", "off", "false", "terminal", "default"):
        return None
    try:
        num = int(text)
    except ValueError:
        return DEFAULT_BACKGROUND
    return num if 0 <= num <= 255 else DEFAULT_BACKGROUND


# --------------------------------------------------------------------------
# output: one-shot, JSON, TUI
# --------------------------------------------------------------------------

def ansi(style: str, background: int | None) -> str:
    fg, _, attr = style_spec(style)
    codes = ["0", f"38;5;{fg}"]
    if background is not None:
        codes.append(f"48;5;{background}")
    if attr == "bold":
        codes.append("1")
    return f"\033[{';'.join(codes)}m"


def print_rows(rows: list[list[Seg]], color: bool, background: int | None,
               width: int) -> None:
    for row in rows:
        if not color:
            print("".join(text for text, _ in row).rstrip())
            continue
        parts = [ansi(style, background) + text for text, style in row]
        used = sum(len(text) for text, _ in row)
        if background is not None and used < width:
            parts.append(ansi("", background) + " " * (width - used))
        print("".join(parts) + "\033[0m")


def backend_to_dict(be: Backend) -> dict:
    data = {
        "kind": be.kind, "name": be.name, "state": be.state, "detail": be.detail,
        "model": be.model, "ctx": be.ctx, "memory_bytes": be.mem,
        "memory_kind": be.mem_kind, "cpu_percent": be.cpu,
        "gpu_memory_bytes": be.gpu_mem, "gpu_percent": be.gpu_util,
        "gpu_percent_estimated": be.gpu_estimated,
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


def build_pairs(curses, background: int | None) -> tuple[dict[str, int], int]:
    """Map style names onto curses attributes. Returns (pairs, base attribute)."""
    pairs: dict[str, int] = {}
    if not curses.has_colors():
        return pairs, 0
    curses.start_color()
    try:
        curses.use_default_colors()
        default_bg = -1
    except curses.error:
        default_bg = curses.COLOR_BLACK
    rich = curses.COLORS >= 256
    bg = background if (rich and background is not None) else default_bg
    basic = {"green": curses.COLOR_GREEN, "yellow": curses.COLOR_YELLOW,
             "red": curses.COLOR_RED, "cyan": curses.COLOR_CYAN,
             "blue": curses.COLOR_BLUE, "magenta": curses.COLOR_MAGENTA}
    plain = -1 if default_bg == -1 else curses.COLOR_WHITE
    styles = [*PALETTE, *(f"grad{i}" for i in range(GRAD_N))]
    for number, style in enumerate(styles, start=1):
        if number >= curses.COLOR_PAIRS:
            break
        fg, fallback, attr = style_spec(style)
        curses.init_pair(number, fg if rich else basic.get(fallback, plain), bg)
        value = curses.color_pair(number)
        if attr == "bold":
            value |= curses.A_BOLD
        elif attr == "dim" and not rich:
            value |= curses.A_DIM
        pairs[style] = value
    return pairs, pairs.get("", 0)


def run_tui(collector: Collector, interval: float, ascii_mode: bool,
            graph_height: str | int, background: int | None) -> int:
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
            except Exception as exc:  # collection must never kill the display
                state["err"] = f"{type(exc).__name__}: {exc}"
            state["force"].wait(state["interval"])
            state["force"].clear()

    def draw(stdscr) -> int:
        curses.curs_set(0)
        stdscr.nodelay(True)
        pairs, base = build_pairs(curses, background)
        stdscr.bkgd(" ", base)  # paints the background into every cell
        renderer = Renderer(ascii_mode, graph_height)
        threading.Thread(target=worker, daemon=True).start()

        while True:
            while True:  # drain every pending key
                try:
                    key = stdscr.getch()
                except curses.error:
                    key = -1
                if key == -1:
                    break
                if key == curses.KEY_RESIZE:
                    curses.update_lines_cols()
                    stdscr.clear()
                elif key in (ord("q"), ord("Q"), 27):
                    state["stop"] = True
                    state["force"].set()
                    return 0
                elif key in (ord("+"), ord("=")):
                    state["interval"] = min(60.0, state["interval"] + 0.5)
                elif key == ord("-"):
                    state["interval"] = max(0.5, state["interval"] - 0.5)
                elif key in (ord("r"), ord("R"), ord(" ")):
                    state["force"].set()

            height, width = stdscr.getmaxyx()
            stdscr.erase()
            snap = state["snap"]
            if width < 40 or height < 8:
                try:
                    stdscr.addnstr(0, 0, "terminal too small", max(0, width - 1), base)
                except curses.error:
                    pass
            elif snap is None:
                stdscr.addnstr(0, 0, "llmtop is collecting ...", width - 1, base)
            else:
                renderer.feed(snap)
                rows = renderer.rows(snap, width - 1, height - 1, state["interval"], True)
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
                                           pairs.get(style, base))
                        except curses.error:
                            pass
                        x += len(chunk)
            if state["err"]:
                try:
                    stdscr.addnstr(height - 1, 0, f"error: {state['err']}"[:width - 1],
                                   width - 1, pairs.get("bad", base))
                except curses.error:
                    pass
            stdscr.refresh()
            # select, not curses.napms: napms does not release the GIL and
            # would starve the collector thread.
            try:
                select.select([sys.stdin], [], [], 0.12)
            except (OSError, ValueError):
                time.sleep(0.12)

    return curses.wrapper(draw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmtop",
        description="State of local LLM backends: llama.cpp, Ollama, Lemonade, "
                    "plus GPU and NPU.")
    parser.add_argument("-1", "--once", action="store_true",
                        help="print once instead of running the TUI")
    parser.add_argument("--json", action="store_true", help="print JSON and exit")
    parser.add_argument("-n", "--interval", type=float, default=None,
                        help="refresh interval in seconds (default 2)")
    parser.add_argument("--graph-height", default=None,
                        help="graph rows: auto (default) or 1-4")
    parser.add_argument("--background", default=None,
                        help="xterm-256 background colour index, or 'none' "
                             f"for the terminal's own (default {DEFAULT_BACKGROUND})")
    parser.add_argument("--ascii", action="store_true",
                        help="ASCII only, no braille or box drawing")
    parser.add_argument("--no-color", action="store_true",
                        help="no colour (with --once)")
    parser.add_argument("--config", help="path to a configuration file")
    parser.add_argument("--version", action="version", version=f"llmtop {VERSION}")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    interval = args.interval if args.interval is not None else float(cfg["ui"]["interval"])
    ascii_mode = args.ascii or bool(cfg["ui"].get("ascii"))
    graph_height = args.graph_height or cfg["ui"].get("graph_height", "auto")
    background = parse_background(args.background if args.background is not None
                                  else cfg["ui"].get("background"))
    collector = Collector(cfg)

    def sampled() -> Snapshot:
        """Measure twice: CPU percentages and tok/s only exist as a delta."""
        collector.snapshot()
        time.sleep(min(1.0, max(0.3, interval / 2)))
        return collector.snapshot()

    if args.json:
        print(json.dumps(snapshot_to_dict(sampled()), indent=2, ensure_ascii=False))
        return 0

    if args.once:
        size = shutil.get_terminal_size((100, 30))
        color = sys.stdout.isatty() and not args.no_color
        width = size.columns - 1
        # One-shot output is not bound by the window height.
        rows = Renderer(ascii_mode, graph_height if graph_height != "auto" else 1).rows(
            sampled(), width, 10_000, interval, False)
        print_rows(rows, color, background, width)
        return 0

    if not sys.stdout.isatty():
        print("llmtop: not a terminal, use --once or --json", file=sys.stderr)
        return 2
    try:
        return run_tui(collector, interval, ascii_mode, graph_height, background)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
