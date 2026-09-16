# llmtop

An htop-style overview of local LLM backends — **llama.cpp**, **Ollama** and
**Lemonade Server** — showing which model is loaded, what it costs in memory and
what is going through it right now, plus integrated GPU and NPU state.

Braille history graphs with a colour gradient, like btop. One file, Python 3.11+,
no dependencies.

```
llmtop 0.2.0  workstation  up 1d16h  load 0.02 0.41 1.07
CPU ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀    3%   RAM ⣿⣿⣿⣿⣿⡇⣀⣀⣀⣀⣀⣀⣀⣀ 49.1 GiB/124.9 GiB
    ⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀
GPU ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣶⣿⣿⣿⣶⣶⣶   88%   GTT ⣿⣿⣿⣿⣿⣀⣀⣀⣀⣀⣀⣀⣀⣀ 41.9 GiB/120.0 GiB  56°C 85W
    ⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣿⣿⣿⣿⣿⣿⣿
NPU idle   amdxdna · fw 1.1.2.65 · D3hot · AMD RYZEN AI MAX+ 395 w/ Radeon 8060S

── llama.cpp ──────────────────────────────────────────────────────────────────
  ● llama-qwen         running  :8091 pid 3168734 up 1m54
    qwen3-30b-a3b · ctx 32k · 18.5 GiB GPU · 98.8 MiB RSS · fa · ngl 999
    cpu  32.8%   gpu  97.9%   slots 1/1   36.2 tok/s ⣀⣠⣴⣶⣿⣿⣷
  ○ llama-qwen36       asleep   socket 8090  :8090 idle 21h19
    qwen3.6-35b-a3b · ctx 64k · draft draft-mtp · fa · ngl 999
  ○ llama-v4           stopped  dead
    deepseek-v4-flash · ctx 64k · ngl 999 · model file missing

── Ollama ─────────────────────────────────────────────────────────────────────
  ● ollama             running  :11434 pid 3154906 up 24m41
    granite4.2:latest · ctx 128k · 25.1 GiB GPU · 8.8B · Q4_K_M
    cpu   0.6%   slots 1/1   54.5 tok/s ⣀⣠⣾⣿⣿⣷⣄   unloads in 29m
```

## Why not just extend btop

btop has no extension point — `shown_boxes` accepts only `cpu mem net proc` and
`gpu0`…`gpu5`, and the boxes are hard-wired in its C++ source. A box of your own
means a fork that needs rebasing on every release. So llmtop runs *next to* btop
rather than inside it, and borrows its look: braille graphs, gradient colours,
meters that fill left to right.

For Ollama alone there are already good tools —
[otop](https://github.com/TiniLLM/ollama-token-monitor),
[ollama-tui](https://github.com/hughdbrown/ollama-tui),
[OllamaManager](https://github.com/tleclaire/OllamaManager). None of them knows
about llama.cpp or Lemonade, and none is prepared for socket activation.

## Socket-activated backends

This is why a generic tool is not enough here. When a llama.cpp backend hangs off
a `systemd` socket unit, **an HTTP status check is itself enough to load the
model** — twenty seconds and twenty gigabytes to answer "is this running?".

llmtop never talks to a socket port. State comes from systemd, and measurements
only from the internal backend port, and only while the service is already up.
The socket ports go on a block list before the first HTTP call is made.

Model, context size and options of a *sleeping* backend are read from the unit:
`ExecStart` via `systemctl show` (where `%h` and friends are already expanded),
and if that points at a start script, the script is parsed — including the case
where its arguments are collected in a bash array first. If the model file has
since disappeared, it says so.

## Install

```bash
git clone https://github.com/huppiflupp/llmtop.git
install -m 755 llmtop/llmtop.py ~/.local/bin/llmtop
llmtop
```

## Usage

```
llmtop                  # TUI with live graphs
llmtop -n 1             # refresh every second
llmtop --once           # print once and exit (meters instead of history)
llmtop --json           # machine readable, for scripts and status bars
llmtop --graph-height 3 # taller graphs; default adapts to the window
llmtop --ascii          # no braille, plain ASCII
```

Keys: `q` quit, `+`/`-` interval, `r` refresh now.

The layout follows the window. Graphs and meters grow and shrink with it, columns
are dropped in priority order when space runs short, and on a narrow terminal the
memory meters move to lines of their own. Resizing keeps the history: the sample
buffer is far wider than any terminal, so a wider window simply reveals more past.

## Where the numbers come from

| Reading | Source |
|---|---|
| llama.cpp: state, model, context | `systemctl show` on service and socket, start script |
| llama.cpp: slots, tok/s | `GET /slots` on the internal port, delta of `n_decoded` |
| Ollama: model, memory, unload timer | `GET /api/ps` (`size_vram`, `context_length`, `expires_at`) |
| Ollama: model ↔ runner process | manifests under `models/manifests`, blob digest of the model layer |
| Lemonade: models, backends, idle time | `GET /api/v1/health`, `last_use` against `/proc/uptime` |
| Lemonade: throughput | `GET /api/v1/stats`, else `/slots` of the matching llama.cpp |
| Memory per process | `/proc/<pid>/fdinfo`, `drm-resident-gtt` + `drm-resident-vram` |
| GPU time per process | `/proc/<pid>/fdinfo`, delta of `drm-engine-*` |
| GPU overall | `/sys/class/drm/card*/device`, else `nvidia-smi` |
| NPU | `/sys/class/accel/*`, `xrt-smi examine`, open `/dev/accel/*` handles |

`tokens_predicted_total` from `/metrics` is only a fallback: that counter is
written when a task finishes and stands still during generation. `/slots` counts
along live.

### Unified memory

On unified-memory systems (AMD Strix Halo and relatives) the model lives in GTT
and does **not** appear in RSS at all — a 30B model shows up there as 99 MiB. Only
`drm-resident-gtt` reveals the real 18.5 GiB. llmtop reports both, separately.

## Limits

- **NPU utilisation in percent** is only exposed by `amdxdna` through debugfs,
  which needs root. Without it llmtop shows whether anything holds `/dev/accel/*`
  open, plus driver, firmware version and power state.
- **Per-process memory and GPU time** are only readable for your own processes.
  If Ollama runs as its own system user, what remains for its runners is the
  figure from `/api/ps` — good, but coarser.
- With **several Ollama models loaded at once**, matching model to process works
  only when the manifests are readable.

## Configuration

Without any configuration the usual addresses are tried and services are found by
their processes; `OLLAMA_HOST`, `OLLAMA_MODELS` and `LEMONADE_URL` are honoured.
For anything unusual: `~/.config/llmtop/config.toml`, see
[`config.toml.example`](config.toml.example).

## License

MIT
