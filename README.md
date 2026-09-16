# llmtop

An htop-style overview of local LLM backends — **llama.cpp**, **Ollama** and
**Lemonade Server** — showing which model is loaded, what it costs in memory and
what is going through it right now, plus integrated GPU and NPU state.

Drawn like btop: framed panels with titles set into the border, braille history
graphs with a colour gradient, a clock, and a background of its own. One file,
Python 3.11+, no dependencies.

```
╭─┐tower · GMKtec NucBox EVO-X2┌─────────────────┐11:19:05┌──────────────────────────────────────┐up 1d18h┌─╮
│      AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 32 th… │      Radeon 8060S · 2737Mhz                         │
│ CPU  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀    9% │ GPU  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿  100% │
│      ⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀       │      ⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣿⣿⣿⣿⣿⣿⣿⣿⣿       │
│ RAM  46.6 GiB/124.9 GiB                             │ GTT  40.3 GiB/120.0 GiB                  63°C   85W │
│      ⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤   37% │      ⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤   34% │
│ load 1.93  2.27  1.65                               │ NPU  idle          amdxdna · fw 1.1.2.65 · D3hot    │
╰─────────────────────────────────────────────────────┴─────────────────────────────────────────────────────╯
╭─┐llama.cpp┌─────────────────────────────────────────┬──────────────────────────────┐2 asleep · 1 stopped┌─╮
│ ○ llama-qwen     asleep        :8091  socket-activ… │                                                     │
│   qwen3-30b-a3b · fa · ngl 999                      │         -      ctx   32k  slots     -  idle 2h18    │
│ ○ llama-qwen36   asleep        :8090  socket-activ… │                                                     │
│   qwen3.6-35b-a3b · draft draft-mtp · fa · ngl 999  │         -      ctx   64k  slots     -  idle 23h15   │
│ ○ llama-v4       stopped              dead          │                                                     │
│   deepseek-v4-flash · ngl 999 · model file missing  │         -      ctx   64k  slots     -               │
╰─────────────────────────────────────────────────────┴─────────────────────────────────────────────────────╯
╭─┐Ollama┌────────────────────────────────────────────┬────────────────────────────────┐1 busy · 1 running┌─╮
│ ● ollama         running  busy :11434               │ cpu 197.1%  gpu      -    23.7 tok/s ⣀⣀⣀⣀⣀⣸⣿⣿⣿⣿⣿⣿⣿⣿ │
│   granite4.2:latest · v0.34.1-igpu-trust-vram       │  25.1 GiB GPU  ctx  128k  slots   1/1  up 58s       │
╰─────────────────────────────────────────────────────┴─────────────────────────────────────────────────────╯
╭─┐Lemonade┌──────────────────────────────────────────┬────────────────────────────────┐1 busy · 1 running┌─╮
│ ● lemonade       running  busy :13311 front: Gemma… │ cpu   0.0%  gpu      -       - tok/s                │
│   2 models · v11.6.0 · 188.1k tok total             │ 959.6 MiB RSS  ctx     -  slots     -  up 20h56     │
│   ● SDXL-Turbo   running       :8002  ready         │ cpu   0.0%  gpu      -       - tok/s                │
│     SDXL-Turbo · sd-cpp/gpu · image                 │  21.3 MiB RSS  ctx   32k  slots     -  used 2h34 a… │
│   ● Gemma-4-E4B… running  busy :8001                │ cpu  33.5%  gpu      -    44.5 tok/s ⣀⣀⣀⣀⣀⣿⣿⣿⣿⣿⣿⣿⣿⣿ │
│     Gemma-4-E4B-it-GGUF · llamacpp/gpu              │ 938.2 MiB RSS  ctx  128k  slots   2/4               │
╰─┘q quit  +/- interval  r refresh└───────────────────┴───────────────────────────┘llmtop 0.5.0 · every 1s└─╯
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
llmtop --background 16  # another xterm-256 background, or "none"
llmtop --ascii          # no braille or box drawing, plain ASCII
```

Keys: `q` quit, `+`/`-` interval, `r` refresh now.

### Layout

Each panel spans the full width and, from 90 columns on, is split down the
middle: identity and system on the left, GPU and live measurements on the right.
Every field has a fixed column - values are padded, never shifted - so nothing
wanders as numbers change length. Below 90 columns the two halves are stacked
inside the same panels.

Graphs and meters grow and shrink with the window, trailing fields are cut with an
ellipsis rather than pushed aside, and the graphs take up whatever rows the
panels leave free, so the screen fills at any window height. As in btop, load
and memory share that space: CPU with RAM, GPU with GTT. On short terminals the
graphs flatten to one row before the bottom panels would get cut off. Resizing keeps the history:
the sample buffer is far wider than any terminal, so a wider window simply reveals
more of the past.

The top border carries host and machine name, the clock and uptime; the CPU and
GPU names head their halves of the system panel. Each backend panel lists a
summary such as `1 busy · 2 asleep` in its border, and the bottom border carries
the key bindings.

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
