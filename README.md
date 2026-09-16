# llmtop

Eine htop-artige Übersicht über lokale LLM-Backends: **llama.cpp**, **Ollama** und
**Lemonade Server** — welches Modell geladen ist, was es an Speicher belegt, was
gerade durchgeht. Dazu iGPU- und NPU-Zustand.

*English: a terminal dashboard for locally running LLM servers (llama.cpp, Ollama,
Lemonade), showing loaded models, memory, live tokens/s, GPU and NPU state. Python
stdlib only, no dependencies. The UI is in German.*

```
llmtop 0.1.0  workstation  up 1t16h  load 2.60 1.47 1.26
CPU ██░░░░░░░░░░   16%   RAM 112.1 GiB/124.9 GiB (90%)
GPU ████████████  100%   GTT 103.9 GiB/120.0 GiB  64°C 85W
NPU frei  (Auslastung nur via debugfs/root)   amdxdna · FW 1.1.2.65 · D0

── llama.cpp ─────────────────────────────────────────────────────────────────
  ● llama-qwen         laeuft   aktiv :8091 pid 3168734 seit 1m54
    qwen3-30b-a3b · ctx 32k · 18.5 GiB GPU · 98.8 MiB RSS · fa · ngl 999
    CPU  32.8%   GPU  97.9%   Slots 1/1   36.2 tok/s
  ○ llama-qwen36       schlaeft Socket 8090  :8090 ruht 20h56
    qwen3.6-35b-a3b · ctx 64k · draft draft-mtp · fa · ngl 999
  ○ llama-v4           gestoppt dead
    deepseek-v4-flash · ctx 64k · ngl 999 · Modelldatei fehlt

── Ollama ────────────────────────────────────────────────────────────────────
  ● ollama             laeuft   :11434 pid 3154906 seit 24m41
    granite4.2:latest · ctx 128k · 25.1 GiB GPU · 8.8B · Q4_K_M
    CPU   0.6%   Slots 1/1   54.5 tok/s   entlaedt in 29m
```

## Warum nicht einfach btop

btop hat keine Erweiterungsschnittstelle — `shown_boxes` akzeptiert ausschließlich
`cpu mem net proc` und `gpu0`…`gpu5`, die Boxen stecken fest im C++-Quelltext. Eine
eigene Box bedeutet einen Fork, der bei jedem Release neu rebasiert werden will.
llmtop läuft deshalb *neben* btop statt darin.

Für Ollama allein gibt es bereits gute Werkzeuge —
[otop](https://github.com/TiniLLM/ollama-token-monitor),
[ollama-tui](https://github.com/hughdbrown/ollama-tui),
[OllamaManager](https://github.com/tleclaire/OllamaManager). Keines davon kennt
llama.cpp oder Lemonade, und keines ist auf Socket-Aktivierung vorbereitet.

## Socket-aktivierte Backends

Das ist der Grund, warum ein allgemeines Werkzeug hier nicht genügt: Hängt ein
llama.cpp-Backend an einer `systemd`-Socket-Unit, **löst schon ein Statuscheck per
HTTP das Laden des Modells aus** — bei einem 35B-Modell zwanzig Sekunden und zwanzig
Gigabyte für die Frage „läuft das gerade?".

llmtop fragt Socket-Ports grundsätzlich nicht an. Der Zustand kommt aus systemd, und
Messwerte holt es ausschließlich vom internen Backend-Port, und nur dann, wenn der
Dienst ohnehin schon läuft. Die Socket-Ports stehen vor dem ersten HTTP-Aufruf auf
einer Sperrliste.

Modell, Kontextgröße und Optionen eines *schlafenden* Backends liest llmtop aus der
Unit: `ExecStart` über `systemctl show` (dort sind `%h` und Co. bereits aufgelöst),
und zeigt die Zeile auf ein Startskript, wird dessen Inhalt ausgewertet — auch wenn
die Argumente darin erst in einem Bash-Array gesammelt werden. Fehlt die Modelldatei
inzwischen, steht das dabei.

## Installation

Eine Datei, Python ≥ 3.11, keine Abhängigkeiten.

```bash
git clone https://github.com/huppiflupp/llmtop.git
install -m 755 llmtop/llmtop.py ~/.local/bin/llmtop
llmtop
```

## Aufruf

```
llmtop              # TUI
llmtop -n 1         # Aktualisierung jede Sekunde
llmtop --once       # einmal ausgeben und beenden
llmtop --json       # Maschinenlesbar, für Skripte und Statuszeilen
llmtop --ascii      # ohne Blockzeichen
```

Tasten: `q` beenden, `+`/`-` Intervall, `r` sofort aktualisieren.

## Woher die Zahlen kommen

| Angabe | Quelle |
|---|---|
| llama.cpp: Zustand, Modell, Kontext | `systemctl show` auf Service und Socket, Startskript |
| llama.cpp: Slots, tok/s | `GET /slots` am internen Port, Delta von `n_decoded` |
| Ollama: Modell, Speicher, Entladezeit | `GET /api/ps` (`size_vram`, `context_length`, `expires_at`) |
| Ollama: Modell ↔ Runner-Prozess | Manifeste unter `models/manifests`, Blob-Digest der Modellschicht |
| Lemonade: Modelle, Backends, Leerlauf | `GET /api/v1/health`, `last_use` gegen `/proc/uptime` |
| Lemonade: Durchsatz | `GET /api/v1/stats`, sonst `/slots` des zugehörigen llama.cpp |
| Speicher je Prozess | `/proc/<pid>/fdinfo`, `drm-resident-gtt` + `drm-resident-vram` |
| GPU-Zeit je Prozess | `/proc/<pid>/fdinfo`, Delta von `drm-engine-*` |
| GPU gesamt | `/sys/class/drm/card*/device`, ersatzweise `nvidia-smi` |
| NPU | `/sys/class/accel/*`, `xrt-smi examine`, offene `/dev/accel/*` |

`tokens_predicted_total` aus `/metrics` wird nur als Rückfall benutzt: Der Zähler
wird erst beim Auftragsende fortgeschrieben und steht während der Generierung still.
`/slots` zählt dagegen live mit.

### Gemeinsamer Speicher

Auf Systemen mit Unified Memory (AMD Strix Halo und Verwandte) liegt das Modell im
GTT und taucht in RSS **gar nicht** auf — ein 30B-Modell erscheint dort als 99 MiB.
Erst `drm-resident-gtt` zeigt die echten 18,5 GiB. llmtop weist beides getrennt aus.

## Grenzen

- **NPU-Auslastung in Prozent** gibt `amdxdna` nur über debugfs heraus, das root
  braucht. Ohne root zeigt llmtop stattdessen, ob jemand `/dev/accel/*` geöffnet hat,
  dazu Treiber, Firmware-Stand und Energiezustand.
- **Speicher und GPU-Zeit je Prozess** stehen nur für eigene Prozesse in `/proc`.
  Läuft Ollama als eigener Systembenutzer, bleibt für dessen Runner die Angabe aus
  `/api/ps` übrig — die ist gut, aber gröber.
- Bei **mehreren gleichzeitig geladenen Ollama-Modellen** klappt die Zuordnung
  Modell ↔ Prozess nur, wenn die Manifeste lesbar sind.

## Konfiguration

Ohne Konfiguration werden die üblichen Adressen probiert und Dienste anhand ihrer
Prozesse erkannt; `OLLAMA_HOST`, `OLLAMA_MODELS` und `LEMONADE_URL` werden beachtet.
Für abweichende Aufbauten: `~/.config/llmtop/config.toml`, siehe
[`config.toml.example`](config.toml.example).

## Lizenz

MIT
