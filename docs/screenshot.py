#!/usr/bin/env python3
"""Render llmtop's live view to a PNG, for the README.

Samples the collector for a while so the graphs carry history, then draws
the frame cell by cell with a monospace font that has braille and box
drawing (Adwaita Mono, DejaVu Sans Mono). Nothing is faked: the picture is
whatever the machine shows at the time, but names can be masked, since a
screenshot on GitHub need not carry hostnames and addresses.

    docs/screenshot.py --seconds 60 --out docs/screenshot.png
    docs/screenshot.py --menu endpoints --host tower --replace 192.168.1.5=node-b
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import time

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = [
    ("/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf",
     "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Bold.ttf"),
    ("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
     "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
]


def load_llmtop():
    spec = importlib.util.spec_from_file_location("llmtop", os.path.join(HERE, "..", "llmtop.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["llmtop"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def xterm_rgb(index: int) -> tuple[int, int, int]:
    base = [(0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0), (0, 0, 238), (205, 0, 205),
            (0, 205, 205), (229, 229, 229), (127, 127, 127), (255, 0, 0), (0, 255, 0),
            (255, 255, 0), (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)]
    if index < 16:
        return base[index]
    if index < 232:
        i = index - 16
        cube = (0, 95, 135, 175, 215, 255)
        return (cube[i // 36], cube[(i // 6) % 6], cube[i % 6])
    v = 8 + 10 * (index - 232)
    return (v, v, v)


def grid_from_rows(rows, width: int, height: int):
    """rows of (text, style) segments -> height x width cells of (char, style)."""
    grid = [[(" ", "")] * width for _ in range(height)]
    for y, row in enumerate(rows[:height]):
        x = 0
        for text, style in row:
            for ch in text:
                if x >= width:
                    break
                grid[y][x] = (ch, style)
                x += 1
    return grid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "screenshot.png"))
    ap.add_argument("--config", help="llmtop config file (default: the usual one)")
    ap.add_argument("--theme", default=None)
    ap.add_argument("--graph-colors", default=None, choices=["theme", "default"])
    ap.add_argument("--width", type=int, default=120, help="columns")
    ap.add_argument("--height", type=int, default=38, help="rows")
    ap.add_argument("--seconds", type=float, default=45, help="how long to sample first")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--menu", choices=["options", "endpoints"], help="draw the menu over the panels")
    ap.add_argument("--host", help="show this instead of the machine's host name")
    ap.add_argument("--replace", action="append", default=[], metavar="OLD=NEW",
                    help="mask a string wherever it appears (menu URLs, names)")
    ap.add_argument("--font-size", type=int, default=15)
    args = ap.parse_args()

    lt = load_llmtop()
    cfg = lt.load_config(args.config)
    graph_colors = args.graph_colors or str(cfg["ui"].get("graph_colors") or "theme")
    theme_name = args.theme or str(cfg["ui"].get("theme") or "default")
    lt.set_theme(lt.load_theme(theme_name, graph_colors))
    background = lt.resolve_background(lt.parse_background(cfg["ui"].get("background")))
    masks = [tuple(pair.split("=", 1)) for pair in args.replace if "=" in pair]

    collector = lt.Collector(cfg)
    renderer = lt.Renderer(False, cfg["ui"].get("graph_height", "auto"))
    deadline = time.monotonic() + args.seconds
    snap = None
    while True:
        snap = collector.snapshot()
        renderer.feed(snap)
        if time.monotonic() >= deadline:
            break
        time.sleep(args.interval)
        print(f"\rsampling … {deadline - time.monotonic():4.0f}s left", end="", file=sys.stderr)
    print(file=sys.stderr)
    assert snap is not None
    if args.host:
        snap.system["host"] = args.host
    for be in snap.backends:
        for node in (be, *be.children):
            for old, new in masks:
                node.name = node.name.replace(old, new)
                node.model = node.model.replace(old, new)

    width, height = args.width, args.height
    rows = renderer.rows(snap, width, height, args.interval, True)
    grid = grid_from_rows(rows, width, height)

    if args.menu:
        ui = {"interval": args.interval, "ascii": False,
              "graph_height": cfg["ui"].get("graph_height", "auto"),
              "background": lt.parse_background(cfg["ui"].get("background")),
              "theme": theme_name, "graph_colors": graph_colors}
        entries = cfg["endpoints"].get("urls") or []
        menu = lt.Menu(ui, entries, collector, lambda key: None)
        menu.tab = 1 if args.menu == "endpoints" else 0
        for i in range(len(menu.endpoints)):
            menu.test(i)
        time.sleep(4)  # the tests run in threads
        for entry in menu.endpoints:
            result = menu.tests.get(entry["url"], "")
            for old, new in masks:
                entry["url"] = entry["url"].replace(old, new)
                if entry.get("name"):
                    entry["name"] = entry["name"].replace(old, new)
                result = result.replace(old, new)
            menu.tests[entry["url"]] = result
        y0, x0, overlay = menu.rows(width, height, renderer)
        for i, row in enumerate(overlay):
            if y0 + i >= height:
                break
            x = x0
            for text, style in row:
                for ch in text:
                    if x >= width:
                        break
                    grid[y0 + i][x] = (ch, style)
                    x += 1

    regular = bold = None
    for reg_path, bold_path in FONTS:
        if os.path.isfile(reg_path):
            regular = ImageFont.truetype(reg_path, args.font_size)
            bold = ImageFont.truetype(bold_path if os.path.isfile(bold_path) else reg_path,
                                      args.font_size)
            break
    if regular is None:
        print("no monospace font with braille found", file=sys.stderr)
        return 1
    cell_w = int(round(regular.getlength("M")))
    ascent, descent = regular.getmetrics()
    cell_h = ascent + descent + 2
    pad = 12
    bg = xterm_rgb(background) if background is not None else (24, 24, 24)
    img = Image.new("RGB", (width * cell_w + 2 * pad, height * cell_h + 2 * pad), bg)
    draw = ImageDraw.Draw(img)
    for y, row in enumerate(grid):
        for x, (ch, style) in enumerate(row):
            if ch == " ":
                continue
            fg, _, attr = lt.style_spec(style)
            colour = xterm_rgb(fg)
            if attr == "dim" and background is not None:
                colour = tuple((c * 3 + b) // 4 for c, b in zip(colour, bg))
            draw.text((pad + x * cell_w, pad + y * cell_h + 1), ch,
                      font=bold if attr == "bold" else regular, fill=colour)
    img.save(args.out)
    print(f"wrote {args.out} ({img.width}x{img.height})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
