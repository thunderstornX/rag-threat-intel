#!/usr/bin/env python3
"""Render the README + paper banner to a PNG.

Same PIL-based ANSI-aware renderer used elsewhere in the portfolio.
Run from the repo root: ``python -m scripts.render_banner``."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


_ANSI16 = {
    30: (40, 42, 46),     31: (224, 92, 84),    32: (140, 188, 80),
    33: (236, 188, 80),   34: (96, 160, 224),   35: (192, 124, 200),
    36: (104, 188, 184),  37: (200, 200, 200),
    90: (104, 104, 104),  91: (240, 124, 116),  92: (160, 208, 120),
    93: (240, 220, 120),  94: (124, 184, 240),  95: (220, 156, 224),
    96: (140, 220, 220),  97: (240, 240, 240),
}
_BG_DEFAULT = (24, 24, 28)
_FG_DEFAULT = (220, 220, 220)
_ANSI_RE = re.compile(r"\x1b\[([0-9;]*)m")


# Bookshelf-with-search-beam motif. The blocks on the left are
# bookshelf rows of a corpus; the >>>>> arrow + cursor is the
# retrieval beam aimed at one of them.
_BANNER = r"""
    ___    _   ___   _____ _   ___ ___   _ _____   ___ _  _ _____ ___ _
   | _ \  /_\ / __| |_   _| |_| _ \ __| /_\_   _| |_ _| \| |_   _| __| |
   |   / / _ \\__ \   | | |  _|   / _| / _ \| |    | || .` | | | | _|| |__
   |_|_\/_/ \_\___/   |_|  \__|_|_\___/_/ \_\_|   |___|_|\_| |_| |___|____|

   ▓▓▓▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓
   ▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓▓
       NVD CVEs        ·        Security PDFs        ·        OSINT corpus

           >>>>>  retrieve  ·  rerank  ·  generate  ·  cite  >>>>>

         ~ AMB · ORCID 0009-0007-2787-943X · v1.0 · 2026 ~
""".strip("\n")


def _stylise(text: str) -> str:
    out = []
    for line in text.splitlines():
        styled = []
        for ch in line:
            if ch in "_":
                styled.append(f"\x1b[1;36m{ch}\x1b[0m")
            elif ch in "/\\|":
                styled.append(f"\x1b[36m{ch}\x1b[0m")
            elif ch in "▓":
                styled.append(f"\x1b[1;94m{ch}\x1b[0m")
            elif ch in "│":
                styled.append(f"\x1b[90m{ch}\x1b[0m")
            elif ch == ">":
                styled.append(f"\x1b[1;93m{ch}\x1b[0m")
            elif ch in "·":
                styled.append(f"\x1b[94m{ch}\x1b[0m")
            elif ch == "~":
                styled.append(f"\x1b[90m{ch}\x1b[0m")
            else:
                styled.append(f"\x1b[3;97m{ch}\x1b[0m")
        out.append("".join(styled))
    return "\n".join(out)


def _parse_ansi(text):
    fg, bg, bold = _FG_DEFAULT, _BG_DEFAULT, False
    pos = 0
    for m in _ANSI_RE.finditer(text):
        if m.start() > pos:
            yield text[pos:m.start()], fg, bg, bold
        codes = [int(x) if x else 0 for x in m.group(1).split(";")] or [0]
        for c in codes:
            if c == 0:
                fg, bg, bold = _FG_DEFAULT, _BG_DEFAULT, False
            elif c == 1: bold = True
            elif c == 22: bold = False
            elif c == 39: fg = _FG_DEFAULT
            elif c in _ANSI16: fg = _ANSI16[c]
        pos = m.end()
    if pos < len(text):
        yield text[pos:], fg, bg, bold


def render(out_path: Path, *, font_size: int = 16, padding: int = 20) -> None:
    text = _stylise(_BANNER)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
    font = ImageFont.truetype(font_path, font_size)
    bold = ImageFont.truetype(bold_path, font_size)
    tmp = Image.new("RGB", (1, 1))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), "M", font=font)
    cell_w = bbox[2] - bbox[0]
    cell_h = int((bbox[3] - bbox[1]) * 1.55)
    lines = text.splitlines() or [""]
    max_cols = max((len(_ANSI_RE.sub("", ln)) for ln in lines), default=1)
    img = Image.new("RGB",
                     (max_cols * cell_w + 2 * padding,
                      len(lines) * cell_h + 2 * padding), _BG_DEFAULT)
    d = ImageDraw.Draw(img)
    for row, line in enumerate(lines):
        x = padding
        y = padding + row * cell_h
        for chunk, fg, bg, is_bold in _parse_ansi(line):
            chunk_w = len(chunk) * cell_w
            d.text((x, y), chunk, font=(bold if is_bold else font), fill=fg)
            x += chunk_w
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="paper/figures/banner.png")
    args = p.parse_args(argv)
    render(Path(args.out))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
