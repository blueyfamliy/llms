#!/usr/bin/env python3
"""Generate the LLMS Studio app icon.

Draws the mark at every size macOS asks for and writes an ``.iconset``
directory, which ``iconutil`` turns into ``assets/icon.icns``. Also writes a
1024px PNG for Linux/Windows builds and for the README.

    python scripts/make_icon.py

The mark is three rounded bars of decreasing width -- lines of text, which also
read as stacked layers -- on a squircle with an indigo-to-violet gradient, plus
a cyan caret suggesting generation in progress. Each size is drawn from the
geometry rather than downsampled from one big render, so the 16px version stays
crisp instead of turning to mush.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

# Sizes macOS expects inside an .iconset, as (pixel size, filename).
ICONSET_SIZES = [
    (16, "icon_16x16.png"),
    (32, "icon_16x16@2x.png"),
    (32, "icon_32x32.png"),
    (64, "icon_32x32@2x.png"),
    (128, "icon_128x128.png"),
    (256, "icon_128x128@2x.png"),
    (256, "icon_256x256.png"),
    (512, "icon_256x256@2x.png"),
    (512, "icon_512x512.png"),
    (1024, "icon_512x512@2x.png"),
]

TOP = (49, 46, 129)  # indigo
BOTTOM = (109, 40, 217)  # violet
BAR = (255, 255, 255)
CARET = (34, 211, 238)  # cyan


def _gradient(size: int) -> Image.Image:
    """A vertical TOP -> BOTTOM gradient, drawn one row at a time."""
    grad = Image.new("RGB", (1, size))
    pixels = grad.load()
    for y in range(size):
        t = y / max(1, size - 1)
        pixels[0, y] = tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM))
    return grad.resize((size, size))


def draw_icon(size: int) -> Image.Image:
    """Render the icon at ``size`` x ``size`` pixels."""
    # Supersample, then downsample once at the end: the only way to get smooth
    # curves out of Pillow's non-antialiased shape drawing.
    scale = 8 if size <= 128 else 2
    s = size * scale
    icon = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    # macOS icons sit in a transparent margin rather than filling the canvas.
    inset = round(s * 0.055)
    box = (inset, inset, s - inset - 1, s - inset - 1)
    side = box[2] - box[0]
    radius = round(side * 0.225)  # close to Apple's squircle

    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    icon.paste(_gradient(s), (0, 0), mask)

    draw = ImageDraw.Draw(icon)

    # Three text lines / layers. Thick bars with tight gaps so the three stay
    # distinguishable down at 16px instead of blurring into one block.
    bar_h = round(side * 0.115)
    gap = round(side * 0.077)
    widths = (0.56, 0.46, 0.34)
    block_h = 3 * bar_h + 2 * gap
    # Centered on the widest bar, and left-aligned like real lines of text.
    x0 = box[0] + round(side * (1 - widths[0]) / 2)
    y = box[1] + (side - block_h) // 2

    for fraction in widths:
        draw.rounded_rectangle(
            (x0, y, x0 + round(side * fraction), y + bar_h),
            radius=bar_h / 2,
            fill=BAR,
        )
        y += bar_h + gap

    # A vertical cyan cursor after the last line: text still being generated.
    # Vertical, not another horizontal blob, so it reads as a caret and adds
    # visual weight on the right to balance the left-aligned lines.
    caret_w = round(side * 0.052)
    caret_x = x0 + round(side * widths[-1]) + round(side * 0.06)
    caret_y = y - bar_h - gap - round(bar_h * 0.28)
    draw.rounded_rectangle(
        (caret_x, caret_y, caret_x + caret_w, caret_y + round(bar_h * 1.56)),
        radius=caret_w / 2,
        fill=CARET,
    )

    return icon.resize((size, size), Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("assets"))
    args = parser.parse_args()

    out_dir = args.out_dir
    iconset = out_dir / "icon.iconset"
    iconset.mkdir(parents=True, exist_ok=True)

    for size, name in ICONSET_SIZES:
        draw_icon(size).save(iconset / name)
    print(f"wrote {len(ICONSET_SIZES)} PNGs to {iconset}")

    draw_icon(1024).save(out_dir / "icon.png")
    print(f"wrote {out_dir / 'icon.png'}")

    # iconutil is macOS-only; on other platforms the PNG is enough.
    try:
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(out_dir / "icon.icns")],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        print("iconutil not found (not macOS) -- skipping icon.icns")
        return
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"iconutil failed: {exc.stderr.decode()}") from exc
    print(f"wrote {out_dir / 'icon.icns'}")


if __name__ == "__main__":
    main()
