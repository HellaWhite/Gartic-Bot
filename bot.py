#!/usr/bin/env python3
"""Gartic Phone HyperDraw Bot.

Designed to outperform simple DrawBot-style implementations with:
- Perceptual color matching (CIEDE2000)
- Optional adaptive image palette extraction
- Fast stroke planning with run-length encoding + path-aware ordering
- Real swatch sampling during calibration for accurate color clicks
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyautogui
from PIL import Image, ImageDraw

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

CONFIG_PATH = Path("config.json")
ARTIFACTS_DIR = Path("artifacts")


@dataclass
class Segment:
    y: int
    x0: int
    x1: int

    @property
    def length(self) -> int:
        return self.x1 - self.x0 + 1


@dataclass
class Calibration:
    canvas_tl: Tuple[int, int]
    canvas_br: Tuple[int, int]
    swatches: List[Tuple[int, int]]
    palette_rgb: List[Tuple[int, int, int]]

    @property
    def canvas_size(self) -> Tuple[int, int]:
        return self.canvas_br[0] - self.canvas_tl[0], self.canvas_br[1] - self.canvas_tl[1]


@dataclass
class DrawPlan:
    palette_rgb: np.ndarray
    indexed: np.ndarray
    segments_per_color: List[List[Segment]]
    pixel_count_per_color: List[int]


def wait_for_enter(prompt: str) -> Tuple[int, int]:
    input(f"{prompt}: hover mouse and press Enter...")
    pos = pyautogui.position()
    return pos.x, pos.y


def sample_pixel_rgb(pos: Tuple[int, int]) -> Tuple[int, int, int]:
    shot = pyautogui.screenshot(region=(pos[0], pos[1], 1, 1))
    rgb = shot.getpixel((0, 0))
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def calibrate(args: argparse.Namespace) -> None:
    print("Starting calibration")
    canvas_tl = wait_for_enter("Canvas top-left")
    canvas_br = wait_for_enter("Canvas bottom-right")

    swatch_a = wait_for_enter("Palette swatch top-left")
    swatch_b = wait_for_enter("Palette swatch bottom-right")

    cols, rows = args.palette_cols, args.palette_rows
    swatches: List[Tuple[int, int]] = []
    palette_rgb: List[Tuple[int, int, int]] = []

    dx = 0 if cols == 1 else (swatch_b[0] - swatch_a[0]) / (cols - 1)
    dy = 0 if rows == 1 else (swatch_b[1] - swatch_a[1]) / (rows - 1)

    for r in range(rows):
        for c in range(cols):
            x = int(round(swatch_a[0] + c * dx))
            y = int(round(swatch_a[1] + r * dy))
            swatches.append((x, y))
            palette_rgb.append(sample_pixel_rgb((x, y)))

    payload = {
        "canvas_tl": list(canvas_tl),
        "canvas_br": list(canvas_br),
        "swatches": [list(p) for p in swatches],
        "palette_rgb": [list(rgb) for rgb in palette_rgb],
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(swatches)} swatches to {CONFIG_PATH}")


def load_calibration() -> Calibration:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing config.json. Run `python bot.py calibrate` first.")
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return Calibration(
        canvas_tl=tuple(raw["canvas_tl"]),
        canvas_br=tuple(raw["canvas_br"]),
        swatches=[tuple(x) for x in raw["swatches"]],
        palette_rgb=[tuple(x) for x in raw["palette_rgb"]],
    )


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)

    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    xyz = x @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)

    eps = 216 / 24389
    kappa = 24389 / 27
    fxyz = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)

    l = 116 * fxyz[:, 1] - 16
    a = 500 * (fxyz[:, 0] - fxyz[:, 1])
    b = 200 * (fxyz[:, 1] - fxyz[:, 2])
    return np.stack([l, a, b], axis=1)


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 distance matrix between arrays [n,3] and [m,3]."""
    L1, a1, b1 = lab1[:, None, 0], lab1[:, None, 1], lab1[:, None, 2]
    L2, a2, b2 = lab2[None, :, 0], lab2[None, :, 1], lab2[None, :, 2]

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt((Cbar**7) / (Cbar**7 + 25**7 + 1e-12)))

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where((C1p * C2p) == 0, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    Lbarp = (L1 + L2) / 2
    Cbarp = (C1p + C2p) / 2

    hbarp = (h1p + h2p) / 2
    hbarp = np.where(np.abs(h1p - h2p) > 180, hbarp + 180, hbarp)
    hbarp = hbarp % 360
    hbarp = np.where((C1p * C2p) == 0, h1p + h2p, hbarp)

    T = (
        1
        - 0.17 * np.cos(np.radians(hbarp - 30))
        + 0.24 * np.cos(np.radians(2 * hbarp))
        + 0.32 * np.cos(np.radians(3 * hbarp + 6))
        - 0.20 * np.cos(np.radians(4 * hbarp - 63))
    )

    Sl = 1 + (0.015 * (Lbarp - 50) ** 2) / np.sqrt(20 + (Lbarp - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T

    dtheta = 30 * np.exp(-((hbarp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt((Cbarp**7) / (Cbarp**7 + 25**7 + 1e-12))
    Rt = -Rc * np.sin(2 * np.radians(dtheta))

    kl = kc = kh = 1
    dE = np.sqrt(
        (dLp / (kl * Sl)) ** 2
        + (dCp / (kc * Sc)) ** 2
        + (dHp / (kh * Sh)) ** 2
        + Rt * (dCp / (kc * Sc)) * (dHp / (kh * Sh))
    )
    return dE


def kmeans_palette(rgb: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    if flat.shape[0] <= k:
        return np.unique(flat.astype(np.uint8), axis=0)

    rng = np.random.default_rng(1337)
    centers = flat[rng.choice(flat.shape[0], size=k, replace=False)]

    for _ in range(iters):
        d = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d, axis=1)
        for i in range(k):
            pts = flat[idx == i]
            if len(pts):
                centers[i] = pts.mean(axis=0)

    return np.clip(centers, 0, 255).astype(np.uint8)


def quantize(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3)
    lab_img = srgb_to_lab(flat)
    lab_pal = srgb_to_lab(palette)
    d = delta_e_ciede2000(lab_img, lab_pal)
    return np.argmin(d, axis=1).reshape(rgb.shape[:2])


def extract_segments(mask: np.ndarray, min_run: int) -> List[Segment]:
    h, w = mask.shape
    out: List[Segment] = []
    for y in range(h):
        x = 0
        row = mask[y]
        while x < w:
            if not row[x]:
                x += 1
                continue
            x0 = x
            while x < w and row[x]:
                x += 1
            x1 = x - 1
            if x1 - x0 + 1 >= min_run:
                out.append(Segment(y=y, x0=x0, x1=x1))
    return out


def order_segments(segments: List[Segment]) -> List[Segment]:
    """Greedy path ordering to reduce pen travel while staying fast."""
    by_row: Dict[int, List[Segment]] = {}
    for seg in segments:
        by_row.setdefault(seg.y, []).append(seg)
    for row in by_row.values():
        row.sort(key=lambda s: s.x0)

    ordered: List[Segment] = []
    cursor_x, cursor_y = 0, 0
    pending_rows = sorted(by_row.keys())

    while pending_rows:
        row = min(pending_rows, key=lambda r: abs(r - cursor_y))
        row_segs = by_row[row]
        if not row_segs:
            pending_rows.remove(row)
            continue

        nearest_i = min(range(len(row_segs)), key=lambda i: min(abs(row_segs[i].x0 - cursor_x), abs(row_segs[i].x1 - cursor_x)))
        seg = row_segs.pop(nearest_i)
        if nearest_i % 2 == 1:
            seg = Segment(y=seg.y, x0=seg.x1, x1=seg.x0)
        ordered.append(seg)
        cursor_y = seg.y
        cursor_x = seg.x1

    return ordered


def plan(
    image: Path,
    width: int,
    height: int,
    palette_rgb: np.ndarray,
    adaptive_palette: int,
    min_run: int,
    min_color_pixels: int,
    preview: bool,
) -> DrawPlan:
    pil = Image.open(image).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    rgb = np.array(pil, dtype=np.uint8)

    palette = kmeans_palette(rgb, adaptive_palette) if adaptive_palette > 0 else palette_rgb
    indexed = quantize(rgb, palette)

    segments_per_color: List[List[Segment]] = []
    pixel_count_per_color: List[int] = []

    for ci in range(len(palette)):
        mask = indexed == ci
        px_count = int(mask.sum())
        pixel_count_per_color.append(px_count)
        if px_count < min_color_pixels:
            segments_per_color.append([])
            continue
        segments = extract_segments(mask, min_run=min_run)
        segments_per_color.append(order_segments(segments))

    if preview:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        preview_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for i, c in enumerate(palette):
            preview_rgb[indexed == i] = c
        Image.fromarray(preview_rgb).save(ARTIFACTS_DIR / "quantized_preview.png")

        overlay = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(overlay)
        for ci, segs in enumerate(segments_per_color):
            color = tuple(int(v) for v in palette[ci])
            for seg in segs:
                draw.line([(seg.x0, seg.y), (seg.x1, seg.y)], fill=color)
        overlay.save(ARTIFACTS_DIR / "segment_overlay.png")

        stats = {
            "width": width,
            "height": height,
            "palette_size": int(len(palette)),
            "total_segments": int(sum(len(s) for s in segments_per_color)),
            "pixel_count_per_color": pixel_count_per_color,
        }
        (ARTIFACTS_DIR / "plan_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return DrawPlan(palette_rgb=palette, indexed=indexed, segments_per_color=segments_per_color, pixel_count_per_color=pixel_count_per_color)


def click_swatch(pos: Tuple[int, int]) -> None:
    pyautogui.click(pos[0], pos[1])


def execute(plan: DrawPlan, calibration: Calibration, speed: float, dry_run: bool) -> None:
    h, w = plan.indexed.shape
    cw, ch = calibration.canvas_size
    sx = cw / max(w - 1, 1)
    sy = ch / max(h - 1, 1)

    total_segments = sum(len(v) for v in plan.segments_per_color)
    print(f"Total segments: {total_segments}")

    if dry_run:
        return

    duration = max(0.0, 0.01 / max(speed, 0.1))

    color_order = sorted(
        range(len(plan.palette_rgb)),
        key=lambda i: len(plan.segments_per_color[i]),
        reverse=True,
    )

    for ci in color_order:
        segments = plan.segments_per_color[ci]
        if not segments:
            continue
        if ci < len(calibration.swatches):
            click_swatch(calibration.swatches[ci])
        else:
            continue

        for seg in segments:
            x0 = int(calibration.canvas_tl[0] + seg.x0 * sx)
            x1 = int(calibration.canvas_tl[0] + seg.x1 * sx)
            y = int(calibration.canvas_tl[1] + seg.y * sy)
            pyautogui.moveTo(x0, y)
            if x0 == x1:
                pyautogui.click()
            else:
                pyautogui.dragTo(x1, y, duration=duration, button="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HyperDraw bot for Gartic Phone")
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="Capture canvas + swatch coordinates")
    c.add_argument("--palette-cols", type=int, default=10)
    c.add_argument("--palette-rows", type=int, default=2)

    d = sub.add_parser("draw", help="Generate plan and draw image")
    d.add_argument("image", type=Path)
    d.add_argument("--width", type=int, default=420)
    d.add_argument("--height", type=int, default=320)
    d.add_argument("--adaptive-palette", type=int, default=0, help="0=disabled, or number of colors to extract")
    d.add_argument("--min-run", type=int, default=2)
    d.add_argument("--min-color-pixels", type=int, default=8)
    d.add_argument("--speed", type=float, default=2.0)
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--preview", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "calibrate":
        calibrate(args)
        return

    calibration = load_calibration()
    palette_rgb = np.array(calibration.palette_rgb, dtype=np.uint8)

    t0 = time.perf_counter()
    draw_plan = plan(
        image=args.image,
        width=args.width,
        height=args.height,
        palette_rgb=palette_rgb,
        adaptive_palette=args.adaptive_palette,
        min_run=args.min_run,
        min_color_pixels=args.min_color_pixels,
        preview=args.preview,
    )
    t1 = time.perf_counter()
    print(f"Planning completed in {t1 - t0:.3f}s")

    execute(draw_plan, calibration, speed=args.speed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
