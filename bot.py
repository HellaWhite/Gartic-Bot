#!/usr/bin/env python3
"""Gartic Phone HyperDraw Bot.

Designed to outperform simple DrawBot-style implementations with:
- Perceptual color matching (CIEDE2000)
- Optional adaptive image palette extraction
- Fast stroke planning with run-length encoding + path-aware ordering
- Flexible calibration that works even when screenshot APIs fail
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _pyautogui():
    import pyautogui  # type: ignore

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
    return pyautogui

CONFIG_PATH = Path("config.json")
ARTIFACTS_DIR = Path("artifacts")

# Fallback colors for common Gartic-like palettes if screen sampling fails.
DEFAULT_GARTIC_PALETTE = [
    (20, 20, 20),
    (245, 245, 245),
    (220, 60, 70),
    (245, 140, 65),
    (245, 220, 85),
    (90, 190, 75),
    (70, 165, 215),
    (95, 105, 205),
    (170, 95, 185),
    (210, 105, 155),
    (130, 75, 55),
    (220, 180, 150),
]


@dataclass
class Segment:
    y: int
    x0: int
    x1: int


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


def wait_for_enter(prompt: str) -> Tuple[int, int]:
    input(f"{prompt}: hover mouse and press Enter...")
    pos = _pyautogui().position()
    return int(pos.x), int(pos.y)


def sample_pixel_rgb(pos: Tuple[int, int]) -> Optional[Tuple[int, int, int]]:
    """Try to sample 1 pixel from the desktop; returns None when unsupported."""
    try:
        shot = _pyautogui().screenshot(region=(pos[0], pos[1], 1, 1))
        rgb = shot.getpixel((0, 0))
        return int(rgb[0]), int(rgb[1]), int(rgb[2])
    except Exception:
        return None


def _interpolate_grid(a: Tuple[int, int], b: Tuple[int, int], cols: int, rows: int) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    dx = 0 if cols == 1 else (b[0] - a[0]) / (cols - 1)
    dy = 0 if rows == 1 else (b[1] - a[1]) / (rows - 1)
    for r in range(rows):
        for c in range(cols):
            x = int(round(a[0] + c * dx))
            y = int(round(a[1] + r * dy))
            points.append((x, y))
    return points


def calibrate(args: argparse.Namespace) -> None:
    print("Starting calibration")
    canvas_tl = wait_for_enter("Canvas top-left")
    canvas_br = wait_for_enter("Canvas bottom-right")

    if args.palette_mode == "grid":
        swatch_a = wait_for_enter("Palette swatch top-left")
        swatch_b = wait_for_enter("Palette swatch bottom-right")
        swatches = _interpolate_grid(swatch_a, swatch_b, args.palette_cols, args.palette_rows)
    else:
        swatches = []
        print(f"Manual mode: mark {args.palette_count} swatches in the same order they appear in UI.")
        for i in range(args.palette_count):
            swatches.append(wait_for_enter(f"Swatch {i + 1}/{args.palette_count}"))

    sampled: List[Optional[Tuple[int, int, int]]] = [sample_pixel_rgb(p) for p in swatches]
    sampled_ok = [c for c in sampled if c is not None]

    if len(sampled_ok) == len(swatches):
        palette_rgb = [c for c in sampled if c is not None]
        print(f"Sampled all {len(palette_rgb)} swatch RGB values successfully.")
    else:
        print(
            "Warning: screenshot-based color sampling failed for some/all swatches. "
            "Using fallback palette values; positions are still saved."
        )
        fallback = (DEFAULT_GARTIC_PALETTE * ((len(swatches) // len(DEFAULT_GARTIC_PALETTE)) + 1))[: len(swatches)]
        palette_rgb = list(fallback)

    payload = {
        "canvas_tl": list(canvas_tl),
        "canvas_br": list(canvas_br),
        "swatches": [list(p) for p in swatches],
        "palette_rgb": [list(rgb) for rgb in palette_rgb],
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved calibration to {CONFIG_PATH}")


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
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    eps = 216 / 24389
    kappa = 24389 / 27
    fxyz = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    l = 116 * fxyz[:, 1] - 16
    a = 500 * (fxyz[:, 0] - fxyz[:, 1])
    b = 200 * (fxyz[:, 1] - fxyz[:, 2])
    return np.stack([l, a, b], axis=1)


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    l1, a1, b1 = lab1[:, None, 0], lab1[:, None, 1], lab1[:, None, 2]
    l2, a2, b2 = lab2[None, :, 0], lab2[None, :, 1], lab2[None, :, 2]

    c1 = np.sqrt(a1**2 + b1**2)
    c2 = np.sqrt(a2**2 + b2**2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt((c_bar**7) / (c_bar**7 + 25**7 + 1e-12)))

    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = np.sqrt(a1p**2 + b1**2)
    c2p = np.sqrt(a2p**2 + b2**2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dlp = l2 - l1
    dcp = c2p - c1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where((c1p * c2p) == 0, 0, dhp)
    dhp_term = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp / 2))

    lbar = (l1 + l2) / 2
    cbar = (c1p + c2p) / 2
    hbar = (h1p + h2p) / 2
    hbar = np.where(np.abs(h1p - h2p) > 180, hbar + 180, hbar) % 360
    hbar = np.where((c1p * c2p) == 0, h1p + h2p, hbar)

    t = (
        1
        - 0.17 * np.cos(np.radians(hbar - 30))
        + 0.24 * np.cos(np.radians(2 * hbar))
        + 0.32 * np.cos(np.radians(3 * hbar + 6))
        - 0.20 * np.cos(np.radians(4 * hbar - 63))
    )
    sl = 1 + (0.015 * (lbar - 50) ** 2) / np.sqrt(20 + (lbar - 50) ** 2)
    sc = 1 + 0.045 * cbar
    sh = 1 + 0.015 * cbar * t
    dtheta = 30 * np.exp(-((hbar - 275) / 25) ** 2)
    rc = 2 * np.sqrt((cbar**7) / (cbar**7 + 25**7 + 1e-12))
    rt = -rc * np.sin(2 * np.radians(dtheta))

    return np.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhp_term / sh) ** 2 + rt * (dcp / sc) * (dhp_term / sh))


def kmeans_palette(rgb: np.ndarray, k: int, iters: int = 18) -> np.ndarray:
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
    img_lab = srgb_to_lab(flat)
    pal_lab = srgb_to_lab(palette.astype(np.uint8))
    d = delta_e_ciede2000(img_lab, pal_lab)
    return np.argmin(d, axis=1).reshape(rgb.shape[:2])


def extract_segments(mask: np.ndarray, min_run: int) -> List[Segment]:
    h, w = mask.shape
    segments: List[Segment] = []
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
                segments.append(Segment(y=y, x0=x0, x1=x1))
    return segments


def order_segments(segments: List[Segment]) -> List[Segment]:
    by_row: Dict[int, List[Segment]] = {}
    for seg in segments:
        by_row.setdefault(seg.y, []).append(seg)

    ordered: List[Segment] = []
    for y in sorted(by_row):
        row_segments = sorted(by_row[y], key=lambda s: s.x0)
        if y % 2 == 1:
            row_segments.reverse()
        ordered.extend(row_segments)
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
    for ci in range(len(palette)):
        mask = indexed == ci
        if int(mask.sum()) < min_color_pixels:
            segments_per_color.append([])
            continue
        segments_per_color.append(order_segments(extract_segments(mask, min_run=min_run)))

    if preview:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        preview_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for i, c in enumerate(palette):
            preview_rgb[indexed == i] = c
        Image.fromarray(preview_rgb).save(ARTIFACTS_DIR / "quantized_preview.png")

        overlay = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(overlay)
        for i, segs in enumerate(segments_per_color):
            color = tuple(int(v) for v in palette[i])
            for seg in segs:
                draw.line([(seg.x0, seg.y), (seg.x1, seg.y)], fill=color)
        overlay.save(ARTIFACTS_DIR / "segment_overlay.png")

    return DrawPlan(palette_rgb=palette, indexed=indexed, segments_per_color=segments_per_color)


def execute(plan_data: DrawPlan, calib: Calibration, speed: float, dry_run: bool) -> None:
    h, w = plan_data.indexed.shape
    cw, ch = calib.canvas_size
    sx = cw / max(w - 1, 1)
    sy = ch / max(h - 1, 1)

    total_segments = sum(len(s) for s in plan_data.segments_per_color)
    print(f"Total segments: {total_segments}")

    if dry_run:
        return

    duration = max(0.0, 0.01 / max(speed, 0.1))

    color_order = sorted(range(len(plan_data.palette_rgb)), key=lambda i: len(plan_data.segments_per_color[i]), reverse=True)
    for ci in color_order:
        segs = plan_data.segments_per_color[ci]
        if not segs or ci >= len(calib.swatches):
            continue

        pg = _pyautogui()
        pg.click(calib.swatches[ci][0], calib.swatches[ci][1])
        for seg in segs:
            x0 = int(calib.canvas_tl[0] + seg.x0 * sx)
            x1 = int(calib.canvas_tl[0] + seg.x1 * sx)
            y = int(calib.canvas_tl[1] + seg.y * sy)
            pg.moveTo(x0, y)
            if x0 == x1:
                pg.click()
            else:
                pg.dragTo(x1, y, duration=duration, button="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HyperDraw bot for Gartic Phone")
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="Capture canvas + swatches")
    c.add_argument("--palette-mode", choices=["grid", "manual"], default="manual")
    c.add_argument("--palette-cols", type=int, default=10, help="Used in grid mode")
    c.add_argument("--palette-rows", type=int, default=2, help="Used in grid mode")
    c.add_argument("--palette-count", type=int, default=12, help="Used in manual mode")

    d = sub.add_parser("draw", help="Generate plan and draw image")
    d.add_argument("image", type=Path)
    d.add_argument("--width", type=int, default=420)
    d.add_argument("--height", type=int, default=320)
    d.add_argument("--adaptive-palette", type=int, default=0)
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
    palette = np.array(calibration.palette_rgb, dtype=np.uint8)

    t0 = time.perf_counter()
    draw_plan = plan(
        image=args.image,
        width=args.width,
        height=args.height,
        palette_rgb=palette,
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
