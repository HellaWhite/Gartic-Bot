# Gartic Phone HyperDraw Bot

An upgraded DrawBot-style bot for **Gartic Phone** with stronger color accuracy and faster drawing execution.

## What is better vs basic bots

- **Real swatch sampling from your screen** during calibration (no blind hardcoded palette).
- **Perceptual color matching** using **CIEDE2000**, which is notably better than raw RGB distance.
- **Faster path planning** with run-length strokes + travel-aware ordering.
- **Noise reduction controls** (`--min-run`, `--min-color-pixels`) to avoid wasting time on tiny details.
- **Preview artifacts** for debugging quantization and stroke maps.

> ⚠️ Educational use only. Automation may violate platform/game rules.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Calibrate

```bash
python bot.py calibrate --palette-cols 10 --palette-rows 2
```

You will capture:

1. canvas top-left
2. canvas bottom-right
3. first palette swatch (top-left swatch)
4. last palette swatch (bottom-right swatch)

The bot interpolates all swatch positions and samples actual RGB values from the screen into `config.json`.

## 2) Draw

```bash
python bot.py draw ./example.png --width 420 --height 320 --speed 2.2 --preview
```

Helpful flags:

- `--adaptive-palette 16` → extract 16 colors from image via k-means.
- `--min-run 2` → ignore single-pixel runs for speed.
- `--min-color-pixels 10` → skip tiny color regions.
- `--dry-run` → planning and stats only, no mouse control.

## Performance notes

This bot accelerates by reducing drag count and cursor travel:

1. quantize image into a small palette,
2. build per-color masks,
3. convert masks into horizontal segments,
4. order segments to reduce pen travel,
5. draw colors with most work first.

## Artifacts

When using `--preview`, outputs are saved in `artifacts/`:

- `quantized_preview.png`
- `segment_overlay.png`
- `plan_stats.json`

## Safety

- `pyautogui.FAILSAFE` is enabled (move mouse to a screen corner to interrupt).
- Start with small canvas sizes first.
- Keep game window stable; UI shifts can break swatch targeting.
