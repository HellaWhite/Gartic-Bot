# Gartic Phone HyperDraw Bot

A refined DrawBot-style bot for **Gartic Phone** with better color matching, faster stroke planning, and easier calibration.

## Highlights

- **18 swatches by default** (`--palette-count 18`).
- **Screenshot calibration UI** (default): click points directly on a screenshot instead of hover+Enter.
- **Hover fallback** if screenshot UI is unavailable.
- **Manual or grid palette capture**.
- **Crash-tolerant swatch RGB sampling** with fallback palette values.
- **Perceptual color matching** (CIEDE2000).
- **Fast drawing** using run-length segments + serpentine ordering.

> ⚠️ Educational use only. Automation may violate platform/game rules.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Calibrate

### Recommended (default screenshot click-menu)

```bash
python bot.py calibrate --palette-mode manual --palette-count 18
```

This opens a screenshot-based click menu and asks for:
1. canvas top-left
2. canvas bottom-right
3. swatch 1..18 (in UI order)

### Grid mode (if swatches are perfectly aligned)

```bash
python bot.py calibrate --palette-mode grid --palette-cols 9 --palette-rows 2
```

### If screenshot APIs are unstable

Use hover mode:

```bash
python bot.py calibrate --calibration-ui hover --palette-mode manual --palette-count 18
```

Skip swatch RGB screenshot sampling completely:

```bash
python bot.py calibrate --palette-mode manual --palette-count 18 --skip-swatch-rgb-sampling
```

## 2) Draw

```bash
python bot.py draw ./my-image.png --width 420 --height 320 --speed 2.2 --preview
```

Helpful flags:
- `--adaptive-palette 16`
- `--min-run 2`
- `--min-color-pixels 10`
- `--dry-run`

## Notes on “AI auto-detect everything”

This version adds a screenshot click-menu to make calibration much easier and less error-prone.
A fully automatic “internet-search + screen understanding” mode is intentionally not used here because it is unreliable across themes/resolutions and can break unpredictably.

## Troubleshooting

- `unrecognized arguments`: pull latest code and re-run.
- `Image not found`: pass a real image path (e.g. `./my-image.png`).
- Screenshot errors: use `--calibration-ui hover` and/or `--skip-swatch-rgb-sampling`.

## Safety

- `pyautogui.FAILSAFE` is enabled (move mouse to corner to abort).
- Start at low resolution first.
- Keep the game window fixed while calibrating and drawing.
