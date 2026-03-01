# Gartic Phone HyperDraw Bot

An upgraded DrawBot-style bot for **Gartic Phone** with stronger color matching and faster stroke execution.

## Why this version is better

- **Manual swatch calibration mode** (recommended): you mark every palette swatch position yourself.
- **Grid calibration mode**: mark top-left and bottom-right swatches and auto-interpolate the rest.
- **Screenshot-failure tolerant**: if desktop screenshot sampling fails on Linux/Wayland/X11, bot still works using fallback palette values.
- **Perceptual color matching** via CIEDE2000.
- **Fast drawing** using run-length segments + serpentine row ordering.

> ⚠️ Educational use only. Automation may violate game rules/TOS.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Calibrate (important)

### Recommended: manual mode (asks for more than 2 points)

```bash
python bot.py calibrate --palette-mode manual --palette-count 12
```

What it asks you to mark:
1. canvas top-left
2. canvas bottom-right
3. each swatch center one by one (12 times by default)

Use this mode when your palette is not a perfect grid, or when the screenshot API is unstable.

If screenshot APIs are broken on your desktop, add:

```bash
python bot.py calibrate --palette-mode manual --palette-count 12 --skip-swatch-rgb-sampling
```


### Optional: grid mode (only 2 palette points)

```bash
python bot.py calibrate --palette-mode grid --palette-cols 10 --palette-rows 2
```

What it asks:
1. canvas top-left
2. canvas bottom-right
3. palette top-left swatch
4. palette bottom-right swatch

## 2) Draw an image

```bash
python bot.py draw ./example.png --width 420 --height 320 --speed 2.2 --preview
```

Helpful flags:
- `--adaptive-palette 16` (extract 16 colors from source image)
- `--min-run 2` (skip tiny runs)
- `--min-color-pixels 10` (skip tiny regions)
- `--dry-run` (plan only, no mouse movement)

## If screenshot sampling crashes

If you see screenshot errors during calibration (GNOME/Wayland/X11 pixbuf errors), the bot now continues by saving swatch positions and using fallback palette RGB values.
If your local script still says `unrecognized arguments: --palette-mode`, you are running an older file—pull latest changes in this repo first.


You can still draw because click positions are the most important part for palette selection.

## Safety

- `pyautogui.FAILSAFE` is enabled (slam mouse to corner to abort).
- Start with low size first (`--width 220 --height 160`).
- Keep the game window fixed in place.
