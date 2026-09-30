---
title: Tessera SRM
emoji: 🛰️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
short_description: Sentinel-2 10 m to 2.5 m super-resolution mapping (SIH26142)
---

# Tessera SRM backend

Super-resolution mapping for Sentinel-2 imagery, from 10 m to 2.5 m, built for
SIH26142 (National Technical Research Organisation).

The model is a 2× SwinIR applied twice (10 m - 5 m - 2.5 m). Tiled inference
removes seam artifacts, and an uncertainty map is computed from flipped copies
of the input.

## How to use this Space

- **Combine bands:** upload the four single-band files (B02, B03, B04, B08, all
  10 m) from a Sentinel-2 product's `IMG_DATA/R10m` folder and get one 4-band
  GeoTIFF back.
- **Enhance:** upload a 4-band GeoTIFF, choose the band order and crop size, and
  get the bicubic and super-resolved images, an NDVI image, an optional
  uncertainty map, a report, and a georeferenced 2.5 m GeoTIFF.

## Limits

- 5 m output is validated against real VENUS imagery on unseen tiles.
- 2.5 m output is extrapolated, since no real 2.5 m reference exists in the
  training data.
- Uncertainty is a guide to where the model is unsure, not a confidence interval.
- Enhance works on a crop from the top-left corner (64 to 256 px).

## Links

- Website: https://steady-bienenstitch-b08fa0.netlify.app

This Space is also the backend for the Workbench section of the Tessera website.
