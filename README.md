# Tessera: Deep Learning Super-Resolution Mapping for Sentinel-2

Built for **SIH26142** - *Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries*, National Technical Research Organisation (NTRO), Smart India Hackathon.

Tessera takes medium-resolution Sentinel-2 imagery (10 m) and resolves it to 2.5 m using a deep generative model, while explicitly tracking which parts of the output are validated against real ground truth and which are model-inferred. Every output comes with a georeference check and an uncertainty map, because some of the detail is inferred rather than directly observed - a distinction the problem statement specifically asks for.

- **Live website:** https://steady-bienenstitch-b08fa0.netlify.app
- **Backend (Hugging Face Space):** https://huggingface.co/spaces/VaishnaviSingh08/tessera-srm

## What it does

- Takes a four-band Sentinel-2 GeoTIFF (Blue, Green, Red, Near-Infrared), or four separate band files (B02, B03, B04, B08) that it combines first.
- Returns a 2.5 m float32 GeoTIFF in the original reflectance units, with CRS, pixel size and origin checked against the input.
- Also returns a bicubic 4× comparison, an NDVI image, an optional uncertainty map, and a short report.

## Method

1. **Training data.** Real, same-day paired Sentinel-2 (10 m) and VENUS (5 m) tiles from the SEN2VENUS dataset - 1,000 pairs, sampled at random across the dataset's 29 global sites rather than sequentially, to avoid biasing toward any one location. Split by image (70/15/15 train/val/test) so no image contributes patches to more than one split, and the test set is never used for training or checkpoint selection.
2. **Bands.** Four channels - Blue (B02), Green (B03), Red (B04), Near-Infrared (B08) - confirmed against SEN2VENUS's official band table rather than assumed. NIR was originally misidentified as a different band (Red Edge) during development; this was caught with a spectral sanity check (vegetation reflectance should be much higher in true NIR than in Red) before any training ran on it.
3. **Normalization.** Each band is scaled by its own 99.5th percentile, computed from the training split. An earlier version used one fixed scale for all four bands; this saturated the NIR channel (whose reflectance range is roughly 2–3× that of the visible bands) and produced misleadingly flat NDVI maps. Per-band scaling fixed this.
4. **Model.** A 2× SwinIR (transformer-based super-resolution network), `in_chans=4`, trained with L1 loss for 60 epochs, Adam optimizer with cosine annealing. The checkpoint with the lowest validation loss is kept, not simply the last epoch.
5. **Cascade for 4×.** The trained 2× model is applied twice - 10 m -> 5 m -> 2.5 m - rather than training a separate direct-4× model. This was a deliberate choice after comparing both: direct-4× has no real 2.5 m supervision anywhere in the dataset (VENUS itself only reaches 5 m), so its training target has to be synthesized, whereas the cascade's first stage is fully supervised by real data. On a small-scale ablation, the cascade also produced sharper, more structurally faithful output (higher SSIM) than a directly-trained 4× model.
6. **Tiled inference.** Images are processed in overlapping tiles with a trimmed border, which removes the seam artifacts that SwinIR's windowed attention otherwise introduces at tile edges.
7. **Uncertainty.** Four flipped copies of the input are run through the model (test-time augmentation); where the predictions disagree, the uncertainty map is bright. This is computed for the 5 m stage only, since it is faster and dropout-based uncertainty was checked and found not to apply (the model has no active dropout at inference).
8. **Packaging.** Weights, per-band scales and settings are saved together as one model bundle, so the service starts without needing the training data.

## Results

Measured at 5 m on 124 held-out test tiles against real VENUS imagery - the untouched test split, never used during training or model selection. Mean ± standard deviation across tiles; the columns below also report the paired 95% confidence interval on the SwinIR - Bicubic difference, and the share of individual tiles where SwinIR wins.

| Metric | Bicubic 2× | SwinIR 2× (stage 1, 5 m) | Paired gain (95% CI) | Tiles where SwinIR wins |
|---|---|---|---|---|
| PSNR (dB) | 29.617 ± 2.923 | **30.257** ± 2.749 | +0.64 [+0.52, +0.76] | 85% |
| SSIM | 0.888 ± 0.056 | **0.898** ± 0.053 | +0.010 [+0.008, +0.012] | 90% |
| RMSE | 0.0349 ± 0.0113 | **0.0322** ± 0.0101 | −0.0027 [−0.0031, −0.0022] | 85% |
| NDVI error (MAE) | **0.0286** ± 0.0229 | 0.0290 ± 0.0333 | +0.0004 [−0.0017, +0.0025] (not significant) | 69% |
| Spectral angle (°) | 1.721 ± 0.894 | **1.663** ± 1.067 | −0.058 [−0.113, −0.004] | 73% |

PSNR, SSIM and RMSE gains are consistent and statistically clear. The NDVI result is more nuanced: per-band RMSE is lower for SwinIR on every one of the four bands individually, and SwinIR's NDVI is closer to ground truth on 69% of tiles - but a handful of outlier tiles pull the mean difference back to roughly zero. We report this honestly rather than only the favourable per-tile figure, since NDVI (a ratio of two correlated bands) is more sensitive to those outliers than the RMSE metrics are.

Figures are in `website/figures/`: qualitative comparisons, the uncertainty map next to actual error, the training curve, and NDVI comparisons.

## A finding worth flagging: training data vs. real deployment data

When the trained model was run on real Sentinel-2 tiles downloaded directly from the Copernicus Browser (rather than through SEN2VENUS), the per-band normalization scales learned from the training data did not transfer directly - the real tiles' raw reflectance values were often outside the range seen in training (most severely for a tile containing bright cloud/water, where the mismatch was over 7×). This caused visible blocky artifacts in the output.

We diagnosed this as a genuine radiometric domain gap between the SEN2VENUS training distribution and raw Copernicus L2A exports, not a bug in the model itself. The fix - computing each real tile's normalization scale from its own pixel statistics at inference time, rather than reusing the fixed training-derived scale — brought the output's internal consistency back in line with what the model achieves on the SEN2VENUS test set. We consider this an important limitation to disclose rather than something to smooth over: it is exactly the kind of uncertainty/error behaviour the problem statement asks a solution to account for.

## Limits

- **Uncertainty is a guide, not a confidence interval.** It is computed for the 5 m stage only. Its correlation with actual error is positive but weak (0.25 ± 0.13 over 20 test tiles) — useful as a rough indicator of where the model is extrapolating more, not as a calibrated error bound.
- **Vegetation index.** NDVI error is on par with bicubic overall (see Results above); read the enhanced output as sharper in structure, not as more accurate in NDVI specifically.
- **Dataset coverage.** SEN2VENUS's 29 sites are weighted toward agricultural/ecological monitoring locations; urban areas are comparatively underrepresented relative to what the problem statement's use cases (small buildings, narrow roads) call for.
- **Fine detail** is model-inferred. Treat it as a lead for analysts, not a measurement.

## Repository layout

```
.
├── README.md
├── space/                  # backend, deployed on Hugging Face
│   ├── app.py
│   ├── requirements.txt
│   ├── model_bundle.pt
│   └── README.md           # Space configuration
├── website/                # static site, deployed on Netlify
│   ├── index.html
│   ├── data/final_results.json
│   └── figures/
└── notebooks/               # training and evaluation notebooks
```

## Training and evaluation notebook

`notebooks/` contains the full pipeline used to produce the results above: data loading and caching from SEN2VENUS, per-band normalization, model training with checkpointing, tiled inference, the bicubic baseline and paired statistical comparison, uncertainty computation, GeoTIFF I/O with georeference verification, and the real-tile evaluation described in the domain-gap section above.

## Notes on the hosted demo

- The Space runs on ZeroGPU, so anonymous visitors get a small daily GPU allowance. After a quiet period, the first request can take a minute while the service wakes up.
- Enhance works on a crop from the top-left corner of the input (up to 256 px) to keep requests fast.

## Acknowledgements

- [SwinIR](https://github.com/JingyunLiang/SwinIR) by Liang et al., the model architecture.
- The SEN2VENUS dataset, for paired Sentinel-2 and VENUS imagery.
- Copernicus Sentinel-2 data.

## Team

Team Name - Error142 
Team Members - Satvik Varshney, Vaishnavi Singh, Marvi Kulkarni, Kunal Patil, Tanisha Tembhare, Tanushree Sheth

