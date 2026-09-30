# Tessera: super-resolution mapping for Sentinel-2

Deep-learning super-resolution mapping (SRM) that takes medium-resolution Sentinel-2 imagery at 10 m and resolves it to 2.5 m. Built for **SIH26142**, National Technical Research Organisation (NTRO), Smart India Hackathon.

Each output comes with a georeference check and an uncertainty map, because some of the detail is inferred rather than observed.

- **Live website:** https://steady-bienenstitch-b08fa0.netlify.app
- **Backend (Hugging Face Space):** https://huggingface.co/spaces/VaishnaviSingh08/tessera-srm

## What it does

- Takes a four-band Sentinel-2 GeoTIFF (blue, green, red, near-infrared), or four separate band files (B02, B03, B04, B08) that it combines first.
- Returns a 2.5 m float32 GeoTIFF in the original reflectance units, with CRS, pixel size and origin checked against the input.
- Also returns a bicubic 4× comparison, an NDVI image, an optional uncertainty map, and a short report.

## Method

1. **Training data.** Pairs of Sentinel-2 and VENUS tiles from the SEN2VENUS dataset, four bands, split by image so test tiles are never seen in training.
2. **Model.** A 2× SwinIR (transformer-based), trained with L1 loss for 60 epochs. Each band is scaled by its own 99.5th percentile so near-infrared does not saturate.
3. **Cascade.** The 2× model is applied twice: 10 m to 5 m to 2.5 m.
4. **Tiled inference.** 48 px tiles with a 4 px border trimmed, which removes seam artifacts.
5. **Uncertainty.** Four flipped copies of the input are run through the model. Where they disagree, the uncertainty map is bright.
6. **Packaging.** Weights, band scales and settings are saved as one 11.7 MB model bundle (`model_bundle.pt`), so the service starts without the training data.

## Results

Measured at 5 m on 124 held-out test tiles against real VENUS imagery, mean ± standard deviation across tiles.

| Method | PSNR (dB) | SSIM | RMSE | NDVI error | Spectral angle (°) |
|---|---|---|---|---|---|
| Bicubic 2× | 29.617 ± 2.923 | 0.888 ± 0.056 | 0.0349 ± 0.0113 | **0.0286** ± 0.0229 | 1.721 ± 0.894 |
| SwinIR 2× (stage 1, 5 m) | **30.257** ± 2.749 | **0.898** ± 0.053 | **0.0322** ± 0.0101 | 0.0290 ± 0.0333 | **1.663** ± 1.067 |

Best value in each column is in bold. Against bicubic, SwinIR gains about 0.64 dB PSNR and 0.010 SSIM.

Figures are in `website/figures/`: qualitative comparisons, the uncertainty map next to actual error, and the training curve.

## Limits

- **5 m output is validated.** It is compared pixel by pixel with real VENUS imagery on unseen tiles.
- **2.5 m output is extrapolated.** No real 2.5 m reference exists in the training data. We check only that averaging the result back to 10 m stays close to the input. That check gives an RMSE of 0.0242 for the cascade against 0.0055 for bicubic. Bicubic matches the input almost by construction, so this catches drift and does not rank the methods.
- **Uncertainty is a guide, not a confidence interval.** It is computed for the 5 m stage only. Its link to real error is positive but weak (correlation 0.25 ± 0.13 over 20 test tiles).
- **Vegetation index.** NDVI error is on par with bicubic (about 0.029 for both). Do not read the enhanced NDVI as more accurate; the measured gain is in structure, not in the index.
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
└── notebooks/              # training and evaluation notebooks
```

## Run it locally

**Website**

```bash
cd website
python3 -m http.server 8000
```

Open http://localhost:8000. Opening `index.html` by double-click will not work, because the page fetches `data/final_results.json`. The Workbench section talks to the hosted Space, set by the `SPACE` line in `index.html`.

**Backend**

```bash
cd space
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860. On first start, `app.py` downloads `network_swinir.py` from the official SwinIR repository. Without a GPU it runs on CPU, which is slower.

**API**

The backend exposes two routes through the Gradio client:

- `/combine`: four single-band files (`b02`, `b03`, `b04`, `b08`) and a `crop` size in, one 4-band GeoTIFF out.
- `/enhance`: a GeoTIFF (`file`), `band_order` (`copernicus` or `sen2venus`), `crop` (64 to 256 px) and `uncertainty` (true or false) in. The outputs are the bicubic image, the super-resolved image, the NDVI image, the uncertainty image, the report and the 2.5 m GeoTIFF.

## Notes on the hosted demo

- The Space runs on ZeroGPU, so anonymous visitors get a small daily GPU allowance. After a quiet period, the first request can take a minute while the service wakes up.
- Enhance works on a crop from the top-left corner of the input (up to 256 px) to keep requests fast.

## Acknowledgements

- [SwinIR](https://github.com/JingyunLiang/SwinIR) by Liang et al., the model architecture.
- The SEN2VENUS dataset, for paired Sentinel-2 and VENUS imagery.
- Copernicus Sentinel-2 data.

## Team

Add team name and members here.

## License

Add a license here before making the repository public.
