import os, sys, urllib.request
import numpy as np, torch, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from rasterio.transform import Affine
from rasterio.windows import Window
from skimage.transform import resize

# SwinIR's model code is one file with no other project dependencies, so it's
# fetched once at startup instead of needing a Dockerfile step to clone it.
SWINIR_URL = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/models/network_swinir.py"
if not os.path.exists("network_swinir.py"):
    urllib.request.urlretrieve(SWINIR_URL, "network_swinir.py")
sys.path.insert(0, ".")
from network_swinir import SwinIR

device = "cuda" if torch.cuda.is_available() else "cpu"
B = torch.load("model_bundle.pt", map_location="cpu", weights_only=False)
BAND_SCALE, FINAL_BANDS = np.asarray(B["band_scale"], np.float32), tuple(B["bands"])
model = SwinIR(upscale=B["scale"], in_chans=4, img_size=B["lr_patch"], window_size=8, img_range=1.,
               depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2,
               upsampler="pixelshuffledirect", resi_connection="1conv")
model.load_state_dict(B["model"]); model = model.to(device).eval()

# ---- inference functions, copied unchanged from the notebook ----
@torch.no_grad()
def sr2x_tiled(net, img_hwc, patch=48, margin=4, batch=16):
    net.eval(); s = 2
    h, w, c = img_hwc.shape
    core = patch - 2 * margin
    ny, nx = -(-h // core), -(-w // core)
    P = np.pad(img_hwc, ((margin, ny*core + margin - h), (margin, nx*core + margin - w), (0, 0)), mode="reflect")
    out = np.zeros((ny*core*s, nx*core*s, c), np.float32)
    coords = [(y*core, x*core) for y in range(ny) for x in range(nx)]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i+batch]
        t = torch.from_numpy(np.stack([P[y:y+patch, x:x+patch] for y, x in chunk]).transpose(0, 3, 1, 2)).float().to(device)
        sr = net(t).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
        for (y, x), p in zip(chunk, sr):
            out[y*s:(y+core)*s, x*s:(x+core)*s] = p[margin*s:(patch-margin)*s, margin*s:(patch-margin)*s]
    return out[:h*s, :w*s]

def cascade_tile(net, img, steps=2):
    outs, cur = [], img
    for _ in range(steps):
        cur = sr2x_tiled(net, cur); outs.append(cur)
    return outs

def tile_uncertainty(net, img):
    preds = []
    for fy in (False, True):
        for fx in (False, True):
            x = img[::-1] if fy else img
            x = x[:, ::-1] if fx else x
            p = sr2x_tiled(net, np.ascontiguousarray(x))
            p = p[:, ::-1] if fx else p
            p = p[::-1] if fy else p
            preds.append(p)
    preds = np.stack(preds)
    return preds.mean(0), preds.std(0).mean(-1)

def ndvi_dn(img, eps=1e-6):
    d = img * BAND_SCALE
    return (d[..., 3] - d[..., 2]) / (d[..., 3] + d[..., 2] + eps)

def bicubic_up(img, f):
    h, w, c = img.shape
    return np.clip(resize(img, (h*f, w*f, c), order=3, mode="edge", anti_aliasing=False), 0, 1)

def block_down(img, f):
    h, w, c = img.shape
    return img.reshape(h//f, f, w//f, f, c).mean((1, 3))

def rgb_disp(img, ref=None):
    ref = img if ref is None else ref
    r = ref[..., [2, 1, 0]]; lo, hi = np.percentile(r, 2), np.percentile(r, 98)
    return np.clip((img[..., [2, 1, 0]] - lo) / (hi - lo + 1e-6), 0, 1)

def read_tile(path, bands=(1, 2, 3, 4), win=None):
    with rasterio.open(path) as src:
        x0, y0, s = win
        w = Window(x0, y0, min(s, src.width - x0), min(s, src.height - y0))
        arr = src.read(list(bands), window=w).astype(np.float32)
        crs, tf = src.crs, src.window_transform(w)
    if arr.max() <= 5:
        arr = arr * 10000.0
    return np.clip(arr.transpose(1, 2, 0) / BAND_SCALE, 0, 1), crs, tf

def run_cascade_geotiff(net, tile_path, out_path, bands, steps=2, win=None):
    img, crs, tf = read_tile(tile_path, bands, win)
    outs = cascade_tile(net, img, steps); f = 2 ** steps
    dn = (outs[-1] * BAND_SCALE).astype(np.float32).transpose(2, 0, 1)
    with rasterio.open(out_path, "w", driver="GTiff", height=dn.shape[1], width=dn.shape[2], count=dn.shape[0],
                       dtype="float32", crs=crs, transform=tf * Affine.scale(1 / f, 1 / f)) as dst:
        dst.write(dn)
    return img, outs, crs, tf

def verify_sr_georef(img, crs, tf, sr_path, f, tol=1e-6):
    with rasterio.open(sr_path) as sr:
        assert sr.crs == crs, "CRS mismatch"
        assert abs(sr.transform.a - tf.a / f) < tol * max(1, abs(tf.a)), "Pixel size scale wrong"
        assert abs(sr.transform.c - tf.c) < 1e-3 and abs(sr.transform.f - tf.f) < 1e-3, "Origin shifted"
        assert sr.width == img.shape[1] * f and sr.height == img.shape[0] * f, "Output dimensions wrong"

def _cm(arr, cmap, vmin, vmax):
    a = np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)
    return (plt.get_cmap(cmap)(a)[..., :3] * 255).astype(np.uint8)

# ---- the function Gradio exposes, both as a web form and as an API ----
def enhance(file, band_order, crop, uncertainty):
    if file is None:
        raise gr.Error("Upload a GeoTIFF first.")
    crop = max(64, min(int(crop), 256))
    bands = (1, 2, 3, 4) if band_order == "copernicus" else FINAL_BANDS
    out_path = "/tmp/tessera_SR_2p5m.tif"
    try:
        img, outs, crs, tf = run_cascade_geotiff(model, file, out_path, bands, win=(0, 0, crop))
        st2 = outs[-1]; bc4 = bicubic_up(img, 4)
        try:
            verify_sr_georef(img, crs, tf, out_path, 4); geo = "passed (CRS, pixel size, origin, footprint)"
        except AssertionError as e:
            geo = f"FAILED: {e}"
        unc_img, unc_txt = None, "not computed"
        if uncertainty:
            _, u = tile_uncertainty(model, img)
            unc_img, unc_txt = _cm(u, "inferno", 0, max(float(u.max()), 1e-9)), f"mean {u.mean():.5f}"
        cons = float(np.sqrt(np.mean((block_down(st2, 4) - img) ** 2)))
        cons_bc = float(np.sqrt(np.mean((block_down(bc4, 4) - img) ** 2)))
        report = (f"Output: {st2.shape[1]} x {st2.shape[0]} px, about 2.5 m, top-left crop of {crop} px\n"
                  f"Georeferencing check: {geo}\n"
                  f"Consistency RMSE: {cons:.5f} (bicubic {cons_bc:.5f}; bicubic matches by construction)\n"
                  f"Stage-1 uncertainty (flip std): {unc_txt}\n"
                  f"NDVI mean: {ndvi_dn(st2).mean():.3f}\n"
                  f"No high-resolution reference for an uploaded tile, so PSNR and SSIM are not reported. "
                  f"Fine details are model-inferred, not directly observed.")
        return ((rgb_disp(bc4, bc4) * 255).astype(np.uint8), (rgb_disp(st2, bc4) * 255).astype(np.uint8),
                _cm(ndvi_dn(st2), "RdYlGn", -1, 1), unc_img, report, out_path)
    except Exception as e:
        raise gr.Error(f"Could not process this file: {e}")

demo = gr.Interface(
    fn=enhance,
    inputs=[
        gr.File(label="Sentinel-2 GeoTIFF (4 bands)", type="filepath", file_types=[".tif", ".tiff"]),
        gr.Radio(["copernicus", "sen2venus"], value="copernicus", label="Band order"),
        gr.Slider(64, 256, value=128, step=32, label="Crop from top-left (px)"),
        gr.Checkbox(value=True, label="Include uncertainty map"),
    ],
    outputs=[
        gr.Image(label="Bicubic 4x"), gr.Image(label="Super-resolved (2.5 m)"),
        gr.Image(label="NDVI"), gr.Image(label="Uncertainty"),
        gr.Textbox(label="Report", lines=6), gr.File(label="Download GeoTIFF"),
    ],
    title="Tessera SRM backend",
    description="Called by the Tessera website. You can also test it directly here.",
    api_name="enhance",
)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
