#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_assets_rgb_only.py
-----------------------
Create slide-ready mining-compliance assets from a **visual** image (JPG/PNG/TIF).
No spectral bands required. Heuristics use RGB + texture.

Outputs for each site/crop prefix:
  <prefix>_rgb.png
  <prefix>_mask.png
  <prefix>_overlay.png
  <prefix>_table.csv
  <prefix>_table.png
  <prefix>_3d.png
  <prefix>_hero.png  (mask+overlay+table stitched)

Usage examples:
  python make_assets_rgb_only.py --image rajasthan.jpg --site rajasthan
  python make_assets_rgb_only.py --image odisha.jpg --site odisha --resize 1600
  # Crop a specific box (x,y,w,h):
  python make_assets_rgb_only.py --image rajasthan.jpg --site raj --crop 200 300 1200 900
  # Make a 2x2 grid of crops:
  python make_assets_rgb_only.py --image rajasthan.jpg --site raj --grid 2 2

Dependencies:
  pip install numpy pillow matplotlib shapely opencv-python pandas
"""
import os, argparse, math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import pandas as pd
import cv2

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_rgb(path, resize=None):
    im = Image.open(path).convert("RGB")
    if resize:
        # maintain aspect ratio by width
        w = int(resize)
        h = int(round(im.height * (w / im.width)))
        im = im.resize((w, h), Image.Resampling.LANCZOS)
    return im

def crop_grid(im, gx, gy):
    W, H = im.size
    cw, ch = W // gx, H // gy
    crops = []
    for j in range(gy):
        for i in range(gx):
            box = (i*cw, j*ch, (i+1)*cw if i<gx-1 else W, (j+1)*ch if j<gy-1 else H)
            crops.append((f"r{j}c{i}", im.crop(box)))
    return crops

def crop_box(im, x, y, w, h):
    W, H = im.size
    box = (int(x), int(y), min(int(x+w), W), min(int(y+h), H))
    return [("crop", im.crop(box))]

def proxy_indices_from_rgb(rgb):
    # RGB in float32 0..1
    R = rgb[...,0]; G = rgb[...,1]; B = rgb[...,2]
    ndvi_like = (G - R) / (G + R + 1e-6)   # green vs red
    ndbi_like = (R - B) / (R + B + 1e-6)   # red vs blue
    gy, gx = np.gradient(cv2.GaussianBlur(R, (0,0), 1.2))
    tex = np.hypot(gx, gy)                 # texture magnitude
    return ndvi_like, ndbi_like, tex

def mining_mask(ndvi_like, ndbi_like, tex):
    m = (ndvi_like < 0.05) & (ndbi_like > -0.05) & (tex > np.percentile(tex, 65))
    m_u8 = (m*255).astype('uint8')
    m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return m_u8

def mask_to_polygons(mask_u8, min_area_px=500):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) >= 3:
            pts = [(float(p[0][0]), float(p[0][1])) for p in c]
            poly = Polygon(pts)
            if poly.is_valid and poly.area > min_area_px:
                polys.append(poly)
    return polys, contours

def fake_lease_polys(shape_hw, seed=7, num=3):
    h, w = shape_hw
    rng = np.random.default_rng(seed)
    leases = []
    for _ in range(num):
        cx, cy = rng.integers(int(0.2*w), int(0.8*w)), rng.integers(int(0.2*h), int(0.8*h))
        rx, ry = rng.integers(int(0.12*w), int(0.25*w)), rng.integers(int(0.10*h), int(0.22*h))
        theta = float(rng.random()*np.pi)
        pts = []
        for t in np.linspace(0, 2*np.pi, 60, endpoint=False):
            x = cx + rx*np.cos(t)*np.cos(theta) - ry*np.sin(t)*np.sin(theta)
            y = cy + rx*np.cos(t)*np.sin(theta) + ry*np.cos(t)*np.sin(theta)
            pts.append((float(x), float(y)))
        leases.append(Polygon(pts))
    return leases

def plot_geom(ax, geom, color, lw=2):
    if isinstance(geom, Polygon):
        xs, ys = geom.exterior.xy
        ax.plot(xs, ys, lw=lw, color=color)
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, lw=lw, color=color)

def pseudo_dem_from_rgb(rgb):
    # luminance + blur; normalized 0..1
    R,G,B = rgb[...,0], rgb[...,1], rgb[...,2]
    lum = 0.3*R + 0.59*G + 0.11*B
    dem = cv2.GaussianBlur(lum, (0,0), 3)
    return dem

def depth_volume_from_dem(dem, mask_u8, px_area_m2=100.0):
    # rim-floor approach (toy)
    blk = mask_u8>0
    if blk.sum()<1:
        return 0.0, 0.0
    dil = cv2.dilate(blk.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1)>0
    rim_zone = dil & (~blk)
    floor = float(np.nanmedian(dem[blk]))
    rim = float(np.nanpercentile(dem[rim_zone], 95)) if rim_zone.any() else floor
    depth = rim - floor
    depth_map = np.clip(rim - dem, 0, None)
    volume = float(depth_map[blk].sum() * px_area_m2)
    area = float(blk.sum() * px_area_m2)
    return depth, volume, area

def save_table_png(df, title, outpath):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 1 + 0.35*max(1,len(df))))
    ax.axis('off')
    ax.set_title(title, pad=10)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.4)
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close(fig)

def save_3d(prefix, dem, rgb, outdir):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    h, w = dem.shape
    step = max(1, min(h,w)//240)
    X, Y = np.meshgrid(np.arange(0,w,step), np.arange(0,h,step))
    Z = dem[::step, ::step]
    tex = rgb[::step, ::step]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, -Y, Z, facecolors=tex, linewidth=0, antialiased=False, shade=False)
    ax.set_title(f"{prefix}: 3D Visualization (prototype DEM)")
    ax.set_axis_off()
    out = os.path.join(outdir, f"{prefix}_3d.png")
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close(fig)

def process_image(im, prefix, outdir, lease_seed=7):
    ensure_dir(outdir)

    # save rgb
    rgb_path = os.path.join(outdir, f"{prefix}_rgb.png")
    im.save(rgb_path)

    rgb = np.array(im).astype("float32")/255.0
    ndvi_like, ndbi_like, tex = proxy_indices_from_rgb(rgb)
    mask_u8 = mining_mask(ndvi_like, ndbi_like, tex)

    # save mask png
    plt.figure(figsize=(6,6)); plt.imshow(mask_u8>0, cmap='gray'); plt.title(f"{prefix}: Mining mask (prototype)"); plt.axis('off')
    mask_path = os.path.join(outdir, f"{prefix}_mask.png")
    plt.tight_layout(); plt.savefig(mask_path, dpi=200); plt.close()

    # polygons
    polys, contours = mask_to_polygons(mask_u8)
    if not polys:
        print(f"[{prefix}] no polygons detected; still saved mask.")
        return

    # lease
    h, w = mask_u8.shape
    lease_polys = fake_lease_polys((h,w), seed=lease_seed, num=3)
    lease_union = unary_union(lease_polys)

    # split
    inside, outside = [], []
    for p in polys:
        in_part = p.intersection(lease_union)
        out_part = p.difference(lease_union)
        if not in_part.is_empty: inside.append(in_part)
        if not out_part.is_empty: outside.append(out_part)

    # overlay
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(im)
    for p in inside: plot_geom(ax, p, 'lime', lw=2)
    for p in outside: plot_geom(ax, p, 'red', lw=2)
    ax.set_title(f"{prefix}: Mining (green=inside lease, red=outside)"); ax.axis('off')
    overlay_path = os.path.join(outdir, f"{prefix}_overlay.png")
    plt.tight_layout(); plt.savefig(overlay_path, dpi=200); plt.close(fig)

    # pseudo DEM & stats per contour
    dem = pseudo_dem_from_rgb(rgb)
    px_area_m2 = 100.0  # assume 10m GSD
    rows = []
    for i, c in enumerate(contours, 1):
        blk = np.zeros(mask_u8.shape, dtype=np.uint8)
        cv2.drawContours(blk, [c], -1, 255, -1)
        if blk.sum() < 800:  # ignore tiny
            continue
        depth, volume, area = depth_volume_from_dem(dem, blk, px_area_m2=px_area_m2)
        rows.append([f"Block-{i}", round(area,1), round(depth*100,2), round(volume*100,1)])
    df = pd.DataFrame(rows, columns=["Block","Area_m2","MeanDepth_m","Volume_m3"])
    csv_path = os.path.join(outdir, f"{prefix}_table.csv"); df.to_csv(csv_path, index=False)
    table_png_path = os.path.join(outdir, f"{prefix}_table.png")
    save_table_png(df, f"{prefix}: Mining Analytics (prototype)", table_png_path)

    # 3D
    save_3d(prefix, dem, rgb, outdir)

    # hero
    try:
        mask_img = Image.open(mask_path)
        ov_img   = Image.open(overlay_path)
        tbl_img  = Image.open(table_png_path)
        W = max(mask_img.width, ov_img.width)
        left = Image.new("RGB", (W, mask_img.height + ov_img.height), (255,255,255))
        left.paste(mask_img, (0,0))
        left.paste(ov_img, (0,mask_img.height))
        hero_h = max(left.height, tbl_img.height)
        hero = Image.new("RGB", (left.width + tbl_img.width, hero_h), (255,255,255))
        hero.paste(left, (0,0))
        hero.paste(tbl_img, (left.width, 0))
        hero.save(os.path.join(outdir, f"{prefix}_hero.png"))
    except Exception as e:
        print(f"[{prefix}] could not build hero panel: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input JPG/PNG/TIF (visual)")
    ap.add_argument("--site", required=True, help="Prefix for outputs (e.g., rajasthan)")
    ap.add_argument("--outdir", default="assets", help="Output directory")
    ap.add_argument("--resize", type=int, default=1600, help="Resize width (keeps aspect). 0=original")
    ap.add_argument("--crop", nargs=4, type=int, help="x y w h crop box (pixels)")
    ap.add_argument("--grid", nargs=2, type=int, help="grid cols rows for automatic crops")
    ap.add_argument("--lease-seed", type=int, default=7, help="Random seed for fake lease polygons")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    base = load_rgb(args.image, resize=None if args.resize==0 else args.resize)

    crops = []
    if args.crop:
        x,y,w,h = args.crop
        crops = crop_box(base, x,y,w,h)
    elif args.grid:
        gx, gy = args.grid
        crops = crop_grid(base, gx, gy)
    else:
        crops = [("full", base)]

    for tag, cim in crops:
        prefix = f"{args.site}_{tag}" if tag else args.site
        process_image(cim, prefix, args.outdir, lease_seed=args.lease_seed)

if __name__ == "__main__":
    main()
