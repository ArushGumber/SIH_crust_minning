import os
import io
import glob
import zipfile
import pandas as pd
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Mining Compliance Demo", layout="wide")

ASSETS = "assets"

st.markdown("### One-click mining compliance: detect, quantify, report")
st.caption("Prototype • RGB-only heuristic for mask • Fake lease polygons • DEM optional/pseudo")

# --- discover available prefixes from assets
def find_prefixes():
    paths = glob.glob(os.path.join(ASSETS, "*_overlay.png"))
    prefs = [os.path.basename(p).replace("_overlay.png", "") for p in paths]
    # nice ordering: odisha first, rajasthan next, then others
    prefs.sort(key=lambda x: (0 if "odisha" in x else 1 if "rajasthan" in x else 2, x))
    return prefs

prefixes = find_prefixes()
if not prefixes:
    st.error("No assets found. Run make_assets_rgb_only.py to generate outputs into ./assets/")
    st.stop()

# --- sidebar controls
sel = st.sidebar.selectbox("Select a site/crop", prefixes, index=0)
st.sidebar.write("Assets prefix:", f"`{sel}`")

def path(kind):
    return os.path.join(ASSETS, f"{sel}_{kind}.png")

def csv_path():
    return os.path.join(ASSETS, f"{sel}_table.csv")

# --- top info row
c1, c2 = st.columns([2,1])
with c1:
    st.info("Pipeline: Upload/choose lease → Detect mining (mask) → Split legal/illegal → Depth & volume from DEM → 2D/3D → Report")
with c2:
    st.success("This demo uses pre-generated assets from your script")

# --- show visuals
st.markdown("#### Visuals")
tab1, tab2, tab3 = st.tabs(["RGB", "Mask", "Overlay + 3D"])

with tab1:
    rgb = path("rgb")
    if os.path.exists(rgb):
        st.image(Image.open(rgb), caption="True Color (visual)")
    else:
        st.warning("RGB image not found.")

with tab2:
    mk = path("mask")
    if os.path.exists(mk):
        st.image(Image.open(mk), caption="Detected Mining Mask (prototype)")
    else:
        st.warning("Mask not found.")

with tab3:
    colA, colB = st.columns(2)
    ov = path("overlay")
    z3 = path("3d")
    with colA:
        if os.path.exists(ov):
            st.image(Image.open(ov), caption="Legal (green) vs Outside Lease (red)")
        else:
            st.warning("Overlay not found.")
    with colB:
        if os.path.exists(z3):
            st.image(Image.open(z3), caption="3D Terrain View (prototype)")
        else:
            st.warning("3D view not found.")

# --- analytics
st.markdown("#### Analytics")
tpng = path("table")
tcsv = csv_path()
if os.path.exists(tcsv):
    df = pd.read_csv(tcsv)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("Table CSV not found.")

if os.path.exists(tpng):
    st.image(Image.open(tpng), caption="Analytics Table (slide-ready)")

# --- hero panel (stitched)
hero = path("hero")
st.markdown("#### Slide-ready Hero")
if os.path.exists(hero):
    st.image(Image.open(hero), caption="Mask + Overlay + Table stitched")
else:
    st.info("Run the asset script again to create a hero panel.")

# --- download zip (report pack)
st.markdown("---")
def inmem_zip_for_prefix(prefix):
    # bundle the common assets for this prefix into a single zip
    files = [
        path("rgb"), path("mask"), path("overlay"),
        path("table"), path("3d"), hero, tcsv
    ]
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if os.path.exists(f):
                z.write(f, arcname=os.path.basename(f))
    mem.seek(0)
    return mem

zip_buf = inmem_zip_for_prefix(sel)
st.download_button(
    label="Download Report Pack (ZIP)",
    data=zip_buf,
    file_name=f"{sel}_report_pack.zip",
    mime="application/zip"
)

st.caption("Tip: Take screenshots from here for your PPT. For ablation, add your TOAM-YOLO panel as a separate image on the slide.")
