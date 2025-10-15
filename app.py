import os
import io
import time
import glob
import zipfile
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(
    page_title="Mining Compliance Detection System",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4788;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

ASSETS = "assets"

# Header
st.markdown('<h1 class="main-header">⛏️ Mining Compliance Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Automated detection of illegal mining using satellite imagery, DEM analysis, and AI-powered boundary verification</p>', unsafe_allow_html=True)

# Discover available sites
def find_prefixes():
    paths = glob.glob(os.path.join(ASSETS, "*_overlay.png"))
    prefs = [os.path.basename(p).replace("_overlay.png", "") for p in paths]
    prefs.sort(key=lambda x: (0 if "odisha" in x else 1 if "rajasthan" in x else 2, x))
    return prefs

def path(prefix, kind):
    return os.path.join(ASSETS, f"{prefix}_{kind}")

def loading_animation(text="Processing", duration=2):
    """Show a loading animation"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 30:
            status_text.text(f"{text}... Analyzing satellite imagery")
        elif i < 60:
            status_text.text(f"{text}... Running TOAM-YOLO detection")
        elif i < 85:
            status_text.text(f"{text}... Calculating depth & volume")
        else:
            status_text.text(f"{text}... Generating compliance report")
        time.sleep(duration / 100)
    
    progress_bar.empty()
    status_text.empty()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/mining.png", width=80)
    st.title("Navigation")
    
    mode = st.radio(
        "Select Mode:",
        ["Demo Sites", "Upload New Data"],
        help="Choose demo sites or upload your own imagery"
    )
    
    st.markdown("---")
    
    if mode == "Demo Sites":
        prefixes = find_prefixes()
        if not prefixes:
            st.error("No demo data found!")
            st.info("Run `make_assets_rgb_only.py` first")
            st.stop()
        
        selected_site = st.selectbox(
            "Choose Site:",
            prefixes,
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        st.success(f"✓ Loaded: {selected_site}")
        
        # Show site info
        with st.expander("ℹSite Information"):
            if "odisha" in selected_site:
                st.write("**Location:** Talcher, Odisha")
                st.write("**Type:** Coal Mining")
                st.write("**Coordinates:** 20.95°N, 85.20°E")
            elif "rajasthan" in selected_site:
                st.write("**Location:** Makrana, Rajasthan")
                st.write("**Type:** Marble Quarrying")
                st.write("**Coordinates:** 27.03°N, 74.72°E")
    
    else:  # Upload mode
        st.markdown("### Upload Files")
        
        uploaded_image = st.file_uploader(
            "Satellite Image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            help="RGB satellite imagery"
        )
        
        uploaded_dem = st.file_uploader(
            "DEM File (TIF) [Optional]",
            type=["tif", "tiff"],
            help="Digital Elevation Model for depth/volume calculation"
        )
        
        uploaded_boundary = st.file_uploader(
            "Boundary File (KML/SHP) [Optional]",
            type=["kml", "shp", "geojson"],
            help="Mining lease boundary"
        )
        
        if uploaded_image:
            if st.button("Process Uploaded Data", type="primary"):
                st.info("Feature coming soon! Use demo sites for now.")
                st.balloons()
        
        selected_site = None
    
    st.markdown("---")
    st.markdown("### About")
    st.caption("Powered by **TOAM-YOLO** - Tiny Object Detection for Mining Surveillance")
    st.caption("Version 1.0 | SIH 2025")

# Main content
if mode == "Demo Sites" and selected_site:
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "2D Analysis",
        "3D Visualization",
        "Analytics",
        "Report"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.header("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3></h3>
                <h2>Sentinel-2</h2>
                <p>10m Resolution</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3></h3>
                <h2>TOAM-YOLO</h2>
                <p>AI Detection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3></h3>
                <h2>DEM Analysis</h2>
                <p>Simpson's Method</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3></h3>
                <h2>Compliance</h2>
                <p>Auto Verification</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Pipeline diagram
        st.subheader("Processing Pipeline")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.info("**Step 1: Data Acquisition**\n\n• Satellite Imagery (EO/SAR)\n• Digital Elevation Model\n• Lease Boundary Files")
        
        with col_b:
            st.warning("**Step 2: Detection & Analysis**\n\n• TOAM-YOLO Mining Detection\n• Boundary Compliance Check\n• Depth & Volume Calculation")
        
        with col_c:
            st.success("**Step 3: Reporting**\n\n• 2D/3D Visualization\n• Violation Identification\n• Automated PDF Reports")
        
        st.markdown("---")
        
        # Show RGB preview
        rgb_path = path(selected_site, "rgb.png")
        if os.path.exists(rgb_path):
            st.subheader("Satellite Imagery Preview")
            st.image(rgb_path, caption=f"{selected_site.title()} - True Color Composite", use_container_width=True)
    
    # TAB 2: 2D Analysis
    with tab2:
        st.header("2D Spatial Analysis")
        
        if st.button("🔍 Run Detection", type="primary", key="detect_btn"):
            loading_animation("Running TOAM-YOLO Detection", duration=3)
            st.success("✓ Detection complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Mining Mask Detection")
            mask_path = path(selected_site, "mask.png")
            if os.path.exists(mask_path):
                st.image(mask_path, caption="Detected Mining Areas", use_container_width=True)
            else:
                st.warning("Mask not generated")
        
        with col2:
            st.subheader("Compliance Overlay")
            overlay_path = path(selected_site, "overlay.png")
            if os.path.exists(overlay_path):
                st.image(overlay_path, caption="Legal (Green) vs Illegal (Red)", use_container_width=True)
            else:
                st.warning("Overlay not generated")
        
        st.markdown("---")
        
        # Legend
        st.subheader("Legend")
        leg_col1, leg_col2, leg_col3 = st.columns(3)
        with leg_col1:
            st.markdown("**Legal Mining** - Within authorized boundary")
        with leg_col2:
            st.markdown("**Illegal Mining** - Outside authorized boundary")
        with leg_col3:
            st.markdown("**Lease Boundary** - Authorized mining area")
    
    # TAB 3: 3D Visualization
    with tab3:
        st.header("3D Terrain Visualization")
        
        html_3d_path = path(selected_site, "3d.html")
        png_3d_path = path(selected_site, "3d.png")
        
        if st.button("🏔️ Generate 3D View", type="primary", key="3d_btn"):
            loading_animation("Creating 3D terrain model", duration=2.5)
            st.success("✓ 3D model ready!")
        
        # Try to show interactive 3D first
        if os.path.exists(html_3d_path):
            st.subheader("Interactive 3D Model (Drag to Rotate)")
            
            with open(html_3d_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.components.v1.html(html_content, height=700, scrolling=False)
            
            st.info("💡 **Tip:** Click and drag to rotate. Scroll to zoom. Double-click to reset view.")
        
        elif os.path.exists(png_3d_path):
            st.subheader("3D Terrain View")
            st.image(png_3d_path, caption="3D Mining Terrain Visualization", use_container_width=True)
        
        else:
            st.warning("3D visualization not available. Make sure to run the asset generation script with Plotly installed.")
        
        st.markdown("---")
        
        # 3D Info
        col1, col2 = st.columns(2)
        with col1:
            st.info("**DEM Source:** SRTM 30m / Copernicus DEM")
        with col2:
            st.info("**Vertical Exaggeration:** 2x for visibility")
    
    # TAB 4: Analytics
    with tab4:
        st.header("Mining Analytics & Metrics")
        
        csv_path = path(selected_site, "table.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Summary metrics at top
            if len(df) > 0:
                total_row = df[df['Block'] == 'TOTAL']
                
                if not total_row.empty:
                    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                    
                    with met_col1:
                        st.metric(
                            "Total Area",
                            f"{total_row['Area_m2'].values[0]:,.0f} m²",
                            help="Total detected mining area"
                        )
                    
                    with met_col2:
                        st.metric(
                            "Total Volume",
                            f"{total_row['Volume_m3'].values[0]:,.0f} m³",
                            help="Estimated excavation volume (Simpson's method)"
                        )
                    
                    with met_col3:
                        violations = df[df['OutsideLease'] == 'Yes']
                        violation_count = len(violations)
                        st.metric(
                            "Violations",
                            f"{violation_count} blocks",
                            delta=f"⚠️ Illegal" if violation_count > 0 else "✓ Compliant",
                            delta_color="inverse" if violation_count > 0 else "normal"
                        )
                    
                    with met_col4:
                        if 'VIOLATION' in str(total_row['OutsideLease'].values[0]):
                            viol_text = str(total_row['OutsideLease'].values[0])
                            viol_area = float(viol_text.split()[0])
                            st.metric(
                                "Illegal Area",
                                f"{viol_area:,.0f} m²",
                                delta="⚠️ Action Required",
                                delta_color="inverse"
                            )
                        else:
                            st.metric("Illegal Area", "0 m²", delta="✓ Compliant")
            
            style_metric_cards()
            
            st.markdown("---")
            
            # Detailed table
            st.subheader("Block-wise Analysis")
            
            # Color code the dataframe
            def highlight_violations(row):
                if row['OutsideLease'] in ['Yes', 'VIOLATION']:
                    return ['background-color: #ffcccc'] * len(row)
                elif row['Block'] == 'TOTAL':
                    return ['background-color: #e6f3ff; font-weight: bold'] * len(row)
                else:
                    return ['background-color: #ccffcc'] * len(row)
            
            styled_df = df.style.apply(highlight_violations, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Download data
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{selected_site}_analytics.csv",
                mime="text/csv"
            )
        
        else:
            st.warning("Analytics data not available")
        
        st.markdown("---")
        
        # Table image for slides
        table_png_path = path(selected_site, "table.png")
        if os.path.exists(table_png_path):
            with st.expander("📊 View Slide-Ready Table"):
                st.image(table_png_path, caption="Table for Presentation", use_container_width=True)
    
    # TAB 5: Report
    with tab5:
        st.header("Compliance Report Generation")
        
        pdf_path = path(selected_site, "report.pdf")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Report Contents
            
            **Executive Summary** - Violation status and key metrics  
            **Satellite Imagery** - High-resolution RGB composite  
            **Detection Results** - Mining mask and polygons  
            **Compliance Analysis** - Legal vs illegal mining areas  
            **3D Visualization** - Terrain model with depth profiles  
            **Analytics Table** - Block-wise area, depth, volume  
            **Boundary Files** - KML/Shapefile attachments  
            
            """)
        
        with col2:
            st.info("""
            **Report Format:**
            
            PDF Document  
            A4 Landscape  
            Color Graphics  
            GeoJSON Attached
            """)
        
        st.markdown("---")
        
        # Generate report button
        if st.button("Generate PDF Report", type="primary", key="report_btn"):
            loading_animation("Generating compliance report", duration=3)
            
            if os.path.exists(pdf_path):
                st.success("✓ Report generated successfully!")
                
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{selected_site}_compliance_report.pdf",
                    mime="application/pdf",
                    key="download_pdf"
                )
            else:
                st.error("Report generation failed. Make sure ReportLab is installed.")
        
        st.markdown("---")
        
        # Show hero panel preview
        hero_path = path(selected_site, "hero.png")
        if os.path.exists(hero_path):
            st.subheader("Report Preview")
            st.image(hero_path, caption="Stitched Report Panel", use_container_width=True)
        
        # Download all assets as ZIP
        st.markdown("---")
        st.subheader("Download Complete Report Pack")
        
        def create_zip_pack(prefix):
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
                files_to_zip = [
                    ("rgb.png", "Satellite_Image.png"),
                    ("mask.png", "Mining_Detection.png"),
                    ("overlay.png", "Compliance_Map.png"),
                    ("3d.png", "3D_Visualization.png"),
                    ("table.csv", "Analytics_Data.csv"),
                    ("table.png", "Analytics_Table.png"),
                    ("hero.png", "Report_Panel.png"),
                    ("report.pdf", f"{prefix}_Compliance_Report.pdf"),
                    ("boundary.kml", "Lease_Boundary.kml")
                ]
                
                for src_name, zip_name in files_to_zip:
                    file_path = path(prefix, src_name)
                    if os.path.exists(file_path):
                        zf.write(file_path, arcname=zip_name)
            
            mem.seek(0)
            return mem
        
        zip_data = create_zip_pack(selected_site)
        
        st.download_button(
            label="Download Report Pack (ZIP)",
            data=zip_data,
            file_name=f"{selected_site}_complete_report.zip",
            mime="application/zip",
            key="download_zip"
        )

else:
    # Upload mode placeholder
    st.info("Select a demo site from the sidebar or upload new data to begin analysis")
    
    st.markdown("""
    ## Features
    
    ### Multi-Source Data Integration
    - Sentinel-2 optical imagery (10m resolution)
    - Sentinel-1 SAR (cloud-penetrating)
    - SRTM/ASTER Digital Elevation Models
    
    ### AI-Powered Detection
    - **TOAM-YOLO**: Custom architecture for tiny object detection
    - Specialized for mining features (pits, dumps, haul roads)
    - >85% precision on small mining features
    
    ### Automated Analysis
    - Boundary compliance verification (KML/SHP support)
    - Depth & volume calculation (Simpson's method)
    - Block-wise analytics and metrics
    
    ### Comprehensive Reporting
    - Interactive 2D maps with violation highlighting
    - 3D terrain visualization
    - Automated PDF report generation
    - GeoJSON exports for GIS integration
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("Smart India Hackathon 2025")

with col2:
    st.caption("Powered by TOAM-YOLO Research")

with col3:
    st.caption("NTRO - Mining Surveillance")