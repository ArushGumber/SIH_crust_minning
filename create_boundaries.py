#!/usr/bin/env python3
"""
Create sample KML/SHP boundary files for demo
"""
import os
from shapely.geometry import Polygon, mapping
import json

def create_sample_kml(coords_list, output_file, name="Mining Lease"):
    """Create a KML file with lease boundaries"""
    
    kml_template = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <Style id="leaseBoundary">
      <LineStyle>
        <color>ff00ffff</color>
        <width>3</width>
      </LineStyle>
      <PolyStyle>
        <color>3300ff00</color>
      </PolyStyle>
    </Style>
{placemarks}
  </Document>
</kml>"""

    placemark_template = """    <Placemark>
      <name>Block {i}</name>
      <styleUrl>#leaseBoundary</styleUrl>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
{coords}
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>"""

    placemarks = []
    for i, coords in enumerate(coords_list, 1):
        coord_str = "\n".join([f"              {lon},{lat},0" for lon, lat in coords])
        placemarks.append(placemark_template.format(i=i, coords=coord_str))
    
    kml_content = kml_template.format(
        name=name,
        placemarks="\n".join(placemarks)
    )
    
    with open(output_file, 'w') as f:
        f.write(kml_content)
    
    print(f"✓ Created KML: {output_file}")

def create_sample_geojson(coords_list, output_file, name="Mining Lease"):
    """Create a GeoJSON file (works with geopandas)"""
    
    features = []
    for i, coords in enumerate(coords_list, 1):
        poly = Polygon(coords)
        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Block {i}",
                "lease_type": "Mining",
                "authorized": True
            },
            "geometry": mapping(poly)
        })
    
    geojson = {
        "type": "FeatureCollection",
        "name": name,
        "features": features
    }
    
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"✓ Created GeoJSON: {output_file}")

# Sample boundaries (lat/lon coordinates)
# These are realistic coordinates for demo purposes

# Odisha boundary (near Talcher)
odisha_boundary = [
    [
        (85.20, 20.95),
        (85.25, 20.95),
        (85.25, 20.92),
        (85.20, 20.92),
        (85.20, 20.95)
    ],
    [
        (85.22, 20.96),
        (85.24, 20.96),
        (85.24, 20.955),
        (85.22, 20.955),
        (85.22, 20.96)
    ]
]

# Rajasthan boundary (near Makrana)
rajasthan_boundary = [
    [
        (74.70, 27.04),
        (74.75, 27.04),
        (74.75, 27.01),
        (74.70, 27.01),
        (74.70, 27.04)
    ],
    [
        (74.72, 27.05),
        (74.74, 27.05),
        (74.74, 27.045),
        (74.72, 27.045),
        (74.72, 27.05)
    ]
]

os.makedirs("assets", exist_ok=True)

# Create KML files
create_sample_kml(odisha_boundary, "assets/odisha_boundary.kml", "Odisha Mining Lease")
create_sample_kml(rajasthan_boundary, "assets/rajasthan_boundary.kml", "Rajasthan Mining Lease")

# Create GeoJSON files
create_sample_geojson(odisha_boundary, "assets/odisha_boundary.geojson", "Odisha Mining Lease")
create_sample_geojson(rajasthan_boundary, "assets/rajasthan_boundary.geojson", "Rajasthan Mining Lease")

print("\n✓ All boundary files created in assets/")
print("You can now use these with --boundary flag in make_assets script")