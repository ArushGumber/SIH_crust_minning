#!/usr/bin/env python3
"""
Download DEM files for Odisha and Rajasthan mining sites
Uses OpenTopography API for SRTM data
"""
import requests
import os

def download_dem(north, south, east, west, output_file):
    """
    Download SRTM 30m DEM from OpenTopography
    """
    url = "https://portal.opentopography.org/API/globaldem"
    
    params = {
        'demtype': 'SRTMGL1',  # SRTM 30m
        'south': south,
        'north': north,
        'west': west,
        'east': east,
        'outputFormat': 'GTiff',
        'API_Key': 'demoapikeyot2022'  # Public demo key
    }
    
    print(f"Downloading DEM for {output_file}...")
    print(f"Bounds: N:{north}, S:{south}, E:{east}, W:{west}")
    
    try:
        response = requests.get(url, params=params, stream=True)
        
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✓ Downloaded: {output_file}")
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# Odisha (Talcher coal mines area)
download_dem(
    north=21.00,
    south=20.90,
    east=85.30,
    west=85.15,
    output_file="assets/odisha_dem.tif"
)

# Rajasthan (Makrana marble quarries)
download_dem(
    north=27.10,
    south=27.00,
    east=74.80,
    west=74.65,
    output_file="assets/rajasthan_dem.tif"
)

print("\n✓ All DEMs downloaded to assets/")